"use client";

import { LoaderCircle } from "lucide-react";
import { startTransition, useCallback, useEffect, useRef, useState } from "react";

import { apiFetchJson, wsUrl } from "@/lib/api";
import type { RecruiterFeedEvent, RecruiterScorecardSummary, ScorecardModel } from "@/lib/types";

import { InjectQuestionModal } from "./recruiter/InjectQuestionModal";
import { LiveFeed } from "./recruiter/LiveFeed";
import { ScorecardView } from "./ScorecardView";

type RecruiterDashboardProps = {
  jobId: string;
};

function recommendationClass(action: string): string {
  if (action === "advance") return "text-emerald-300";
  if (action === "hold") return "text-amber-200";
  return "text-rose-300";
}

export function RecruiterDashboard({ jobId }: RecruiterDashboardProps) {
  const wsRef = useRef<WebSocket | null>(null);
  const selectedIdRef = useRef("");
  const [summaries, setSummaries] = useState<RecruiterScorecardSummary[]>([]);
  const [scorecards, setScorecards] = useState<Record<string, ScorecardModel>>({});
  const [selectedId, setSelectedId] = useState<string>("");
  const [feedOpen, setFeedOpen] = useState(true);
  const [feedEvents, setFeedEvents] = useState<RecruiterFeedEvent[]>([]);
  const [activeSessions, setActiveSessions] = useState<string[]>([]);
  const [loading, setLoading] = useState(true);
  const [finalizing, setFinalizing] = useState(false);
  const [loadError, setLoadError] = useState("");
  const [injectModalOpen, setInjectModalOpen] = useState(false);
  const [injectQuestion, setInjectQuestion] = useState("");

  const selectedScorecard = selectedId ? scorecards[selectedId] ?? null : null;

  function sessionIdFrom(event: RecruiterFeedEvent): string {
    return typeof event.session_id === "string" ? event.session_id : "";
  }

  const loadScorecards = useCallback(async (preferredCandidateId?: string) => {
    setLoading(true);
    setLoadError("");
    try {
      const list = await apiFetchJson<RecruiterScorecardSummary[]>(`/recruiter/${jobId}/scorecards`);
      const details = await Promise.all(
        list.map((item) => apiFetchJson<ScorecardModel>(`/recruiter/${jobId}/scorecard/${item.candidate_id}`)),
      );
      const nextCards = Object.fromEntries(details.map((item) => [item.candidate_id, item]));
      const nextList = list.map((item) => ({
        ...item,
        candidate_name: nextCards[item.candidate_id]?.candidate_name ?? item.candidate_id,
        session_id: nextCards[item.candidate_id]?.session_id,
      }));
      startTransition(() => {
        setScorecards(nextCards);
        setSummaries(nextList);
        setSelectedId(preferredCandidateId || selectedIdRef.current || nextList[0]?.candidate_id || "");
      });
    } catch (error) {
      setLoadError(error instanceof Error ? error.message : "Unable to load scorecards.");
    } finally {
      setLoading(false);
    }
  }, [jobId]);

  useEffect(() => {
    void loadScorecards();
  }, [loadScorecards]);

  useEffect(() => {
    selectedIdRef.current = selectedId;
  }, [selectedId]);

  useEffect(() => {
    const socket = new WebSocket(wsUrl(`/ws/recruiter/${jobId}`));
    wsRef.current = socket;
    socket.onmessage = (event) => {
      const payload = JSON.parse(event.data) as RecruiterFeedEvent;
      const sessionId = sessionIdFrom(payload);
      setFeedEvents((current) => [payload, ...current].slice(0, 12));
      if (payload.event === "session_started" && sessionId) {
        setActiveSessions((current) => (current.includes(sessionId) ? current : [sessionId, ...current]));
      }
      if (["session_completed", "scorecard_ready", "scorecard_generated"].includes(payload.event) && sessionId) {
        setActiveSessions((current) => current.filter((item) => item !== sessionId));
      }
      if (["session_completed", "scorecard_ready", "scorecard_generated", "shortlist_ready"].includes(payload.event)) {
        void loadScorecards(selectedIdRef.current);
      }
    };
    return () => socket.close();
  }, [jobId, loadScorecards]);

  async function finalizeShortlist() {
    if (!window.confirm(`This will lock all ${summaries.length} scorecards. Continue?`)) return;
    setFinalizing(true);
    try {
      await apiFetchJson(`/recruiter/${jobId}/finalize`, { method: "POST" });
      await loadScorecards(selectedId);
    } finally {
      setFinalizing(false);
    }
  }

  function injectFollowUp() {
    if (!selectedScorecard || !injectQuestion.trim() || wsRef.current?.readyState !== WebSocket.OPEN) return;
    wsRef.current.send(
      JSON.stringify({
        type: "hitl_inject_question",
        data: { session_id: selectedScorecard.session_id, question_text: injectQuestion.trim() },
      }),
    );
    setInjectQuestion("");
    setInjectModalOpen(false);
  }

  return (
    <section className="space-y-6">
      <div className="flex items-center justify-between gap-4 rounded-[2rem] border border-slate-800 bg-slate-950/85 p-6 text-slate-50">
        <div>
          <p className="text-xs uppercase tracking-[0.28em] text-slate-400">Recruiter Review</p>
          <h1 className="mt-3 text-3xl font-semibold">Job {jobId}</h1>
        </div>
        <button
          className="rounded-full bg-brand-600 px-5 py-3 text-sm font-medium text-white disabled:opacity-50"
          disabled={finalizing || summaries.length === 0}
          onClick={() => void finalizeShortlist()}
          type="button"
        >
          {finalizing ? "Publishing..." : "Publish shortlist"}
        </button>
      </div>

      <div className="grid gap-6 xl:grid-cols-[0.9fr_1.1fr]">
        <section className="rounded-[2rem] border border-slate-800 bg-slate-950/85 p-5 text-slate-50">
          <div className="grid grid-cols-[0.55fr_1.6fr_0.9fr_1fr_0.8fr_0.8fr] gap-3 border-b border-slate-800 px-3 pb-3 text-xs uppercase tracking-[0.2em] text-slate-400">
            <p>Rank</p><p>Name</p><p>Score</p><p>Recommendation</p><p>Bias</p><p>Integrity</p>
          </div>
          <div className="mt-3 space-y-2">
            {loading ? <p className="px-3 py-10 text-sm text-slate-400">Loading ranked candidates...</p> : null}
            {!loading && summaries.map((item) => (
              <button
                key={item.candidate_id}
                className={`grid w-full grid-cols-[0.55fr_1.6fr_0.9fr_1fr_0.8fr_0.8fr] gap-3 rounded-3xl px-3 py-4 text-left ${selectedId === item.candidate_id ? "bg-brand-500/15" : "bg-slate-900/80"}`}
                onClick={() => setSelectedId(item.candidate_id)}
                type="button"
              >
                <p>{item.final_rank ?? "-"}</p>
                <p className="font-medium">{item.candidate_name}</p>
                <p>{(item.weighted_total * 100).toFixed(1)}%</p>
                <p className={recommendationClass(item.recommended_action)}>{item.recommended_action}</p>
                <p>{item.bias_flag_count}</p>
                <p>{item.integrity_flag_count}</p>
              </button>
            ))}
            {!loading && summaries.length === 0 ? <p className="px-3 py-10 text-sm text-slate-400">No scorecards are available yet.</p> : null}
          </div>
        </section>

        <section className="space-y-6">
          <LiveFeed
            activeSessions={activeSessions}
            feedEvents={feedEvents}
            feedOpen={feedOpen}
            onInject={() => setInjectModalOpen(true)}
            onToggle={() => setFeedOpen((open) => !open)}
            selectedLocked={!selectedScorecard || selectedScorecard.is_finalized}
            summaries={summaries}
          />

          {loadError ? <p className="rounded-3xl border border-rose-500/20 bg-rose-500/10 p-4 text-sm text-rose-100">{loadError}</p> : null}
          {!selectedScorecard && loading ? <div className="flex justify-center rounded-[2rem] border border-slate-200/10 bg-white/95 p-10"><LoaderCircle className="h-6 w-6 animate-spin text-slate-400" /></div> : null}
          {selectedScorecard ? <ScorecardView onRefresh={() => loadScorecards(selectedScorecard.candidate_id)} scorecard={selectedScorecard} /> : null}
        </section>
      </div>

      <InjectQuestionModal
        onChange={setInjectQuestion}
        onClose={() => setInjectModalOpen(false)}
        onSend={injectFollowUp}
        open={injectModalOpen}
        question={injectQuestion}
      />
    </section>
  );
}
