"use client";

import { LiveKitRoom } from "@livekit/components-react";
import { FormEvent, useEffect, useRef, useState } from "react";

import { apiFetchJson, wsUrl } from "@/lib/api";
import { CompletionPanel, StatusPanel, TranscriptPanel, WaitingRoomPanel } from "@/components/interview/RoomPanels";
import { asRealtimeState, FocusTracker, transcriptText, type MicTone, VoiceBridge } from "@/components/interview/RealtimeSupport";
import type { InterviewRealtimeState, InterviewSessionInfo, LiveKitTokenResponse } from "@/lib/types";

type InterviewRoomProps = {
  sessionId: string;
};

export function InterviewRoom({ sessionId }: InterviewRoomProps) {
  const wsRef = useRef<WebSocket | null>(null);
  const [sessionInfo, setSessionInfo] = useState<InterviewSessionInfo | null>(null);
  const [tokenBundle, setTokenBundle] = useState<LiveKitTokenResponse | null>(null);
  const [transportMode, setTransportMode] = useState<"voice" | "text">("voice");
  const [voiceTone, setVoiceTone] = useState<MicTone>("grey");
  const [realtime, setRealtime] = useState<InterviewRealtimeState>({ transcript: [] });
  const [draft, setDraft] = useState("");
  const [loading, setLoading] = useState(true);
  const [starting, setStarting] = useState(false);
  const [error, setError] = useState("");
  const [streamReady, setStreamReady] = useState(false);
  const [completionSent, setCompletionSent] = useState(false);

  const started = Boolean(tokenBundle) || transportMode === "text";
  const completed = sessionInfo?.status === "completed" || Boolean(realtime.interview_complete);
  const micTone = realtime.answer_is_shallow ? "amber" : voiceTone;
  const maxQuestions = realtime.max_questions ?? sessionInfo?.max_questions ?? 10;
  const questionCount =
    realtime.question_index ??
    realtime.transcript.filter((turn) => turn.role.toLowerCase() === "agent").length;

  useEffect(() => {
    async function loadSession() {
      try {
        setSessionInfo(await apiFetchJson<InterviewSessionInfo>(`/sessions/${sessionId}/info`));
      } catch (loadError) {
        setError(loadError instanceof Error ? loadError.message : "Unable to load session.");
      } finally {
        setLoading(false);
      }
    }
    void loadSession();
  }, [sessionId]);

  useEffect(() => {
    if (!started || completed) return;
    const socket = new WebSocket(wsUrl(`/ws/interviews/${sessionId}`));
    wsRef.current = socket;
    socket.onopen = () => setStreamReady(true);
    socket.onclose = () => setStreamReady(false);
    socket.onmessage = (event) => {
      const nextState = asRealtimeState(JSON.parse(event.data) as unknown);
      setRealtime((current) => ({
        ...current,
        ...nextState,
        transcript: nextState.transcript ?? current.transcript,
      }));
    };
    return () => {
      socket.close();
      wsRef.current = null;
      setStreamReady(false);
    };
  }, [completed, sessionId, started]);

  useEffect(() => {
    if (!realtime.interview_complete || completionSent) return;
    setCompletionSent(true);
    void apiFetchJson(`/sessions/${sessionId}/complete`, {
      method: "POST",
      body: JSON.stringify({ transcript: transcriptText(realtime.transcript) }),
    }).catch(() => undefined);
  }, [completionSent, realtime.interview_complete, realtime.transcript, sessionId]);

  async function startInterview() {
    if (!sessionInfo) return;
    setStarting(true);
    setError("");
    try {
      const query = new URLSearchParams({
        session_id: sessionId,
        candidate_id: sessionInfo.candidate_id,
      });
      const bundle = await apiFetchJson<LiveKitTokenResponse>(`/livekit/token?${query.toString()}`, {
        method: "POST",
      });
      setTokenBundle(bundle);
      setTransportMode("voice");
      setSessionInfo((current) => (current ? { ...current, status: "in_progress" } : current));
    } catch (startError) {
      setError(startError instanceof Error ? startError.message : "Unable to start interview.");
    } finally {
      setStarting(false);
    }
  }

  function sendTextAnswer(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!draft.trim() || wsRef.current?.readyState !== WebSocket.OPEN) return;
    wsRef.current.send(draft.trim());
    setDraft("");
  }

  if (loading) {
    return <div className="rounded-[2rem] border border-slate-800 bg-slate-950/80 p-8 text-slate-200">Loading interview session...</div>;
  }

  if (!sessionInfo) {
    return <div className="rounded-[2rem] border border-rose-500/20 bg-rose-500/10 p-8 text-rose-100">{error || "Session not found."}</div>;
  }

  if (completed) {
    return <CompletionPanel candidateName={sessionInfo.candidate_name} roleTitle={sessionInfo.role_title} />;
  }

  return (
    <section className="space-y-6">
      {started && !completed ? <FocusTracker sessionId={sessionId} wsRef={wsRef} /> : null}
      {tokenBundle && transportMode === "voice" ? (
        <LiveKitRoom audio connect serverUrl={tokenBundle.server_url} token={tokenBundle.token} className="contents">
          <VoiceBridge setVoiceTone={setVoiceTone} />
        </LiveKitRoom>
      ) : null}

      <header className="rounded-[2.5rem] border border-slate-800 bg-slate-950/85 p-8 text-slate-50">
        <p className="text-xs uppercase tracking-[0.28em] text-slate-400">{sessionInfo.role_title}</p>
        <h1 className="mt-3 text-4xl font-semibold">{sessionInfo.candidate_name}</h1>
      </header>

      {!started ? (
        <WaitingRoomPanel
          disabled={sessionInfo.status === "error"}
          error={error}
          onStart={() => void startInterview()}
          starting={starting}
        />
      ) : (
        <div className="grid gap-6 xl:grid-cols-[0.9fr_1.1fr]">
          <StatusPanel
            maxQuestions={maxQuestions}
            micTone={micTone}
            onSwitchToText={() => setTransportMode("text")}
            questionCount={questionCount}
          />
          <TranscriptPanel
            draft={draft}
            onDraftChange={setDraft}
            onSubmit={sendTextAnswer}
            streamReady={streamReady}
            transcript={realtime.transcript}
            transportMode={transportMode}
          />
        </div>
      )}
    </section>
  );
}
