"use client";

import { ChevronDown, Radio } from "lucide-react";

import type { RecruiterFeedEvent, RecruiterScorecardSummary } from "@/lib/types";

type LiveFeedProps = {
  activeSessions: string[];
  feedEvents: RecruiterFeedEvent[];
  feedOpen: boolean;
  summaries: RecruiterScorecardSummary[];
  selectedLocked: boolean;
  onInject: () => void;
  onToggle: () => void;
};

function sessionIdFrom(event: RecruiterFeedEvent): string {
  return typeof event.session_id === "string" ? event.session_id : "";
}

export function LiveFeed({
  activeSessions,
  feedEvents,
  feedOpen,
  summaries,
  selectedLocked,
  onInject,
  onToggle,
}: LiveFeedProps) {
  return (
    <>
      <div className="flex items-center justify-between gap-4 rounded-[2rem] border border-slate-200/10 bg-white/95 p-4 text-slate-950 shadow-2xl shadow-slate-950/10">
        <button className="inline-flex items-center gap-2 text-sm font-medium" onClick={onToggle} type="button">
          <ChevronDown className={`h-4 w-4 transition ${feedOpen ? "" : "-rotate-90"}`} />
          Live feed
        </button>
        <button
          className="rounded-full bg-slate-950 px-4 py-2 text-sm font-medium text-white disabled:opacity-50"
          disabled={selectedLocked}
          onClick={onInject}
          type="button"
        >
          Inject question
        </button>
      </div>

      {feedOpen ? (
        <section className="rounded-[2rem] border border-slate-200/10 bg-white/95 p-5 text-slate-950 shadow-2xl shadow-slate-950/10">
          <div className="flex flex-wrap gap-3">
            {activeSessions.length === 0 ? <p className="text-sm text-slate-500">No interviews are currently in progress.</p> : null}
            {activeSessions.map((sessionId) => (
              <div key={sessionId} className="inline-flex items-center gap-2 rounded-full bg-emerald-500/10 px-3 py-2 text-sm text-emerald-700">
                <Radio className="h-4 w-4 animate-pulse" />
                {summaries.find((item) => item.session_id === sessionId)?.candidate_name ?? sessionId}
              </div>
            ))}
          </div>
          <div className="mt-4 space-y-3">
            {feedEvents.map((event, index) => (
              <article key={`${event.event}-${index}`} className="rounded-3xl border border-slate-200 bg-slate-50 px-4 py-3">
                <p className="text-xs uppercase tracking-[0.2em] text-slate-500">{event.event}</p>
                <p className="mt-2 text-sm text-slate-700">{event.detail || sessionIdFrom(event) || "Job event received."}</p>
              </article>
            ))}
          </div>
        </section>
      ) : null}
    </>
  );
}
