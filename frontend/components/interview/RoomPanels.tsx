"use client";

import { LoaderCircle, MessageSquareText, Mic } from "lucide-react";
import { FormEvent } from "react";

import type { MicTone } from "./RealtimeSupport";
import type { TranscriptTurn } from "@/lib/types";

export function CompletionPanel({
  candidateName,
  roleTitle,
}: {
  candidateName: string;
  roleTitle: string;
}) {
  return (
    <section className="rounded-[2.5rem] border border-emerald-500/20 bg-slate-950/85 p-10 text-center text-slate-50">
      <p className="text-xs uppercase tracking-[0.28em] text-emerald-300">Interview Complete</p>
      <h1 className="mt-4 text-4xl font-semibold">Thank you, {candidateName}.</h1>
      <p className="mt-3 text-slate-300">Your interview for {roleTitle} has been submitted for recruiter review.</p>
    </section>
  );
}

export function WaitingRoomPanel({
  disabled,
  error,
  starting,
  onStart,
}: {
  disabled: boolean;
  error: string;
  starting: boolean;
  onStart: () => void;
}) {
  return (
    <article className="rounded-[2.5rem] border border-slate-800 bg-white/95 p-8 text-slate-950 shadow-2xl shadow-slate-950/10">
      <p className="text-sm uppercase tracking-[0.24em] text-slate-500">Waiting Room</p>
      <h2 className="mt-4 text-3xl font-semibold">Your interview is ready when you are.</h2>
      <p className="mt-3 max-w-2xl text-slate-600">When you start, the interview room will connect audio, activate lightweight integrity tracking, and begin streaming the transcript.</p>
      <button
        className="mt-8 rounded-full bg-brand-600 px-6 py-3 text-sm font-medium text-white disabled:opacity-50"
        disabled={starting || disabled}
        onClick={onStart}
        type="button"
      >
        {starting ? "Starting..." : "Start Interview"}
      </button>
      {error ? <p className="mt-4 text-sm text-rose-600">{error}</p> : null}
    </article>
  );
}

export function StatusPanel({
  maxQuestions,
  micTone,
  questionCount,
  onSwitchToText,
}: {
  maxQuestions: number;
  micTone: MicTone;
  questionCount: number;
  onSwitchToText: () => void;
}) {
  return (
    <article className="rounded-[2.5rem] border border-slate-800 bg-slate-950/85 p-8 text-slate-50">
      <p className="text-xs uppercase tracking-[0.28em] text-slate-400">Live Interview</p>
      <div className="mt-8 flex flex-col items-center text-center">
        <div
          className={`flex h-40 w-40 items-center justify-center rounded-full border-2 ${
            micTone === "green" ? "border-emerald-400 bg-emerald-500/10 animate-pulse" : ""
          } ${micTone === "grey" ? "border-slate-500 bg-slate-500/10" : ""} ${micTone === "amber" ? "border-amber-400 bg-amber-500/10" : ""}`}
        >
          <Mic className="h-16 w-16" />
        </div>
        <p className="mt-6 text-lg font-medium">
          {micTone === "green" ? "Your turn to respond" : micTone === "grey" ? "Listen to the prompt" : "Follow-up incoming"}
        </p>
        <p className="mt-2 text-sm text-slate-300">Question {Math.max(questionCount, 1)} of ~{maxQuestions}</p>
        <button
          className="mt-6 inline-flex items-center gap-2 text-sm text-brand-200 underline underline-offset-4"
          onClick={onSwitchToText}
          type="button"
        >
          <MessageSquareText className="h-4 w-4" />
          Having trouble with audio?
        </button>
      </div>
    </article>
  );
}

export function TranscriptPanel({
  draft,
  streamReady,
  transcript,
  transportMode,
  onDraftChange,
  onSubmit,
}: {
  draft: string;
  streamReady: boolean;
  transcript: TranscriptTurn[];
  transportMode: "voice" | "text";
  onDraftChange: (value: string) => void;
  onSubmit: (event: FormEvent<HTMLFormElement>) => void;
}) {
  return (
    <article className="rounded-[2.5rem] border border-white/20 bg-white/95 p-6 text-slate-950 shadow-2xl shadow-slate-950/10">
      <div className="flex items-center justify-between gap-4">
        <div>
          <p className="text-xs uppercase tracking-[0.28em] text-slate-500">Transcript Feed</p>
          <p className="mt-2 text-sm text-slate-500">{streamReady ? "Realtime stream connected" : "Connecting to interview stream"}</p>
        </div>
        {!streamReady ? <LoaderCircle className="h-5 w-5 animate-spin text-slate-400" /> : null}
      </div>

      <div className="mt-6 space-y-3">
        {transcript.slice(-4).map((turn, index) => (
          <article key={`${turn.role}-${index}`} className="rounded-3xl border border-slate-200 bg-slate-50 px-4 py-3">
            <p className="text-xs uppercase tracking-[0.22em] text-slate-500">{turn.role}</p>
            <p className="mt-2 text-sm leading-6 text-slate-700">{turn.content}</p>
          </article>
        ))}
        {transcript.length === 0 ? <p className="rounded-3xl border border-dashed border-slate-300 px-4 py-8 text-sm text-slate-500">The interview transcript will appear here once the websocket connects.</p> : null}
      </div>

      {transportMode === "text" ? (
        <form className="mt-5 space-y-3" onSubmit={onSubmit}>
          <textarea
            value={draft}
            onChange={(event) => onDraftChange(event.target.value)}
            placeholder="Type your answer here if audio is unavailable."
            rows={5}
            className="w-full rounded-3xl border border-slate-300 px-4 py-3 text-sm outline-none"
          />
          <button className="rounded-full bg-slate-950 px-5 py-3 text-sm font-medium text-white disabled:opacity-50" disabled={!streamReady || !draft.trim()} type="submit">
            Send Answer
          </button>
        </form>
      ) : null}
    </article>
  );
}
