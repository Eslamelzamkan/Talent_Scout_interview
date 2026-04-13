"use client";

import { ShieldAlert } from "lucide-react";

import type { RecruiterOverride, ScorecardDimensionScore, ScorecardModel } from "@/lib/types";

import { ScorecardOverrideForm } from "../ScorecardOverrideForm";

const scoreLabels: Record<1 | 2 | 3, string> = { 1: "Poor", 2: "OK", 3: "Excellent" };
const scoreBadgeClass: Record<1 | 2 | 3, string> = {
  1: "bg-rose-500/15 text-rose-300 border-rose-500/30",
  2: "bg-amber-500/15 text-amber-200 border-amber-500/30",
  3: "bg-emerald-500/15 text-emerald-200 border-emerald-500/30",
};

type DimensionCardProps = {
  dimensionName: string;
  dimension: ScorecardDimensionScore;
  override?: RecruiterOverride;
  scorecard: ScorecardModel;
  onOverride: (payload: {
    dimensionName: string;
    newScore: number;
    justification: string;
  }) => Promise<void>;
};

export function DimensionCard({
  dimension,
  dimensionName,
  override,
  scorecard,
  onOverride,
}: DimensionCardProps) {
  const judgeNotes = scorecard.judge_ensemble_raw.judges.map((judge, index) => ({
    label: `J${index + 1}`,
    text:
      judge.votes.find((vote) => vote.dimension === dimensionName)?.reasoning ?? dimension.reasoning,
  }));
  const score = dimension.score as 1 | 2 | 3;

  return (
    <details className="rounded-[2rem] border border-slate-800 bg-slate-950/80 p-5 text-slate-50">
      <summary className="flex cursor-pointer list-none items-center justify-between gap-4">
        <div className="space-y-3">
          <div className="flex flex-wrap items-center gap-3">
            <h3 className="text-xl font-semibold">{dimensionName}</h3>
            <span className={`rounded-full border px-3 py-1 text-sm ${scoreBadgeClass[score]}`}>
              {score} | {scoreLabels[score]}
            </span>
            {dimension.is_flagged ? (
              <span className="inline-flex items-center gap-2 rounded-full border border-amber-500/30 bg-amber-500/15 px-3 py-1 text-sm text-amber-200">
                <ShieldAlert className="h-4 w-4" />
                Bias warning
              </span>
            ) : null}
            {override ? (
              <span className="rounded-full border border-brand-400/30 bg-brand-500/15 px-3 py-1 text-sm text-brand-100">
                {override.original_score} {"->"} {override.new_score} (overridden)
              </span>
            ) : null}
          </div>
          <p className="text-sm text-slate-300">Weight {(dimension.weight * 100).toFixed(0)}%</p>
        </div>
        <span className="text-sm text-slate-400">Expand</span>
      </summary>

      <div className="mt-5 grid gap-4 lg:grid-cols-[1fr_0.95fr]">
        <div className="space-y-3">
          {judgeNotes.map((judge) => (
            <article key={judge.label} className="rounded-3xl border border-slate-800 bg-slate-900/90 p-4">
              <p className="text-xs uppercase tracking-[0.24em] text-slate-400">{judge.label}</p>
              <p className="mt-2 text-sm leading-6 text-slate-200">{judge.text}</p>
            </article>
          ))}
        </div>

        <ScorecardOverrideForm
          defaultScore={dimension.score}
          dimensionName={dimensionName}
          disabled={scorecard.is_finalized}
          onSubmit={onOverride}
        />
      </div>
    </details>
  );
}
