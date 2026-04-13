"use client";

import { AlertTriangle } from "lucide-react";
import { PolarAngleAxis, PolarGrid, Radar, RadarChart, ResponsiveContainer } from "recharts";

import { apiFetchJson } from "@/lib/api";
import type { IntegrityFlag, RecruiterOverride, ScorecardModel } from "@/lib/types";

import { DimensionCard } from "./scorecard/DimensionCard";

type ScorecardViewProps = {
  scorecard: ScorecardModel;
  onRefresh?: () => Promise<void>;
};

function normalisedScore(score: number): number {
  return ((score - 1) / 2) * 100;
}

function latestOverrides(overrides: RecruiterOverride[]): Record<string, RecruiterOverride> {
  return overrides.reduce<Record<string, RecruiterOverride>>((accumulator, override) => {
    if (override.dimension_name) accumulator[override.dimension_name] = override;
    return accumulator;
  }, {});
}

function flagText(flag: IntegrityFlag): { label: string; timestamp: string } {
  if (typeof flag === "string") {
    return { label: flag.replaceAll("_", " "), timestamp: "Timestamp unavailable" };
  }
  return {
    label: (flag.flag_type ?? "integrity flag").replaceAll("_", " "),
    timestamp: flag.timestamp ?? "Timestamp unavailable",
  };
}

function recommendationClass(action: string): string {
  if (action === "advance") return "text-emerald-300";
  if (action === "hold") return "text-amber-200";
  return "text-rose-300";
}

export function ScorecardView({ scorecard, onRefresh }: ScorecardViewProps) {
  const overrides = latestOverrides(scorecard.recruiter_overrides);
  const dimensions = Object.entries(scorecard.interview_dimension_scores).map(([name, value]) => ({
    name,
    ...value,
  }));
  const chartData = dimensions.map((dimension) => ({
    subject: dimension.name,
    score: normalisedScore(dimension.score),
    fullMark: 100,
  }));

  async function submitOverride(payload: {
    dimensionName: string;
    newScore: number;
    justification: string;
  }) {
    await apiFetchJson(`/recruiter/${scorecard.job_id}/override`, {
      method: "POST",
      body: JSON.stringify({
        candidate_id: scorecard.candidate_id,
        dimension_name: payload.dimensionName,
        new_score: payload.newScore,
        justification: payload.justification,
      }),
    });
    if (onRefresh) await onRefresh();
  }

  return (
    <section className="space-y-6">
      <div className="grid gap-6 xl:grid-cols-[0.9fr_1.1fr]">
        <article className="rounded-[2rem] border border-slate-800 bg-slate-950/85 p-6 text-slate-50 shadow-2xl shadow-slate-950/30">
          <p className="text-xs uppercase tracking-[0.28em] text-slate-400">Candidate Scorecard</p>
          <h2 className="mt-4 text-3xl font-semibold">{scorecard.candidate_name}</h2>
          <p className="mt-2 text-sm text-slate-400">Candidate ID: {scorecard.candidate_id}</p>
          <div className="mt-8 rounded-[1.75rem] bg-slate-900/90 p-5">
            <p className="text-sm text-slate-400">Weighted total</p>
            <p className="mt-2 text-5xl font-semibold">{(scorecard.weighted_total * 100).toFixed(1)}%</p>
            <p className={`mt-3 text-sm uppercase tracking-[0.22em] ${recommendationClass(scorecard.recommended_action)}`}>
              {scorecard.recommended_action}
            </p>
          </div>
          <div className="mt-6 grid gap-3 text-sm text-slate-300 sm:grid-cols-2">
            <p>Rank: {scorecard.final_rank ?? "Pending"}</p>
            <p>Bias flags: {scorecard.bias_flags.length}</p>
            <p>Strengths: {scorecard.strengths.join(", ") || "None"}</p>
            <p>Gaps: {scorecard.gaps.join(", ") || "None"}</p>
          </div>
        </article>

        <article className="rounded-[2rem] border border-slate-200/10 bg-white/95 p-6 text-slate-950 shadow-2xl shadow-slate-950/10">
          <p className="text-xs uppercase tracking-[0.28em] text-slate-500">Radar View</p>
          <div className="mt-6 h-80">
            <ResponsiveContainer width="100%" height={320}>
              <RadarChart data={chartData}>
                <PolarGrid />
                <PolarAngleAxis dataKey="subject" tick={{ fontSize: 12 }} />
                <Radar name="Candidate" dataKey="score" fill="#4f8ef7" fillOpacity={0.3} stroke="#4f8ef7" />
              </RadarChart>
            </ResponsiveContainer>
          </div>
        </article>
      </div>

      {scorecard.integrity_flags.length > 0 ? (
        <section className="rounded-[2rem] border border-amber-500/30 bg-amber-500/10 p-5 text-amber-50" title="Flags are signals, not conclusions. Recruiter review required.">
          <div className="flex items-start gap-3">
            <AlertTriangle className="mt-0.5 h-5 w-5 text-amber-300" />
            <div>
              <p className="font-medium">{scorecard.integrity_flags.length} tab-switch event(s) detected during interview</p>
              <div className="mt-3 space-y-2 text-sm text-amber-100">
                {scorecard.integrity_flags.map((flag, index) => {
                  const display = flagText(flag);
                  return <p key={`${display.label}-${index}`}>{display.label} | {display.timestamp}</p>;
                })}
              </div>
            </div>
          </div>
        </section>
      ) : null}

      <section className="space-y-4">
        {dimensions.map((dimension) => (
          <DimensionCard
            key={dimension.name}
            dimension={dimension}
            dimensionName={dimension.name}
            onOverride={submitOverride}
            override={overrides[dimension.name]}
            scorecard={scorecard}
          />
        ))}
      </section>
    </section>
  );
}
