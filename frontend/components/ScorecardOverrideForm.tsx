"use client";

import { FormEvent, useEffect, useState } from "react";

interface ScorecardOverrideFormProps {
  dimensionName: string;
  defaultScore: number;
  disabled: boolean;
  onSubmit: (payload: {
    dimensionName: string;
    newScore: number;
    justification: string;
  }) => Promise<void>;
}

export function ScorecardOverrideForm({
  dimensionName,
  defaultScore,
  disabled,
  onSubmit,
}: ScorecardOverrideFormProps) {
  const [newScore, setNewScore] = useState(defaultScore);
  const [justification, setJustification] = useState("");
  const [error, setError] = useState("");
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    setNewScore(defaultScore);
  }, [defaultScore]);

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (justification.trim().length < 10) {
      setError("Justification must be at least 10 characters.");
      return;
    }
    setSaving(true);
    setError("");
    try {
      await onSubmit({
        dimensionName,
        newScore,
        justification: justification.trim(),
      });
      setJustification("");
    } catch (submitError) {
      setError(submitError instanceof Error ? submitError.message : "Override failed.");
    } finally {
      setSaving(false);
    }
  }

  return (
    <form className="mt-5 space-y-4 rounded-3xl bg-slate-950/95 p-5 text-slate-50" onSubmit={submit}>
      <div className="flex items-center justify-between gap-4">
        <div>
          <p className="text-xs uppercase tracking-[0.24em] text-slate-400">Recruiter Override</p>
          <p className="mt-1 text-sm text-slate-200">{dimensionName}</p>
        </div>
        <button
          type="submit"
          disabled={disabled || saving}
          className="rounded-full bg-brand-600 px-4 py-2 text-sm font-medium text-white disabled:opacity-50"
        >
          {saving ? "Saving..." : "Apply Override"}
        </button>
      </div>

      <div className="grid gap-2 sm:grid-cols-3">
        {[
          { value: 1, label: "Poor" },
          { value: 2, label: "OK" },
          { value: 3, label: "Excellent" },
        ].map((option) => (
          <label
            key={option.value}
            className={`rounded-2xl border px-3 py-3 text-sm ${
              newScore === option.value
                ? "border-brand-400 bg-brand-500/15 text-white"
                : "border-slate-700 bg-slate-900/60 text-slate-300"
            }`}
          >
            <input
              checked={newScore === option.value}
              className="sr-only"
              disabled={disabled}
              name={`score-${dimensionName}`}
              onChange={() => setNewScore(option.value)}
              type="radio"
            />
            {option.value}. {option.label}
          </label>
        ))}
      </div>

      <textarea
        value={justification}
        onChange={(event) => setJustification(event.target.value)}
        rows={3}
        placeholder="Explain why this score should change."
        disabled={disabled}
        className="w-full rounded-2xl border border-slate-700 bg-slate-900/60 px-4 py-3 text-sm text-slate-100 outline-none"
      />
      {error ? <p className="text-sm text-amber-300">{error}</p> : null}
    </form>
  );
}
