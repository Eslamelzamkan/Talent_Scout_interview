"use client";

type InjectQuestionModalProps = {
  question: string;
  open: boolean;
  onChange: (value: string) => void;
  onClose: () => void;
  onSend: () => void;
};

export function InjectQuestionModal({
  open,
  question,
  onChange,
  onClose,
  onSend,
}: InjectQuestionModalProps) {
  if (!open) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/60 px-6">
      <div className="w-full max-w-xl rounded-[2rem] bg-white p-6 text-slate-950 shadow-2xl">
        <p className="text-xs uppercase tracking-[0.28em] text-slate-500">HITL Inject Question</p>
        <textarea
          value={question}
          onChange={(event) => onChange(event.target.value)}
          rows={6}
          className="mt-4 w-full rounded-3xl border border-slate-300 px-4 py-3 text-sm outline-none"
          placeholder="Add a recruiter follow-up question."
        />
        <div className="mt-4 flex justify-end gap-3">
          <button className="rounded-full border border-slate-300 px-4 py-2 text-sm" onClick={onClose} type="button">Cancel</button>
          <button className="rounded-full bg-slate-950 px-4 py-2 text-sm font-medium text-white disabled:opacity-50" disabled={!question.trim()} onClick={onSend} type="button">Send</button>
        </div>
      </div>
    </div>
  );
}
