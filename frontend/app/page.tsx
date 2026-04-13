import Link from "next/link";

const reviewJobId = "demo-review-20260409";
const interviewSessionId = "4c9b5d26-807c-4286-8d6e-94539f41afa5";

export default function HomePage() {
  return (
    <main className="min-h-screen bg-[radial-gradient(circle_at_top,_rgba(76,110,245,0.22),_transparent_30%),linear-gradient(135deg,_#020617_0%,_#0f172a_45%,_#111827_100%)] px-6 py-10 text-slate-50">
      <div className="mx-auto flex min-h-[calc(100vh-5rem)] max-w-6xl flex-col justify-center gap-8">
        <section className="rounded-[2.5rem] border border-white/10 bg-slate-950/70 p-8 shadow-2xl shadow-slate-950/40 backdrop-blur">
          <p className="text-xs uppercase tracking-[0.3em] text-brand-200">Interview Engine</p>
          <h1 className="mt-4 max-w-3xl text-4xl font-semibold tracking-tight text-white sm:text-5xl">
            Open the live recruiter review or jump straight into the candidate interview flow.
          </h1>
          <p className="mt-4 max-w-2xl text-base text-slate-300 sm:text-lg">
            The demo services are already running locally. Use the entry points below instead of the raw root URL.
          </p>
        </section>

        <section className="grid gap-6 md:grid-cols-2">
          <Link
            className="group rounded-[2rem] border border-white/10 bg-white/8 p-6 transition hover:-translate-y-1 hover:border-brand-300/60 hover:bg-white/12"
            href={`/review/${reviewJobId}`}
          >
            <p className="text-sm uppercase tracking-[0.24em] text-slate-400">Recruiter Demo</p>
            <h2 className="mt-3 text-2xl font-semibold text-white">Review scorecards</h2>
            <p className="mt-3 text-sm text-slate-300">
              Inspect the seeded shortlist, watch live session events, and open candidate scorecards.
            </p>
            <p className="mt-6 text-sm font-medium text-brand-200 group-hover:text-brand-100">
              /review/{reviewJobId}
            </p>
          </Link>

          <Link
            className="group rounded-[2rem] border border-white/10 bg-white/8 p-6 transition hover:-translate-y-1 hover:border-cyan-300/60 hover:bg-white/12"
            href={`/interview/${interviewSessionId}`}
          >
            <p className="text-sm uppercase tracking-[0.24em] text-slate-400">Candidate Demo</p>
            <h2 className="mt-3 text-2xl font-semibold text-white">Join interview room</h2>
            <p className="mt-3 text-sm text-slate-300">
              Launch the candidate session, connect to the websocket stream, and test the live interview loop.
            </p>
            <p className="mt-6 text-sm font-medium text-cyan-200 group-hover:text-cyan-100">
              /interview/{interviewSessionId}
            </p>
          </Link>
        </section>
      </div>
    </main>
  );
}
