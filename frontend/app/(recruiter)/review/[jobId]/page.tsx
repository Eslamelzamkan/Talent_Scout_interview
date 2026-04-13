import { RecruiterDashboard } from "@/components/RecruiterDashboard";

type RecruiterPageProps = {
  params: Promise<{ jobId: string }>;
};

export default async function RecruiterReviewPage({ params }: RecruiterPageProps) {
  const { jobId } = await params;

  return (
    <main className="min-h-screen bg-[radial-gradient(circle_at_top,_rgba(15,23,42,0.1),_transparent_35%),linear-gradient(180deg,_#f8fafc,_#e2e8f0)] px-6 py-10 text-slate-950">
      <div className="mx-auto max-w-7xl">
        <RecruiterDashboard jobId={jobId} />
      </div>
    </main>
  );
}
