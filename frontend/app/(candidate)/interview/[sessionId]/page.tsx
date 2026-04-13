import { InterviewRoom } from "@/components/InterviewRoom";

type CandidatePageProps = {
  params: Promise<{ sessionId: string }>;
};

export default async function CandidateInterviewPage({ params }: CandidatePageProps) {
  const { sessionId } = await params;

  return (
    <main className="min-h-screen bg-[radial-gradient(circle_at_top,_rgba(79,142,247,0.2),_transparent_36%),linear-gradient(180deg,_#0f172a,_#020617)] px-6 py-10">
      <div className="mx-auto max-w-6xl">
        <InterviewRoom sessionId={sessionId} />
      </div>
    </main>
  );
}
