"use client";

import { RoomAudioRenderer, useSpeakingParticipants } from "@livekit/components-react";
import { useEffect } from "react";
import type { RefObject } from "react";

import { useTabFocus } from "@/components/hooks/useTabFocus";
import type { InterviewRealtimeState, TranscriptTurn } from "@/lib/types";

export type MicTone = "green" | "grey" | "amber";

export function FocusTracker({
  sessionId,
  wsRef,
}: {
  sessionId: string;
  wsRef: RefObject<WebSocket | null>;
}) {
  useTabFocus(sessionId, wsRef);
  return null;
}

export function VoiceBridge({ setVoiceTone }: { setVoiceTone: (value: MicTone) => void }) {
  const speakers = useSpeakingParticipants();

  useEffect(() => {
    setVoiceTone(speakers.some((participant) => !participant.isLocal) ? "grey" : "green");
  }, [setVoiceTone, speakers]);

  return <RoomAudioRenderer />;
}

export function asRealtimeState(payload: unknown): Partial<InterviewRealtimeState> {
  if (!payload || typeof payload !== "object") return {};
  const record = payload as Record<string, unknown>;
  const data = typeof record.data === "object" && record.data ? (record.data as Record<string, unknown>) : record;
  return {
    transcript: Array.isArray(data.transcript) ? (data.transcript as TranscriptTurn[]) : undefined,
    current_question: typeof data.current_question === "string" ? data.current_question : undefined,
    current_dimension: typeof data.current_dimension === "string" ? data.current_dimension : undefined,
    question_index: typeof data.question_index === "number" ? data.question_index : undefined,
    max_questions: typeof data.max_questions === "number" ? data.max_questions : undefined,
    answer_is_shallow: typeof data.answer_is_shallow === "boolean" ? data.answer_is_shallow : undefined,
    interview_complete:
      typeof data.interview_complete === "boolean"
        ? data.interview_complete
        : record.event === "interview_complete",
  };
}

export function transcriptText(turns: TranscriptTurn[]): string {
  return turns.map((turn) => `${turn.role}: ${turn.content}`).join("\n");
}
