export type TranscriptTurn = {
  role: string;
  content: string;
  dimension?: string | null;
  timestamp?: string;
};

export type InterviewSessionInfo = {
  session_id: string;
  candidate_id: string;
  candidate_name: string;
  role_title: string;
  room_name?: string | null;
  max_questions?: number;
  status: "scheduled" | "in_progress" | "completed" | "abandoned" | "error" | string;
};

export type LiveKitTokenResponse = {
  session_id: string;
  room_name: string;
  server_url: string;
  token: string;
};

export type InterviewRealtimeState = {
  transcript: TranscriptTurn[];
  current_question?: string | null;
  current_dimension?: string | null;
  question_index?: number;
  max_questions?: number;
  answer_is_shallow?: boolean;
  interview_complete?: boolean;
};

export type JudgeVote = {
  dimension: string;
  score: number;
  reasoning: string;
};

export type JudgeModelResult = {
  model: string;
  votes: JudgeVote[];
};

export type JudgeEnsembleRaw = {
  judges: JudgeModelResult[];
};

export type RecruiterOverride = {
  reviewer_id: string;
  override_score: number;
  override_action: string;
  notes: string;
  created_at: string;
  dimension_name?: string;
  original_score?: number;
  new_score?: number;
  justification?: string;
};

export type IntegrityFlag = string | { flag_type?: string; timestamp?: string };

export type ScorecardDimensionScore = {
  score: number;
  weight: number;
  judge_votes: number[];
  reasoning: string;
  is_flagged: boolean;
};

export type ScorecardModel = {
  candidate_id: string;
  candidate_name: string;
  job_id: string;
  session_id: string;
  final_rank: number | null;
  weighted_total: number;
  interview_dimension_scores: Record<string, ScorecardDimensionScore>;
  assessment_score: number;
  speech_score: number | null;
  screening_score: number;
  strengths: string[];
  gaps: string[];
  recommended_action: string;
  integrity_flags: IntegrityFlag[];
  bias_flags: string[];
  judge_ensemble_raw: JudgeEnsembleRaw;
  recruiter_overrides: RecruiterOverride[];
  interview_transcript?: string | null;
  is_finalized: boolean;
};

export type RecruiterScorecardSummary = {
  candidate_id: string;
  candidate_name?: string;
  session_id?: string;
  final_rank: number | null;
  weighted_total: number;
  recommended_action: string;
  bias_flag_count: number;
  integrity_flag_count: number;
  is_finalized: boolean;
};

export type RecruiterFeedEvent = {
  event: string;
  detail?: string;
  session_id?: string;
  [key: string]: unknown;
};
