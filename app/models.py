from __future__ import annotations

import uuid
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any

from pydantic import Field, field_validator, model_validator
from sqlalchemy import Column, DateTime
from sqlalchemy.dialects.postgresql import JSONB
from sqlmodel import Field as SQLField
from sqlmodel import SQLModel


def _uuid() -> uuid.UUID:
    return uuid.uuid4()


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _jsonb_field(default_factory: Callable[[], Any]) -> Any:
    return SQLField(default_factory=default_factory, sa_column=Column(JSONB, nullable=False))


def _utc_datetime_field(*, alias: str | None = None) -> Any:
    return SQLField(
        default_factory=_utcnow,
        alias=alias,
        sa_column=Column(DateTime(timezone=True), nullable=False),
    )


def _optional_datetime_field(*, alias: str | None = None) -> Any:
    return SQLField(default=None, alias=alias, sa_column=Column(DateTime(timezone=True)))


class SessionStatus(StrEnum):
    SCHEDULED = "scheduled"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    ABANDONED = "abandoned"
    ERROR = "error"


class RecommendedAction(StrEnum):
    ADVANCE = "advance"
    HOLD = "hold"
    REJECT = "reject"


class RubricDimension(SQLModel):
    name: str
    weight: float = Field(gt=0.0, le=1.0)
    description: str
    score_anchor_3: str
    score_anchor_2: str
    score_anchor_1: str
    few_shot_3: str
    few_shot_2: str
    few_shot_1: str
    sample_questions: list[str] = Field(default_factory=list)


class RubricModel(SQLModel):
    dimensions: list[RubricDimension] = Field(default_factory=list)
    required_skills: list[str] = Field(default_factory=list)
    role_level: str = "unknown"

    @model_validator(mode="after")
    def validate_weights(self) -> RubricModel:
        total = round(sum(item.weight for item in self.dimensions), 6)
        if self.dimensions and not 0.99 <= total <= 1.01:
            raise ValueError("rubric dimension weights must sum to 1.0")
        return self


class JudgeDimensionVote(SQLModel):
    dimension: str
    score: int = Field(ge=1, le=3)
    reasoning: str


class JudgeModelResult(SQLModel):
    model: str
    votes: list[JudgeDimensionVote] = Field(default_factory=list)


class JudgeEnsembleRaw(SQLModel):
    judges: list[JudgeModelResult] = Field(default_factory=list)


class DimensionScore(SQLModel):
    score: int = Field(ge=1, le=3)
    weight: float = Field(gt=0.0, le=1.0)
    judge_votes: list[int] = Field(default_factory=list)
    reasoning: str
    is_flagged: bool = False


class CandidateIntake(SQLModel):
    model_config = {"populate_by_name": True, "extra": "ignore"}

    candidate_id: str
    candidate_name: str = Field(alias="name")
    candidate_email: str = Field(alias="email")
    screening_score: float = Field(ge=0.0, le=1.0)
    assessment_score: float = Field(ge=0.0, le=1.0)
    assessment_type: str
    weak_areas: list[str] = Field(default_factory=list)
    extracted_skills: list[str] = Field(default_factory=list)
    summary: str | None = None
    project_highlights: list[str] = Field(default_factory=list)
    work_highlights: list[str] = Field(default_factory=list)
    behavioral_highlights: list[str] = Field(default_factory=list)
    achievements: list[str] = Field(default_factory=list)
    motivations: list[str] = Field(default_factory=list)

    @property
    def name(self) -> str:
        return self.candidate_name

    @property
    def email(self) -> str:
        return self.candidate_email

    @property
    def candidate_context_payload(self) -> dict[str, Any]:
        return CandidateContext(
            summary=self.summary,
            project_highlights=self.project_highlights,
            work_highlights=self.work_highlights,
            behavioral_highlights=self.behavioral_highlights,
            achievements=self.achievements,
            motivations=self.motivations,
        ).model_dump(mode="json")


class JobIntake(SQLModel):
    model_config = {"populate_by_name": True, "extra": "ignore"}

    job_id: str
    role_title: str = Field(alias="title")
    job_description: str = Field(alias="jd_text")
    requires_coding: bool = False

    @property
    def title(self) -> str:
        return self.role_title

    @property
    def jd_text(self) -> str:
        return self.job_description


class BatchIntakeRequest(SQLModel):
    job: JobIntake
    candidates: list[CandidateIntake] = Field(default_factory=list)
    recruiter_id: str


class BatchIntakeResponse(SQLModel):
    job_id: str
    interview_session_ids: list[str] = Field(default_factory=list)
    status: SessionStatus = SessionStatus.SCHEDULED


class CandidateScorecard(SQLModel):
    candidate_id: str
    candidate_name: str
    job_id: str
    final_rank: int | None = None
    weighted_total: float
    interview_dimension_scores: dict[str, DimensionScore] = Field(default_factory=dict)
    assessment_score: float
    speech_score: float | None = None
    screening_score: float
    strengths: list[str] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)
    recommended_action: RecommendedAction
    integrity_flags: list[str] = Field(default_factory=list)
    bias_flags: list[str] = Field(default_factory=list)
    recruiter_overrides: list[dict[str, Any]] = Field(default_factory=list)
    interview_transcript: str | None = None


class ScoreOverrideRequest(SQLModel):
    candidate_id: str
    dimension_name: str
    new_score: int = Field(ge=1, le=3)
    justification: str = Field(min_length=10)


class RecruiterOverride(SQLModel):
    reviewer_id: str
    override_score: float = Field(ge=0.0, le=1.0)
    override_action: RecommendedAction | str
    notes: str
    created_at: datetime = Field(default_factory=_utcnow)


class TranscriptTurn(SQLModel):
    role: str
    content: str
    dimension_targeted: str | None = Field(default=None, alias="dimension")
    timestamp: str


class TurnRequest(SQLModel):
    candidate_message: str | None = None


class TurnResponse(SQLModel):
    session_id: str
    agent_message: str | None
    active_dimension: str | None = None
    is_complete: bool = False
    transcript: list[TranscriptTurn] = Field(default_factory=list)


class LiveKitTokenResponse(SQLModel):
    session_id: str
    room_name: str
    server_url: str
    token: str


class IntegrityFlagPayload(SQLModel):
    model_config = {"populate_by_name": True}

    event_type: str
    timestamp: datetime
    event_metadata: dict[str, Any] | None = Field(default=None, alias="metadata")


class OverridePayload(SQLModel):
    override_score: float = Field(ge=0.0, le=1.0)
    override_action: RecommendedAction | str
    notes: str
    reviewer_id: str


class CandidateContext(SQLModel):
    model_config = {"extra": "ignore"}

    summary: str | None = None
    project_highlights: list[str] = Field(default_factory=list)
    work_highlights: list[str] = Field(default_factory=list)
    behavioral_highlights: list[str] = Field(default_factory=list)
    achievements: list[str] = Field(default_factory=list)
    motivations: list[str] = Field(default_factory=list)
    live_answer_evidence: list[str] = Field(default_factory=list)

    def evidence_snippets(self) -> list[str]:
        snippets: list[str] = []
        if self.summary:
            snippets.append(self.summary)
        snippets.extend(self.project_highlights[:3])
        snippets.extend(self.work_highlights[:2])
        snippets.extend(self.behavioral_highlights[:2])
        snippets.extend(self.achievements[:2])
        snippets.extend(self.motivations[:1])
        snippets.extend(self.live_answer_evidence[-3:])
        return [item for item in snippets if item]


def _normalize_rubric(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    raw_dimensions = value.get("dimensions", value) if isinstance(value, Mapping) else value
    return [
        RubricDimension.model_validate(item).model_dump(mode="json") for item in raw_dimensions
    ]


def _normalize_dimension_scores(value: Any) -> dict[str, dict[str, Any]]:
    if value is None:
        return {}
    return {
        str(name): DimensionScore.model_validate(score).model_dump(mode="json")
        for name, score in dict(value).items()
    }


def _normalize_judge_ensemble(value: Any) -> dict[str, Any]:
    return JudgeEnsembleRaw.model_validate(value or {}).model_dump(mode="json")


class ParsedJobContextBase(SQLModel):
    model_config = {"populate_by_name": True, "validate_assignment": True, "extra": "ignore"}

    job_id: str = SQLField(index=True, unique=True, alias="external_job_id")
    role_title: str = SQLField(alias="title")
    seniority: str = "unknown"
    domain: str = "unknown"
    must_have_skills: list[str] = _jsonb_field(list)
    rubric: list[dict[str, Any]] = _jsonb_field(list)
    question_seed_topics: list[str] = _jsonb_field(list)
    requires_coding: bool = False
    linting_warnings: list[dict[str, Any]] = _jsonb_field(list)
    parsed_at: datetime = _utc_datetime_field()

    @field_validator("rubric", mode="before")
    @classmethod
    def validate_rubric(cls, value: Any) -> list[dict[str, Any]]:
        return _normalize_rubric(value)

    @property
    def rubric_dimensions(self) -> list[RubricDimension]:
        return [RubricDimension.model_validate(item) for item in self.rubric]

    @property
    def rubric_model(self) -> RubricModel:
        return RubricModel(
            dimensions=self.rubric_dimensions,
            required_skills=list(self.must_have_skills),
            role_level=self.seniority,
        )

    @property
    def external_job_id(self) -> str:
        return self.job_id

    @property
    def title(self) -> str:
        return self.role_title

    @property
    def jd_text(self) -> str:
        return "\n".join(
            [
                f"Role: {self.role_title}",
                f"Seniority: {self.seniority}",
                f"Domain: {self.domain}",
                f"Must-have skills: {', '.join(self.must_have_skills)}",
            ]
        )


class ParsedJobContext(ParsedJobContextBase, table=True):
    __tablename__ = "parsed_job_contexts"

    id: uuid.UUID = SQLField(default_factory=_uuid, primary_key=True)


class InterviewSessionBase(SQLModel):
    model_config = {"populate_by_name": True, "validate_assignment": True, "extra": "ignore"}

    candidate_id: str = SQLField(index=True, alias="external_candidate_id")
    job_id: str = SQLField(index=True)
    recruiter_id: str
    candidate_name: str = SQLField(alias="name")
    candidate_email: str = SQLField(alias="email")
    livekit_room_name: str = SQLField(unique=True, alias="livekit_room")
    langgraph_thread_id: str | None = SQLField(default=None, index=True)
    status: SessionStatus = SessionStatus.SCHEDULED
    current_round: int = 1
    screening_score: float
    assessment_score: float
    assessment_type: str
    weak_areas: list[str] = _jsonb_field(list)
    extracted_skills: list[str] = _jsonb_field(list)
    candidate_context: dict[str, Any] = _jsonb_field(dict)
    speech_score: float | None = None
    integrity_flags: list[str] = _jsonb_field(list)
    transcript: str | None = None
    started_at: datetime | None = _optional_datetime_field()
    completed_at: datetime | None = _optional_datetime_field()
    created_at: datetime = _utc_datetime_field()

    @model_validator(mode="before")
    @classmethod
    def default_thread_id(cls, data: Any) -> Any:
        if isinstance(data, Mapping):
            payload = dict(data)
            payload["langgraph_thread_id"] = payload.get("langgraph_thread_id") or payload.get(
                "candidate_id"
            )
            return payload
        return data

    @field_validator("candidate_context", mode="before")
    @classmethod
    def validate_candidate_context(cls, value: Any) -> dict[str, Any]:
        return CandidateContext.model_validate(value or {}).model_dump(mode="json")

    @property
    def external_candidate_id(self) -> str:
        return self.candidate_id

    @property
    def name(self) -> str:
        return self.candidate_name

    @property
    def email(self) -> str:
        return self.candidate_email

    @property
    def livekit_room(self) -> str:
        return self.livekit_room_name

    @property
    def candidate_context_model(self) -> CandidateContext:
        return CandidateContext.model_validate(self.candidate_context or {})


class InterviewSession(InterviewSessionBase, table=True):
    __tablename__ = "interview_sessions"

    id: uuid.UUID = SQLField(default_factory=_uuid, primary_key=True)


class InterviewMessageBase(SQLModel):
    model_config = {"populate_by_name": True, "validate_assignment": True, "extra": "ignore"}

    session_id: uuid.UUID = SQLField(foreign_key="interview_sessions.id", index=True)
    role: str
    content: str
    dimension_targeted: str | None = SQLField(default=None, alias="dimension")
    lane: str | None = None
    focus: str | None = None
    follow_up: bool = False
    scoreable: bool | None = None
    sequence_number: int = 0
    created_at: datetime = _utc_datetime_field(alias="timestamp")

    @property
    def dimension(self) -> str | None:
        return self.dimension_targeted

    @property
    def timestamp(self) -> datetime:
        return self.created_at


class InterviewMessage(InterviewMessageBase, table=True):
    __tablename__ = "interview_messages"

    id: uuid.UUID = SQLField(default_factory=_uuid, primary_key=True)


class ScorecardBase(SQLModel):
    model_config = {"populate_by_name": True, "validate_assignment": True, "extra": "ignore"}

    candidate_id: str = SQLField(index=True)
    job_id: str = SQLField(index=True)
    session_id: uuid.UUID = SQLField(foreign_key="interview_sessions.id", unique=True, index=True)
    final_rank: int | None = None
    weighted_total: float
    interview_dimension_scores: dict[str, Any] = _jsonb_field(dict)
    assessment_score: float
    speech_score: float | None = None
    screening_score: float
    strengths: list[str] = _jsonb_field(list)
    gaps: list[str] = _jsonb_field(list)
    recommended_action: RecommendedAction = RecommendedAction.HOLD
    integrity_flags: list[str] = _jsonb_field(list)
    bias_flags: list[str] = _jsonb_field(list)
    judge_ensemble_raw: dict[str, Any] = _jsonb_field(dict)
    recruiter_overrides: list[dict[str, Any]] = _jsonb_field(list)
    is_finalized: bool = False
    created_at: datetime = _utc_datetime_field()
    updated_at: datetime = _utc_datetime_field()

    @field_validator("interview_dimension_scores", mode="before")
    @classmethod
    def validate_dimension_scores(cls, value: Any) -> dict[str, dict[str, Any]]:
        return _normalize_dimension_scores(value)

    @field_validator("judge_ensemble_raw", mode="before")
    @classmethod
    def validate_judge_ensemble(cls, value: Any) -> dict[str, Any]:
        return _normalize_judge_ensemble(value)

    @property
    def dimension_scores_model(self) -> dict[str, DimensionScore]:
        return {
            name: DimensionScore.model_validate(score)
            for name, score in self.interview_dimension_scores.items()
        }

    @property
    def judge_ensemble_model(self) -> JudgeEnsembleRaw:
        return JudgeEnsembleRaw.model_validate(self.judge_ensemble_raw)

    @property
    def recruiter_overrides_model(self) -> list[RecruiterOverride]:
        return [RecruiterOverride.model_validate(item) for item in self.recruiter_overrides]


class Scorecard(ScorecardBase, table=True):
    __tablename__ = "scorecards"

    id: uuid.UUID = SQLField(default_factory=_uuid, primary_key=True)


class AuditLogBase(SQLModel):
    model_config = {"populate_by_name": True, "validate_assignment": True, "extra": "ignore"}

    event_type: str = SQLField(alias="action")
    entity_type: str
    entity_id: str
    actor_type: str = SQLField(default="system", alias="actor")
    actor_id: str | None = None
    payload: dict[str, Any] = SQLField(
        default_factory=dict, alias="detail", sa_column=Column(JSONB, nullable=False)
    )
    created_at: datetime = _utc_datetime_field()


class AuditLog(AuditLogBase, table=True):
    # APPEND-ONLY: the application role should receive INSERT-only grants for this table.
    __tablename__ = "audit_logs"

    id: uuid.UUID = SQLField(default_factory=_uuid, primary_key=True)
