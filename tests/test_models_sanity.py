from __future__ import annotations

import os

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlmodel import SQLModel, select
from sqlmodel.ext.asyncio.session import AsyncSession

from app.core.config import get_settings
from app.models import (
    AuditLog,
    InterviewMessage,
    InterviewSession,
    JudgeDimensionVote,
    JudgeEnsembleRaw,
    JudgeModelResult,
    ParsedJobContext,
    RecommendedAction,
    RubricDimension,
    Scorecard,
)


def _test_db_url() -> str:
    return os.environ.get("TEST_POSTGRES_URL") or get_settings().postgres_url


@pytest.mark.asyncio
async def test_model_round_trip_sanity() -> None:
    engine = create_async_engine(_test_db_url(), pool_pre_ping=True)
    session_factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async with engine.begin() as connection:
        await connection.run_sync(SQLModel.metadata.drop_all)
        await connection.run_sync(SQLModel.metadata.create_all)

    rubric = [
        RubricDimension(
            name="Communication",
            weight=1.0,
            description="Clear, structured communication.",
            score_anchor_3="Specific, concise, and calibrated.",
            score_anchor_2="Generally clear but inconsistent.",
            score_anchor_1="Vague or difficult to follow.",
            few_shot_3="I framed the tradeoffs, aligned the team, and documented the decision.",
            few_shot_2="I explained the plan and answered questions.",
            few_shot_1="I gave a generic answer without concrete detail.",
            sample_questions=["Tell me about a difficult stakeholder conversation."],
        )
    ]
    ensemble = JudgeEnsembleRaw(
        judges=[
            JudgeModelResult(
                model="openai/gpt-4.1",
                votes=[
                    JudgeDimensionVote(
                        dimension="Communication",
                        score=3,
                        reasoning="Clear and well structured answer.",
                    )
                ],
            )
        ]
    )

    async with session_factory() as session:
        job = ParsedJobContext(
            job_id="job-123",
            role_title="Senior Backend Engineer",
            seniority="senior",
            domain="platform",
            must_have_skills=["Python", "Postgres"],
            rubric=rubric,
            question_seed_topics=["incident response", "async Python"],
            requires_coding=True,
            linting_warnings=[
                {
                    "type": "missing_benefits",
                    "message": "Benefits section not found.",
                }
            ],
        )
        interview = InterviewSession(
            candidate_id="cand-123",
            job_id="job-123",
            recruiter_id="rec-123",
            candidate_name="Alex Doe",
            candidate_email="alex@example.com",
            livekit_room_name="job-123-cand-123",
            screening_score=0.82,
            assessment_score=0.77,
            assessment_type="verbal",
            weak_areas=["system design"],
            extracted_skills=["Python", "Redis"],
            transcript="agent: hello\ncandidate: hello",
        )
        session.add(job)
        session.add(interview)
        await session.flush()

        session.add(
            InterviewMessage(
                session_id=interview.id,
                role="agent",
                content="Tell me about a difficult incident.",
                dimension_targeted="Communication",
                sequence_number=1,
            )
        )
        session.add(
            Scorecard(
                candidate_id=interview.candidate_id,
                job_id=interview.job_id,
                session_id=interview.id,
                final_rank=1,
                weighted_total=0.84,
                interview_dimension_scores={
                    "Communication": {
                        "score": 3,
                        "weight": 1.0,
                        "judge_votes": [3, 3, 3],
                        "reasoning": "Consistent consensus.",
                        "is_flagged": False,
                    }
                },
                assessment_score=interview.assessment_score,
                speech_score=0.73,
                screening_score=interview.screening_score,
                strengths=["Communication"],
                gaps=[],
                recommended_action=RecommendedAction.ADVANCE,
                integrity_flags=["tab_blur"],
                bias_flags=[],
                judge_ensemble_raw=ensemble.model_dump(mode="json"),
                recruiter_overrides=[],
                is_finalized=True,
            )
        )
        session.add(
            AuditLog(
                event_type="scorecard_generated",
                entity_type="scorecard",
                entity_id=interview.candidate_id,
                actor_type="system",
                payload={"source": "sanity-test"},
            )
        )
        await session.commit()

    async with session_factory() as session:
        stored_job = (await session.exec(select(ParsedJobContext))).one()
        stored_session = (await session.exec(select(InterviewSession))).one()
        stored_message = (await session.exec(select(InterviewMessage))).one()
        stored_scorecard = (await session.exec(select(Scorecard))).one()
        stored_audit = (await session.exec(select(AuditLog))).one()

    assert stored_job.rubric_dimensions[0].name == "Communication"
    assert stored_job.rubric_model.required_skills == ["Python", "Postgres"]
    assert stored_session.langgraph_thread_id == stored_session.candidate_id
    assert stored_message.dimension_targeted == "Communication"
    assert stored_scorecard.dimension_scores_model["Communication"].score == 3
    assert stored_scorecard.judge_ensemble_model.judges[0].votes[0].dimension == "Communication"
    assert stored_audit.event_type == "scorecard_generated"

    await engine.dispose()
