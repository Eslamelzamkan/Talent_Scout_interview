"""Recruiter-facing API routes — scorecards, overrides, finalization, HITL WebSocket."""

from __future__ import annotations

import asyncio
import json
from collections import Counter
from datetime import datetime, timezone
from typing import Any
from uuid import UUID

import structlog
from fastapi import APIRouter, Depends, HTTPException, WebSocket
from fastapi import status as http_status
from fastapi.websockets import WebSocketDisconnect
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

import app.core as core
from app.core.auth import AuthUser, Role, get_ws_user, require_role
from app.core.config import get_settings
from app.core.db import AsyncSessionFactory, get_session
from app.models import (
    AuditLog,
    BatchIntakeRequest,
    BatchIntakeResponse,
    DimensionScore,
    InterviewMessage,
    InterviewSession,
    ParsedJobContext,
    RecruiterOverride,
    Scorecard,
    ScoreOverrideRequest,
    SessionStatus,
)
from app.pipeline import evaluation
from app.pipeline.interview import graph_invoke_config, messages_to_transcript
from app.pipeline.jd import parse_and_seed

router = APIRouter()
settings = get_settings()
log = structlog.get_logger()
TERMINAL_STATUSES = {
    SessionStatus.COMPLETED,
    SessionStatus.ABANDONED,
    SessionStatus.ERROR,
}


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _match_name(target: str, names: list[str]) -> str | None:
    lookup = {item.casefold(): item for item in names}
    return lookup.get(target.strip().casefold())


def _scorecard_sort_key(scorecard: Scorecard) -> tuple[int, int, float]:
    rank = scorecard.final_rank if scorecard.final_rank is not None else 1_000_000
    return (0 if scorecard.final_rank is not None else 1, rank, -scorecard.weighted_total)


def _strengths_from_scores(scores: dict[str, DimensionScore]) -> list[str]:
    ranked = sorted(
        scores.items(),
        key=lambda item: (-((item[1].score - 1) / 2), -item[1].weight, item[0]),
    )
    return [name for name, _ in ranked[:3]]


async def _job_exists(session: AsyncSession, job_id: str) -> bool:
    return (
        await session.exec(select(ParsedJobContext.id).where(ParsedJobContext.job_id == job_id))
    ).first() is not None


async def _transcript_text(session: AsyncSession, interview: InterviewSession) -> str | None:
    if interview.transcript:
        return interview.transcript
    messages = (
        await session.exec(
            select(InterviewMessage)
            .where(InterviewMessage.session_id == interview.id)
            .order_by(InterviewMessage.sequence_number, InterviewMessage.created_at)
        )
    ).all()
    if not messages:
        return None
    return "\n".join(f"{item.role}: {item.content}" for item in messages)


def _parsed_scorecard_payload(
    scorecard: Scorecard,
    interview: InterviewSession,
    transcript: str | None,
) -> dict[str, Any]:
    return {
        "candidate_id": scorecard.candidate_id,
        "candidate_name": interview.candidate_name,
        "job_id": scorecard.job_id,
        "session_id": str(scorecard.session_id),
        "final_rank": scorecard.final_rank,
        "weighted_total": scorecard.weighted_total,
        "interview_dimension_scores": {
            name: value.model_dump(mode="json")
            for name, value in scorecard.dimension_scores_model.items()
        },
        "assessment_score": scorecard.assessment_score,
        "speech_score": scorecard.speech_score,
        "screening_score": scorecard.screening_score,
        "strengths": list(scorecard.strengths or []),
        "gaps": list(scorecard.gaps or []),
        "recommended_action": scorecard.recommended_action,
        "integrity_flags": list(scorecard.integrity_flags or interview.integrity_flags or []),
        "bias_flags": list(scorecard.bias_flags or []),
        "judge_ensemble_raw": scorecard.judge_ensemble_model.model_dump(mode="json"),
        "recruiter_overrides": [
            item.model_dump(mode="json") for item in scorecard.recruiter_overrides_model
        ],
        "is_finalized": scorecard.is_finalized,
        "interview_transcript": transcript,
        "created_at": scorecard.created_at.isoformat(),
        "updated_at": scorecard.updated_at.isoformat(),
    }


def _redis_event_payload(data: Any) -> dict[str, Any]:
    if isinstance(data, bytes):
        data = data.decode("utf-8")
    if isinstance(data, str):
        try:
            parsed = json.loads(data)
        except json.JSONDecodeError:
            return {"event": data}
        if isinstance(parsed, dict):
            return parsed
        return {"event": "message", "data": parsed}
    if isinstance(data, dict):
        return data
    return {"event": "message", "data": data}


async def _hitl_session(job_id: str, session_id: str) -> InterviewSession:
    try:
        session_key = UUID(session_id)
    except ValueError as exc:
        raise ValueError("invalid session_id") from exc
    async with AsyncSessionFactory() as db:
        interview = await db.get(InterviewSession, session_key)
    if interview is None or interview.job_id != job_id:
        raise ValueError("session not found for job")
    return interview


async def _hitl_state_response(interview: InterviewSession) -> dict[str, Any]:
    if core.interview_graph is None:
        raise RuntimeError("interview graph is not initialised")
    config = graph_invoke_config(
        interview.langgraph_thread_id or str(interview.id),
        str(interview.id),
    )
    snapshot = await core.interview_graph.aget_state(config)
    next_nodes = tuple(getattr(snapshot, "next", ()) or ())
    if "hitl" not in next_nodes:
        raise ValueError("session is not waiting for recruiter input")
    return snapshot.values or {}


# ── REST endpoints ──────────────────────────────────────────────────────────


@router.post("/intake/batch", response_model=BatchIntakeResponse)
async def intake_batch(
    payload: BatchIntakeRequest,
    session: AsyncSession = Depends(get_session),
    _user: AuthUser = Depends(require_role(Role.RECRUITER, Role.SYSTEM)),
) -> BatchIntakeResponse:
    try:
        job = await parse_and_seed(
            payload.job.job_id,
            payload.job.job_description,
            session,
            commit=False,
            requires_coding=payload.job.requires_coding,
        )
        if not job.rubric:
            await session.rollback()
            raise HTTPException(
                status_code=http_status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail={
                    "message": "job description could not be parsed",
                    "linting_warnings": job.linting_warnings,
                },
            )
        interviews: list[InterviewSession] = []
        for candidate in payload.candidates:
            interview = InterviewSession(
                candidate_id=candidate.candidate_id,
                job_id=job.job_id,
                recruiter_id=payload.recruiter_id,
                candidate_name=candidate.candidate_name,
                candidate_email=candidate.candidate_email,
                livekit_room_name=f"{payload.job.job_id}-{candidate.candidate_id}".lower(),
                screening_score=candidate.screening_score,
                assessment_score=candidate.assessment_score,
                assessment_type=candidate.assessment_type,
                weak_areas=candidate.weak_areas,
                extracted_skills=candidate.extracted_skills,
                candidate_context=candidate.candidate_context_payload,
            )
            interview.langgraph_thread_id = str(interview.id)
            session.add(interview)
            interviews.append(interview)
        session.add(
            AuditLog(
                event_type="intake_received",
                entity_type="job",
                entity_id=job.job_id,
                actor_type="recruiter",
                actor_id=payload.recruiter_id,
                payload={
                    "candidate_count": len(payload.candidates),
                    "linting_warnings": job.linting_warnings,
                },
            )
        )
        await session.commit()
    except HTTPException:
        await session.rollback()
        raise
    except ValueError as exc:
        await session.rollback()
        raise HTTPException(
            status_code=http_status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"message": str(exc), "linting_warnings": []},
        ) from exc
    except Exception:
        await session.rollback()
        raise
    return BatchIntakeResponse(
        job_id=payload.job.job_id,
        interview_session_ids=[str(item.id) for item in interviews],
    )


@router.get("/intake/status/{job_id}")
async def intake_status(
    job_id: str,
    session: AsyncSession = Depends(get_session),
    _user: AuthUser = Depends(require_role(Role.RECRUITER, Role.SYSTEM)),
) -> dict[str, Any]:
    statuses = (
        await session.exec(select(InterviewSession.status).where(InterviewSession.job_id == job_id))
    ).all()
    if not statuses and not await _job_exists(session, job_id):
        raise HTTPException(status_code=http_status.HTTP_404_NOT_FOUND, detail="job not found")
    counts = Counter(
        item.value if isinstance(item, SessionStatus) else str(item) for item in statuses
    )
    return {
        "job_id": job_id,
        "counts": {state.value: counts.get(state.value, 0) for state in SessionStatus},
    }


@router.get("/recruiter/{job_id}/scorecards")
async def recruiter_scorecards(
    job_id: str,
    session: AsyncSession = Depends(get_session),
    _user: AuthUser = Depends(require_role(Role.RECRUITER, Role.SYSTEM)),
) -> list[dict[str, Any]]:
    rows = (
        await session.exec(
            select(Scorecard, InterviewSession)
            .join(InterviewSession, Scorecard.session_id == InterviewSession.id)
            .where(Scorecard.job_id == job_id)
        )
    ).all()
    return [
        {
            "candidate_id": scorecard.candidate_id,
            "final_rank": scorecard.final_rank,
            "weighted_total": scorecard.weighted_total,
            "recommended_action": scorecard.recommended_action,
            "bias_flag_count": len(scorecard.bias_flags or []),
            "integrity_flag_count": len(
                scorecard.integrity_flags or interview.integrity_flags or []
            ),
            "is_finalized": scorecard.is_finalized,
        }
        for scorecard, interview in sorted(rows, key=lambda item: _scorecard_sort_key(item[0]))
    ]


@router.get("/recruiter/{job_id}/scorecard/{candidate_id}")
async def recruiter_scorecard_detail(
    job_id: str,
    candidate_id: str,
    session: AsyncSession = Depends(get_session),
    _user: AuthUser = Depends(require_role(Role.RECRUITER, Role.SYSTEM)),
) -> dict[str, Any]:
    row = (
        await session.exec(
            select(Scorecard, InterviewSession)
            .join(InterviewSession, Scorecard.session_id == InterviewSession.id)
            .where(Scorecard.job_id == job_id)
            .where(Scorecard.candidate_id == candidate_id)
        )
    ).first()
    if row is None:
        raise HTTPException(
            status_code=http_status.HTTP_404_NOT_FOUND,
            detail="scorecard not found",
        )
    scorecard, interview = row
    return _parsed_scorecard_payload(
        scorecard,
        interview,
        await _transcript_text(session, interview),
    )


@router.post("/recruiter/{job_id}/override")
async def recruiter_override(
    job_id: str,
    payload: ScoreOverrideRequest,
    session: AsyncSession = Depends(get_session),
    _user: AuthUser = Depends(require_role(Role.RECRUITER, Role.SYSTEM)),
) -> dict[str, Any]:
    row = (
        await session.exec(
            select(Scorecard, InterviewSession)
            .join(InterviewSession, Scorecard.session_id == InterviewSession.id)
            .where(Scorecard.job_id == job_id)
            .where(Scorecard.candidate_id == payload.candidate_id)
        )
    ).first()
    if row is None:
        raise HTTPException(
            status_code=http_status.HTTP_404_NOT_FOUND,
            detail="scorecard not found",
        )
    scorecard, interview = row
    if scorecard.is_finalized:
        raise HTTPException(
            status_code=http_status.HTTP_409_CONFLICT,
            detail="scorecard already finalized",
        )
    dimension_scores = scorecard.dimension_scores_model
    matched_dimension = _match_name(payload.dimension_name, list(dimension_scores))
    if matched_dimension is None:
        raise HTTPException(
            status_code=http_status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="dimension does not exist",
        )
    old_total = scorecard.weighted_total
    old_action = scorecard.recommended_action
    original = dimension_scores[matched_dimension]
    dimension_scores[matched_dimension] = DimensionScore(
        score=payload.new_score,
        weight=original.weight,
        judge_votes=list(original.judge_votes),
        reasoning=original.reasoning,
        is_flagged=original.is_flagged,
    )
    interview_signal = sum(
        ((item.score - 1) / 2) * item.weight for item in dimension_scores.values()
    )
    new_total = evaluation.weighted_total(
        interview_signal,
        scorecard.assessment_score,
        scorecard.speech_score or 0.0,
        scorecard.screening_score,
    )
    new_action = evaluation.recommended_action(new_total)
    override = RecruiterOverride(
        reviewer_id="recruiter",
        override_score=new_total,
        override_action=new_action,
        notes=f"{matched_dimension}: {payload.justification}",
    ).model_dump(mode="json")
    override.update(
        {
            "dimension_name": matched_dimension,
            "original_score": original.score,
            "new_score": payload.new_score,
            "justification": payload.justification,
        }
    )
    scorecard.interview_dimension_scores = {
        name: value.model_dump(mode="json") for name, value in dimension_scores.items()
    }
    scorecard.weighted_total = new_total
    scorecard.recommended_action = new_action
    scorecard.strengths = _strengths_from_scores(dimension_scores)
    scorecard.recruiter_overrides = [*(scorecard.recruiter_overrides or []), override]
    scorecard.updated_at = _now()
    session.add(scorecard)
    session.add(
        AuditLog(
            event_type="score_overridden",
            entity_type="scorecard",
            entity_id=str(scorecard.id),
            actor_type="recruiter",
            actor_id=interview.recruiter_id,
            payload={
                "candidate_id": payload.candidate_id,
                "dimension_name": matched_dimension,
                "original": {
                    "score": original.score,
                    "weighted_total": old_total,
                    "recommended_action": old_action,
                },
                "new": {
                    "score": payload.new_score,
                    "weighted_total": new_total,
                    "recommended_action": new_action,
                },
                "justification": payload.justification,
            },
        )
    )
    await session.commit()
    await evaluation.rank_candidates(job_id)
    return {"new_total": new_total, "new_action": new_action}


@router.post("/recruiter/{job_id}/finalize")
async def finalize_shortlist(
    job_id: str,
    session: AsyncSession = Depends(get_session),
    _user: AuthUser = Depends(require_role(Role.RECRUITER, Role.SYSTEM)),
) -> dict[str, Any]:
    scorecards = (await session.exec(select(Scorecard).where(Scorecard.job_id == job_id))).all()
    if not scorecards:
        raise HTTPException(
            status_code=http_status.HTTP_404_NOT_FOUND,
            detail="scorecards not found",
        )
    for scorecard in scorecards:
        scorecard.is_finalized = True
        scorecard.updated_at = _now()
        session.add(scorecard)
    session.add(
        AuditLog(
            event_type="shortlist_finalized",
            entity_type="job",
            entity_id=job_id,
            actor_type="recruiter",
            payload={"scorecard_count": len(scorecards)},
        )
    )
    await session.commit()
    return {"status": "finalized", "scorecard_count": len(scorecards)}


@router.get("/output/{job_id}")
async def output_scorecards(
    job_id: str,
    session: AsyncSession = Depends(get_session),
    _user: AuthUser = Depends(require_role(Role.RECRUITER, Role.SYSTEM)),
) -> list[dict[str, Any]]:
    rows = (
        await session.exec(
            select(Scorecard, InterviewSession)
            .join(InterviewSession, Scorecard.session_id == InterviewSession.id)
            .where(Scorecard.job_id == job_id)
            .where(Scorecard.is_finalized)
        )
    ).all()
    if not rows:
        raise HTTPException(
            status_code=http_status.HTTP_425_TOO_EARLY,
            detail="no finalized scorecards available",
        )
    ordered = sorted(rows, key=lambda item: _scorecard_sort_key(item[0]))
    return [
        _parsed_scorecard_payload(scorecard, interview, await _transcript_text(session, interview))
        for scorecard, interview in ordered
    ]


# ── WebSocket — replaces 100ms busy-loop polling with concurrent tasks ──────


@router.websocket("/ws/recruiter/{job_id}")
async def recruiter_ws(websocket: WebSocket, job_id: str) -> None:
    await websocket.accept()
    if core.interview_graph is None:
        await websocket.send_json({"event": "error", "detail": "interview graph unavailable"})
        await websocket.close(code=1011)
        return
    # Authenticate WebSocket
    try:
        _user = await get_ws_user(websocket)
    except HTTPException:
        await websocket.send_json({"event": "error", "detail": "authentication failed"})
        await websocket.close(code=4001)
        return
    pubsub = core.redis.pubsub() if core.redis is not None else None
    channel = f"job_events:{job_id}"
    if pubsub is not None:
        await pubsub.subscribe(channel)
    else:
        await websocket.send_json({"event": "warning", "detail": "redis unavailable"})
    await websocket.send_json({"event": "subscribed", "job_id": job_id})

    # ── Concurrent tasks instead of 100ms busy-loop ──

    closed = asyncio.Event()

    async def _redis_listener() -> None:
        """Forward Redis pub/sub events to the WebSocket."""
        if pubsub is None:
            return
        try:
            while not closed.is_set():
                message = await pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0)
                if message and message.get("data") is not None:
                    await websocket.send_json(_redis_event_payload(message["data"]))
        except (WebSocketDisconnect, Exception):
            closed.set()

    async def _ws_listener() -> None:
        """Handle incoming recruiter WebSocket commands."""
        try:
            while not closed.is_set():
                try:
                    payload = await websocket.receive_json()
                except WebSocketDisconnect:
                    closed.set()
                    return

                try:
                    command, body = _ws_command(payload)
                    interview = await _hitl_session(job_id, str(body.get("session_id") or ""))
                    await _hitl_state_response(interview)
                    config = graph_invoke_config(
                        interview.langgraph_thread_id or str(interview.id),
                        str(interview.id),
                    )
                    if command == "hitl_inject_question":
                        question_text = body.get("question_text")
                        if not isinstance(question_text, str) or not question_text.strip():
                            raise ValueError("question_text is required")
                        await core.interview_graph.aupdate_state(
                            config,
                            {"recruiter_injected_question": question_text.strip()},
                        )
                        state = await core.interview_graph.ainvoke(None, config)
                    elif command == "hitl_approve_continue":
                        await core.interview_graph.aupdate_state(
                            config,
                            {
                                "hitl_requested": False,
                                "recruiter_injected_question": None,
                                "current_question": None,
                                "current_dimension": None,
                                "candidate_answer": None,
                                "answer_is_shallow": False,
                            },
                        )
                        await core.interview_graph.ainvoke(None, config)
                        state = await core.interview_graph.ainvoke(None, config)
                    else:
                        raise ValueError("unsupported command")
                    await websocket.send_json(
                        {
                            "event": "hitl_state",
                            "data": {
                                "session_id": str(interview.id),
                                "current_question": state.get("current_question"),
                                "current_dimension": state.get("current_dimension"),
                                "interview_complete": bool(state.get("interview_complete")),
                                "transcript": messages_to_transcript(state.get("messages", [])),
                            },
                        }
                    )
                except ValueError as exc:
                    await websocket.send_json({"event": "error", "detail": str(exc)})
                except Exception:
                    log.exception("recruiter_ws_command_failed", job_id=job_id)
                    await websocket.send_json({"event": "error", "detail": "command failed"})
        except Exception:
            closed.set()

    try:
        await asyncio.gather(_redis_listener(), _ws_listener())
    finally:
        closed.set()
        if pubsub is not None:
            await pubsub.unsubscribe(channel)
            await pubsub.aclose()


def _ws_command(payload: Any) -> tuple[str, dict[str, Any]]:
    if isinstance(payload, dict) and len(payload) == 1:
        command, body = next(iter(payload.items()))
        return str(command), dict(body or {})
    if isinstance(payload, dict) and "type" in payload:
        return str(payload["type"]), dict(payload.get("data") or payload.get("payload") or {})
    raise ValueError("invalid websocket payload")
