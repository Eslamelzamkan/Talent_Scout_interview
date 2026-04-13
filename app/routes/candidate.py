"""Candidate-facing API routes — interview session, WebSocket, integrity flags."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any
from uuid import UUID

import structlog
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, WebSocket
from fastapi import status as http_status
from fastapi.websockets import WebSocketDisconnect
from livekit import api as livekit_api
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

import app.core as core
from app.core.auth import AuthUser, Role, get_current_user, get_ws_user, require_role
from app.core.config import get_settings
from app.core.db import AsyncSessionFactory, get_session
from app.models import (
    AuditLog,
    InterviewMessage,
    InterviewSession,
    LiveKitTokenResponse,
    ParsedJobContext,
    SessionStatus,
)
from app.pipeline import evaluation
from app.pipeline.interview import (
    build_interview_seed,
    graph_invoke_config,
    messages_to_transcript,
)
from app.routes.schemas import CompleteSessionPayload, IntegrityFlagPayload

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


def _parse_timestamp(value: str | None) -> datetime:
    if not value:
        return _now()
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _dedupe_strings(values: list[str]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for value in values:
        item = value.strip()
        if not item:
            continue
        key = item.casefold()
        if key in seen:
            continue
        seen.add(key)
        ordered.append(item)
    return ordered


async def _graph_state_values(session_id: UUID) -> dict[str, Any] | None:
    if core.interview_graph is None:
        return None
    async with AsyncSessionFactory() as db:
        interview = await db.get(InterviewSession, session_id)
        if interview is None:
            return None
        config = graph_invoke_config(
            interview.langgraph_thread_id or str(interview.id),
            str(interview.id),
        )
    try:
        snapshot = await core.interview_graph.aget_state(config)
    except Exception:
        log.exception("graph_state_load_failed", session_id=str(session_id))
        return None
    return snapshot.values or None


def _graph_config(interview: InterviewSession) -> dict[str, Any]:
    return graph_invoke_config(
        interview.langgraph_thread_id or str(interview.id),
        str(interview.id),
    )


def _interview_state_payload(
    interview: InterviewSession,
    state: dict[str, Any],
) -> dict[str, Any]:
    transcript = messages_to_transcript(state.get("messages", []))
    question_index = sum(1 for item in transcript if item.get("role") == "agent")
    return {
        "session_id": str(interview.id),
        "current_question": state.get("current_question"),
        "current_dimension": state.get("current_dimension"),
        "question_index": question_index,
        "max_questions": settings.max_questions,
        "answer_is_shallow": bool(state.get("answer_is_shallow")),
        "interview_complete": bool(state.get("interview_complete")),
        "transcript": transcript,
    }


async def _candidate_session_bundle(
    session_id: UUID,
    *,
    session: AsyncSession | None = None,
) -> tuple[InterviewSession, ParsedJobContext]:
    async with AsyncSessionFactory() as db:
        effective_db = session if session is not None else db
        row = (
            await effective_db.exec(
                select(InterviewSession, ParsedJobContext)
                .join(ParsedJobContext, InterviewSession.job_id == ParsedJobContext.job_id)
                .where(InterviewSession.id == session_id)
            )
        ).first()
        if row is None:
            raise ValueError("session not found")
        return row


async def _mark_session_started(
    session_id: UUID,
) -> tuple[InterviewSession, ParsedJobContext, bool]:
    async with AsyncSessionFactory() as db:
        interview, job = await _candidate_session_bundle(session_id, session=db)
        started_now = False
        if interview.status == SessionStatus.SCHEDULED:
            interview.status = SessionStatus.IN_PROGRESS
            started_now = True
        if interview.started_at is None:
            interview.started_at = _now()
            started_now = True
        if started_now:
            db.add(interview)
            await db.commit()
            await db.refresh(interview)
        return interview, job, started_now


async def _candidate_graph_state(
    interview: InterviewSession,
    job: ParsedJobContext,
) -> dict[str, Any]:
    if core.interview_graph is None:
        raise RuntimeError("interview graph is not initialised")
    config = _graph_config(interview)
    snapshot = await core.interview_graph.aget_state(config)
    state = snapshot.values or {}
    if not state:
        state = await core.interview_graph.ainvoke(
            build_interview_seed(interview, job, job.rubric_model),
            config,
        )
    elif (
        not state.get("interview_complete")
        and not state.get("current_question")
        and not state.get("candidate_answer")
    ):
        state = await core.interview_graph.ainvoke(None, config)
    return state


async def _persist_graph_messages(
    session: AsyncSession,
    session_id: UUID,
    state: dict[str, Any] | None,
) -> list[str]:
    if not state:
        return []
    exists = (
        await session.exec(
            select(InterviewMessage.id).where(InterviewMessage.session_id == session_id)
        )
    ).first()
    transcript_rows = messages_to_transcript(state.get("messages", []))
    if exists is None:
        sequence_number = 0
        for row in transcript_rows:
            role = str(row.get("role") or "")
            if role == "agent":
                sequence_number += 1
            session.add(
                InterviewMessage(
                    session_id=session_id,
                    role=role,
                    content=str(row.get("content") or ""),
                    dimension_targeted=row.get("dimension"),
                    lane=row.get("lane"),
                    focus=row.get("focus"),
                    follow_up=bool(row.get("follow_up")),
                    scoreable=row.get("scoreable"),
                    sequence_number=sequence_number,
                    created_at=_parse_timestamp(row.get("timestamp")),
                )
            )
    return [str(item) for item in state.get("extracted_skills", []) if str(item).strip()]


async def _record_integrity_flag(
    session_id: UUID,
    candidate_id: str,
    flag_type: str,
    *,
    timestamp: str | None = None,
) -> None:
    async with AsyncSessionFactory() as session:
        interview = await session.get(InterviewSession, session_id)
        if interview is None:
            return
        interview.integrity_flags = _dedupe_strings([*(interview.integrity_flags or []), flag_type])
        session.add(interview)
        session.add(
            AuditLog(
                event_type="integrity_flag_raised",
                entity_type="interview_session",
                entity_id=str(session_id),
                actor_type="candidate",
                actor_id=candidate_id,
                payload={
                    "flag_type": flag_type,
                    "timestamp": (
                        _parse_timestamp(timestamp).isoformat() if timestamp else _now().isoformat()
                    ),
                },
            )
        )
        await session.commit()


async def _publish_job_event(job_id: str, payload: dict[str, Any]) -> None:
    if core.redis is None:
        return
    try:
        await core.redis.publish(f"job_events:{job_id}", json.dumps(payload, default=str))
    except Exception:
        log.exception("job_event_publish_failed", job_id=job_id, payload=payload)


async def _post_interview_pipeline(session_id: str, job_id: str) -> None:
    try:
        await evaluation.build_scorecard(session_id)
        await _publish_job_event(
            job_id,
            {"event": "scorecard_generated", "job_id": job_id, "session_id": session_id},
        )
        async with AsyncSessionFactory() as session:
            statuses = (
                await session.exec(
                    select(InterviewSession.status).where(InterviewSession.job_id == job_id)
                )
            ).all()
        if statuses and all(status in TERMINAL_STATUSES for status in statuses):
            await evaluation.rank_candidates(job_id)
    except Exception:
        log.exception(
            "post_interview_pipeline_failed",
            session_id=session_id,
            job_id=job_id,
        )


# ── REST endpoints ──────────────────────────────────────────────────────────


@router.post("/livekit/token", response_model=LiveKitTokenResponse)
async def livekit_token(
    session_id: UUID = Query(...),
    candidate_id: str = Query(...),
    session: AsyncSession = Depends(get_session),
    _user: AuthUser = Depends(require_role(Role.CANDIDATE, Role.SYSTEM)),
) -> LiveKitTokenResponse:
    interview = await session.get(InterviewSession, session_id)
    if interview is None or interview.candidate_id != candidate_id:
        raise HTTPException(status_code=http_status.HTTP_404_NOT_FOUND, detail="session not found")
    token = (
        livekit_api.AccessToken(settings.livekit_api_key, settings.livekit_api_secret)
        .with_identity(candidate_id)
        .with_name(interview.candidate_name)
        .with_grants(
            livekit_api.VideoGrants(
                room_join=True,
                room=interview.livekit_room_name,
                can_publish=True,
                can_subscribe=True,
            )
        )
        .to_jwt()
    )
    return LiveKitTokenResponse(
        session_id=str(interview.id),
        room_name=interview.livekit_room_name,
        server_url=settings.livekit_url,
        token=token,
    )


@router.get("/sessions/{session_id}/info")
async def session_info(
    session_id: UUID,
    session: AsyncSession = Depends(get_session),
    _user: AuthUser = Depends(get_current_user),
) -> dict[str, Any]:
    row = (
        await session.exec(
            select(InterviewSession, ParsedJobContext)
            .join(ParsedJobContext, InterviewSession.job_id == ParsedJobContext.job_id)
            .where(InterviewSession.id == session_id)
        )
    ).first()
    if row is None:
        raise HTTPException(status_code=http_status.HTTP_404_NOT_FOUND, detail="session not found")
    interview, job = row
    return {
        "session_id": str(interview.id),
        "candidate_id": interview.candidate_id,
        "candidate_name": interview.candidate_name,
        "role_title": job.role_title,
        "room_name": interview.livekit_room_name,
        "max_questions": settings.max_questions,
        "status": interview.status.value,
    }


@router.post("/sessions/{session_id}/complete")
async def complete_session(
    session_id: UUID,
    payload: CompleteSessionPayload,
    background: BackgroundTasks,
    session: AsyncSession = Depends(get_session),
    _user: AuthUser = Depends(require_role(Role.CANDIDATE, Role.SYSTEM)),
) -> dict[str, str]:
    graph_state = await _graph_state_values(session_id)
    interview = await session.get(InterviewSession, session_id)
    if interview is None:
        raise HTTPException(status_code=http_status.HTTP_404_NOT_FOUND, detail="session not found")
    extracted_skills = await _persist_graph_messages(session, session_id, graph_state)
    if extracted_skills:
        interview.extracted_skills = _dedupe_strings(
            [*(interview.extracted_skills or []), *extracted_skills]
        )
    if graph_state:
        graph_flags = [
            str(item).strip()
            for item in graph_state.get("integrity_flags", [])
            if str(item).strip()
        ]
        if graph_flags:
            interview.integrity_flags = _dedupe_strings(
                [*(interview.integrity_flags or []), *graph_flags]
            )
        candidate_context = graph_state.get("candidate_context")
        if candidate_context is not None:
            interview.candidate_context = candidate_context
    interview.status = SessionStatus.COMPLETED
    interview.transcript = payload.transcript
    interview.completed_at = _now()
    session.add(interview)
    session.add(
        AuditLog(
            event_type="session_completed",
            entity_type="interview_session",
            entity_id=str(session_id),
            actor_type="candidate",
            actor_id=interview.candidate_id,
            payload={"transcript_length": len(payload.transcript)},
        )
    )
    await session.commit()
    await _publish_job_event(
        interview.job_id,
        {
            "event": "session_completed",
            "job_id": interview.job_id,
            "session_id": str(session_id),
            "candidate_id": interview.candidate_id,
        },
    )
    background.add_task(_post_interview_pipeline, str(session_id), interview.job_id)
    return {"status": "evaluation_queued"}


@router.post("/sessions/{session_id}/integrity_flag")
async def integrity_flag(
    session_id: UUID,
    payload: IntegrityFlagPayload,
    session: AsyncSession = Depends(get_session),
    _user: AuthUser = Depends(require_role(Role.CANDIDATE, Role.SYSTEM)),
) -> dict[str, str]:
    interview = await session.get(InterviewSession, session_id)
    if interview is None:
        raise HTTPException(status_code=http_status.HTTP_404_NOT_FOUND, detail="session not found")
    interview.integrity_flags = _dedupe_strings(
        [*(interview.integrity_flags or []), payload.flag_type.strip()]
    )
    session.add(interview)
    session.add(
        AuditLog(
            event_type="integrity_flag_raised",
            entity_type="interview_session",
            entity_id=str(session_id),
            actor_type="candidate",
            actor_id=interview.candidate_id,
            payload={
                "flag_type": payload.flag_type.strip(),
                "timestamp": _parse_timestamp(payload.timestamp).isoformat(),
            },
        )
    )
    await session.commit()
    return {"status": "recorded"}


# ── WebSocket ───────────────────────────────────────────────────────────────


@router.websocket("/ws/interviews/{session_id}")
async def candidate_ws(websocket: WebSocket, session_id: UUID) -> None:
    await websocket.accept()
    if core.interview_graph is None:
        await websocket.send_json({"event": "error", "detail": "interview graph unavailable"})
        await websocket.close(code=1011)
        return
    # Authenticate WebSocket (degrades gracefully in dev mode)
    try:
        _user = await get_ws_user(websocket)
    except HTTPException:
        await websocket.send_json({"event": "error", "detail": "authentication failed"})
        await websocket.close(code=4001)
        return
    try:
        interview, job, started_now = await _mark_session_started(session_id)
    except ValueError:
        await websocket.send_json({"event": "error", "detail": "session not found"})
        await websocket.close(code=1008)
        return
    if started_now:
        await _publish_job_event(
            interview.job_id,
            {
                "event": "session_started",
                "job_id": interview.job_id,
                "session_id": str(interview.id),
                "candidate_id": interview.candidate_id,
            },
        )
    state = await _candidate_graph_state(interview, job)
    try:
        await websocket.send_json(
            {
                "event": (
                    "interview_complete" if state.get("interview_complete") else "interview_state"
                ),
                "data": _interview_state_payload(interview, state),
            }
        )
    except WebSocketDisconnect:
        return
    while True:
        try:
            raw = await websocket.receive_text()
        except WebSocketDisconnect:
            break
        if not raw.strip():
            continue
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            payload = raw
        try:
            if isinstance(payload, str):
                config = _graph_config(interview)
                await core.interview_graph.aupdate_state(
                    config,
                    {"candidate_answer": payload.strip()},
                )
                state = await core.interview_graph.ainvoke(None, config)
            elif isinstance(payload, dict) and payload.get("event") == "integrity_flag":
                flag_type = payload.get("flag_type")
                if not isinstance(flag_type, str) or not flag_type.strip():
                    raise ValueError("flag_type is required")
                flag_value = flag_type.strip()
                state = await _graph_state_values(interview.id) or {}
                flags = [str(item) for item in state.get("integrity_flags", []) if str(item)]
                config = _graph_config(interview)
                await core.interview_graph.aupdate_state(
                    config,
                    {"integrity_flags": _dedupe_strings([*flags, flag_value])},
                )
                state = await core.interview_graph.aget_state(config)
                state = state.values or {}
                await _record_integrity_flag(
                    interview.id,
                    interview.candidate_id,
                    flag_value,
                    timestamp=(
                        payload.get("timestamp")
                        if isinstance(payload.get("timestamp"), str)
                        else None
                    ),
                )
            else:
                raise ValueError("unsupported websocket payload")
            await websocket.send_json(
                {
                    "event": (
                        "interview_complete"
                        if state.get("interview_complete")
                        else "interview_state"
                    ),
                    "data": _interview_state_payload(interview, state),
                }
            )
        except WebSocketDisconnect:
            break
        except ValueError as exc:
            await websocket.send_json({"event": "error", "detail": str(exc)})
        except Exception:
            log.exception("candidate_ws_command_failed", session_id=str(session_id))
            await websocket.send_json({"event": "error", "detail": "command failed"})
