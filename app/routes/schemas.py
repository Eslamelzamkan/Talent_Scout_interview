"""Pydantic request/response models for API payloads — replacing raw dict[str, Any]."""

from __future__ import annotations

from pydantic import Field
from sqlmodel import SQLModel


class CompleteSessionPayload(SQLModel):
    """POST /sessions/{session_id}/complete"""

    transcript: str = Field(min_length=1)


class IntegrityFlagPayload(SQLModel):
    """POST /sessions/{session_id}/integrity_flag"""

    flag_type: str = Field(min_length=1)
    timestamp: str


class WebSocketAnswerPayload(SQLModel):
    """Candidate answer submitted over WebSocket."""

    answer: str


class WebSocketIntegrityPayload(SQLModel):
    """Integrity flag submitted over WebSocket."""

    event: str = "integrity_flag"
    flag_type: str
    timestamp: str | None = None
