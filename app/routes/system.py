"""System routes — health checks and token generation."""

from __future__ import annotations

from typing import Any

import structlog
from fastapi import APIRouter

import app.core as core
from app.core.auth import Role, create_token

router = APIRouter()
log = structlog.get_logger()


@router.get("/health")
@router.get("/healthz", include_in_schema=False)
async def health() -> dict[str, Any]:
    checks: dict[str, bool] = {"llm": False, "redis": False, "chroma": False}
    if core.llm is not None:
        try:
            checks["llm"] = await core.llm.health_check()
        except Exception:
            log.exception("health_llm_check_failed")
    if core.redis is not None:
        try:
            checks["redis"] = bool(await core.redis.ping())  # type: ignore[misc]
        except Exception:
            log.exception("health_redis_check_failed")
    chroma_client = getattr(core.chroma, "_chroma", None)
    if chroma_client is not None:
        try:
            await chroma_client.list_collections()
            checks["chroma"] = True
        except Exception:
            log.exception("health_chroma_check_failed")
    return {"status": "ok" if all(checks.values()) else "degraded", "checks": checks}


@router.post("/auth/token")
async def generate_dev_token(
    sub: str = "dev-user",
    role: str = "system",
    session_id: str | None = None,
    job_id: str | None = None,
) -> dict[str, str]:
    """Generate a JWT for development/testing. In production, tokens should be
    issued by an external identity provider.
    """
    try:
        resolved_role = Role(role)
    except ValueError:
        resolved_role = Role.CANDIDATE
    token = create_token(
        sub,
        resolved_role,
        session_id=session_id,
        job_id=job_id,
    )
    return {"token": token}
