"""JWT authentication and role-based access control."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any

import structlog
from fastapi import Depends, HTTPException, WebSocket, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.core.config import get_settings

log = structlog.get_logger()
settings = get_settings()

_bearer_scheme = HTTPBearer(auto_error=False)

try:
    import jwt as pyjwt  # PyJWT
except ImportError:  # pragma: no cover
    pyjwt = None  # type: ignore[assignment]


class Role(str, Enum):
    CANDIDATE = "candidate"
    RECRUITER = "recruiter"
    SYSTEM = "system"


class AuthUser:
    """Lightweight identity extracted from a verified JWT."""

    def __init__(
        self,
        sub: str,
        role: Role,
        *,
        session_id: str | None = None,
        job_id: str | None = None,
    ) -> None:
        self.sub = sub
        self.role = role
        self.session_id = session_id
        self.job_id = job_id


def _decode_token(token: str) -> dict[str, Any]:
    """Decode and verify a JWT. Returns the payload dict or raises."""
    if pyjwt is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="JWT support unavailable — install PyJWT",
        )
    if not settings.jwt_secret_key:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="JWT_SECRET_KEY is not configured",
        )
    try:
        return pyjwt.decode(
            token,
            settings.jwt_secret_key,
            algorithms=[settings.jwt_algorithm],
        )
    except pyjwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="token expired"
        ) from None
    except pyjwt.InvalidTokenError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail=f"invalid token: {exc}"
        ) from exc


def create_token(
    sub: str,
    role: Role,
    *,
    session_id: str | None = None,
    job_id: str | None = None,
    extra: dict[str, Any] | None = None,
) -> str:
    """Issue a signed JWT for a given identity and role."""
    if pyjwt is None:
        raise RuntimeError("PyJWT is not installed")
    now = datetime.now(timezone.utc)
    payload: dict[str, Any] = {
        "sub": sub,
        "role": role.value,
        "iat": now,
        "exp": now + timedelta(minutes=settings.jwt_expiry_minutes),
    }
    if session_id:
        payload["session_id"] = session_id
    if job_id:
        payload["job_id"] = job_id
    if extra:
        payload.update(extra)
    return pyjwt.encode(payload, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)


def _extract_user(payload: dict[str, Any]) -> AuthUser:
    sub = payload.get("sub")
    role_str = payload.get("role", "candidate")
    if not sub:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="missing sub claim")
    try:
        role = Role(role_str)
    except ValueError:
        role = Role.CANDIDATE
    return AuthUser(
        sub=str(sub),
        role=role,
        session_id=payload.get("session_id"),
        job_id=payload.get("job_id"),
    )


# ---------------------------------------------------------------------------
# FastAPI dependencies
# ---------------------------------------------------------------------------


async def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
) -> AuthUser:
    """Dependency: extract and verify the JWT from the Authorization header.

    If JWT_SECRET_KEY is not set, authentication is disabled and a default
    system user is returned (development mode).
    """
    if not settings.jwt_secret_key:
        # Dev mode — no auth required
        return AuthUser(sub="dev", role=Role.SYSTEM)
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="missing Authorization header",
        )
    payload = _decode_token(credentials.credentials)
    return _extract_user(payload)


async def get_ws_user(websocket: WebSocket) -> AuthUser:
    """Extract user identity from a WebSocket query param ``token``.

    If JWT_SECRET_KEY is not set, authentication is disabled.
    """
    if not settings.jwt_secret_key:
        return AuthUser(sub="dev", role=Role.SYSTEM)
    token = websocket.query_params.get("token")
    if not token:
        await websocket.close(code=4001)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="missing token query parameter",
        )
    payload = _decode_token(token)
    return _extract_user(payload)


def require_role(*roles: Role) -> Callable[..., Any]:
    """Return a dependency that enforces the user has one of the given roles."""

    async def checker(user: AuthUser = Depends(get_current_user)) -> AuthUser:
        if user.role not in roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"role {user.role.value!r} not in {[r.value for r in roles]}",
            )
        return user

    return checker
