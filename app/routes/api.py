"""API router — aggregates candidate, recruiter, and system sub-routers.

The original monolithic 1,077-line api.py has been split into:
  - routes/candidate.py  — candidate interview, WebSocket, integrity flags
  - routes/recruiter.py  — intake, scorecards, overrides, HITL WebSocket
  - routes/system.py     — health checks, auth token generation
  - routes/schemas.py    — shared Pydantic request/response models
"""

from __future__ import annotations

from fastapi import APIRouter

from app.routes.candidate import router as candidate_router
from app.routes.recruiter import router as recruiter_router
from app.routes.system import router as system_router

router = APIRouter()
router.include_router(candidate_router, tags=["candidate"])
router.include_router(recruiter_router, tags=["recruiter"])
router.include_router(system_router, tags=["system"])
