from __future__ import annotations

from typing import Any

import redis.asyncio as aioredis

from app.core.llm import LLMClient

llm: LLMClient | None = None
redis: aioredis.Redis | None = None
interview_graph: Any = None

from . import chroma  # noqa: E402

__all__ = ["llm", "redis", "interview_graph", "chroma"]
