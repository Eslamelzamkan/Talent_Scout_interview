from __future__ import annotations

import asyncio
import logging
import os
import sys
from contextlib import asynccontextmanager

import chromadb
import redis.asyncio as aioredis
import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

import app.core as core
from app.core.config import get_settings
from app.core.db import ensure_schema_ready
from app.core.llm import LLMClient
from app.pipeline.interview import build_interview_graph
from app.routes.api import router

settings = get_settings()
log = structlog.get_logger()

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


def _configure_logging() -> None:
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


@asynccontextmanager
async def lifespan(_: FastAPI):
    _configure_logging()
    if settings.langsmith_api_key:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGSMITH_API_KEY"] = settings.langsmith_api_key
        os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project
    await ensure_schema_ready()
    core.redis = aioredis.from_url(settings.redis_url, decode_responses=True)
    core.llm = LLMClient(settings, redis_client=core.redis)
    chroma_client = await chromadb.AsyncHttpClient(
        host=settings.chroma_host,
        port=settings.chroma_port,
    )
    core.chroma.init(chroma_client, core.llm)
    if not await core.llm.health_check():
        log.warning("llm_health_check_failed", model=settings.model_fast)
    pg_url = settings.postgres_url.replace("postgresql+asyncpg://", "postgresql://")
    async with AsyncPostgresSaver.from_conn_string(pg_url) as checkpointer:
        await checkpointer.setup()
        core.interview_graph = build_interview_graph(checkpointer)
        yield
    if core.redis is not None:
        await core.redis.aclose()
    if core.llm is not None:
        await core.llm.aclose()


def create_app() -> FastAPI:
    app = FastAPI(title="Interview Engine", version="0.1.0", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(router, prefix="/api")
    return app


app = create_app()
