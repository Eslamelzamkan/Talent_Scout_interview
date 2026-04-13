from __future__ import annotations

import os
from unittest.mock import AsyncMock

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlmodel import SQLModel, select
from sqlmodel.ext.asyncio.session import AsyncSession

from app.core.config import get_settings
from app.models import ParsedJobContext
from app.pipeline import jd as jd_module


def _test_db_url() -> str:
    return os.environ.get("TEST_POSTGRES_URL") or get_settings().postgres_url


def _extracted_payload() -> dict[str, object]:
    return {
        "role_title": "Senior Backend Engineer",
        "seniority": "senior",
        "domain": "platform",
        "must_have_skills": ["Python", "Postgres"],
        "question_seed_topics": ["async io", "postgres", "api design", "debugging", "ownership"],
        "rubric_dimensions": [
            {
                "name": "Technical Reasoning",
                "weight": 0.5,
                "description": "Explains trade-offs and constraints clearly.",
                "score_anchor_3": "Explains architecture and trade-offs with measurable detail.",
                "score_anchor_2": "Explains a workable approach but misses key trade-offs.",
                "score_anchor_1": "Gives generic or unsupported technical claims.",
                "few_shot_3": "I compared two designs and justified the complexity trade-off.",
                "few_shot_2": "I described the rough implementation path.",
                "few_shot_1": "I kept the explanation generic and skipped constraints.",
                "sample_questions": [
                    "Talk through how you would debug a production latency spike."
                ],
            },
            {
                "name": "Communication",
                "weight": 0.5,
                "description": "Communicates technical decisions with clear evidence.",
                "score_anchor_3": "Uses concrete examples, constraints, and outcomes.",
                "score_anchor_2": "Generally clear but uneven on specifics.",
                "score_anchor_1": "Relies on vague statements with little evidence.",
                "few_shot_3": "I aligned stakeholders by walking through the rollout risks.",
                "few_shot_2": "I explained the plan and answered follow-up questions.",
                "few_shot_1": "I repeated generic points without clarifying the decision.",
                "sample_questions": ["Describe a technical decision you had to defend."],
            },
        ],
        "requires_coding": True,
    }


def test_weight_normalisation() -> None:
    weights = [{"weight": 0.3}, {"weight": 0.3}, {"weight": 0.3}]
    result = jd_module._normalise_weights(weights)
    assert [item["weight"] for item in result] == pytest.approx([1 / 3, 1 / 3, 1 / 3], abs=1e-5)


def test_weight_already_correct() -> None:
    weights = [{"weight": 0.4}, {"weight": 0.6}]
    assert jd_module._normalise_weights(weights) == weights


def test_json_strips_fences() -> None:
    payload = jd_module._parse_json_response('```json\n{"role_title": "Backend Engineer"}\n```')
    assert payload == {"role_title": "Backend Engineer"}


@pytest.mark.asyncio
async def test_idempotent(monkeypatch: pytest.MonkeyPatch) -> None:
    engine = create_async_engine(_test_db_url(), pool_pre_ping=True)
    session_factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    try:
        async with engine.begin() as connection:
            await connection.run_sync(SQLModel.metadata.drop_all)
            await connection.run_sync(SQLModel.metadata.create_all)
    except Exception as exc:
        await engine.dispose()
        pytest.skip(f"postgres unavailable for idempotent test: {exc}")

    monkeypatch.setattr(jd_module, "_extract", AsyncMock(return_value=_extracted_payload()))
    monkeypatch.setattr(jd_module, "_lint", AsyncMock(return_value=[]))
    monkeypatch.setattr(jd_module, "_seed_question_bank", AsyncMock(return_value=30))

    try:
        async with session_factory() as session:
            first = await jd_module.parse_and_seed("job-1", "backend jd", session)
        async with session_factory() as session:
            second = await jd_module.parse_and_seed("job-1", "backend jd", session)
        async with session_factory() as session:
            rows = (await session.exec(select(ParsedJobContext))).all()
    finally:
        await engine.dispose()

    assert len(rows) == 1
    assert first.id == second.id
    assert rows[0].job_id == "job-1"
    assert rows[0].role_title == "Senior Backend Engineer"
