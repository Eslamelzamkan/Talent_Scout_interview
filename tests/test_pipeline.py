from __future__ import annotations

import importlib
import inspect
import os
import uuid
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient
from langchain_core.messages import AIMessage, HumanMessage
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlmodel import SQLModel, select
from sqlmodel.ext.asyncio.session import AsyncSession

from app.core import chroma as chroma_module
from app.core.config import get_settings
from app.core.llm import LLMClient
from app.models import (
    InterviewMessage,
    InterviewSession,
    ParsedJobContext,
    RecommendedAction,
    SessionStatus,
)
from app.pipeline import evaluation, interview
from app.pipeline import jd as jd_module
from app.routes import api as api_routes
from app.routes import candidate as candidate_routes
from app.routes import recruiter as recruiter_routes

llm_module = importlib.import_module("app.core.llm")


def _test_db_url() -> str:
    return os.environ.get("TEST_POSTGRES_URL") or get_settings().postgres_url


def _dimension(
    name: str,
    weight: float = 1 / 3,
    *,
    description: str | None = None,
) -> dict[str, object]:
    return {
        "name": name,
        "weight": weight,
        "description": description or f"Evaluate {name}",
        "score_anchor_3": f"Excellent {name}",
        "score_anchor_2": f"Acceptable {name}",
        "score_anchor_1": f"Weak {name}",
        "few_shot_3": f"Strong {name}",
        "few_shot_2": f"Average {name}",
        "few_shot_1": f"Weak {name}",
        "sample_questions": [f"Tell me about {name}."],
    }


def _job_context() -> dict[str, object]:
    return {
        "job_id": "job-1",
        "role_title": "Backend Engineer",
        "seniority": "senior",
        "rubric": [_dimension("Python"), _dimension("SQL"), _dimension("Design")],
    }


def _state(**overrides: object) -> interview.InterviewState:
    state: interview.InterviewState = {
        "session_id": "sess-1",
        "candidate_id": "cand-1",
        "job_id": "job-1",
        "candidate_name": "Alex",
        "job_context": _job_context(),
        "candidate_context": {},
        "weak_areas": [],
        "extracted_skills": [],
        "requires_coding": False,
        "interview_plan": {},
        "question_lane_counts": {},
        "messages": [],
        "questions_asked": [],
        "dimension_coverage": {"Python": 0, "SQL": 0, "Design": 0},
        "dimension_scores_live": {"Python": 0.0, "SQL": 0.0, "Design": 0.0},
        "current_question": None,
        "current_dimension": None,
        "current_question_lane": None,
        "current_question_focus": None,
        "candidate_answer": None,
        "answer_is_shallow": False,
        "follow_up_signals": [],
        "integrity_flags": [],
        "total_questions_asked": 0,
        "interview_complete": False,
        "hitl_requested": False,
        "recruiter_injected_question": None,
        "stt_fallback_active": False,
        "error": None,
    }
    state.update(overrides)
    return state


def _judge(score: int, model: str = "judge") -> evaluation._JudgeReply:
    return evaluation._JudgeReply(model=model, score=score, reasoning=f"score={score}")


def _settings() -> SimpleNamespace:
    return SimpleNamespace(
        openrouter_api_key="test-key",
        openrouter_base_url="https://openrouter.ai/api/v1",
        gemini_api_key="gemini-key",
        gemini_base_url="https://generativelanguage.googleapis.com/v1beta/openai",
        openrouter_headers={
            "Authorization": "Bearer test-key",
            "HTTP-Referer": "https://talentscout.ai",
            "X-Title": "Talent Scout",
        },
        gemini_headers={"Authorization": "Bearer gemini-key"},
        model_fast="google/gemini-2.0-flash-001",
        model_embedding="openai/text-embedding-3-large",
        rpm_limit_gpt4=500,
        rpm_limit_claude=400,
        rpm_limit_gemini=1000,
    )


def _intake_payload(
    job_id: str, candidate_count: int = 1, *, start_index: int = 0
) -> dict[str, object]:
    return {
        "recruiter_id": "rec-1",
        "job": {
            "job_id": job_id,
            "title": "Backend Engineer",
            "jd_text": "Build async backend services.",
            "requires_coding": True,
        },
        "candidates": [
            {
                "candidate_id": f"cand-{start_index + index}",
                "name": f"Candidate {start_index + index}",
                "email": f"cand-{start_index + index}@example.com",
                "screening_score": 0.8,
                "assessment_score": 0.75,
                "assessment_type": "verbal",
                "weak_areas": ["SQL"],
                "extracted_skills": ["Python"],
            }
            for index in range(candidate_count)
        ],
    }


async def _fake_parse_and_seed(
    job_id: str,
    _: str,
    session: AsyncSession,
    *,
    commit: bool = True,
    requires_coding: bool | None = None,
) -> ParsedJobContext:
    job = (
        await session.exec(select(ParsedJobContext).where(ParsedJobContext.job_id == job_id))
    ).first()
    payload = {
        "role_title": "Backend Engineer",
        "seniority": "senior",
        "domain": "platform",
        "must_have_skills": ["Python", "SQL"],
        "rubric": [_dimension("Python", 0.34), _dimension("SQL", 0.33), _dimension("Design", 0.33)],
        "question_seed_topics": ["python", "sql", "design"],
        "requires_coding": bool(requires_coding),
        "linting_warnings": [],
    }
    if job is None:
        job = ParsedJobContext(job_id=job_id, **payload)
    else:
        for field, value in payload.items():
            setattr(job, field, value)
    session.add(job)
    if commit:
        await session.commit()
    else:
        await session.flush()
    await session.refresh(job)
    return job


@pytest.fixture
async def session_factory():
    engine = create_async_engine(_test_db_url(), pool_pre_ping=True)
    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    try:
        async with engine.begin() as connection:
            await connection.run_sync(SQLModel.metadata.drop_all)
            await connection.run_sync(SQLModel.metadata.create_all)
    except Exception as exc:
        await engine.dispose()
        pytest.skip(f"postgres unavailable for integration tests: {exc}")
    try:
        yield factory
    finally:
        async with engine.begin() as connection:
            await connection.run_sync(SQLModel.metadata.drop_all)
        await engine.dispose()


def _test_app(session_factory: async_sessionmaker[AsyncSession]) -> FastAPI:
    app = FastAPI()
    app.include_router(api_routes.router)

    async def override_get_session():
        async with session_factory() as session:
            yield session

    from app.core.db import get_session
    app.dependency_overrides[get_session] = override_get_session
    return app


async def _seed_candidate_interview(
    session_factory: async_sessionmaker[AsyncSession],
    *,
    status: SessionStatus = SessionStatus.SCHEDULED,
) -> uuid.UUID:
    async with session_factory() as session:
        session.add(
            ParsedJobContext(
                job_id="job-candidate-ui",
                role_title="Backend Engineer",
                seniority="senior",
                domain="platform",
                must_have_skills=["Python"],
                rubric=[_dimension("Python", 1.0)],
                question_seed_topics=["python"],
                requires_coding=False,
                linting_warnings=[],
            )
        )
        interview_row = InterviewSession(
            candidate_id="cand-ui",
            job_id="job-candidate-ui",
            recruiter_id="rec-1",
            candidate_name="Taylor",
            candidate_email="taylor@example.com",
            livekit_room_name="job-candidate-ui-cand-ui",
            screening_score=0.8,
            assessment_score=0.75,
            assessment_type="verbal",
            status=status,
        )
        session.add(interview_row)
        await session.commit()
        await session.refresh(interview_row)
        return interview_row.id


class TestWeightNormalisation:
    def test_proportional_normalisation(self) -> None:
        result = jd_module._normalise_weights([{"weight": 0.3}, {"weight": 0.3}, {"weight": 0.3}])
        assert [item["weight"] for item in result] == pytest.approx([1 / 3, 1 / 3, 1 / 3], abs=1e-5)

    def test_idempotent_if_correct(self) -> None:
        assert jd_module._normalise_weights([{"weight": 0.4}, {"weight": 0.6}]) == [
            {"weight": 0.4},
            {"weight": 0.6},
        ]

    def test_single_dimension(self) -> None:
        assert jd_module._normalise_weights([{"weight": 0.5}]) == [{"weight": 1.0}]


class TestJudgeEnsemble:
    def test_majority_two_to_one(self) -> None:
        result = evaluation._aggregate_judges([_judge(3, "a"), _judge(3, "b"), _judge(2, "c")])
        assert result["score"] == 3 and result["is_flagged"] is False

    def test_flag_large_spread(self) -> None:
        result = evaluation._aggregate_judges([_judge(1, "a"), _judge(3, "b"), _judge(2, "c")])
        assert result["score"] == 1 and result["is_flagged"] is True

    def test_all_agree(self) -> None:
        result = evaluation._aggregate_judges([_judge(2, "a"), _judge(2, "b"), _judge(2, "c")])
        assert result["score"] == 2 and result["is_flagged"] is False

    @pytest.mark.parametrize(
        ("text", "expected", "warns"),
        [
            ("Reasoning\nSCORE: 2", 2, False),
            ("SCORE:3", 3, False),
            ("score: 1 - poor", 1, False),
            ("No score line", 2, True),
        ],
    )
    def test_score_parse_formats(
        self, monkeypatch: pytest.MonkeyPatch, text: str, expected: int, warns: bool
    ) -> None:
        warning = MagicMock()
        monkeypatch.setattr(evaluation, "log", SimpleNamespace(warning=warning))
        assert evaluation._parse_score(text) == expected
        assert warning.called is warns


class TestDimensionPicking:
    def test_weak_areas_first(self) -> None:
        assert (
            interview._pick_next_dimension(
                _state(weak_areas=["SQL"], dimension_coverage={"SQL": 0, "Python": 1, "Design": 1})
            )
            == "SQL"
        )

    def test_uncovered_beats_low_score(self) -> None:
        state = _state(
            dimension_coverage={"Python": 1, "SQL": 1, "Design": 0},
            dimension_scores_live={"Python": 1.0, "SQL": 1.0, "Design": 3.0},
        )
        assert interview._pick_next_dimension(state) == "Design"

    def test_lowest_score_when_all_covered(self) -> None:
        state = _state(
            dimension_coverage={"Python": 1, "SQL": 1, "Design": 1},
            dimension_scores_live={"Python": 2.5, "SQL": 1.2, "Design": 2.0},
        )
        assert interview._pick_next_dimension(state) == "SQL"

    def test_respects_cap_at_two(self) -> None:
        state = _state(
            dimension_coverage={"Python": 2, "SQL": 1, "Design": 1},
            dimension_scores_live={"Python": 0.1, "SQL": 1.0, "Design": 2.0},
        )
        assert interview._pick_next_dimension(state) == "SQL"

    def test_prefers_lane_compatible_dimension(self) -> None:
        state = _state(
            requires_coding=False,
            job_context={
                "job_id": "job-1",
                "role_title": "Product Analyst",
                "seniority": "senior",
                "rubric": [
                    _dimension("System Design", description="Technical architecture and trade-offs"),
                    _dimension(
                        "Communication",
                        description="Communication, collaboration, and stakeholder alignment",
                    ),
                ],
            },
            dimension_coverage={"System Design": 0, "Communication": 0},
            dimension_scores_live={"System Design": 0.0, "Communication": 0.0},
        )
        assert interview._pick_next_dimension(state, "behavioral") == "Communication"


class TestInterviewPlanning:
    def test_build_interview_plan_for_coding_role_includes_mix(self) -> None:
        state = _state(requires_coding=True)
        plan = interview._build_interview_plan(state)
        assert plan["target_total"] >= 4
        assert "technical_fundamentals" in plan["lane_sequence"]
        assert "project_deep_dive" in plan["lane_sequence"]

    def test_build_interview_plan_for_behavioral_dimension_includes_behavioral_lane(self) -> None:
        state = _state(
            job_context={
                "job_id": "job-1",
                "role_title": "Engineering Manager",
                "seniority": "senior",
                "rubric": [
                    _dimension("Architecture", description="System design and architecture"),
                    _dimension(
                        "Leadership",
                        description="Leadership, mentoring, ownership, and collaboration",
                    ),
                ],
            },
            dimension_coverage={"Architecture": 0, "Leadership": 0},
            dimension_scores_live={"Architecture": 0.0, "Leadership": 0.0},
        )
        plan = interview._build_interview_plan(state)
        assert "behavioral" in plan["lane_sequence"]

    def test_next_question_lane_reuses_current_lane_for_follow_up(self) -> None:
        state = _state(
            answer_is_shallow=True,
            current_question_lane="project_deep_dive",
            interview_plan={
                "target_total": 4,
                "lane_sequence": ["technical_fundamentals", "project_deep_dive"],
                "lane_targets": {"technical_fundamentals": 1, "project_deep_dive": 1},
            },
            total_questions_asked=1,
        )
        assert interview._next_question_lane(state) == "project_deep_dive"

    def test_process_answer_route_skips_live_scoring_for_follow_up(self) -> None:
        assert interview._after_process_answer_route(_state(answer_is_shallow=True)) == "generate_question"
        assert interview._after_process_answer_route(_state(answer_is_shallow=False)) == "evaluate_answer"


class _FakeResult:
    def __init__(self, value: object) -> None:
        self._value = value

    def first(self) -> object:
        return self._value

    def all(self) -> object:
        return self._value


class _FakeSession:
    def __init__(self, results: list[object]) -> None:
        self.results = results
        self.calls = 0
        self.committed = False

    async def exec(self, _: object) -> _FakeResult:
        result = _FakeResult(self.results[self.calls])
        self.calls += 1
        return result

    def add(self, _: object) -> None:
        return None

    async def commit(self) -> None:
        self.committed = True

    async def refresh(self, _: object) -> None:
        return None


class _Factory:
    def __init__(self, session: _FakeSession) -> None:
        self.session = session
        self.calls = 0

    def __call__(self) -> _Factory:
        self.calls += 1
        return self

    async def __aenter__(self) -> _FakeSession:
        return self.session

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> bool:
        return False


class _FakeInterviewGraph:
    def __init__(self) -> None:
        self.states: dict[str, dict[str, object]] = {}

    def _thread_id(self, config: dict[str, object]) -> str:
        configurable = config.get("configurable", {})
        assert isinstance(configurable, dict)
        return str(configurable["thread_id"])

    async def aget_state(self, config: dict[str, object]) -> SimpleNamespace:
        return SimpleNamespace(values=self.states.get(self._thread_id(config), {}), next=())

    async def aupdate_state(self, config: dict[str, object], updates: dict[str, object]) -> None:
        thread_id = self._thread_id(config)
        state = dict(self.states.get(thread_id, {}))
        state.update(updates)
        self.states[thread_id] = state

    async def ainvoke(
        self, payload: dict[str, object] | None, config: dict[str, object]
    ) -> dict[str, object]:
        thread_id = self._thread_id(config)
        state = dict(self.states.get(thread_id, {}))
        if payload:
            state.update(payload)
        if not state.get("messages"):
            question = "Tell me about your Python backend work."
            state.update(
                {
                    "current_question": question,
                    "current_dimension": "Python",
                    "messages": [
                        AIMessage(
                            content=question,
                            additional_kwargs={
                                "dimension": "Python",
                                "timestamp": "2026-04-09T00:00:00+00:00",
                            },
                        )
                    ],
                    "questions_asked": [question],
                    "total_questions_asked": 1,
                    "candidate_answer": None,
                    "answer_is_shallow": False,
                    "interview_complete": False,
                }
            )
        elif state.get("candidate_answer"):
            answer = str(state["candidate_answer"])
            state["messages"] = [
                *list(state["messages"]),
                HumanMessage(
                    content=answer,
                    additional_kwargs={
                        "dimension": str(state.get("current_dimension") or "Python"),
                        "timestamp": "2026-04-09T00:01:00+00:00",
                    },
                ),
            ]
            state.update(
                {
                    "candidate_answer": None,
                    "current_question": None,
                    "answer_is_shallow": False,
                    "interview_complete": True,
                }
            )
        self.states[thread_id] = state
        return state


class TestAggregatorBoundaries:
    def test_advance_at_threshold(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(evaluation.settings, "advance_threshold", 0.75)
        assert evaluation.recommended_action(0.75) == RecommendedAction.ADVANCE

    def test_hold_at_threshold(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(evaluation.settings, "advance_threshold", 0.75)
        monkeypatch.setattr(evaluation.settings, "hold_threshold", 0.50)
        assert evaluation.recommended_action(0.50) == RecommendedAction.HOLD

    def test_reject_below_hold(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(evaluation.settings, "hold_threshold", 0.50)
        assert evaluation.recommended_action(0.499) == RecommendedAction.REJECT

    def test_clamping(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(evaluation.settings, "interview_weight", 1.0)
        monkeypatch.setattr(evaluation.settings, "assessment_weight", 0.0)
        monkeypatch.setattr(evaluation.settings, "speech_weight", 0.0)
        monkeypatch.setattr(evaluation.settings, "screening_weight", 0.0)
        assert evaluation.weighted_total(1.0000001, 0.0, 0.0, 0.0) == 1.0

    @pytest.mark.asyncio
    async def test_build_scorecard_uses_session_factory(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        session_id = uuid.uuid4()
        interview_row = InterviewSession(
            candidate_id="cand-1",
            job_id="job-1",
            recruiter_id="rec-1",
            candidate_name="Alex",
            candidate_email="alex@example.com",
            livekit_room_name="job-1-cand-1",
            screening_score=0.8,
            assessment_score=0.7,
            assessment_type="verbal",
            extracted_skills=["Python"],
        )
        interview_row.id = session_id
        job_row = ParsedJobContext(
            job_id="job-1",
            role_title="Backend Engineer",
            seniority="senior",
            domain="platform",
            must_have_skills=["Python", "SQL"],
            rubric=[_dimension("Python", 1.0)],
            question_seed_topics=["python"],
            requires_coding=False,
            linting_warnings=[],
        )
        messages = [
            InterviewMessage(
                session_id=session_id,
                role="agent",
                content="Tell me about Python.",
                dimension_targeted="Python",
                sequence_number=1,
            ),
            InterviewMessage(
                session_id=session_id,
                role="candidate",
                content="I optimize async services.",
                dimension_targeted="Python",
                sequence_number=1,
            ),
        ]
        factory = _Factory(_FakeSession([(interview_row, job_row), messages, None]))
        monkeypatch.setattr(evaluation, "AsyncSessionFactory", factory)
        monkeypatch.setattr(
            evaluation,
            "run_ensemble_judge",
            AsyncMock(
                return_value={
                    "dimension": "Python",
                    "score": 3,
                    "votes": [3, 3, 2],
                    "reasoning": "Strong.",
                    "all_reasons": ["Strong."] * 3,
                    "is_flagged": False,
                    "judge_results": [
                        {"model": "a", "score": 3, "reasoning": "Strong."},
                        {"model": "b", "score": 3, "reasoning": "Strong."},
                        {"model": "c", "score": 2, "reasoning": "Acceptable."},
                    ],
                }
            ),
        )
        scorecard = await evaluation.build_scorecard(str(session_id))
        assert factory.calls == 1 and scorecard.recommended_action == RecommendedAction.ADVANCE
        assert "AsyncSessionFactory()" in inspect.getsource(evaluation.build_scorecard)
        assert "get_session" not in inspect.getsource(evaluation.build_scorecard)

    @pytest.mark.asyncio
    async def test_build_scorecard_uses_graph_metadata_to_skip_unscoreable_answers(
        self, monkeypatch: pytest.MonkeyPatch, session_factory
    ) -> None:
        async with session_factory() as session:
            job_row = ParsedJobContext(
                job_id="job-graph",
                role_title="Backend Engineer",
                seniority="senior",
                domain="platform",
                must_have_skills=["Python"],
                rubric=[_dimension("Python", 1.0)],
                question_seed_topics=["python"],
                requires_coding=False,
                linting_warnings=[],
            )
            interview_row = InterviewSession(
                candidate_id="cand-graph",
                job_id="job-graph",
                recruiter_id="rec-1",
                candidate_name="Alex",
                candidate_email="alex@example.com",
                livekit_room_name="job-graph-cand-graph",
                screening_score=0.8,
                assessment_score=0.7,
                assessment_type="verbal",
            )
            interview_row.langgraph_thread_id = str(interview_row.id)
            session.add(job_row)
            session.add(interview_row)
            await session.commit()
            session_id = interview_row.id

        fake_graph = _FakeInterviewGraph()
        fake_graph.states[str(session_id)] = {
            "messages": [
                AIMessage(
                    content="Tell me about Python.",
                    additional_kwargs={"dimension": "Python", "timestamp": "2026-04-09T00:00:00+00:00"},
                ),
                HumanMessage(
                    content="It was good.",
                    additional_kwargs={
                        "dimension": "Python",
                        "timestamp": "2026-04-09T00:01:00+00:00",
                        "scoreable": False,
                    },
                ),
                AIMessage(
                    content="Give me a concrete backend example.",
                    additional_kwargs={
                        "dimension": "Python",
                        "timestamp": "2026-04-09T00:02:00+00:00",
                        "follow_up": True,
                    },
                ),
                HumanMessage(
                    content="I improved async API latency by 35% after rewriting the query path.",
                    additional_kwargs={
                        "dimension": "Python",
                        "timestamp": "2026-04-09T00:03:00+00:00",
                        "scoreable": True,
                    },
                ),
            ]
        }
        judge = AsyncMock(
            return_value={
                "dimension": "Python",
                "score": 3,
                "votes": [3, 3, 3],
                "reasoning": "Strong.",
                "all_reasons": ["Strong."] * 3,
                "is_flagged": False,
                "judge_results": [
                    {"model": "a", "score": 3, "reasoning": "Strong."},
                    {"model": "b", "score": 3, "reasoning": "Strong."},
                    {"model": "c", "score": 3, "reasoning": "Strong."},
                ],
            }
        )
        monkeypatch.setattr(evaluation, "AsyncSessionFactory", session_factory)
        monkeypatch.setattr(evaluation.core, "interview_graph", fake_graph)
        monkeypatch.setattr(evaluation, "run_ensemble_judge", judge)

        scorecard = await evaluation.build_scorecard(str(session_id))

        assert judge.await_count == 1
        assert judge.await_args.args[:2] == (
            "Give me a concrete backend example.",
            "I improved async API latency by 35% after rewriting the query path.",
        )
        assert scorecard.interview_dimension_scores["Python"]["score"] == 3


class TestIntakeAPI:
    @pytest.mark.asyncio
    async def test_batch_creates_sessions(
        self, monkeypatch: pytest.MonkeyPatch, session_factory
    ) -> None:
        monkeypatch.setattr(recruiter_routes, "parse_and_seed", _fake_parse_and_seed)
        async with AsyncClient(
            transport=ASGITransport(app=_test_app(session_factory)), base_url="http://test"
        ) as client:
            response = await client.post("/intake/batch", json=_intake_payload("job-batch", 3))
        async with session_factory() as session:
            sessions = (
                await session.exec(
                    select(InterviewSession).where(InterviewSession.job_id == "job-batch")
                )
            ).all()
        assert (
            response.status_code == 200
            and len(response.json()["interview_session_ids"]) == 3
            and len(sessions) == 3
        )

    @pytest.mark.asyncio
    async def test_jd_parse_runs_before_sessions(
        self, monkeypatch: pytest.MonkeyPatch, session_factory
    ) -> None:
        parsed_jobs: set[str] = set()

        async def tracked_parse(*args, **kwargs):
            job = await _fake_parse_and_seed(*args, **kwargs)
            parsed_jobs.add(job.job_id)
            return job

        def tracked_session(*args, **kwargs):
            assert kwargs["job_id"] in parsed_jobs
            return InterviewSession(*args, **kwargs)

        monkeypatch.setattr(recruiter_routes, "parse_and_seed", tracked_parse)
        monkeypatch.setattr(recruiter_routes, "InterviewSession", tracked_session)
        async with AsyncClient(
            transport=ASGITransport(app=_test_app(session_factory)), base_url="http://test"
        ) as client:
            response = await client.post("/intake/batch", json=_intake_payload("job-ordered", 1))
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_atomic_rollback(self, monkeypatch: pytest.MonkeyPatch, session_factory) -> None:
        calls = 0

        def flaky_session(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("candidate insert failed")
            return InterviewSession(*args, **kwargs)

        monkeypatch.setattr(recruiter_routes, "parse_and_seed", _fake_parse_and_seed)
        monkeypatch.setattr(recruiter_routes, "InterviewSession", flaky_session)
        transport = ASGITransport(app=_test_app(session_factory), raise_app_exceptions=False)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post("/intake/batch", json=_intake_payload("job-rollback", 3))
        async with session_factory() as session:
            sessions = (
                await session.exec(
                    select(InterviewSession).where(InterviewSession.job_id == "job-rollback")
                )
            ).all()
            jobs = (
                await session.exec(
                    select(ParsedJobContext).where(ParsedJobContext.job_id == "job-rollback")
                )
            ).all()
        assert response.status_code == 500 and sessions == [] and jobs == []

    @pytest.mark.asyncio
    async def test_idempotent_jd_parse(
        self, monkeypatch: pytest.MonkeyPatch, session_factory
    ) -> None:
        monkeypatch.setattr(recruiter_routes, "parse_and_seed", _fake_parse_and_seed)
        async with AsyncClient(
            transport=ASGITransport(app=_test_app(session_factory)), base_url="http://test"
        ) as client:
            await client.post("/intake/batch", json=_intake_payload("job-idempotent", 1))
            await client.post(
                "/intake/batch",
                json=_intake_payload("job-idempotent", 1, start_index=1),
            )
        async with session_factory() as session:
            jobs = (
                await session.exec(
                    select(ParsedJobContext).where(ParsedJobContext.job_id == "job-idempotent")
                )
            ).all()
        assert len(jobs) == 1


async def _seed_session(
    session_factory: async_sessionmaker[AsyncSession], status: SessionStatus
) -> uuid.UUID:
    async with session_factory() as session:
        interview_row = InterviewSession(
            candidate_id=f"cand-{status.value}",
            job_id="job-integrity",
            recruiter_id="rec-1",
            candidate_name="Alex",
            candidate_email="alex@example.com",
            livekit_room_name=f"room-{status.value}",
            screening_score=0.8,
            assessment_score=0.7,
            assessment_type="verbal",
            status=status,
        )
        session.add(interview_row)
        await session.commit()
        await session.refresh(interview_row)
        return interview_row.id


class TestIntegrityFlags:
    @pytest.mark.asyncio
    async def test_flag_accepted_on_completed_session(self, session_factory) -> None:
        session_id = await _seed_session(session_factory, SessionStatus.COMPLETED)
        async with AsyncClient(
            transport=ASGITransport(app=_test_app(session_factory)), base_url="http://test"
        ) as client:
            response = await client.post(
                f"/sessions/{session_id}/integrity_flag",
                json={"flag_type": "tab_blur", "timestamp": datetime.now(timezone.utc).isoformat()},
            )
        async with session_factory() as session:
            stored = await session.get(InterviewSession, session_id)
        assert (
            response.status_code == 200
            and stored is not None
            and stored.integrity_flags == ["tab_blur"]
        )

    @pytest.mark.asyncio
    async def test_flag_accepted_on_in_progress_session(self, session_factory) -> None:
        session_id = await _seed_session(session_factory, SessionStatus.IN_PROGRESS)
        async with AsyncClient(
            transport=ASGITransport(app=_test_app(session_factory)), base_url="http://test"
        ) as client:
            response = await client.post(
                f"/sessions/{session_id}/integrity_flag",
                json={"flag_type": "tab_blur", "timestamp": datetime.now(timezone.utc).isoformat()},
            )
        async with session_factory() as session:
            stored = await session.get(InterviewSession, session_id)
        assert (
            response.status_code == 200
            and stored is not None
            and stored.integrity_flags == ["tab_blur"]
        )

    @pytest.mark.asyncio
    async def test_flag_accepted_on_scheduled_session(self, session_factory) -> None:
        session_id = await _seed_session(session_factory, SessionStatus.SCHEDULED)
        async with AsyncClient(
            transport=ASGITransport(app=_test_app(session_factory)), base_url="http://test"
        ) as client:
            response = await client.post(
                f"/sessions/{session_id}/integrity_flag",
                json={"flag_type": "tab_blur", "timestamp": datetime.now(timezone.utc).isoformat()},
            )
        async with session_factory() as session:
            stored = await session.get(InterviewSession, session_id)
        assert (
            response.status_code == 200
            and stored is not None
            and stored.integrity_flags == ["tab_blur"]
        )

    @pytest.mark.asyncio
    async def test_record_integrity_flag_persists(self, monkeypatch: pytest.MonkeyPatch, session_factory) -> None:
        session_id = await _seed_session(session_factory, SessionStatus.IN_PROGRESS)
        monkeypatch.setattr(candidate_routes, "AsyncSessionFactory", session_factory)

        await candidate_routes._record_integrity_flag(session_id, "cand-in_progress", "tab_blur")

        async with session_factory() as session:
            stored = await session.get(InterviewSession, session_id)
        assert stored is not None
        assert stored.integrity_flags == ["tab_blur"]


class TestCandidateInterviewAPI:
    @pytest.mark.asyncio
    async def test_session_info_returns_candidate_metadata(self, session_factory) -> None:
        session_id = await _seed_candidate_interview(session_factory)
        async with AsyncClient(
            transport=ASGITransport(app=_test_app(session_factory)), base_url="http://test"
        ) as client:
            response = await client.get(f"/sessions/{session_id}/info")
        assert response.status_code == 200
        assert response.json() == {
            "session_id": str(session_id),
            "candidate_id": "cand-ui",
            "candidate_name": "Taylor",
            "role_title": "Backend Engineer",
            "room_name": "job-candidate-ui-cand-ui",
            "max_questions": get_settings().max_questions,
            "status": "scheduled",
        }

    @pytest.mark.asyncio
    async def test_candidate_websocket_streams_question_and_completion(
        self,
        monkeypatch: pytest.MonkeyPatch,
        session_factory,
    ) -> None:
        session_id = uuid.uuid4()
        interview_row = InterviewSession(
            candidate_id="cand-ui",
            job_id="job-candidate-ui",
            recruiter_id="rec-1",
            candidate_name="Taylor",
            candidate_email="taylor@example.com",
            livekit_room_name="job-candidate-ui-cand-ui",
            screening_score=0.8,
            assessment_score=0.75,
            assessment_type="verbal",
            status=SessionStatus.IN_PROGRESS,
        )
        interview_row.id = session_id
        job_row = ParsedJobContext(
            job_id="job-candidate-ui",
            role_title="Backend Engineer",
            seniority="senior",
            domain="platform",
            must_have_skills=["Python"],
            rubric=[_dimension("Python", 1.0)],
            question_seed_topics=["python"],
            requires_coding=False,
            linting_warnings=[],
        )

        async def fake_mark_started(_: uuid.UUID):
            return interview_row, job_row, True

        publish_event = AsyncMock()
        monkeypatch.setattr(candidate_routes, "_mark_session_started", fake_mark_started)
        monkeypatch.setattr(candidate_routes, "_publish_job_event", publish_event)
        monkeypatch.setattr(candidate_routes.core, "interview_graph", _FakeInterviewGraph())
        monkeypatch.setattr(candidate_routes.core, "redis", None)
        with TestClient(_test_app(session_factory)) as client:
            with client.websocket_connect(f"/ws/interviews/{session_id}") as websocket:
                initial = websocket.receive_json()
                assert initial["event"] == "interview_state"
                assert initial["data"]["current_question"] == "Tell me about your Python backend work."
                websocket.send_text("I build async APIs with Python and Postgres.")
                completed = websocket.receive_json()
                assert completed["event"] == "interview_complete"
                assert completed["data"]["interview_complete"] is True
                assert completed["data"]["transcript"][-1]["content"] == (
                    "I build async APIs with Python and Postgres."
                )
        assert publish_event.await_count == 1

    @pytest.mark.asyncio
    async def test_complete_session_persists_graph_state_updates(
        self, monkeypatch: pytest.MonkeyPatch, session_factory
    ) -> None:
        session_id = await _seed_candidate_interview(session_factory, status=SessionStatus.IN_PROGRESS)
        graph_state = {
            "candidate_context": {
                "summary": "Backend engineer",
                "live_answer_evidence": ["Reduced latency by 35%"],
            },
            "integrity_flags": ["tab_blur"],
            "extracted_skills": ["Python", "Postgres"],
            "messages": [
                AIMessage(
                    content="Tell me about Python.",
                    additional_kwargs={
                        "dimension": "Python",
                        "lane": "technical_fundamentals",
                        "focus": "reasoning and trade-offs",
                        "timestamp": "2026-04-09T00:00:00+00:00",
                    },
                ),
                HumanMessage(
                    content="I reduced latency by 35% with async APIs.",
                    additional_kwargs={
                        "dimension": "Python",
                        "lane": "technical_fundamentals",
                        "focus": "reasoning and trade-offs",
                        "scoreable": True,
                        "timestamp": "2026-04-09T00:01:00+00:00",
                    },
                ),
            ],
        }
        monkeypatch.setattr(candidate_routes, "_graph_state_values", AsyncMock(return_value=graph_state))
        monkeypatch.setattr(candidate_routes, "_publish_job_event", AsyncMock())
        monkeypatch.setattr(candidate_routes, "_post_interview_pipeline", AsyncMock())

        async with AsyncClient(
            transport=ASGITransport(app=_test_app(session_factory)),
            base_url="http://test",
        ) as client:
            response = await client.post(
                f"/sessions/{session_id}/complete",
                json={"transcript": "candidate transcript"},
            )

        async with session_factory() as session:
            stored = await session.get(InterviewSession, session_id)
            messages = (
                await session.exec(
                    select(InterviewMessage)
                    .where(InterviewMessage.session_id == session_id)
                    .order_by(InterviewMessage.sequence_number, InterviewMessage.created_at)
                )
            ).all()

        assert response.status_code == 200
        assert stored is not None
        assert stored.integrity_flags == ["tab_blur"]
        assert stored.candidate_context["live_answer_evidence"] == ["Reduced latency by 35%"]
        assert stored.extracted_skills == ["Python", "Postgres"]
        assert len(messages) == 2
        assert messages[0].lane == "technical_fundamentals"
        assert messages[1].scoreable is True

    @pytest.mark.asyncio
    async def test_candidate_websocket_integrity_flag_invokes_persistence(
        self, monkeypatch: pytest.MonkeyPatch, session_factory
    ) -> None:
        session_id = uuid.uuid4()
        interview_row = InterviewSession(
            candidate_id="cand-ui",
            job_id="job-candidate-ui",
            recruiter_id="rec-1",
            candidate_name="Taylor",
            candidate_email="taylor@example.com",
            livekit_room_name="job-candidate-ui-cand-ui",
            screening_score=0.8,
            assessment_score=0.75,
            assessment_type="verbal",
            status=SessionStatus.IN_PROGRESS,
        )
        interview_row.id = session_id
        interview_row.langgraph_thread_id = str(session_id)
        job_row = ParsedJobContext(
            job_id="job-candidate-ui",
            role_title="Backend Engineer",
            seniority="senior",
            domain="platform",
            must_have_skills=["Python"],
            rubric=[_dimension("Python", 1.0)],
            question_seed_topics=["python"],
            requires_coding=False,
            linting_warnings=[],
        )

        async def fake_mark_started(_: uuid.UUID):
            return interview_row, job_row, False

        fake_graph = _FakeInterviewGraph()
        record_integrity_flag = AsyncMock()
        monkeypatch.setattr(candidate_routes, "_mark_session_started", fake_mark_started)
        monkeypatch.setattr(candidate_routes, "_publish_job_event", AsyncMock())
        monkeypatch.setattr(candidate_routes, "_record_integrity_flag", record_integrity_flag)
        monkeypatch.setattr(candidate_routes.core, "interview_graph", fake_graph)
        monkeypatch.setattr(candidate_routes.core, "redis", None)

        async with session_factory() as session:
            session.add(job_row)
            session.add(interview_row)
            await session.commit()

        with TestClient(_test_app(session_factory)) as client:
            with client.websocket_connect(f"/ws/interviews/{session_id}") as websocket:
                websocket.receive_json()
                websocket.send_json({"event": "integrity_flag", "flag_type": "tab_blur"})
                state = websocket.receive_json()
                assert state["event"] == "interview_state"
        record_integrity_flag.assert_awaited_once()


class TestChromaSeeding:
    @pytest.mark.asyncio
    async def test_seed_replaces_existing_collection(self, monkeypatch: pytest.MonkeyPatch) -> None:
        collection = MagicMock()
        collection.upsert = AsyncMock()
        chroma_client = MagicMock()
        chroma_client.delete_collection = AsyncMock()
        chroma_client.get_or_create_collection = AsyncMock(return_value=collection)
        llm = MagicMock()
        llm.embed = AsyncMock(return_value=[[0.1, 0.2]])

        monkeypatch.setattr(chroma_module, "_chroma", chroma_client)
        monkeypatch.setattr(chroma_module, "_llm", llm)

        count = await chroma_module.seed(
            "job-1",
            [
                {
                    "text": "Tell me about Python.",
                    "dimension": "Python",
                    "seniority": "senior",
                    "topic": "python",
                    "lane": "technical_fundamentals",
                }
            ],
        )

        assert count == 1
        chroma_client.delete_collection.assert_awaited_once_with("questions_job-1")
        collection.upsert.assert_awaited_once()


class TestRateLimiter:
    @pytest.mark.asyncio
    async def test_rate_limiter_triggers_sleep(self, monkeypatch: pytest.MonkeyPatch) -> None:
        sleep = AsyncMock()
        monkeypatch.setattr(llm_module.asyncio, "sleep", sleep)
        redis_client = MagicMock()
        redis_client.incr = AsyncMock(return_value=int(_settings().rpm_limit_gpt4 * 0.9))
        redis_client.expire = AsyncMock()
        await LLMClient(
            _settings(), redis_client=redis_client, http_client=MagicMock()
        )._apply_rate_limit("openai/gpt-4.1")
        assert sleep.await_count == 1 and sleep.await_args.args[0] > 0

    @pytest.mark.asyncio
    async def test_rate_limiter_no_sleep_below_threshold(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        sleep = AsyncMock()
        monkeypatch.setattr(llm_module.asyncio, "sleep", sleep)
        redis_client = MagicMock()
        redis_client.incr = AsyncMock(return_value=10)
        redis_client.expire = AsyncMock()
        await LLMClient(
            _settings(), redis_client=redis_client, http_client=MagicMock()
        )._apply_rate_limit("openai/gpt-4.1")
        sleep.assert_not_awaited()
