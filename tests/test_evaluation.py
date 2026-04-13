from __future__ import annotations

import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.models import InterviewMessage, InterviewSession, ParsedJobContext, RecommendedAction
from app.pipeline import evaluation


def _dimension() -> dict[str, object]:
    return {
        "name": "Technical Reasoning",
        "weight": 1.0,
        "description": "Explains trade-offs clearly.",
        "score_anchor_3": "Specific trade-offs and constraints.",
        "score_anchor_2": "Workable answer with some gaps.",
        "score_anchor_1": "Generic or unsupported claims.",
        "few_shot_3": "I compared designs and justified the trade-off.",
        "few_shot_2": "I described a plausible implementation path.",
        "few_shot_1": "I stayed generic and skipped constraints.",
        "sample_questions": ["Talk through a production debugging decision."],
    }


def _job_context() -> dict[str, object]:
    return {
        "role_title": "Backend Engineer",
        "seniority": "senior",
        "must_have_skills": ["Python", "Postgres"],
        "rubric": [_dimension()],
    }


class _FakeResult:
    def __init__(self, value: object) -> None:
        self.value = value

    def first(self) -> object:
        return self.value

    def all(self) -> object:
        return self.value


class _FakeSession:
    def __init__(self, results: list[object]) -> None:
        self.results = results
        self.exec_count = 0
        self.added: list[object] = []
        self.committed = False
        self.refreshed: list[object] = []

    async def exec(self, _: object) -> _FakeResult:
        result = _FakeResult(self.results[self.exec_count])
        self.exec_count += 1
        return result

    def add(self, item: object) -> None:
        self.added.append(item)

    async def commit(self) -> None:
        self.committed = True

    async def refresh(self, item: object) -> None:
        self.refreshed.append(item)


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


@pytest.mark.asyncio
async def test_majority_vote_two_to_one(monkeypatch: pytest.MonkeyPatch) -> None:
    replies = iter(
        [
            evaluation._JudgeReply("judge-a", 3, "Strong evidence."),
            evaluation._JudgeReply("judge-b", 3, "Consistent depth."),
            evaluation._JudgeReply("judge-c", 2, "Adequate but thinner."),
        ]
    )

    async def fake_ask(_: str, __: str) -> evaluation._JudgeReply:
        return next(replies)

    monkeypatch.setattr(evaluation, "_ask_judge", fake_ask)
    result = await evaluation.run_ensemble_judge(
        "Question?", "Answer.", _dimension(), _job_context()
    )
    assert result["score"] == 3
    assert result["votes"] == [3, 3, 2]
    assert result["is_flagged"] is False


@pytest.mark.asyncio
async def test_flag_on_large_spread(monkeypatch: pytest.MonkeyPatch) -> None:
    replies = iter(
        [
            evaluation._JudgeReply("judge-a", 1, "Weak."),
            evaluation._JudgeReply("judge-b", 3, "Strong."),
            evaluation._JudgeReply("judge-c", 2, "Mixed."),
        ]
    )

    async def fake_ask(_: str, __: str) -> evaluation._JudgeReply:
        return next(replies)

    monkeypatch.setattr(evaluation, "_ask_judge", fake_ask)
    result = await evaluation.run_ensemble_judge(
        "Question?", "Answer.", _dimension(), _job_context()
    )
    assert result["votes"] == [1, 3, 2]
    assert result["is_flagged"] is True


def test_score_parse_variants() -> None:
    assert evaluation._parse_score("SCORE: 2") == 2
    assert evaluation._parse_score("score:3") == 3
    assert evaluation._parse_score("SCORE:1\n") == 1


def test_default_score_on_malformed() -> None:
    assert evaluation._parse_score("No score here") == 2


@pytest.mark.asyncio
async def test_rank_candidates(monkeypatch: pytest.MonkeyPatch) -> None:
    scorecards = [
        SimpleNamespace(weighted_total=0.9, final_rank=None),
        SimpleNamespace(weighted_total=0.5, final_rank=None),
        SimpleNamespace(weighted_total=0.7, final_rank=None),
        SimpleNamespace(weighted_total=0.6, final_rank=None),
        SimpleNamespace(weighted_total=0.8, final_rank=None),
    ]
    session = _FakeSession([scorecards])
    factory = _Factory(session)
    publish = AsyncMock()
    monkeypatch.setattr(evaluation, "AsyncSessionFactory", factory)
    monkeypatch.setattr(evaluation.core, "redis", SimpleNamespace(publish=publish))

    await evaluation.rank_candidates("job-1")

    assert [item.final_rank for item in scorecards] == [1, 5, 3, 4, 2]
    publish.assert_awaited_once_with("job_events:job-1", "shortlist_ready")


def test_advance_boundary(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(evaluation.settings, "advance_threshold", 0.75)
    assert evaluation.recommended_action(0.75) == RecommendedAction.ADVANCE


def test_hold_boundary(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(evaluation.settings, "hold_threshold", 0.50)
    monkeypatch.setattr(evaluation.settings, "advance_threshold", 0.75)
    assert evaluation.recommended_action(0.50) == RecommendedAction.HOLD
    assert evaluation.recommended_action(0.499) == RecommendedAction.REJECT


@pytest.mark.asyncio
async def test_scorecard_opens_own_session(monkeypatch: pytest.MonkeyPatch) -> None:
    session_id = uuid.uuid4()
    interview = InterviewSession(
        id=session_id,
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
        integrity_flags=["tab_blur"],
    )
    job = ParsedJobContext(
        job_id="job-1",
        role_title="Backend Engineer",
        seniority="senior",
        domain="platform",
        must_have_skills=["Python", "Postgres"],
        rubric=[_dimension()],
        question_seed_topics=["debugging"],
        requires_coding=True,
        linting_warnings=[],
    )
    messages = [
        InterviewMessage(
            session_id=session_id,
            role="agent",
            content="Talk through a trade-off.",
            dimension_targeted="Technical Reasoning",
            sequence_number=1,
        ),
        InterviewMessage(
            session_id=session_id,
            role="candidate",
            content="I compare latency, cost, and operational complexity.",
            dimension_targeted="Technical Reasoning",
            sequence_number=1,
        ),
    ]
    session = _FakeSession([(interview, job), messages, None])
    factory = _Factory(session)
    monkeypatch.setattr(evaluation, "AsyncSessionFactory", factory)
    monkeypatch.setattr(
        evaluation,
        "run_ensemble_judge",
        AsyncMock(
            return_value={
                "dimension": "Technical Reasoning",
                "score": 3,
                "votes": [3, 3, 2],
                "reasoning": "Strong trade-off analysis.",
                "all_reasons": ["Strong trade-off analysis."] * 3,
                "is_flagged": False,
                "judge_results": [
                    {
                        "model": "judge-a",
                        "score": 3,
                        "reasoning": "Strong trade-off analysis.",
                    },
                    {
                        "model": "judge-b",
                        "score": 3,
                        "reasoning": "Strong trade-off analysis.",
                    },
                    {"model": "judge-c", "score": 2, "reasoning": "Acceptable depth."},
                ],
            }
        ),
    )

    scorecard = await evaluation.build_scorecard(str(session_id))

    assert factory.calls == 1
    assert scorecard.recommended_action == RecommendedAction.ADVANCE
    assert scorecard.integrity_flags == ["tab_blur"]
    assert scorecard.gaps == ["Postgres"]
    assert session.committed is True
