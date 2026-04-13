from __future__ import annotations

import asyncio
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import groupby
from typing import Any
from uuid import UUID

import structlog
from langchain_core.messages import AIMessage, BaseMessage
from sqlmodel import select

import app.core as core
from app.core.config import get_settings
from app.core.constants import message_content as _message_content
from app.core.constants import meta_bool as _meta_bool
from app.core.db import AsyncSessionFactory
from app.models import (
    AuditLog,
    DimensionScore,
    InterviewMessage,
    InterviewSession,
    JudgeDimensionVote,
    JudgeEnsembleRaw,
    JudgeModelResult,
    ParsedJobContext,
    RecommendedAction,
    RubricDimension,
    RubricModel,
    Scorecard,
)

settings = get_settings()
log = structlog.get_logger()
JUDGE_PROMPT = """Role: evaluating a job interview answer for {role_title} ({seniority}).
Dimension: {name} \u2014 {description}
Scoring (ONLY 1, 2, or 3):
  3 (Excellent): {anchor_3}
  2 (Acceptable): {anchor_2}
  1 (Poor):       {anchor_1}
Calibration examples (ORDER IS SHUFFLED PER JUDGE \u2014 see bias mitigation):
  Example A: {shot_a}
  Example B: {shot_b}
  Example C: {shot_c}
Question: {question}
Answer:   {answer}
Write reasoning (2\u20133 sentences). Then: SCORE: <1|2|3>"""


@dataclass(frozen=True)
class _JudgeReply:
    model: str
    score: int
    reasoning: str


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)




def _job_context_payload(job_context: ParsedJobContext | dict[str, Any]) -> dict[str, Any]:
    if isinstance(job_context, ParsedJobContext):
        return job_context.model_dump(mode="json", by_alias=True)
    return dict(job_context)


def _judge_models() -> tuple[str, str, str]:
    return settings.model_judge_1, settings.model_judge_2, settings.model_judge_3


def _display_score(value: float) -> int:
    return 1 if value < 1.5 else 2 if value < 2.5 else 3


def _content_text(message: BaseMessage) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    return " ".join(
        item if isinstance(item, str) else str(item.get("text") or "") for item in content or []
    ).strip()




def _transcript_rows_from_messages(messages: list[BaseMessage]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for message in messages:
        meta = message.additional_kwargs or {}
        rows.append(
            {
                "role": "agent" if isinstance(message, AIMessage) else "candidate",
                "content": _content_text(message),
                "dimension": meta.get("dimension"),
                "scoreable": _meta_bool(meta.get("scoreable")),
            }
        )
    return rows


async def _graph_transcript_rows(interview: InterviewSession) -> list[dict[str, Any]] | None:
    if core.interview_graph is None:
        return None
    try:
        snapshot = await core.interview_graph.aget_state(
            {"configurable": {"thread_id": interview.langgraph_thread_id or str(interview.id)}}
        )
    except Exception:
        log.warning("graph_transcript_load_failed", session_id=str(interview.id))
        return None
    values = snapshot.values or {}
    messages = values.get("messages") or []
    if not isinstance(messages, list) or not messages:
        return None
    return _transcript_rows_from_messages(messages)


def _pair_transcript_rows(
    rows: list[dict[str, Any]],
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    question: dict[str, Any] | None = None
    for row in rows:
        role = str(row.get("role") or "").lower()
        if role in {"agent", "assistant", "ai"}:
            question = row
        elif question is not None and role in {"candidate", "user", "human"}:
            pairs.append((question, row))
            question = None
    return pairs


def _session_identity(session_id: str) -> UUID | str:
    try:
        return UUID(session_id)
    except ValueError:
        return session_id


def _pair_messages(
    messages: list[InterviewMessage],
) -> list[tuple[InterviewMessage, InterviewMessage]]:
    ordered = sorted(messages, key=lambda item: (item.sequence_number, item.created_at))
    pairs: list[tuple[InterviewMessage, InterviewMessage]] = []
    for _, group in groupby(ordered, key=lambda item: item.sequence_number):
        question: InterviewMessage | None = None
        for message in group:
            role = message.role.lower()
            if role in {"agent", "assistant", "ai"}:
                question = message
            elif question is not None and role in {"candidate", "user", "human"}:
                pairs.append((question, message))
                question = None
    return pairs


def _build_judge_prompt(
    question: str,
    answer: str,
    dimension: RubricDimension,
    job_context: ParsedJobContext | dict[str, Any],
    *,
    shuffle_seed: int,
) -> str:
    payload = _job_context_payload(job_context)
    shots = [
        getattr(dimension, f"few_shot_{index}")
        for index in random.Random(shuffle_seed).sample([1, 2, 3], 3)
    ]
    return JUDGE_PROMPT.format(
        role_title=payload.get("role_title") or payload.get("title") or "unknown",
        seniority=payload.get("seniority") or payload.get("role_level") or "unknown",
        name=dimension.name,
        description=dimension.description,
        anchor_3=dimension.score_anchor_3,
        anchor_2=dimension.score_anchor_2,
        anchor_1=dimension.score_anchor_1,
        shot_a=shots[0],
        shot_b=shots[1],
        shot_c=shots[2],
        question=question,
        answer=answer,
    )


def _parse_score(text: str) -> int:
    match = re.search(r"SCORE:\s*([123])", text, re.IGNORECASE)
    if match:
        return int(match.group(1))
    log.warning("judge_score_parse_failed", preview=text[:120])
    return 2


def _clean_reasoning(text: str) -> str:
    cleaned = re.sub(r"SCORE:\s*[123]\s*$", "", text.strip(), flags=re.IGNORECASE).strip()
    return cleaned or "Model did not provide reasoning."


async def _ask_judge(model: str, prompt: str) -> _JudgeReply:
    if core.llm is None:
        raise RuntimeError("llm client is not initialised")
    response = await core.llm.chat(
        model,
        [{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=220,
    )
    text = _message_content(response)
    return _JudgeReply(
        model=model,
        score=_parse_score(text),
        reasoning=_clean_reasoning(text),
    )


def _aggregate_judges(results: list[_JudgeReply]) -> dict[str, Any]:
    votes = [item.score for item in results]
    majority = Counter(votes).most_common(1)[0][0]
    return {
        "score": majority,
        "votes": votes,
        "reasoning": next(item.reasoning for item in results if item.score == majority),
        "all_reasons": [item.reasoning for item in results],
        "is_flagged": (max(votes) - min(votes)) >= 2,
        "judge_results": [
            {"model": item.model, "score": item.score, "reasoning": item.reasoning}
            for item in results
        ],
    }


async def run_ensemble_judge(
    question: str,
    answer: str,
    dimension: RubricDimension | dict[str, Any],
    job_context: ParsedJobContext | dict[str, Any],
) -> dict[str, Any]:
    dimension_model = RubricDimension.model_validate(dimension)
    prompts = [
        _build_judge_prompt(question, answer, dimension_model, job_context, shuffle_seed=index)
        for index in range(3)
    ]
    results = await asyncio.gather(
        *[
            _ask_judge(model, prompt)
            for model, prompt in zip(_judge_models(), prompts, strict=True)
        ]
    )
    aggregated = _aggregate_judges(list(results))
    aggregated["dimension"] = dimension_model.name
    return aggregated


def weighted_total(
    interview_score: float,
    assessment_score: float,
    speech_score: float,
    screening_score: float,
) -> float:
    total = (
        interview_score * settings.interview_weight
        + assessment_score * settings.assessment_weight
        + speech_score * settings.speech_weight
        + screening_score * settings.screening_weight
    )
    return max(0.0, min(1.0, total))


def recommended_action(total: float) -> RecommendedAction:
    if total >= settings.advance_threshold:
        return RecommendedAction.ADVANCE
    if total >= settings.hold_threshold:
        return RecommendedAction.HOLD
    return RecommendedAction.REJECT


def aggregate_dimension_scores(
    ensemble: JudgeEnsembleRaw,
    rubric: RubricModel,
) -> tuple[dict[str, DimensionScore], float, list[str], list[str], list[str]]:
    scores: dict[str, DimensionScore] = {}
    interview_signal = 0.0
    strengths: list[str] = []
    gaps: list[str] = []
    bias_flags: list[str] = []
    for dimension in rubric.dimensions:
        votes = [
            vote
            for judge in ensemble.judges
            for vote in judge.votes
            if vote.dimension == dimension.name
        ]
        raw_votes = [vote.score for vote in votes] or [2]
        score = Counter(raw_votes).most_common(1)[0][0]
        flagged = (max(raw_votes) - min(raw_votes)) >= 2
        scores[dimension.name] = DimensionScore(
            score=score,
            weight=dimension.weight,
            judge_votes=raw_votes,
            reasoning=next(
                (vote.reasoning for vote in votes if vote.score == score),
                "Consensus score.",
            ),
            is_flagged=flagged,
        )
        normalised = (score - 1) / 2
        interview_signal += normalised * dimension.weight
        if flagged:
            bias_flags.append(f"bias_flag:{dimension.name}")
        if normalised >= 0.5:
            strengths.append(dimension.name)
        if score == 1:
            gaps.append(dimension.name)
    return scores, interview_signal, strengths[:3], gaps, bias_flags


def _judge_ensemble_raw(results: list[dict[str, Any]]) -> JudgeEnsembleRaw:
    grouped: defaultdict[str, list[JudgeDimensionVote]] = defaultdict(list)
    for result in results:
        for item in result.get("judge_results", []):
            grouped[str(item["model"])].append(
                JudgeDimensionVote(
                    dimension=str(result["dimension"]),
                    score=int(item["score"]),
                    reasoning=str(item["reasoning"]),
                )
            )
    return JudgeEnsembleRaw(
        judges=[JudgeModelResult(model=model, votes=votes) for model, votes in grouped.items()]
    )


async def build_scorecard(session_id: str) -> Scorecard:
    session_key = _session_identity(session_id)
    async with AsyncSessionFactory() as db:
        bundle = (
            await db.exec(
                select(InterviewSession, ParsedJobContext)
                .join(ParsedJobContext, InterviewSession.job_id == ParsedJobContext.job_id)
                .where(InterviewSession.id == session_key)
            )
        ).first()
        if bundle is None:
            raise ValueError(f"interview session not found: {session_id}")
        interview, job = bundle
        rubric = [RubricDimension.model_validate(item) for item in job.rubric]
        pairs = []
        transcript_rows = await _graph_transcript_rows(interview)
        if transcript_rows:
            for question, answer in _pair_transcript_rows(transcript_rows):
                if answer.get("scoreable") is False:
                    continue
                dimension_name = str(question.get("dimension") or "").strip()
                if not dimension_name:
                    continue
                dimension = next((item for item in rubric if item.name == dimension_name), None)
                if dimension is not None:
                    pairs.append((question["content"], answer["content"], dimension))
        else:
            messages = (
                await db.exec(
                    select(InterviewMessage)
                    .where(InterviewMessage.session_id == interview.id)
                    .order_by(InterviewMessage.sequence_number, InterviewMessage.created_at)
                )
            ).all()
            for question, answer in _pair_messages(list(messages)):
                if answer.scoreable is False or not question.dimension_targeted:
                    continue
                dimension = next(
                    (item for item in rubric if item.name == question.dimension_targeted),
                    None,
                )
                if dimension is not None:
                    pairs.append((question.content, answer.content, dimension))
        results = (
            await asyncio.gather(
                *[
                    run_ensemble_judge(question, answer, dimension, job)
                    for question, answer, dimension in pairs
                ]
            )
            if pairs
            else []
        )
        grouped: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
        for result in results:
            grouped[str(result["dimension"])].append(result)
        dimension_scores: dict[str, DimensionScore] = {}
        dimension_ranking: list[tuple[str, float, float]] = []
        bias_flags: list[str] = []
        interview_signal = 0.0
        for dimension in rubric:
            bucket = grouped.get(dimension.name, [])
            average_score = (
                sum(float(item["score"]) for item in bucket) / len(bucket)
                if bucket
                else 2.0
            )
            normalised = (average_score - 1.0) / 2.0
            flagged = any(bool(item["is_flagged"]) for item in bucket)
            if flagged:
                bias_flags.append(f"bias_flag:{dimension.name}")
            dimension_scores[dimension.name] = DimensionScore(
                score=_display_score(average_score),
                weight=dimension.weight,
                judge_votes=[vote for item in bucket for vote in item["votes"]] or [2],
                reasoning=" ".join(str(item["reasoning"]) for item in bucket[:2])
                or "No evaluated answer for this dimension.",
                is_flagged=flagged,
            )
            interview_signal += normalised * dimension.weight
            dimension_ranking.append((dimension.name, normalised, dimension.weight))
        extracted = {skill.casefold() for skill in interview.extracted_skills or []}
        total = weighted_total(
            interview_signal,
            interview.assessment_score,
            interview.speech_score or 0.0,
            interview.screening_score,
        )
        existing = (
            await db.exec(select(Scorecard).where(Scorecard.session_id == interview.id))
        ).first()
        scorecard = existing or Scorecard(
            candidate_id=interview.candidate_id,
            job_id=interview.job_id,
            session_id=interview.id,
            weighted_total=0.0,
            interview_dimension_scores={},
            assessment_score=interview.assessment_score,
            speech_score=interview.speech_score,
            screening_score=interview.screening_score,
            strengths=[],
            gaps=[],
            integrity_flags=list(interview.integrity_flags or []),
            bias_flags=[],
            judge_ensemble_raw={},
            recommended_action=RecommendedAction.HOLD,
        )
        scorecard.interview_dimension_scores = {
            name: value.model_dump(mode="json") for name, value in dimension_scores.items()
        }
        scorecard.judge_ensemble_raw = _judge_ensemble_raw(list(results)).model_dump(mode="json")
        scorecard.weighted_total = total
        scorecard.recommended_action = recommended_action(total)
        scorecard.strengths = [
            item[0]
            for item in sorted(
                dimension_ranking,
                key=lambda entry: (-entry[1], -entry[2], entry[0]),
            )[:3]
        ]
        scorecard.gaps = [
            skill for skill in job.must_have_skills if skill.casefold() not in extracted
        ]
        scorecard.assessment_score = interview.assessment_score
        scorecard.speech_score = interview.speech_score
        scorecard.screening_score = interview.screening_score
        scorecard.integrity_flags = list(interview.integrity_flags or [])
        scorecard.bias_flags = bias_flags
        scorecard.updated_at = _utcnow()
        db.add(scorecard)
        db.add(
            AuditLog(
                event_type="scorecard_generated",
                entity_type="scorecard",
                entity_id=str(scorecard.session_id),
                payload={
                    "recommended_action": scorecard.recommended_action,
                    "weighted_total": scorecard.weighted_total,
                    "bias_flags": scorecard.bias_flags,
                },
            )
        )
        await db.commit()
        await db.refresh(scorecard)
        return scorecard


async def rank_candidates(job_id: str) -> None:
    async with AsyncSessionFactory() as db:
        scorecards = (await db.exec(select(Scorecard).where(Scorecard.job_id == job_id))).all()
        for rank, scorecard in enumerate(
            sorted(scorecards, key=lambda item: item.weighted_total, reverse=True),
            start=1,
        ):
            scorecard.final_rank = rank
            db.add(scorecard)
        await db.commit()
    if core.redis is not None:
        await core.redis.publish(f"job_events:{job_id}", "shortlist_ready")


__all__ = [
    "JUDGE_PROMPT",
    "aggregate_dimension_scores",
    "build_scorecard",
    "recommended_action",
    "rank_candidates",
    "run_ensemble_judge",
    "weighted_total",
]
