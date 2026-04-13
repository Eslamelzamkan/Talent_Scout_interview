from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from typing import Any

import structlog
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

import app.core as core
from app.core.config import get_settings
from app.core.constants import (
    BEHAVIORAL_KEYWORDS,
    BUSINESS_KEYWORDS,
    TECHNICAL_KEYWORDS,
)
from app.core.constants import (
    lane_rules as get_lane_rules,
)
from app.core.constants import (
    message_content as _message_content,
)
from app.models import ParsedJobContext, RubricDimension

log = structlog.get_logger()
settings = get_settings()

JD_EXTRACTION_PROMPT = (
    "Senior talent acquisition specialist extracting a structured rubric.\n"
    "Input: {jd_text}\n"
    "Output: JSON with role_title, seniority, domain, must_have_skills, "
    "question_seed_topics (5-8 topics), rubric_dimensions (list of RubricDimension "
    "objects), requires_coding. Weights must sum to 1.0. Score anchors must be "
    'specific and measurable - reject vague phrases like "good communicator" or '
    '"team player". If requires_coding is true, include at least one technical '
    'dimension with verbal problem-solving anchors (for example: "candidate '
    'articulates time complexity and trade-offs clearly").'
)

JD_LINTING_PROMPT = (
    "Review rubric dimensions for specificity.\n"
    "Input: {rubric_json}\n"
    "Output: JSON list of {{dimension, issue, suggestion}} or [] if all good. "
    "A dimension fails linting if its description or score anchors could apply "
    "to any candidate regardless of their answer."
)

QUESTION_GENERATION_PROMPT = (
    "Generate {n} interview questions for {seniority} {role_title}.\n"
    "Dimension: {dimension}\n"
    "Dimension description: {dimension_description}\n"
    "Topic: {topic}\n"
    "Question lane: {lane}\n"
    "Lane rules: {lane_rules}\n"
    "Rules: open-ended, require specific examples or technical depth, no yes/no, "
    "and avoid generic filler.\n"
    "Output: JSON array of question strings only."
)

_SEED_SEMAPHORE = asyncio.Semaphore(8)  # limit concurrent LLM calls during question seeding


def _string_list(value: Any) -> list[str]:
    return [item for item in (str(part).strip() for part in (value or [])) if item]


def _dimension_lanes(dimension: dict[str, Any], requires_coding: bool) -> list[str]:
    haystack = f"{dimension.get('name', '')} {dimension.get('description', '')}".lower()
    lanes: list[str] = []
    if any(keyword in haystack for keyword in BEHAVIORAL_KEYWORDS):
        lanes.append("behavioral")
    if any(keyword in haystack for keyword in BUSINESS_KEYWORDS):
        lanes.extend(["business_case", "project_deep_dive"])
    tech_match = any(keyword in haystack for keyword in TECHNICAL_KEYWORDS)
    if tech_match or (requires_coding and not lanes):
        lanes.extend(["technical_fundamentals", "project_deep_dive"])
    if not lanes:
        lanes.append("project_deep_dive")
    ordered: list[str] = []
    for lane in lanes:
        if lane not in ordered:
            ordered.append(lane)
    return ordered


def _lane_rules(lane: str) -> str:
    return get_lane_rules(lane)


def _normalise_weights(dimensions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    total = sum(float(item.get("weight", 0.0)) for item in dimensions)
    if not dimensions or abs(total - 1.0) <= 0.001:
        return dimensions
    if total <= 0:
        raise ValueError("rubric dimension weights must sum to a positive value")
    corrected: list[dict[str, Any]] = []
    running_total = 0.0
    for index, item in enumerate(dimensions):
        clone = dict(item)
        if index == len(dimensions) - 1:
            clone["weight"] = round(1.0 - running_total, 6)
        else:
            clone["weight"] = round(float(item["weight"]) / total, 6)
            running_total += float(clone["weight"])
        corrected.append(clone)
    log.warning("jd_weights_normalised", original_total=round(total, 6))
    return corrected


def _parse_json_response(text: str) -> dict[str, Any] | list[Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as exc:
        preview = cleaned[:200]
        raise ValueError(f"failed to parse JSON response: {exc.msg}; preview={preview!r}") from exc


def _validated_dimensions(raw: Any) -> list[dict[str, Any]]:
    if not isinstance(raw, list) or not raw:
        raise ValueError("rubric_dimensions must be a non-empty list")
    return [
        RubricDimension.model_validate(item).model_dump(mode="json")
        for item in _normalise_weights([dict(item) for item in raw])
    ]


def _require_llm() -> Any:
    if core.llm is None:
        raise RuntimeError("llm client is not initialised")
    return core.llm


async def _extract(jd_text: str) -> dict[str, Any]:
    response = await _require_llm().chat(
        settings.model_fast,
        [{"role": "user", "content": JD_EXTRACTION_PROMPT.format(jd_text=jd_text)}],
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    payload = _parse_json_response(_message_content(response))
    if not isinstance(payload, dict):
        raise ValueError("jd extraction response must be a JSON object")
    extracted = {
        "role_title": str(payload.get("role_title") or "unknown").strip() or "unknown",
        "seniority": str(payload.get("seniority") or "unknown").strip() or "unknown",
        "domain": str(payload.get("domain") or "unknown").strip() or "unknown",
        "must_have_skills": _string_list(payload.get("must_have_skills")),
        "question_seed_topics": _string_list(payload.get("question_seed_topics")),
        "rubric_dimensions": _validated_dimensions(payload.get("rubric_dimensions")),
        "requires_coding": bool(payload.get("requires_coding", False)),
    }
    extracted["question_seed_topics"] = extracted["question_seed_topics"] or [
        item["name"] for item in extracted["rubric_dimensions"][:5]
    ]
    return extracted


async def _lint(dimensions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    response = await _require_llm().chat(
        settings.model_fast,
        [{"role": "user", "content": JD_LINTING_PROMPT.format(rubric_json=json.dumps(dimensions))}],
        temperature=0.0,
    )
    payload = _parse_json_response(_message_content(response))
    issues = (
        payload.get("issues", payload.get("warnings", []))
        if isinstance(payload, dict)
        else payload
    )
    if not isinstance(issues, list):
        raise ValueError("jd linting response must be a JSON list")
    return [dict(item) for item in issues]


async def _generate_questions(
    role_title: str,
    seniority: str,
    dimension: str,
    dimension_description: str,
    topic: str,
    lane: str,
    n: int = 3,
) -> list[str]:
    prompt = QUESTION_GENERATION_PROMPT.format(
        n=n,
        seniority=seniority,
        role_title=role_title,
        dimension=dimension,
        dimension_description=dimension_description,
        topic=topic,
        lane=lane,
        lane_rules=get_lane_rules(lane),
    )
    response = await _require_llm().chat(
        settings.model_fast, [{"role": "user", "content": prompt}], temperature=0.2
    )
    payload = _parse_json_response(_message_content(response))
    questions = payload.get("questions", []) if isinstance(payload, dict) else payload
    if not isinstance(questions, list):
        raise ValueError("question generation response must be a JSON list")
    return [question for question in (str(item).strip() for item in questions) if question][:n]


async def _seed_question_bank(job_id: str, extracted: dict[str, Any]) -> int:
    combinations = [
        (dimension["name"], str(dimension.get("description") or ""), topic, lane)
        for dimension in extracted["rubric_dimensions"]
        for topic in extracted["question_seed_topics"][:3]
        for lane in _dimension_lanes(dimension, bool(extracted.get("requires_coding")))
    ]
    async def _limited_generate(
        dimension: str,
        description: str,
        topic: str,
        lane: str,
    ) -> list[str]:
        async with _SEED_SEMAPHORE:
            return await _generate_questions(
                role_title=extracted["role_title"],
                seniority=extracted["seniority"],
                dimension=dimension,
                dimension_description=description,
                topic=topic,
                lane=lane,
                n=1,
            )

    batches = await asyncio.gather(
        *[
            _limited_generate(dimension, description, topic, lane)
            for dimension, description, topic, lane in combinations
        ]
    )
    questions = [
        {
            "text": question,
            "dimension": dimension,
            "seniority": extracted["seniority"],
            "topic": topic,
            "lane": lane,
        }
        for (dimension, _, topic, lane), batch in zip(combinations, batches, strict=True)
        for question in batch
    ]
    return await core.chroma.seed(job_id, questions)


async def parse_and_seed(
    job_id: str,
    jd_text: str,
    session: AsyncSession,
    *,
    commit: bool = True,
    requires_coding: bool | None = None,
) -> ParsedJobContext:
    job = (
        await session.exec(select(ParsedJobContext).where(ParsedJobContext.job_id == job_id))
    ).first()
    requires_coding_flag = bool(
        requires_coding or (job.requires_coding if job is not None else False)
    )
    extracted = await _extract(
        f"{jd_text}\n\nSystem note: requires_coding is true. Include at least one technical rubric "
        "dimension."
        if requires_coding_flag
        else jd_text
    )
    warnings = await _lint(extracted["rubric_dimensions"])
    question_count = await _seed_question_bank(job_id, extracted)
    payload = {
        "role_title": extracted["role_title"],
        "seniority": extracted["seniority"],
        "domain": extracted["domain"],
        "must_have_skills": extracted["must_have_skills"],
        "rubric": extracted["rubric_dimensions"],
        "question_seed_topics": extracted["question_seed_topics"],
        "requires_coding": requires_coding_flag or bool(extracted["requires_coding"]),
        "linting_warnings": warnings,
    }
    if job is None:
        job = ParsedJobContext(job_id=job_id, **payload)
    else:
        for field, value in payload.items():
            if field == "requires_coding":
                value = job.requires_coding or value
            setattr(job, field, value)
        job.parsed_at = datetime.now(timezone.utc)
    session.add(job)
    if commit:
        await session.commit()
    else:
        await session.flush()
    await session.refresh(job)
    log.info(
        "jd_parse_and_seed_completed",
        job_id=job_id,
        question_count=question_count,
        lint_warning_count=len(warnings),
    )
    return job


__all__ = ["parse_and_seed"]
