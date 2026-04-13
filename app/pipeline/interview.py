from __future__ import annotations

import json
import re
from collections.abc import Sequence
from datetime import datetime, timezone
from typing import Annotated, Any, TypedDict, cast

import structlog
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.tracers.langchain import LangChainTracer
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph

import app.core as core
from app.core.chroma import retrieve
from app.core.config import get_settings
from app.core.constants import (
    BEHAVIORAL_KEYWORDS,
    BUSINESS_KEYWORDS,
    QUESTION_LANES,
    TECHNICAL_KEYWORDS,
)
from app.core.constants import (
    lane_rules as get_lane_rules,
)
from app.core.constants import (
    message_content as _raw_message_content,
)
from app.core.constants import (
    meta_bool as _meta_bool,
)
from app.models import (
    CandidateContext,
    InterviewSession,
    ParsedJobContext,
    RubricDimension,
    RubricModel,
)
from app.pipeline import evaluation

log = structlog.get_logger()
settings = get_settings()


class InterviewState(TypedDict):
    session_id: str
    candidate_id: str
    job_id: str
    candidate_name: str
    job_context: dict[str, Any]
    candidate_context: dict[str, Any]
    weak_areas: list[str]
    extracted_skills: list[str]
    requires_coding: bool
    interview_plan: dict[str, Any]
    question_lane_counts: dict[str, int]
    messages: Annotated[list[BaseMessage], add_messages]
    questions_asked: list[str]
    dimension_coverage: dict[str, int]
    dimension_scores_live: dict[str, float]
    current_question: str | None
    current_dimension: str | None
    current_question_lane: str | None
    current_question_focus: str | None
    candidate_answer: str | None
    answer_is_shallow: bool
    follow_up_signals: list[str]
    integrity_flags: list[str]
    total_questions_asked: int
    interview_complete: bool
    hitl_requested: bool
    recruiter_injected_question: str | None
    stt_fallback_active: bool
    error: str | None


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _content_text(message: BaseMessage) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    return " ".join(
        item if isinstance(item, str) else str(item.get("text") or "") for item in content or []
    ).strip()


def _message_text(payload: dict[str, Any]) -> str:
    return _raw_message_content(payload)


def _strip_json(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    return cleaned


def _rubric_dimensions(job_context: dict[str, Any]) -> list[RubricDimension]:
    raw = job_context.get("rubric") or []
    if isinstance(raw, dict):
        raw = raw.get("dimensions", [])
    return [RubricDimension.model_validate(item) for item in raw]


def _match_dimension_name(name: str, names: Sequence[str]) -> str | None:
    lookup = {item.strip().lower(): item for item in names}
    return lookup.get(name.strip().lower())


def _is_technical_dimension(dimension: RubricDimension) -> bool:
    haystack = f"{dimension.name} {dimension.description}".lower()
    return any(keyword in haystack for keyword in TECHNICAL_KEYWORDS)


def _is_business_dimension(dimension: RubricDimension) -> bool:
    haystack = f"{dimension.name} {dimension.description}".lower()
    return any(keyword in haystack for keyword in BUSINESS_KEYWORDS)


def _is_behavioral_dimension(dimension: RubricDimension) -> bool:
    haystack = f"{dimension.name} {dimension.description}".lower()
    return any(keyword in haystack for keyword in BEHAVIORAL_KEYWORDS)


def _candidate_context_model(state: InterviewState) -> CandidateContext:
    return CandidateContext.model_validate(state.get("candidate_context") or {})


def _candidate_evidence_lines(
    state: InterviewState,
    lane: str,
    dimension: str,
) -> list[str]:
    context = _candidate_context_model(state)
    dimension_terms = {
        term.lower() for term in re.split(r"[^a-zA-Z0-9]+", dimension) if term.strip()
    }
    evidence = context.evidence_snippets()
    matched = [item for item in evidence if any(term in item.lower() for term in dimension_terms)]
    lane_pool_map = {
        "technical_fundamentals": [
            *context.project_highlights,
            *context.work_highlights,
            *context.live_answer_evidence[-2:],
        ],
        "project_deep_dive": [
            *context.project_highlights,
            *context.work_highlights,
            *context.achievements,
            *context.live_answer_evidence[-2:],
        ],
        "business_case": [
            *context.work_highlights,
            *context.achievements,
            *context.live_answer_evidence[-2:],
        ],
        "behavioral": [
            *context.behavioral_highlights,
            *context.achievements,
            *context.motivations,
            *context.live_answer_evidence[-2:],
        ],
    }
    lane_pool = lane_pool_map.get(lane, evidence)
    combined = [
        *(matched[:3]),
        *(item for item in lane_pool if item not in matched),
    ]
    if state.get("extracted_skills"):
        combined.append(f"Observed skills: {', '.join(state['extracted_skills'][:6])}")
    return [item for item in combined if item][:5]


def _dimension_supported_lanes(dimension: RubricDimension, requires_coding: bool) -> list[str]:
    lanes: list[str] = []
    if _is_behavioral_dimension(dimension):
        lanes.append("behavioral")
    if _is_business_dimension(dimension):
        lanes.extend(["business_case", "project_deep_dive"])
    if _is_technical_dimension(dimension) or (requires_coding and not lanes):
        lanes.extend(["technical_fundamentals", "project_deep_dive"])
    if not lanes:
        lanes.append("project_deep_dive")
    deduped: list[str] = []
    for lane in lanes:
        if lane not in deduped:
            deduped.append(lane)
    return deduped


def _available_lanes(dimensions: Sequence[RubricDimension], requires_coding: bool) -> list[str]:
    lane_priority = (
        ["technical_fundamentals", "project_deep_dive", "business_case", "behavioral"]
        if requires_coding
        else ["project_deep_dive", "business_case", "behavioral", "technical_fundamentals"]
    )
    supported = {
        lane
        for dimension in dimensions
        for lane in _dimension_supported_lanes(dimension, requires_coding)
    }
    return [lane for lane in lane_priority if lane in supported]


def _build_interview_plan(state: InterviewState) -> dict[str, Any]:
    dimensions = _rubric_dimensions(state["job_context"])
    requires_coding = bool(state.get("requires_coding"))
    available = _available_lanes(dimensions, requires_coding) or ["project_deep_dive"]
    minimum_total = min(settings.max_questions, max(4, len(dimensions) + 1))
    weak_area_bonus = 1 if state.get("weak_areas") else 0
    target_total = min(settings.max_questions, minimum_total + weak_area_bonus)
    primary_order = (
        ["technical_fundamentals", "project_deep_dive", "business_case", "behavioral"]
        if requires_coding
        else ["project_deep_dive", "behavioral", "business_case", "technical_fundamentals"]
    )
    sequence: list[str] = []
    for lane in primary_order:
        if lane in available and len(sequence) < target_total:
            sequence.append(lane)
    cycle = (
        ["project_deep_dive", "technical_fundamentals", "business_case", "behavioral"]
        if requires_coding
        else ["project_deep_dive", "behavioral", "business_case", "technical_fundamentals"]
    )
    while len(sequence) < target_total:
        for lane in cycle:
            if lane not in available or len(sequence) >= target_total:
                continue
            sequence.append(lane)
    targets = {lane: sequence.count(lane) for lane in QUESTION_LANES}
    return {"target_total": target_total, "lane_sequence": sequence, "lane_targets": targets}


def _next_question_lane(state: InterviewState) -> str:
    if state.get("answer_is_shallow") and state.get("current_question_lane"):
        return str(state["current_question_lane"])
    sequence = list((state.get("interview_plan") or {}).get("lane_sequence") or [])
    if not sequence:
        return "project_deep_dive"
    index = min(int(state.get("total_questions_asked", 0)), len(sequence) - 1)
    return str(sequence[index])


def _lane_focus_hint(lane: str, follow_up_signals: Sequence[str]) -> str | None:
    if follow_up_signals:
        return str(follow_up_signals[0])
    defaults = {
        "technical_fundamentals": "reasoning and trade-offs",
        "project_deep_dive": "specific actions and measurable outcomes",
        "business_case": "business impact and prioritization logic",
        "behavioral": "one concrete past situation with ownership",
    }
    return defaults.get(lane)


def _lane_prompt_rules(lane: str) -> str:
    return get_lane_rules(lane)


def _default_template(state: InterviewState, dimension: str, lane: str) -> str:
    evidence = _candidate_evidence_lines(state, lane, dimension)
    evidence_hint = evidence[0] if evidence else f"your experience related to {dimension}"
    templates = {
        "technical_fundamentals": (
            f"Using {evidence_hint}, how do you reason about the main trade-offs in {dimension} "
            "for a role like this?"
        ),
        "project_deep_dive": (
            f"Tell me about a specific project or work example where {dimension} mattered. "
            "What decisions did you make, why, and what was the result?"
        ),
        "business_case": (
            f"Describe a time your work around {dimension} changed a business, stakeholder, or "
            "user outcome. How did you measure the impact?"
        ),
        "behavioral": (
            f"Tell me about a specific situation that best demonstrates your {dimension.lower()}. "
            "What did you personally do, and what was the outcome?"
        ),
    }
    return templates.get(lane, f"Tell me about your experience with {dimension}.")


def _answer_snippet(answer: str) -> str:
    sentence = re.split(r"(?<=[.!?])\s+", answer.strip(), maxsplit=1)[0]
    return sentence[:220].strip()


def _extract_skills(
    answer: str, required_skills: Sequence[str], existing: Sequence[str]
) -> list[str]:
    seen = list(existing)
    lowered = answer.lower()
    for skill in required_skills:
        if skill.lower() in lowered and skill not in seen:
            seen.append(skill)
    return seen


def _latest_exchange(messages: Sequence[BaseMessage]) -> list[BaseMessage]:
    human_idx = next(
        (
            idx
            for idx in range(len(messages) - 1, -1, -1)
            if isinstance(messages[idx], HumanMessage)
        ),
        None,
    )
    if human_idx is None:
        return []
    ai_idx = next(
        (idx for idx in range(human_idx - 1, -1, -1) if isinstance(messages[idx], AIMessage)),
        None,
    )
    start = ai_idx if ai_idx is not None else human_idx
    return list(messages[start : human_idx + 1])


def _latest_question_answer(messages: Sequence[BaseMessage]) -> tuple[str, str] | None:
    exchange = _latest_exchange(messages)
    question = next((item for item in exchange if isinstance(item, AIMessage)), None)
    answer = next((item for item in reversed(exchange) if isinstance(item, HumanMessage)), None)
    if question is None or answer is None:
        return None
    return _content_text(question), _content_text(answer)


def _question_meta(
    dimension: str | None,
    *,
    lane: str | None = None,
    focus: str | None = None,
    follow_up: bool = False,
    scoreable: bool | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "dimension": dimension,
        "lane": lane,
        "focus": focus,
        "follow_up": follow_up,
        "timestamp": _utcnow(),
    }
    if scoreable is not None:
        payload["scoreable"] = scoreable
    return payload


def build_interview_seed(
    session: InterviewSession,
    context: ParsedJobContext,
    rubric: RubricModel,
) -> InterviewState:
    job_context = context.model_dump(mode="json", by_alias=True)
    job_context["rubric"] = [item.model_dump(mode="json") for item in rubric.dimensions]
    return {
        "session_id": str(session.id),
        "candidate_id": session.external_candidate_id,
        "job_id": context.external_job_id,
        "candidate_name": session.name,
        "job_context": job_context,
        "candidate_context": session.candidate_context_model.model_dump(mode="json"),
        "weak_areas": list(session.weak_areas or []),
        "extracted_skills": list(session.extracted_skills or []),
        "requires_coding": context.requires_coding,
        "interview_plan": {},
        "question_lane_counts": {},
        "messages": [],
        "questions_asked": [],
        "dimension_coverage": {},
        "dimension_scores_live": {},
        "current_question": None,
        "current_dimension": None,
        "current_question_lane": None,
        "current_question_focus": None,
        "candidate_answer": None,
        "answer_is_shallow": False,
        "follow_up_signals": [],
        "integrity_flags": list(session.integrity_flags or []),
        "total_questions_asked": 0,
        "interview_complete": False,
        "hitl_requested": False,
        "recruiter_injected_question": None,
        "stt_fallback_active": False,
        "error": None,
    }


def messages_to_transcript(messages: Sequence[BaseMessage]) -> list[dict[str, Any]]:
    transcript: list[dict[str, Any]] = []
    for message in messages:
        meta = message.additional_kwargs or {}
        role = "agent" if isinstance(message, AIMessage) else "candidate"
        transcript.append(
            {
                "role": role,
                "content": _content_text(message),
                "dimension": cast(str | None, meta.get("dimension")),
                "lane": cast(str | None, meta.get("lane")),
                "focus": cast(str | None, meta.get("focus")),
                "follow_up": _meta_bool(meta.get("follow_up")),
                "scoreable": _meta_bool(meta.get("scoreable")),
                "timestamp": cast(str, meta.get("timestamp") or _utcnow()),
            }
        )
    return transcript


def graph_invoke_config(thread_id: str, session_id: str) -> dict[str, Any]:
    callbacks = (
        [LangChainTracer(project_name=settings.langsmith_project)]
        if settings.langsmith_api_key
        else []
    )
    return {
        "configurable": {"thread_id": thread_id},
        "callbacks": callbacks,
        "run_name": f"interview_{session_id}",
        "tags": ["interview-engine", f"session:{session_id}"],
        "metadata": {"session_id": session_id},
    }


def _pick_next_dimension(state: InterviewState, preferred_lane: str | None = None) -> str | None:
    dimensions = _rubric_dimensions(state["job_context"])
    names = [item.name for item in dimensions]
    if not names:
        return None
    compatible = {
        item.name
        for item in dimensions
        if preferred_lane is None
        or preferred_lane in _dimension_supported_lanes(item, bool(state.get("requires_coding")))
    }
    if not compatible:
        compatible = set(names)
    coverage = state.get("dimension_coverage", {})
    scores = state.get("dimension_scores_live", {})
    candidates = [name for name in names if coverage.get(name, 0) < 2 and name in compatible] or [
        name for name in names if name in compatible
    ]
    if not candidates:
        candidates = names
    for weak_area in state.get("weak_areas", []):
        matched = _match_dimension_name(weak_area, names)
        if matched and coverage.get(matched, 0) == 0 and matched in candidates:
            return matched
    if state.get("requires_coding"):
        for dimension in dimensions:
            if _is_technical_dimension(dimension) and coverage.get(dimension.name, 0) == 0:
                if dimension.name in candidates:
                    return dimension.name
    for name in names:
        if name in candidates and coverage.get(name, 0) == 0:
            return name
    if candidates:
        return min(candidates, key=lambda name: (scores.get(name, 0.0), coverage.get(name, 0)))
    return None


async def _adapt_question(
    state: InterviewState,
    dimension: str,
    lane: str,
    template: str,
) -> str:
    focus = _lane_focus_hint(lane, state.get("follow_up_signals", []))
    evidence_lines = _candidate_evidence_lines(state, lane, dimension)
    previous_exchange = _latest_question_answer(state.get("messages", []))
    if core.llm is None:
        return template or _default_template(state, dimension, lane)
    prompt = (
        f"Candidate: {state['candidate_name']}\n"
        f"Role: {
            state['job_context'].get('role_title') or state['job_context'].get('title') or 'unknown'
        }\n"
        f"Weak areas: {', '.join(state.get('weak_areas', [])) or 'none'}\n"
        f"Skills: {', '.join(state.get('extracted_skills', [])) or 'unspecified'}\n"
        f"Dimension: {dimension}\n"
        f"Dimension description: {
            next(
                (
                    item.description
                    for item in _rubric_dimensions(state['job_context'])
                    if item.name == dimension
                ),
                '',
            )
        }\n"
        f"Question lane: {lane}\n"
        f"Question focus: {focus or 'none'}\n"
        f"Template: {template}\n"
        f"Follow-up required: {state.get('answer_is_shallow', False)}\n"
        f"Lane rules: {_lane_prompt_rules(lane)}\n"
        f"Candidate summary: {_candidate_context_model(state).summary or 'none'}\n"
        "Candidate evidence:\n- "
        + ("\n- ".join(evidence_lines) if evidence_lines else "none")
        + "\n"
    )
    if previous_exchange is not None:
        prompt += (
            f"Previous question: {previous_exchange[0]}\nPrevious answer: {previous_exchange[1]}\n"
        )
    if state.get("answer_is_shallow"):
        prompt += (
            "This must be a focused follow-up on the same dimension. Recover the missing detail "
            f"around: {', '.join(state.get('follow_up_signals', [])) or 'specific evidence'}.\n"
        )
    try:
        response = await core.llm.chat(
            settings.model_interview,
            [
                {
                    "role": "system",
                    "content": (
                        "Create one high-quality interview question. Use the template only as a "
                        "starting point. Make the question concrete, role-specific, and aligned "
                        "to the requested interview lane. Avoid generic filler. Output only the "
                        "question."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=180,
        )
        adapted = _message_text(cast(dict[str, Any], response))
        return adapted or template or _default_template(state, dimension, lane)
    except Exception:
        log.warning(
            "question_adaptation_failed",
            session_id=state["session_id"],
            dimension=dimension,
            lane=lane,
        )
        return template or _default_template(state, dimension, lane)


async def _depth_check(answer: str, lane: str) -> dict[str, Any]:
    lowered = answer.lower()
    token_count = len(answer.split())
    fallback_missing: list[str] = []
    if token_count < 25:
        fallback_missing.append("concrete detail")
    if not re.search(r"\b\d+(?:\.\d+)?%?\b", answer):
        fallback_missing.append("measurable impact")
    if not any(
        keyword in lowered for keyword in ("because", "trade-off", "tradeoff", "decided", "instead")
    ):
        fallback_missing.append("reasoning or trade-offs")
    if not any(
        keyword in lowered
        for keyword in (
            "for example",
            "for instance",
            "when i",
            "one time",
            "project",
            "customer",
            "user",
            "stakeholder",
        )
    ):
        fallback_missing.append("specific example")
    ownership_kw = (" i ", " my role", "i led", "i owned", "i was responsible")
    if lane == "behavioral" and not any(keyword in lowered for keyword in ownership_kw):
        fallback_missing.append("personal ownership")
    fallback = token_count >= 25 and len(fallback_missing) <= 1
    if core.llm is None:
        return {
            "is_substantive": fallback,
            "should_follow_up": not fallback,
            "missing_signals": fallback_missing,
            "follow_up_angle": fallback_missing[0] if fallback_missing else None,
        }
    try:
        response = await core.llm.chat(
            settings.model_fast,
            [
                {
                    "role": "system",
                    "content": (
                        "Return JSON with is_substantive (bool), "
                        "should_follow_up (bool), "
                        "missing_signals (array of short strings), "
                        "and follow_up_angle (string or null). "
                        "Assess whether the answer lacks concrete "
                        "examples, metrics, reasoning, ownership, "
                        "or business impact."
                    ),
                },
                {"role": "user", "content": f"Lane: {lane}\nAnswer:\n{answer}"},
            ],
            temperature=0.0,
            max_tokens=140,
            response_format={"type": "json_object"},
        )
        payload = json.loads(_strip_json(_message_text(cast(dict[str, Any], response)) or "{}"))
        return {
            "is_substantive": bool(payload.get("is_substantive")),
            "should_follow_up": bool(payload.get("should_follow_up")),
            "missing_signals": [
                str(item).strip()
                for item in payload.get("missing_signals", [])
                if str(item).strip()
            ]
            or fallback_missing,
            "follow_up_angle": (
                payload.get("follow_up_angle")
                or (fallback_missing[0] if fallback_missing else None)
            ),
        }
    except Exception:
        return {
            "is_substantive": fallback,
            "should_follow_up": not fallback,
            "missing_signals": fallback_missing,
            "follow_up_angle": fallback_missing[0] if fallback_missing else None,
        }


async def init_session(state: InterviewState) -> dict[str, Any]:
    dimensions = _rubric_dimensions(state["job_context"])
    coverage = {
        item.name: int(state.get("dimension_coverage", {}).get(item.name, 0)) for item in dimensions
    }
    scores = {
        item.name: float(state.get("dimension_scores_live", {}).get(item.name, 0.0))
        for item in dimensions
    }
    plan = state.get("interview_plan") or _build_interview_plan(state)
    updates: dict[str, Any] = {
        "dimension_coverage": coverage,
        "dimension_scores_live": scores,
        "interview_plan": plan,
    }
    for key, default in {
        "candidate_context": _candidate_context_model(state).model_dump(mode="json"),
        "question_lane_counts": {lane: 0 for lane in QUESTION_LANES},
        "questions_asked": [],
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
    }.items():
        if key not in state:
            updates[key] = default
    return updates


async def select_dimension(state: InterviewState) -> dict[str, Any]:
    lane = _next_question_lane(state)
    dimension = (
        state.get("current_dimension")
        if state.get("answer_is_shallow") and state.get("current_dimension")
        else _pick_next_dimension(state, lane)
    )
    return {
        "current_dimension": dimension,
        "current_question_lane": lane,
        "current_question_focus": _lane_focus_hint(lane, state.get("follow_up_signals", [])),
    }


async def generate_question(state: InterviewState) -> dict[str, Any]:
    lane = str(state.get("current_question_lane") or _next_question_lane(state))
    focus = _lane_focus_hint(lane, state.get("follow_up_signals", []))
    dimension = state.get("current_dimension") or _pick_next_dimension(state, lane)
    if dimension is None:
        return {
            "interview_complete": True,
            "current_question": None,
            "error": "missing_rubric_dimensions",
        }
    seniority = str(
        state["job_context"].get("seniority") or state["job_context"].get("role_level") or "unknown"
    )
    try:
        templates = await retrieve(
            state["job_id"],
            dimension=dimension,
            seniority=seniority,
            lane=lane,
            n=3,
            exclude=state.get("questions_asked", []),
        )
    except Exception:
        log.warning(
            "question_bank_lookup_failed",
            session_id=state["session_id"],
            dimension=dimension,
            lane=lane,
        )
        templates = []
    if not templates:
        templates = [_default_template(state, dimension, lane)]
    question = await _adapt_question(state, dimension, lane, templates[0])
    is_follow_up = bool(state.get("answer_is_shallow"))
    lane_counts = dict(state.get("question_lane_counts", {}))
    if not is_follow_up:
        lane_counts[lane] = int(lane_counts.get(lane, 0)) + 1
    return {
        "messages": [
            AIMessage(
                content=question,
                additional_kwargs=_question_meta(
                    dimension,
                    lane=lane,
                    focus=focus,
                    follow_up=is_follow_up,
                ),
            )
        ],
        "questions_asked": [*state.get("questions_asked", []), question],
        "current_question": question,
        "current_dimension": dimension,
        "current_question_lane": lane,
        "current_question_focus": focus,
        "question_lane_counts": lane_counts,
        "total_questions_asked": state.get("total_questions_asked", 0) + (0 if is_follow_up else 1),
        "answer_is_shallow": False,
        "follow_up_signals": [],
    }


async def process_answer(state: InterviewState) -> dict[str, Any]:
    answer = (state.get("candidate_answer") or "").strip()
    if not answer:
        return {"candidate_answer": None, "answer_is_shallow": False}
    lane = str(state.get("current_question_lane") or _next_question_lane(state))
    depth = await _depth_check(answer, lane)
    needs_follow_up = (not depth["is_substantive"]) or bool(depth["should_follow_up"])
    skills = _extract_skills(
        answer, state["job_context"].get("must_have_skills", []), state.get("extracted_skills", [])
    )
    candidate_context = _candidate_context_model(state)
    if depth["is_substantive"]:
        snippet = _answer_snippet(answer)
        if snippet and snippet not in candidate_context.live_answer_evidence:
            candidate_context.live_answer_evidence = [
                *candidate_context.live_answer_evidence[-4:],
                snippet,
            ]
    return {
        "messages": [
            HumanMessage(
                content=answer,
                additional_kwargs=_question_meta(
                    state.get("current_dimension"),
                    lane=state.get("current_question_lane"),
                    focus=state.get("current_question_focus"),
                    scoreable=not needs_follow_up,
                ),
            )
        ],
        "candidate_answer": None,
        "current_question": None,
        "current_question_focus": None,
        "answer_is_shallow": needs_follow_up,
        "follow_up_signals": [
            str(item) for item in depth.get("missing_signals", []) if str(item).strip()
        ],
        "extracted_skills": skills,
        "candidate_context": candidate_context.model_dump(mode="json"),
    }


async def evaluate_answer(state: InterviewState) -> dict[str, Any]:
    dimension = state.get("current_dimension")
    latest = _latest_question_answer(state.get("messages", []))
    if dimension is None or latest is None:
        return {}
    dimension_model = next(
        (item for item in _rubric_dimensions(state["job_context"]) if item.name == dimension),
        None,
    )
    if dimension_model is None:
        return {}
    scorecard = await evaluation.run_ensemble_judge(
        latest[0],
        latest[1],
        dimension_model,
        state["job_context"],
    )
    coverage = dict(state.get("dimension_coverage", {}))
    scores = dict(state.get("dimension_scores_live", {}))
    prior_count = coverage.get(dimension, 0)
    coverage[dimension] = prior_count + 1
    scores[dimension] = round(
        ((scores.get(dimension, 0.0) * prior_count) + float(scorecard["score"]))
        / (prior_count + 1),
        4,
    )
    return {"dimension_coverage": coverage, "dimension_scores_live": scores}


async def check_completion(state: InterviewState) -> dict[str, Any]:
    names = [item.name for item in _rubric_dimensions(state["job_context"])]
    coverage = state.get("dimension_coverage", {})
    all_dimensions_covered = all(coverage.get(name, 0) > 0 for name in names)
    weak_targets = {
        matched
        for item in state.get("weak_areas", [])
        if (matched := _match_dimension_name(item, names)) is not None
    }
    weak_areas_probed = all(coverage.get(name, 0) > 0 for name in weak_targets)
    lane_targets = dict((state.get("interview_plan") or {}).get("lane_targets") or {})
    lane_counts = state.get("question_lane_counts", {})
    lane_targets_met = all(
        int(lane_counts.get(lane, 0)) >= int(target) for lane, target in lane_targets.items()
    )
    plan = state.get("interview_plan") or {}
    planned_total = int(plan.get("target_total") or settings.max_questions)
    complete = state.get("total_questions_asked", 0) >= settings.max_questions or (
        state.get("total_questions_asked", 0) >= planned_total
        and all_dimensions_covered
        and weak_areas_probed
        and lane_targets_met
    )
    return {"interview_complete": complete}


async def hitl_checkpoint(state: InterviewState) -> dict[str, Any]:
    question = (state.get("recruiter_injected_question") or "").strip()
    updates: dict[str, Any] = {"hitl_requested": False, "recruiter_injected_question": None}
    if question:
        updates.update(
            {
                "messages": [
                    AIMessage(
                        content=question,
                        additional_kwargs=_question_meta(
                            state.get("current_dimension"),
                            lane=state.get("current_question_lane"),
                            focus="recruiter injected question",
                        ),
                    )
                ],
                "questions_asked": [*state.get("questions_asked", []), question],
                "current_question": question,
                "total_questions_asked": state.get("total_questions_asked", 0) + 1,
                "answer_is_shallow": False,
                "follow_up_signals": [],
            }
        )
    return updates


async def finalise_session(state: InterviewState) -> dict[str, Any]:
    log.info(
        "interview_session_complete",
        session_id=state["session_id"],
        candidate_id=state["candidate_id"],
        total_questions_asked=state.get("total_questions_asked", 0),
    )
    return {"interview_complete": True, "current_question": None}


async def await_input(_: InterviewState) -> dict[str, Any]:
    return {}


def _after_init_route(state: InterviewState) -> str:
    if state.get("interview_complete"):
        return "finalise"
    if state.get("candidate_answer"):
        return "process_answer"
    if state.get("hitl_requested"):
        if state.get("recruiter_injected_question"):
            return "hitl"
        return "await_input"
    if state.get("current_question"):
        return "await_input"
    return "select_dimension"


def _completion_route(state: InterviewState) -> str:
    if state.get("interview_complete"):
        return "finalise"
    if state.get("hitl_requested"):
        return "hitl"
    if state.get("answer_is_shallow"):
        return "generate_question"
    return "select_dimension"


def _after_process_answer_route(state: InterviewState) -> str:
    if state.get("answer_is_shallow"):
        return "generate_question"
    return "evaluate_answer"


def build_graph(checkpointer: Any) -> CompiledStateGraph:
    builder = StateGraph(InterviewState)
    builder.add_node("init_session", init_session)
    builder.add_node("await_input", await_input)
    builder.add_node("select_dimension", select_dimension)
    builder.add_node("generate_question", generate_question)
    builder.add_node("process_answer", process_answer)
    builder.add_node("evaluate_answer", evaluate_answer)
    builder.add_node("check_completion", check_completion)
    builder.add_node("hitl", hitl_checkpoint)
    builder.add_node("finalise", finalise_session)
    builder.add_edge(START, "init_session")
    builder.add_conditional_edges(
        "init_session",
        _after_init_route,
        {
            "await_input": "await_input",
            "process_answer": "process_answer",
            "select_dimension": "select_dimension",
            "finalise": "finalise",
        },
    )
    builder.add_edge("await_input", END)
    builder.add_edge("select_dimension", "generate_question")
    builder.add_edge("generate_question", "process_answer")
    builder.add_conditional_edges(
        "process_answer",
        _after_process_answer_route,
        {
            "generate_question": "generate_question",
            "evaluate_answer": "evaluate_answer",
        },
    )
    builder.add_edge("evaluate_answer", "check_completion")
    builder.add_conditional_edges(
        "check_completion",
        _completion_route,
        {
            "finalise": "finalise",
            "hitl": "hitl",
            "generate_question": "generate_question",
            "select_dimension": "select_dimension",
        },
    )
    builder.add_edge("hitl", "process_answer")
    builder.add_edge("finalise", END)
    graph = builder.compile(
        checkpointer=checkpointer,
        interrupt_before=["hitl"],
        interrupt_after=["generate_question", "hitl"],
    )
    log.info("interview_graph_compiled")
    return graph


def build_interview_graph(checkpointer: Any) -> CompiledStateGraph:
    return build_graph(checkpointer)
