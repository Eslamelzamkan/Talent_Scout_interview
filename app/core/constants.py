"""Shared constants and utility functions used across pipeline modules."""

from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# Keyword classification lists — used by both jd.py and interview.py
# to determine question lanes and dimension classification.
# ---------------------------------------------------------------------------

TECHNICAL_KEYWORDS: tuple[str, ...] = (
    "technical",
    "coding",
    "engineer",
    "architecture",
    "system design",
    "debug",
    "performance",
    "backend",
    "frontend",
    "api",
    "sql",
    "python",
    "data model",
)

BUSINESS_KEYWORDS: tuple[str, ...] = (
    "business",
    "stakeholder",
    "customer",
    "user",
    "product",
    "market",
    "metric",
    "revenue",
    "growth",
    "priorit",
    "domain",
)

BEHAVIORAL_KEYWORDS: tuple[str, ...] = (
    "communication",
    "collaboration",
    "leadership",
    "ownership",
    "learning",
    "adaptability",
    "culture",
    "values",
    "conflict",
    "teamwork",
    "mentoring",
)

QUESTION_LANES: tuple[str, ...] = (
    "technical_fundamentals",
    "project_deep_dive",
    "business_case",
    "behavioral",
)

LANE_RULES: dict[str, str] = {
    "technical_fundamentals": (
        "Focus on principles, architecture, reasoning, constraints, and trade-offs. "
        "Do not ask the candidate to write code."
    ),
    "project_deep_dive": (
        "Ask for one concrete project or work example, including what the candidate did, "
        "why they chose that path, and what happened."
    ),
    "business_case": (
        "Focus on business impact, users, prioritization, metrics, stakeholder trade-offs, "
        "and why the work mattered."
    ),
    "behavioral": (
        "Ask a behavioral event interview question about a specific past situation. "
        "Require ownership and outcome, and avoid generic self-description."
    ),
}


# ---------------------------------------------------------------------------
# Shared utility functions — eliminating duplicates across modules.
# ---------------------------------------------------------------------------


def message_content(response: object) -> str:
    """Extract the text content from an LLM chat-completion response.

    Handles both dict (raw JSON) and object (SDK) response shapes.
    """
    if isinstance(response, dict):
        choices = response.get("choices", [])
        if choices:
            return str(choices[0].get("message", {}).get("content") or "")
        return ""
    choices = getattr(response, "choices", [])
    if choices:
        message = choices[0].message if hasattr(choices[0], "message") else choices[0]
        return str(getattr(message, "content", "") or "")
    return ""


def meta_bool(value: Any) -> bool | None:
    """Coerce a value to a boolean, returning None if it cannot be parsed."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes"}:
            return True
        if lowered in {"false", "0", "no"}:
            return False
    return None


def lane_rules(lane: str) -> str:
    """Return the prompt rules for a given question lane."""
    return LANE_RULES.get(lane, "Ask one focused interview question.")
