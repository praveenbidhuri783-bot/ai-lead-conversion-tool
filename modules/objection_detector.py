"""Objection detection from call-summary free text."""

from __future__ import annotations

from typing import Dict, List, Tuple

NO_MAJOR_OBJECTION = "No Major Objection"


def _text_contains(text: str, keywords: List[str]) -> bool:
    return any(kw in text for kw in keywords)


def detect_objections(text: str, scoring_config: Dict) -> Tuple[str, List[str]]:
    """Detect the primary objection plus any secondary objections in call text.

    The primary objection is the highest-priority category (per
    ``objection_priority_order`` in the scoring config) that matches. All
    other matching categories are returned as secondary objections.
    """
    normalised = (text or "").lower()
    keyword_map: Dict[str, List[str]] = scoring_config.get("objection_keywords", {})
    priority_order: List[str] = scoring_config.get("objection_priority_order", [])

    matched: List[str] = []
    for category in priority_order:
        if category == NO_MAJOR_OBJECTION:
            continue
        keywords = keyword_map.get(category, [])
        if keywords and _text_contains(normalised, keywords):
            matched.append(category)

    if not matched:
        return NO_MAJOR_OBJECTION, []

    primary = matched[0]
    secondary = matched[1:]
    return primary, secondary
