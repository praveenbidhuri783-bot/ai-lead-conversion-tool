"""Sentiment classification for scored leads."""

from __future__ import annotations

from typing import Dict, Optional

POSITIVE = "Positive"
NEUTRAL_POSITIVE = "Neutral-Positive"
NEUTRAL = "Neutral"
NEUTRAL_NEGATIVE = "Neutral-Negative"
NEGATIVE = "Negative"


def classify_sentiment(
    score: float,
    scoring_config: Dict,
    hard_exclusion_key: Optional[str] = None,
) -> str:
    """Classify sentiment primarily from score, with language-driven overrides.

    Unambiguous hard exclusions (wrong number, do-not-call, explicit
    disinterest, fraud) force a Negative sentiment regardless of where the
    numeric score lands, since the language of the call is decisive in these
    cases.
    """
    override_keys = set(scoring_config.get("sentiment_negative_override_exclusions", []))
    if hard_exclusion_key and hard_exclusion_key in override_keys:
        return NEGATIVE

    thresholds = scoring_config.get("sentiment_thresholds", {})
    if score >= thresholds.get("positive_min", 75):
        return POSITIVE
    if score >= thresholds.get("neutral_positive_min", 55):
        return NEUTRAL_POSITIVE
    if score >= thresholds.get("neutral_min", 35):
        return NEUTRAL
    if score >= thresholds.get("neutral_negative_min", 20):
        return NEUTRAL_NEGATIVE
    return NEGATIVE
