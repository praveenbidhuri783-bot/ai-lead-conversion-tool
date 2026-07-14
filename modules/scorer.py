"""Rule-based lead scoring engine.

Computes a 0-100 conversion score for each call record, applies hard
exclusion rules, assigns a lead priority tier, and produces all of the
downstream fields (sentiment, objection, pitch, recommended action/channel/
follow-up time, conversion evidence) required by the build brief.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import pandas as pd

from modules import objection_detector, pitch_generator, sentiment

SCORING_VERSION_KEY = "scoring_version"

PRIORITY_LABELS = {
    "P1": "P1 - Hot",
    "P2": "P2 - Warm",
    "P3": "P3 - Nurture",
    "P4": "P4 - Exclude",
}

CONVERSION_POTENTIAL_BY_PRIORITY = {
    "P1": "High",
    "P2": "Medium",
    "P3": "Low",
    "P4": "None",
}

_DURATION_MMSS_RE = re.compile(r"^(\d+):([0-5]?\d)$")


def _normalise_text(value) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() in ("nan", "none", "<na>"):
        return ""
    return text


def _lower(value) -> str:
    return _normalise_text(value).lower()


def _column_value(row: pd.Series, columns_map: Dict[str, str], canonical: str) -> str:
    col = columns_map.get(canonical)
    if not col or col not in row.index:
        return ""
    return _normalise_text(row[col])


def _value_matches(value: str, match_values: List[str]) -> bool:
    if not value:
        return False
    v = value.strip().lower()
    return v in {m.lower() for m in match_values}


def _keyword_hit(text: str, keywords: List[str]) -> bool:
    return any(kw in text for kw in keywords)


def parse_duration_seconds(value) -> Optional[float]:
    """Parse a call-duration value (seconds, or mm:ss string) into seconds."""
    text = _normalise_text(value)
    if not text:
        return None
    match = _DURATION_MMSS_RE.match(text)
    if match:
        minutes, seconds = int(match.group(1)), int(match.group(2))
        return float(minutes * 60 + seconds)
    try:
        return float(text)
    except ValueError:
        return None


def check_hard_exclusion(
    text: str, row: pd.Series, columns_map: Dict[str, str], config: Dict
) -> Tuple[Optional[str], Optional[str]]:
    """Return (exclusion_key, reason) if any hard-exclusion rule matches, else (None, None)."""
    for key, rule in config.get("hard_exclusions", {}).items():
        if rule.get("source") == "column":
            value = _column_value(row, columns_map, rule.get("column", ""))
            if _value_matches(value, rule.get("match_values", [])):
                return key, rule.get("reason", key)
        keywords = rule.get("keywords", [])
        if keywords and _keyword_hit(text, keywords):
            return key, rule.get("reason", key)
    return None, None


def _positive_signal_score(
    text: str, row: pd.Series, columns_map: Dict[str, str], config: Dict
) -> Tuple[float, List[str]]:
    total = 0.0
    matched_labels: List[str] = []
    for key, rule in config.get("positive_signals", {}).items():
        hit = False
        if rule.get("source") == "column":
            col_key = rule.get("column", "")
            value = _column_value(row, columns_map, col_key)
            if not value:
                continue
            match_values = rule.get("match_values")
            hit = _value_matches(value, match_values) if match_values else bool(value)
        else:
            hit = _keyword_hit(text, rule.get("keywords", []))
        if hit:
            total += rule.get("score", 0)
            matched_labels.append(key.replace("_", " ").capitalize())

    duration_cfg = config.get("call_duration_signals", {})
    duration_col = duration_cfg.get("column")
    if duration_col:
        seconds = parse_duration_seconds(_column_value(row, columns_map, duration_col))
        if seconds is not None:
            if seconds >= 120:
                total += duration_cfg.get("above_120_score", 0)
                matched_labels.append("Call duration above 120 seconds")
            elif 60 <= seconds <= 119:
                total += duration_cfg.get("range_60_119_score", 0)
                matched_labels.append("Call duration between 60 and 119 seconds")

    return total, matched_labels


def _negative_signal_score(text: str, config: Dict) -> Tuple[float, List[str]]:
    total = 0.0
    matched_labels: List[str] = []
    for key, rule in config.get("negative_signals", {}).items():
        if _keyword_hit(text, rule.get("keywords", [])):
            total += rule.get("score", 0)
            matched_labels.append(key.replace("_", " ").capitalize())
    return total, matched_labels


def assign_priority(score: float, hard_excluded: bool, config: Dict) -> str:
    if hard_excluded:
        return "P4"
    thresholds = config.get("priority_thresholds", {})
    if score >= thresholds.get("P1_hot_min", 75):
        return "P1"
    if score >= thresholds.get("P2_warm_min", 55):
        return "P2"
    if score >= thresholds.get("P3_nurture_min", 35):
        return "P3"
    return "P4"


def _recommended_follow_up_time(priority: str, row: pd.Series, columns_map: Dict[str, str], config: Dict) -> str:
    sla = config.get("sla_rules", {}).get(priority, {})
    default_time = sla.get("follow_up_time", "Not applicable")
    if priority in ("P1", "P2"):
        callback_date = _column_value(row, columns_map, "callback_date")
        callback_time = _column_value(row, columns_map, "callback_time")
        if callback_date or callback_time:
            parts = [p for p in (callback_date, callback_time) if p]
            return f"Callback requested for {' '.join(parts)}"
        booking_date = _column_value(row, columns_map, "booking_date")
        booking_time = _column_value(row, columns_map, "booking_time")
        if booking_date or booking_time:
            parts = [p for p in (booking_date, booking_time) if p]
            return f"Booking slot: {' '.join(parts)}"
    return default_time


def _recommended_channel(priority: str, primary_objection: str, config: Dict) -> str:
    sla = config.get("sla_rules", {}).get(priority, {})
    base_channel = sla.get("channel", "None")
    if primary_objection == "Language Barrier":
        return "Call (preferred language agent)"
    if primary_objection == "Unresponsive":
        return "WhatsApp/SMS"
    return base_channel


def _next_best_action(priority: str, primary_objection: str, config: Dict) -> str:
    sla_action = config.get("sla_rules", {}).get(priority, {}).get("action", "")
    if priority == "P4" or primary_objection == "No Major Objection":
        return sla_action
    return f"{sla_action}; specifically address: {primary_objection}"


def score_row(
    row: pd.Series,
    summary_column: str,
    columns_map: Dict[str, str],
    scoring_config: Dict,
    pitch_config: Dict,
) -> Dict:
    """Score a single record and produce every output field for that lead."""
    text = _lower(row.get(summary_column, ""))

    exclusion_key, exclusion_reason = check_hard_exclusion(text, row, columns_map, scoring_config)
    hard_excluded = exclusion_key is not None

    base_score = scoring_config.get("base_score", 25)
    pos_score, pos_labels = _positive_signal_score(text, row, columns_map, scoring_config)
    neg_score, neg_labels = _negative_signal_score(text, scoring_config)

    raw_score = base_score + pos_score + neg_score
    score_min = scoring_config.get("score_min", 0)
    score_max = scoring_config.get("score_max", 100)
    final_score = max(score_min, min(score_max, raw_score))
    if hard_excluded:
        final_score = min(final_score, scoring_config.get("priority_thresholds", {}).get("P4_exclude_max", 34))

    priority = assign_priority(final_score, hard_excluded, scoring_config)
    priority_label = PRIORITY_LABELS[priority]
    conversion_potential = CONVERSION_POTENTIAL_BY_PRIORITY[priority]

    lead_sentiment = sentiment.classify_sentiment(final_score, scoring_config, exclusion_key)

    if not text.strip():
        primary_objection, secondary_objections = "Unresponsive", []
    else:
        primary_objection, secondary_objections = objection_detector.detect_objections(text, scoring_config)

    service_type = _column_value(row, columns_map, "service_type") or _column_value(row, columns_map, "product")
    language = _column_value(row, columns_map, "language")

    pitch = pitch_generator.generate_pitch(
        priority=priority,
        primary_objection=primary_objection,
        hard_excluded=hard_excluded,
        service_type=service_type,
        language=language,
        pitch_config=pitch_config,
    )

    evidence = "; ".join(pos_labels) if pos_labels else "No strong positive signals detected"
    if neg_labels:
        evidence += " | Concerns: " + "; ".join(neg_labels)

    return {
        "Sentiment": lead_sentiment,
        "Lead_Score": round(final_score, 1),
        "Conversion_Potential": conversion_potential,
        "Lead_Priority": priority_label,
        "Primary_Objection": primary_objection,
        "Secondary_Objections": ", ".join(secondary_objections) if secondary_objections else "",
        "Conversion_Evidence": evidence,
        "Recommended_Action": scoring_config.get("sla_rules", {}).get(priority, {}).get("action", ""),
        "Next_Best_Action": _next_best_action(priority, primary_objection, scoring_config),
        "Recommended_Agent_Pitch": pitch,
        "Recommended_Channel": _recommended_channel(priority, primary_objection, scoring_config),
        "Recommended_Follow_Up_Time": _recommended_follow_up_time(priority, row, columns_map, scoring_config),
        "Lead_Exclusion_Reason": exclusion_reason or "",
        "Scoring_Version": scoring_config.get(SCORING_VERSION_KEY, "1.0.0"),
        "_priority_code": priority,
    }


def score_dataframe(
    df: pd.DataFrame,
    summary_column: str,
    columns_map: Dict[str, str],
    scoring_config: Dict,
    pitch_config: Dict,
    progress_callback=None,
) -> pd.DataFrame:
    """Score every row in ``df`` and return a new DataFrame with output columns appended."""
    records: List[Dict] = []
    total = len(df)
    for idx, (_, row) in enumerate(df.iterrows()):
        records.append(score_row(row, summary_column, columns_map, scoring_config, pitch_config))
        if progress_callback and (idx % 250 == 0 or idx == total - 1):
            progress_callback(idx + 1, total)

    scored = pd.DataFrame(records)
    result = pd.concat([df.reset_index(drop=True), scored.reset_index(drop=True)], axis=1)
    result.attrs["scoring_run_timestamp"] = datetime.utcnow().isoformat() + "Z"
    result.attrs["scoring_version"] = scoring_config.get(SCORING_VERSION_KEY, "1.0.0")
    return result
