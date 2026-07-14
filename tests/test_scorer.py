"""Unit tests for the rule-based scoring engine."""

import json
import os

import pandas as pd
import pytest

from modules import scorer

CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config")


@pytest.fixture(scope="module")
def scoring_config():
    with open(os.path.join(CONFIG_DIR, "scoring_config.json"), "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def pitch_config():
    with open(os.path.join(CONFIG_DIR, "pitch_config.json"), "r", encoding="utf-8") as f:
        return json.load(f)


def _row(summary: str, **extra) -> pd.Series:
    data = {"summary": summary}
    data.update(extra)
    return pd.Series(data)


def _score(summary, scoring_config, pitch_config, columns_map=None, **extra):
    return scorer.score_row(_row(summary, **extra), "summary", columns_map or {}, scoring_config, pitch_config)


def test_positive_intent_scoring(scoring_config, pitch_config):
    result = _score(
        "Customer is ready to book and wants to schedule the earliest available slot.",
        scoring_config,
        pitch_config,
    )
    assert result["Lead_Score"] > scoring_config["base_score"]
    assert result["Lead_Priority"] in ("P1 - Hot", "P2 - Warm")


def test_negative_intent_scoring(scoring_config, pitch_config):
    result = _score(
        "Customer said not interested and declined the offer.",
        scoring_config,
        pitch_config,
    )
    assert result["Lead_Score"] < scoring_config["base_score"]


def test_hard_exclusion_wrong_number(scoring_config, pitch_config):
    result = _score("This is the wrong number, please remove it.", scoring_config, pitch_config)
    assert result["Lead_Priority"] == "P4 - Exclude"
    assert result["Lead_Exclusion_Reason"] == "Wrong number"
    assert result["Recommended_Agent_Pitch"] == "No further sales pitch recommended."


def test_hard_exclusion_do_not_call(scoring_config, pitch_config):
    result = _score("Customer asked us to stop calling and not call again.", scoring_config, pitch_config)
    assert result["Lead_Priority"] == "P4 - Exclude"
    assert "no further calls" in result["Lead_Exclusion_Reason"].lower()


def test_callback_detection(scoring_config, pitch_config):
    result = _score("Customer requested a callback for tomorrow evening.", scoring_config, pitch_config)
    assert "Timing or Callback" == result["Primary_Objection"]
    assert "callback" in result["Conversion_Evidence"].lower()


def test_booking_date_column_detection(scoring_config, pitch_config):
    result = _score(
        "Customer confirmed the appointment.",
        scoring_config,
        pitch_config,
        columns_map={"booking_date": "booking_date"},
        booking_date="2026-07-20",
    )
    assert "Booking date captured" in result["Conversion_Evidence"]


def test_slot_issue_detection(scoring_config, pitch_config):
    result = _score(
        "Customer wanted to book but there was a technical issue and the booking failed.",
        scoring_config,
        pitch_config,
    )
    assert result["Primary_Objection"] == "Technical Issue"


def test_wrong_number_detection_case_insensitive(scoring_config, pitch_config):
    result = _score("WRONG NUMBER, this is not the customer we are looking for.", scoring_config, pitch_config)
    assert result["Lead_Exclusion_Reason"] != ""


def test_empty_summary_handling(scoring_config, pitch_config):
    result = _score("", scoring_config, pitch_config)
    assert result["Primary_Objection"] == "Unresponsive"
    assert result["Lead_Score"] == scoring_config["base_score"]


def test_score_threshold_mapping(scoring_config, pitch_config):
    hot = _score(
        "Customer is ready to book, explicitly interested, wants to book immediately, requested booking, asked for callback.",
        scoring_config,
        pitch_config,
    )
    assert hot["Lead_Score"] >= scoring_config["priority_thresholds"]["P1_hot_min"]
    assert hot["Lead_Priority"] == "P1 - Hot"

    exclude = _score("Not interested, do not call again.", scoring_config, pitch_config)
    assert exclude["Lead_Priority"] == "P4 - Exclude"


def test_score_is_clamped_between_0_and_100(scoring_config, pitch_config):
    result = _score(
        "Wrong number, not interested, do not call, stop calling, refused, declined, hung up.",
        scoring_config,
        pitch_config,
    )
    assert 0 <= result["Lead_Score"] <= 100


def test_score_dataframe_duplicate_handling(scoring_config, pitch_config):
    df = pd.DataFrame(
        {
            "summary": [
                "Customer wants to book a slot.",
                "Customer wants to book a slot.",
            ]
        }
    )
    scored = scorer.score_dataframe(df, "summary", {}, scoring_config, pitch_config)
    assert len(scored) == 2
    assert scored.loc[0, "Lead_Score"] == scored.loc[1, "Lead_Score"]
