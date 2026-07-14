"""Unit tests for objection detection."""

import json
import os

import pytest

from modules import objection_detector

CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config")


@pytest.fixture(scope="module")
def scoring_config():
    with open(os.path.join(CONFIG_DIR, "scoring_config.json"), "r", encoding="utf-8") as f:
        return json.load(f)


def test_no_objection_detected(scoring_config):
    primary, secondary = objection_detector.detect_objections(
        "customer is happy and confirmed the booking", scoring_config
    )
    assert primary == "No Major Objection"
    assert secondary == []


def test_timing_objection(scoring_config):
    primary, _ = objection_detector.detect_objections("customer is busy right now, call me later", scoring_config)
    assert primary == "Timing or Callback"


def test_price_objection(scoring_config):
    primary, _ = objection_detector.detect_objections("customer said it is too expensive", scoring_config)
    assert primary == "Price Concern"


def test_wrong_number_takes_priority_over_secondary_signals(scoring_config):
    primary, _ = objection_detector.detect_objections(
        "wrong number, also too expensive and not a good time", scoring_config
    )
    assert primary == "Wrong or Invalid Number"


def test_secondary_objections_captured(scoring_config):
    primary, secondary = objection_detector.detect_objections(
        "too expensive and is this genuine or a scam", scoring_config
    )
    assert primary == "Price Concern"
    assert "Trust Concern" in secondary


def test_language_barrier_detected(scoring_config):
    primary, _ = objection_detector.detect_objections("customer does not understand english", scoring_config)
    assert primary == "Language Barrier"
