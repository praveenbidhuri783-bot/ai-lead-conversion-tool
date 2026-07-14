"""Unit tests for input validation."""

import pandas as pd

from modules import validator


def test_empty_dataframe_is_invalid():
    result = validator.validate_dataframe(pd.DataFrame(), None)
    assert not result.is_valid
    assert result.errors


def test_missing_summary_column_is_invalid():
    df = pd.DataFrame({"foo": ["a", "b"]})
    result = validator.validate_dataframe(df, None)
    assert not result.is_valid
    assert "summary" in result.errors[0].lower()


def test_all_blank_summaries_is_invalid():
    df = pd.DataFrame({"summary": ["", None, "  "]})
    result = validator.validate_dataframe(df, "summary")
    assert not result.is_valid


def test_partial_blank_summaries_is_valid_with_warning():
    df = pd.DataFrame({"summary": ["Customer interested.", "", "Wants to book."]})
    result = validator.validate_dataframe(df, "summary")
    assert result.is_valid
    assert result.blank_summary_count == 1
    assert result.warnings


def test_duplicate_rows_detected():
    df = pd.DataFrame({"summary": ["Same summary text.", "Same summary text.", "Different text."]})
    result = validator.validate_dataframe(df, "summary")
    assert result.duplicate_row_count == 2


def test_invalid_phone_numbers_detected():
    df = pd.DataFrame(
        {
            "summary": ["Customer interested.", "Wants to book.", "Not interested."],
            "mobile": ["9876543210", "abc123", "12"],
        }
    )
    result = validator.validate_dataframe(df, "summary", {"mobile": "mobile"})
    assert result.invalid_phone_count == 2


def test_valid_file_no_errors():
    df = pd.DataFrame({"summary": ["Customer interested.", "Wants to book."], "mobile": ["9876543210", "9123456789"]})
    result = validator.validate_dataframe(df, "summary", {"mobile": "mobile"})
    assert result.is_valid
    assert not result.errors
