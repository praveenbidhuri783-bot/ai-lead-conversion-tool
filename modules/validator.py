"""Data validation utilities for uploaded lead files."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd

PHONE_REGEX = re.compile(r"^\+?\d[\d\s-]{6,14}\d$")


@dataclass
class ValidationResult:
    is_valid: bool = True
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    total_rows: int = 0
    blank_summary_count: int = 0
    duplicate_row_count: int = 0
    invalid_phone_count: int = 0

    def to_summary_dict(self) -> Dict:
        return {
            "Total Rows": self.total_rows,
            "Blank Summaries": self.blank_summary_count,
            "Duplicate Rows": self.duplicate_row_count,
            "Invalid Phone Numbers": self.invalid_phone_count,
            "Errors": len(self.errors),
            "Warnings": len(self.warnings),
        }


def _find_phone_column(df: pd.DataFrame, optional_columns: Dict[str, str]) -> Optional[str]:
    for key in ("mobile", "phone_number"):
        if key in optional_columns:
            return optional_columns[key]
    return None


def validate_dataframe(
    df: pd.DataFrame,
    summary_column: Optional[str],
    optional_columns: Optional[Dict[str, str]] = None,
) -> ValidationResult:
    """Run structural validation on the uploaded dataframe before scoring."""
    optional_columns = optional_columns or {}
    result = ValidationResult(total_rows=len(df))

    if df.empty:
        result.is_valid = False
        result.errors.append("The uploaded file contains no data rows.")
        return result

    if not summary_column or summary_column not in df.columns:
        result.is_valid = False
        result.errors.append(
            "No call-summary column could be detected. Please map one manually."
        )
        return result

    blank_mask = df[summary_column].isna() | (df[summary_column].astype(str).str.strip() == "")
    result.blank_summary_count = int(blank_mask.sum())
    if result.blank_summary_count == len(df):
        result.is_valid = False
        result.errors.append("All rows have a blank call-summary value.")
    elif result.blank_summary_count > 0:
        result.warnings.append(
            f"{result.blank_summary_count} row(s) have a blank call summary and will "
            "receive a default 'Unresponsive' style score."
        )

    duplicate_mask = df.duplicated(keep=False)
    result.duplicate_row_count = int(duplicate_mask.sum())
    if result.duplicate_row_count > 0:
        result.warnings.append(
            f"{result.duplicate_row_count} duplicate row(s) detected. They will still be "
            "scored individually; consider de-duplicating upstream."
        )

    phone_col = _find_phone_column(df, optional_columns)
    if phone_col:
        phones = df[phone_col].dropna().astype(str).str.strip()
        phones = phones[phones != ""]
        invalid = phones[~phones.str.replace(r"[()]", "", regex=True).str.match(PHONE_REGEX)]
        result.invalid_phone_count = int(len(invalid))
        if result.invalid_phone_count > 0:
            result.warnings.append(
                f"{result.invalid_phone_count} row(s) have a phone/mobile number in an "
                "unexpected format."
            )

    if len(df.columns) != len(set(df.columns)):
        result.warnings.append("The file contains duplicate column headers.")

    return result
