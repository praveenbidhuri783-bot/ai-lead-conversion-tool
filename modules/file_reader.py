"""File ingestion and column-detection utilities for the AI Lead Conversion Tool."""

from __future__ import annotations

import io
import json
import os
from typing import Dict, List, Optional, Union

import pandas as pd

SUPPORTED_EXTENSIONS = (".xlsx", ".xls", ".csv")


class FileReadError(Exception):
    """Raised when the uploaded file cannot be parsed."""


def load_column_mapping(config_path: str) -> Dict:
    """Load the column-mapping configuration from disk."""
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _normalise(name: str) -> str:
    return str(name).strip().lower().replace("-", "_").replace(" ", "_")


def read_input_file(file: Union[str, io.BytesIO], filename: Optional[str] = None) -> pd.DataFrame:
    """Read an uploaded .xlsx, .xls, or .csv file into a DataFrame.

    ``file`` may be a filesystem path or a file-like object (e.g. a Streamlit
    UploadedFile). ``filename`` is required when ``file`` is not a path, so the
    correct parser can be selected.
    """
    name = filename or (file if isinstance(file, str) else getattr(file, "name", None))
    if not name:
        raise FileReadError("Could not determine the file name to detect its format.")

    ext = os.path.splitext(name)[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise FileReadError(
            f"Unsupported file type '{ext}'. Please upload a .xlsx, .xls, or .csv file."
        )

    try:
        if ext == ".csv":
            df = pd.read_csv(file, dtype=str, keep_default_na=False, na_values=[""])
        else:
            engine = "openpyxl" if ext == ".xlsx" else None
            df = pd.read_excel(file, dtype=str, engine=engine)
    except Exception as exc:  # noqa: BLE001 - surface a clean, user-facing error
        raise FileReadError(f"Failed to read the file: {exc}") from exc

    if df is None or df.empty:
        raise FileReadError("The uploaded file is empty.")

    df.columns = [str(c).strip() for c in df.columns]
    return df


def detect_summary_column(df: pd.DataFrame, column_mapping: Dict) -> Optional[str]:
    """Guess which column holds the call summary text using configured aliases."""
    aliases = {_normalise(a) for a in column_mapping.get("summary_column_aliases", [])}
    for col in df.columns:
        if _normalise(col) in aliases:
            return col

    # Fallback: pick the free-text column with the greatest average text length.
    candidate, best_len = None, 0
    for col in df.columns:
        sample = df[col].dropna().astype(str).head(200)
        if sample.empty:
            continue
        avg_len = sample.str.len().mean()
        if avg_len and avg_len > best_len and avg_len > 25:
            candidate, best_len = col, avg_len
    return candidate


def detect_optional_columns(df: pd.DataFrame, column_mapping: Dict) -> Dict[str, str]:
    """Map canonical optional field names to the actual column names present in df."""
    found: Dict[str, str] = {}
    normalised_cols = {_normalise(c): c for c in df.columns}
    for canonical, aliases in column_mapping.get("optional_column_aliases", {}).items():
        for alias in aliases:
            key = _normalise(alias)
            if key in normalised_cols:
                found[canonical] = normalised_cols[key]
                break
    return found


def list_all_columns(df: pd.DataFrame) -> List[str]:
    return list(df.columns)
