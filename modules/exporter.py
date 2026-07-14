"""Excel report generation: builds the full multi-sheet workbook download."""

from __future__ import annotations

import io
from datetime import datetime, timezone
from typing import Dict

import pandas as pd

from modules import dashboard

SHEET_ORDER = [
    "Executive Summary",
    "All Lead Scoring",
    "P1 Hot Leads",
    "P2 Warm Leads",
    "P3 Nurture Leads",
    "P4 Excluded Leads",
    "Objection Analysis",
    "Scoring Methodology",
]


def _build_objection_analysis(df: pd.DataFrame, pitch_config: Dict) -> pd.DataFrame:
    total = len(df)
    counts = df["Primary_Objection"].value_counts()
    conversion_opportunity = df.groupby("Primary_Objection")["Lead_Priority"].apply(
        lambda s: int(s.isin(["P1 - Hot", "P2 - Warm"]).sum())
    )
    pitch_map = pitch_config.get("pitch_by_objection", {})

    rows = []
    for objection, count in counts.items():
        rows.append(
            {
                "Objection": objection,
                "Call Count": int(count),
                "Share (%)": round(100.0 * count / total, 1) if total else 0.0,
                "Conversion Opportunity (P1+P2)": int(conversion_opportunity.get(objection, 0)),
                "Recommended Response": pitch_map.get(objection, "") or "Standard priority-based pitch",
            }
        )
    return pd.DataFrame(rows).sort_values("Call Count", ascending=False)


def _build_executive_summary_blocks(df: pd.DataFrame, columns_map: Dict[str, str], run_metadata: Dict):
    kpis = dashboard.compute_kpis(df, columns_map)
    kpi_df = pd.DataFrame(list(kpis.items()), columns=["Metric", "Value"])

    priority_df = df["Lead_Priority"].value_counts().rename_axis("Lead Priority").reset_index(name="Count")
    sentiment_df = df["Sentiment"].value_counts().rename_axis("Sentiment").reset_index(name="Count")
    objection_df = df["Primary_Objection"].value_counts().rename_axis("Primary Objection").reset_index(name="Count")

    actionable = kpis["Actionable Leads (P1+P2)"]
    total = kpis["Total Calls Analysed"]
    recommendations = [
        f"{actionable} of {total} leads ({kpis['Actionable Lead Rate (%)']}%) are actionable (P1/P2) "
        "and should be prioritised for immediate agent follow-up.",
        f"Wrong-number rate is {kpis['Wrong Number Rate (%)']}%; consider a data-hygiene pass on the "
        "source list if this is elevated.",
        f"Callback request rate is {kpis['Callback Request Rate (%)']}% — ensure P1/P2 leads are called "
        "back within the recommended SLA windows.",
        f"Not-interested rate is {kpis['Not Interested Rate (%)']}%; review messaging/targeting for this "
        "segment before the next campaign wave.",
    ]
    recommendations_df = pd.DataFrame({"Key Recommendations": recommendations})

    meta_df = pd.DataFrame(
        {
            "Field": ["Scoring Version", "Report Generated (UTC)", "Total Records"],
            "Value": [run_metadata.get("scoring_version", "1.0.0"), run_metadata.get("timestamp", ""), total],
        }
    )

    return meta_df, kpi_df, priority_df, sentiment_df, objection_df, recommendations_df


def _write_executive_summary(writer: pd.ExcelWriter, df: pd.DataFrame, columns_map: Dict[str, str], run_metadata: Dict) -> None:
    meta_df, kpi_df, priority_df, sentiment_df, objection_df, recommendations_df = _build_executive_summary_blocks(
        df, columns_map, run_metadata
    )
    sheet = "Executive Summary"
    row = 0
    meta_df.to_excel(writer, sheet_name=sheet, startrow=row, index=False)
    row += len(meta_df) + 3
    kpi_df.to_excel(writer, sheet_name=sheet, startrow=row, index=False)
    row += len(kpi_df) + 3
    priority_df.to_excel(writer, sheet_name=sheet, startrow=row, index=False)
    row += len(priority_df) + 3
    sentiment_df.to_excel(writer, sheet_name=sheet, startrow=row, index=False)
    row += len(sentiment_df) + 3
    objection_df.to_excel(writer, sheet_name=sheet, startrow=row, index=False)
    row += len(objection_df) + 3
    recommendations_df.to_excel(writer, sheet_name=sheet, startrow=row, index=False)


def _build_scoring_methodology(scoring_config: Dict) -> pd.DataFrame:
    rows = []
    rows.append({"Category": "Base Score", "Rule": "base_score", "Score": scoring_config.get("base_score"), "Detail": ""})
    for key, rule in scoring_config.get("positive_signals", {}).items():
        detail = ", ".join(rule.get("keywords", [])) if rule.get("source") != "column" else f"column: {rule.get('column')}"
        rows.append({"Category": "Positive Signal", "Rule": key, "Score": rule.get("score"), "Detail": detail})
    duration_cfg = scoring_config.get("call_duration_signals", {})
    if duration_cfg:
        rows.append({
            "Category": "Positive Signal", "Rule": "call_duration_above_120",
            "Score": duration_cfg.get("above_120_score"), "Detail": f"column: {duration_cfg.get('column')}",
        })
        rows.append({
            "Category": "Positive Signal", "Rule": "call_duration_60_119",
            "Score": duration_cfg.get("range_60_119_score"), "Detail": f"column: {duration_cfg.get('column')}",
        })
    for key, rule in scoring_config.get("negative_signals", {}).items():
        rows.append({"Category": "Negative Signal", "Rule": key, "Score": rule.get("score"), "Detail": ", ".join(rule.get("keywords", []))})
    for key, rule in scoring_config.get("hard_exclusions", {}).items():
        detail = ", ".join(rule.get("keywords", [])) if rule.get("source") != "column" else f"column: {rule.get('column')}"
        rows.append({"Category": "Hard Exclusion", "Rule": key, "Score": "Forces P4", "Detail": f"{rule.get('reason')} ({detail})"})

    thresholds = scoring_config.get("priority_thresholds", {})
    rows.append({"Category": "Threshold", "Rule": "P1 Hot minimum score", "Score": thresholds.get("P1_hot_min"), "Detail": ""})
    rows.append({"Category": "Threshold", "Rule": "P2 Warm minimum score", "Score": thresholds.get("P2_warm_min"), "Detail": ""})
    rows.append({"Category": "Threshold", "Rule": "P3 Nurture minimum score", "Score": thresholds.get("P3_nurture_min"), "Detail": ""})
    rows.append({"Category": "Threshold", "Rule": "P4 Exclude maximum score", "Score": thresholds.get("P4_exclude_max"), "Detail": ""})

    return pd.DataFrame(rows)


def build_excel_report(
    df: pd.DataFrame,
    columns_map: Dict[str, str],
    scoring_config: Dict,
    pitch_config: Dict,
    run_metadata: Dict = None,
) -> io.BytesIO:
    """Build the complete multi-sheet Excel workbook and return it as an in-memory buffer."""
    run_metadata = run_metadata or {}
    run_metadata.setdefault("scoring_version", scoring_config.get("scoring_version", "1.0.0"))
    run_metadata.setdefault("timestamp", datetime.now(timezone.utc).isoformat())

    export_df = df.drop(columns=[c for c in df.columns if c.startswith("_")], errors="ignore")

    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        _write_executive_summary(writer, export_df, columns_map, run_metadata)
        export_df.to_excel(writer, sheet_name="All Lead Scoring", index=False)
        export_df[export_df["Lead_Priority"] == "P1 - Hot"].to_excel(writer, sheet_name="P1 Hot Leads", index=False)
        export_df[export_df["Lead_Priority"] == "P2 - Warm"].to_excel(writer, sheet_name="P2 Warm Leads", index=False)
        export_df[export_df["Lead_Priority"] == "P3 - Nurture"].to_excel(writer, sheet_name="P3 Nurture Leads", index=False)
        export_df[export_df["Lead_Priority"] == "P4 - Exclude"].to_excel(writer, sheet_name="P4 Excluded Leads", index=False)
        _build_objection_analysis(export_df, pitch_config).to_excel(writer, sheet_name="Objection Analysis", index=False)
        _build_scoring_methodology(scoring_config).to_excel(writer, sheet_name="Scoring Methodology", index=False)

    buffer.seek(0)
    return buffer


def build_csv_export(df: pd.DataFrame) -> bytes:
    export_df = df.drop(columns=[c for c in df.columns if c.startswith("_")], errors="ignore")
    return export_df.to_csv(index=False).encode("utf-8")


def build_priority_subset_excel(df: pd.DataFrame, priorities) -> io.BytesIO:
    export_df = df.drop(columns=[c for c in df.columns if c.startswith("_")], errors="ignore")
    subset = export_df[export_df["Lead_Priority"].isin(priorities)]
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        subset.to_excel(writer, sheet_name="Leads", index=False)
    buffer.seek(0)
    return buffer
