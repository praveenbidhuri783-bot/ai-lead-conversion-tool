"""AI Lead Sentiment & Conversion Scoring Tool.

Streamlit application entry point. Wires together file ingestion,
validation, rule-based scoring, the executive dashboard, and Excel/CSV
export, per the build brief.
"""

from __future__ import annotations

import copy
import json
import os
from datetime import datetime, timezone

import pandas as pd
import streamlit as st

from modules import dashboard, exporter, file_reader, llm_classifier, scorer, validator

CONFIG_DIR = os.path.join(os.path.dirname(__file__), "config")
SCORING_CONFIG_PATH = os.path.join(CONFIG_DIR, "scoring_config.json")
PITCH_CONFIG_PATH = os.path.join(CONFIG_DIR, "pitch_config.json")
COLUMN_MAPPING_PATH = os.path.join(CONFIG_DIR, "column_mapping.json")

st.set_page_config(page_title="AI Lead Sentiment & Conversion Scoring Tool", layout="wide")


@st.cache_data(show_spinner=False)
def _load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _init_session_state() -> None:
    defaults = {
        "raw_df": None,
        "uploaded_filename": None,
        "summary_column": None,
        "columns_map": {},
        "validation_result": None,
        "scored_df": None,
        "scoring_config": copy.deepcopy(_load_json(SCORING_CONFIG_PATH)),
        "pitch_config": copy.deepcopy(_load_json(PITCH_CONFIG_PATH)),
        "column_mapping": _load_json(COLUMN_MAPPING_PATH),
        "run_metadata": {},
        "reveal_full_numbers": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def render_header() -> None:
    st.title("AI Lead Sentiment & Conversion Scoring Tool")
    st.caption(
        "Analyses AI voice-call summaries, scores conversion likelihood, and recommends "
        "the next-best agent action and pitch for every lead."
    )
    st.markdown(f"**Scoring Version:** `{st.session_state['scoring_config'].get('scoring_version', '1.0.0')}`")
    st.divider()


def render_upload_section() -> None:
    st.header("1. Upload")
    uploaded = st.file_uploader("Upload call-summary file", type=["xlsx", "xls", "csv"])

    if uploaded is None:
        st.info("Upload a .xlsx, .xls, or .csv file containing AI call summaries to get started.")
        return

    if st.session_state["uploaded_filename"] != uploaded.name:
        try:
            df = file_reader.read_input_file(uploaded, filename=uploaded.name)
        except file_reader.FileReadError as exc:
            st.error(str(exc))
            return

        st.session_state["raw_df"] = df
        st.session_state["uploaded_filename"] = uploaded.name
        st.session_state["summary_column"] = file_reader.detect_summary_column(df, st.session_state["column_mapping"])
        st.session_state["columns_map"] = file_reader.detect_optional_columns(df, st.session_state["column_mapping"])
        st.session_state["validation_result"] = None
        st.session_state["scored_df"] = None

    df = st.session_state["raw_df"]
    st.success(f"Loaded **{uploaded.name}** — {len(df):,} rows, {len(df.columns)} columns.")

    col_left, col_right = st.columns(2)
    with col_left:
        st.markdown("**Detected Column Mapping**")
        mapping_rows = [{"Field": "Call Summary (auto-detected)", "Column": st.session_state["summary_column"] or "Not detected"}]
        for canonical, actual in st.session_state["columns_map"].items():
            mapping_rows.append({"Field": canonical, "Column": actual})
        st.dataframe(pd.DataFrame(mapping_rows), use_container_width=True, hide_index=True)

    with col_right:
        st.markdown("**Manual Field Mapping**")
        options = ["-- None --"] + list(df.columns)
        current = st.session_state["summary_column"]
        index = options.index(current) if current in options else 0
        chosen = st.selectbox("Call Summary Column", options, index=index)
        st.session_state["summary_column"] = None if chosen == "-- None --" else chosen

        with st.expander("Override optional field mappings"):
            for canonical in st.session_state["column_mapping"].get("optional_column_aliases", {}):
                current_val = st.session_state["columns_map"].get(canonical, "-- None --")
                opts = ["-- None --"] + list(df.columns)
                idx = opts.index(current_val) if current_val in opts else 0
                selection = st.selectbox(canonical, opts, index=idx, key=f"map_{canonical}")
                if selection == "-- None --":
                    st.session_state["columns_map"].pop(canonical, None)
                else:
                    st.session_state["columns_map"][canonical] = selection

    if st.button("Validate File", type="primary"):
        result = validator.validate_dataframe(
            df, st.session_state["summary_column"], st.session_state["columns_map"]
        )
        st.session_state["validation_result"] = result

    result = st.session_state["validation_result"]
    if result is not None:
        st.markdown("**Validation Summary**")
        st.dataframe(pd.DataFrame([result.to_summary_dict()]), use_container_width=True, hide_index=True)
        for err in result.errors:
            st.error(err)
        for warn in result.warnings:
            st.warning(warn)
        if result.is_valid:
            st.success("File passed validation. You can proceed to configuration and analysis.")


def render_configuration_section() -> dict:
    st.header("2. Configuration")
    cfg = st.session_state["scoring_config"]

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Rule Groups**")
        apply_positive = st.checkbox("Apply positive signal scoring", value=True)
        apply_negative = st.checkbox("Apply negative signal scoring", value=True)
        apply_exclusions = st.checkbox("Apply hard exclusion rules", value=True)
        apply_duration = st.checkbox("Include call-duration signals", value=True)

    with col2:
        st.markdown("**Priority Thresholds**")
        thresholds = cfg["priority_thresholds"]
        p1_min = st.slider("P1 Hot minimum score", 50, 100, thresholds["P1_hot_min"])
        p2_min = st.slider("P2 Warm minimum score", 30, p1_min - 1, min(thresholds["P2_warm_min"], p1_min - 1))
        p3_min = st.slider("P3 Nurture minimum score", 0, p2_min - 1, min(thresholds["P3_nurture_min"], p2_min - 1))

    col3, col4, col5 = st.columns(3)
    with col3:
        campaign_objective = st.selectbox(
            "Campaign Objective", ["Booking / Appointment", "Consultation", "Awareness", "Renewal"]
        )
    with col4:
        transaction_type = st.selectbox("Transaction Type", ["Booking", "Purchase", "Subscription", "Enquiry"])
    with col5:
        service_type_default = st.text_input("Default Service Type (used when column is missing)", value="")

    st.markdown("**Optional AI Analysis**")
    llm_available = llm_classifier.is_llm_available()
    use_llm = st.checkbox(
        "Use Advanced AI Analysis (LLM enrichment for top leads)",
        value=False,
        disabled=not llm_available,
        help="Requires ANTHROPIC_API_KEY or OPENAI_API_KEY to be set in the environment."
        if not llm_available
        else "Enriches the top-scoring leads with an LLM-based sentiment/intent/pitch review.",
    )
    llm_top_n = 0
    if use_llm:
        llm_top_n = st.number_input("Number of top leads to enrich with AI", min_value=1, max_value=500, value=50)
    if not llm_available:
        st.caption("No LLM API key detected — the tool will run fully on rule-based scoring.")

    run_config = copy.deepcopy(cfg)
    run_config["priority_thresholds"]["P1_hot_min"] = p1_min
    run_config["priority_thresholds"]["P2_warm_min"] = p2_min
    run_config["priority_thresholds"]["P3_nurture_min"] = p3_min
    if not apply_positive:
        run_config["positive_signals"] = {}
    if not apply_negative:
        run_config["negative_signals"] = {}
    if not apply_exclusions:
        run_config["hard_exclusions"] = {}
    if not apply_duration:
        run_config["call_duration_signals"] = {}

    return {
        "run_config": run_config,
        "campaign_objective": campaign_objective,
        "transaction_type": transaction_type,
        "service_type_default": service_type_default,
        "use_llm": use_llm,
        "llm_top_n": int(llm_top_n),
    }


def render_analysis_section(config_state: dict) -> None:
    st.header("3. Analysis")
    df = st.session_state["raw_df"]
    if df is None:
        st.info("Upload and validate a file first.")
        return

    result = st.session_state["validation_result"]
    if result is None or not result.is_valid:
        st.warning("Please validate the file (step 1) before running analysis.")
        return

    st.write(f"Records ready for analysis: **{len(df):,}**")

    if st.button("Analyse Leads", type="primary"):
        progress_bar = st.progress(0, text="Scoring leads...")

        def _update(done: int, total: int) -> None:
            progress_bar.progress(min(done / total, 1.0), text=f"Scoring leads... {done:,}/{total:,}")

        working_df = df.copy()
        service_col = st.session_state["columns_map"].get("service_type")
        if not service_col and config_state["service_type_default"]:
            working_df["service_type"] = config_state["service_type_default"]
            st.session_state["columns_map"]["service_type"] = "service_type"

        scored = scorer.score_dataframe(
            working_df,
            st.session_state["summary_column"],
            st.session_state["columns_map"],
            config_state["run_config"],
            st.session_state["pitch_config"],
            progress_callback=_update,
        )
        progress_bar.progress(1.0, text="Scoring complete.")

        if config_state["use_llm"] and config_state["llm_top_n"] > 0:
            _apply_llm_enrichment(scored, config_state["llm_top_n"])

        st.session_state["scored_df"] = scored
        st.session_state["run_metadata"] = {
            "scoring_version": config_state["run_config"].get("scoring_version", "1.0.0"),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "campaign_objective": config_state["campaign_objective"],
            "transaction_type": config_state["transaction_type"],
        }
        st.success(f"Analysis complete for {len(scored):,} records.")


def _apply_llm_enrichment(scored: pd.DataFrame, top_n: int) -> None:
    summary_col = st.session_state["summary_column"]
    top_leads = scored.sort_values("Lead_Score", ascending=False).head(top_n)

    ai_sentiment, ai_probability, ai_pitch, ai_reason = {}, {}, {}, {}
    progress = st.progress(0, text="Running AI enrichment...")
    for i, (idx, row) in enumerate(top_leads.iterrows()):
        try:
            result = llm_classifier.classify_with_llm(str(row.get(summary_col, "")))
            ai_sentiment[idx] = result.get("sentiment", "")
            ai_probability[idx] = result.get("conversion_probability", "")
            ai_pitch[idx] = result.get("recommended_pitch", "")
            ai_reason[idx] = result.get("reason", "")
        except llm_classifier.LLMClassificationError:
            continue
        progress.progress((i + 1) / len(top_leads), text=f"AI enrichment {i + 1}/{len(top_leads)}")

    scored["AI_Sentiment"] = pd.Series(ai_sentiment)
    scored["AI_Conversion_Probability"] = pd.Series(ai_probability)
    scored["AI_Recommended_Pitch"] = pd.Series(ai_pitch)
    scored["AI_Reason"] = pd.Series(ai_reason)


def render_dashboard_section() -> None:
    st.header("4. Dashboard")
    scored = st.session_state["scored_df"]
    if scored is None:
        st.info("Run analysis (step 3) to see the executive dashboard.")
        return

    filtered = dashboard.render_filters(scored, st.session_state["columns_map"])
    dashboard.render_kpis(filtered, st.session_state["columns_map"])
    dashboard.render_charts(filtered, st.session_state["columns_map"])

    st.session_state["reveal_full_numbers"] = st.checkbox(
        "Reveal full mobile numbers in table (authorised users only)",
        value=st.session_state["reveal_full_numbers"],
    )
    dashboard.render_lead_table(filtered, st.session_state["columns_map"], st.session_state["reveal_full_numbers"])
    st.session_state["filtered_df"] = filtered


def render_export_section() -> None:
    st.header("5. Export")
    scored = st.session_state["scored_df"]
    if scored is None:
        st.info("Run analysis (step 3) to enable exports.")
        return

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        buffer = exporter.build_excel_report(
            scored,
            st.session_state["columns_map"],
            st.session_state["scoring_config"],
            st.session_state["pitch_config"],
            st.session_state["run_metadata"],
        )
        st.download_button(
            "Download Full Excel Report",
            data=buffer,
            file_name=f"lead_scoring_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    with col2:
        p1_buffer = exporter.build_priority_subset_excel(scored, ["P1 - Hot"])
        st.download_button(
            "Download P1 Leads",
            data=p1_buffer,
            file_name="p1_hot_leads.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    with col3:
        p1p2_buffer = exporter.build_priority_subset_excel(scored, ["P1 - Hot", "P2 - Warm"])
        st.download_button(
            "Download P1 + P2 Leads",
            data=p1p2_buffer,
            file_name="p1_p2_actionable_leads.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    with col4:
        csv_bytes = exporter.build_csv_export(scored)
        st.download_button("Download CSV", data=csv_bytes, file_name="lead_scoring_all.csv", mime="text/csv")


def main() -> None:
    _init_session_state()
    render_header()
    render_upload_section()
    st.divider()
    config_state = render_configuration_section()
    st.divider()
    render_analysis_section(config_state)
    st.divider()
    render_dashboard_section()
    st.divider()
    render_export_section()


if __name__ == "__main__":
    main()
