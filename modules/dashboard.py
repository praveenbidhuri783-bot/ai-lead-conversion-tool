"""Executive dashboard: KPIs, filters, and charts for scored leads."""

from __future__ import annotations

from typing import Dict, Optional

import pandas as pd
import plotly.express as px
import streamlit as st

PRIORITY_ORDER = ["P1 - Hot", "P2 - Warm", "P3 - Nurture", "P4 - Exclude"]
SENTIMENT_ORDER = ["Positive", "Neutral-Positive", "Neutral", "Neutral-Negative", "Negative"]


NOT_SPECIFIED = "(Not Specified)"


def _filled_str(series: pd.Series) -> pd.Series:
    """Coerce a column to string for filtering, mapping blanks/NaN to a selectable placeholder.

    Using ``.astype(str)`` directly on a NaN produces the literal string "nan",
    which never matches dropdown options built from ``.dropna()`` — silently
    dropping rows with missing values from every "select all" filter. Mapping
    blanks to an explicit, selectable placeholder keeps them included by default.
    """
    return series.where(series.notna() & (series.astype(str).str.strip() != ""), NOT_SPECIFIED).astype(str)


def mask_mobile(value: str) -> str:
    """Mask a mobile/phone number, keeping only the last 4 digits visible."""
    digits = "".join(ch for ch in str(value) if ch.isdigit())
    if len(digits) < 4:
        return "XXXXXX" + digits
    return "X" * max(0, len(digits) - 4) + digits[-4:]


def render_filters(df: pd.DataFrame, columns_map: Dict[str, str]) -> pd.DataFrame:
    """Render sidebar filter widgets and return the filtered dataframe."""
    st.subheader("Filters")
    filtered = df.copy()

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        priorities = sorted(filtered["Lead_Priority"].dropna().unique().tolist())
        selected_priorities = st.multiselect("Lead Priority", priorities, default=priorities)
        sentiments = sorted(filtered["Sentiment"].dropna().unique().tolist())
        selected_sentiments = st.multiselect("Sentiment", sentiments, default=sentiments)
    with col_b:
        objections = sorted(filtered["Primary_Objection"].dropna().unique().tolist())
        selected_objections = st.multiselect("Primary Objection", objections, default=objections)
        potentials = sorted(filtered["Conversion_Potential"].dropna().unique().tolist())
        selected_potentials = st.multiselect("Conversion Potential", potentials, default=potentials)
    with col_c:
        score_min, score_max = int(filtered["Lead_Score"].min()), int(filtered["Lead_Score"].max())
        score_range = st.slider("Score Range", 0, 100, (score_min, score_max))

    if selected_priorities:
        filtered = filtered[filtered["Lead_Priority"].isin(selected_priorities)]
    if selected_sentiments:
        filtered = filtered[filtered["Sentiment"].isin(selected_sentiments)]
    if selected_objections:
        filtered = filtered[filtered["Primary_Objection"].isin(selected_objections)]
    if selected_potentials:
        filtered = filtered[filtered["Conversion_Potential"].isin(selected_potentials)]
    filtered = filtered[
        (filtered["Lead_Score"] >= score_range[0]) & (filtered["Lead_Score"] <= score_range[1])
    ]

    with st.expander("More filters (campaign, product, language, dates)"):
        col_d, col_e, col_f = st.columns(3)
        with col_d:
            for canonical, label in (("campaign_name", "Campaign"), ("campaign_id", "Campaign ID")):
                col = columns_map.get(canonical)
                if col and col in filtered.columns:
                    values = _filled_str(filtered[col])
                    options = sorted(values.unique().tolist())
                    chosen = st.multiselect(label, options, default=options, key=f"filter_{canonical}")
                    if chosen:
                        filtered = filtered[values.isin(chosen)]
        with col_e:
            for canonical, label in (("partner", "Partner"), ("product", "Product"), ("service_type", "Service Type")):
                col = columns_map.get(canonical)
                if col and col in filtered.columns:
                    values = _filled_str(filtered[col])
                    options = sorted(values.unique().tolist())
                    chosen = st.multiselect(label, options, default=options, key=f"filter_{canonical}")
                    if chosen:
                        filtered = filtered[values.isin(chosen)]
        with col_f:
            lang_col = columns_map.get("language")
            if lang_col and lang_col in filtered.columns:
                values = _filled_str(filtered[lang_col])
                options = sorted(values.unique().tolist())
                chosen = st.multiselect("Language", options, default=options, key="filter_language")
                if chosen:
                    filtered = filtered[values.isin(chosen)]
            callback_col = columns_map.get("callback_request")
            if callback_col and callback_col in filtered.columns:
                only_callback = st.checkbox("Callback requested only", key="filter_callback")
                if only_callback:
                    filtered = filtered[
                        filtered[callback_col].astype(str).str.lower().isin(["true", "yes", "y", "1", "requested"])
                    ]

    return filtered


def _rate(numerator: int, denominator: int) -> float:
    return round(100.0 * numerator / denominator, 1) if denominator else 0.0


def compute_kpis(df: pd.DataFrame, columns_map: Dict[str, str]) -> Dict[str, float]:
    total = len(df)
    p1 = int((df["Lead_Priority"] == "P1 - Hot").sum())
    p2 = int((df["Lead_Priority"] == "P2 - Warm").sum())
    p3 = int((df["Lead_Priority"] == "P3 - Nurture").sum())
    p4 = int((df["Lead_Priority"] == "P4 - Exclude").sum())
    actionable = p1 + p2

    positive_sentiment = int(df["Sentiment"].isin(["Positive", "Neutral-Positive"]).sum())

    callback_col = columns_map.get("callback_request")
    if callback_col and callback_col in df.columns:
        callback_count = int(
            df[callback_col].astype(str).str.lower().isin(["true", "yes", "y", "1", "requested"]).sum()
        )
    else:
        callback_count = int(df["Conversion_Evidence"].str.contains("callback", case=False, na=False).sum())

    booking_col = columns_map.get("booking_date")
    if booking_col and booking_col in df.columns:
        booking_count = int((df[booking_col].astype(str).str.strip() != "").sum())
    else:
        booking_count = int(df["Conversion_Evidence"].str.contains("book", case=False, na=False).sum())

    wrong_number_count = int(df["Lead_Exclusion_Reason"].astype(str).str.contains("Wrong number", na=False).sum())
    early_disconnect_count = int((df["Primary_Objection"] == "Early Disconnect").sum())
    not_interested_count = int(
        (
            df["Lead_Exclusion_Reason"].astype(str).str.contains("not interested", case=False, na=False)
            | (df["Primary_Objection"] == "Not Interested")
        ).sum()
    )

    return {
        "Total Calls Analysed": total,
        "P1 Hot Leads": p1,
        "P2 Warm Leads": p2,
        "P3 Nurture Leads": p3,
        "P4 Excluded Leads": p4,
        "Actionable Leads (P1+P2)": actionable,
        "Actionable Lead Rate (%)": _rate(actionable, total),
        "Average Lead Score": round(df["Lead_Score"].mean(), 1) if total else 0.0,
        "Positive Sentiment Rate (%)": _rate(positive_sentiment, total),
        "Callback Request Rate (%)": _rate(callback_count, total),
        "Booking Intent Rate (%)": _rate(booking_count, total),
        "Wrong Number Rate (%)": _rate(wrong_number_count, total),
        "Early Disconnect Rate (%)": _rate(early_disconnect_count, total),
        "Not Interested Rate (%)": _rate(not_interested_count, total),
    }


def render_kpis(df: pd.DataFrame, columns_map: Dict[str, str]) -> None:
    kpis = compute_kpis(df, columns_map)
    st.subheader("Key Performance Indicators")
    keys = list(kpis.items())
    for row_start in range(0, len(keys), 4):
        cols = st.columns(4)
        for col, (label, value) in zip(cols, keys[row_start : row_start + 4]):
            col.metric(label, value)


def _campaign_column(columns_map: Dict[str, str]) -> Optional[str]:
    return columns_map.get("campaign_name") or columns_map.get("campaign_id")


def render_charts(df: pd.DataFrame, columns_map: Dict[str, str]) -> None:
    st.subheader("Analytics")

    row1_col1, row1_col2 = st.columns(2)
    with row1_col1:
        priority_counts = df["Lead_Priority"].value_counts().reindex(PRIORITY_ORDER).dropna()
        fig = px.pie(
            names=priority_counts.index, values=priority_counts.values, title="Lead Priority Distribution", hole=0.4
        )
        st.plotly_chart(fig, use_container_width=True)
    with row1_col2:
        sentiment_counts = df["Sentiment"].value_counts().reindex(SENTIMENT_ORDER).dropna()
        fig = px.bar(x=sentiment_counts.index, y=sentiment_counts.values, title="Sentiment Distribution")
        fig.update_layout(xaxis_title="Sentiment", yaxis_title="Count")
        st.plotly_chart(fig, use_container_width=True)

    row2_col1, row2_col2 = st.columns(2)
    with row2_col1:
        objection_counts = df["Primary_Objection"].value_counts().head(10)
        fig = px.bar(
            x=objection_counts.values, y=objection_counts.index, orientation="h", title="Top Objections"
        )
        fig.update_layout(xaxis_title="Count", yaxis_title="Objection")
        st.plotly_chart(fig, use_container_width=True)
    with row2_col2:
        fig = px.histogram(df, x="Lead_Score", nbins=20, title="Lead Score Distribution")
        st.plotly_chart(fig, use_container_width=True)

    campaign_col = _campaign_column(columns_map)
    if campaign_col and campaign_col in df.columns:
        row3_col1, row3_col2 = st.columns(2)
        with row3_col1:
            grp = df.groupby([campaign_col, "Conversion_Potential"]).size().reset_index(name="Count")
            fig = px.bar(
                grp, x=campaign_col, y="Count", color="Conversion_Potential",
                title="Conversion Potential by Campaign", barmode="stack",
            )
            st.plotly_chart(fig, use_container_width=True)
        with row3_col2:
            actionable = df[df["Lead_Priority"].isin(["P1 - Hot", "P2 - Warm"])]
            grp = actionable.groupby([campaign_col, "Lead_Priority"]).size().reset_index(name="Count")
            fig = px.bar(
                grp, x=campaign_col, y="Count", color="Lead_Priority",
                title="P1 and P2 Leads by Campaign", barmode="group",
            )
            st.plotly_chart(fig, use_container_width=True)

    service_col = columns_map.get("service_type") or columns_map.get("product")
    if service_col and service_col in df.columns:
        actionable = df[df["Lead_Priority"].isin(["P1 - Hot", "P2 - Warm"])]
        grp = actionable.groupby([service_col, "Lead_Priority"]).size().reset_index(name="Count")
        fig = px.bar(
            grp, x=service_col, y="Count", color="Lead_Priority",
            title="P1 and P2 Leads by Service Type", barmode="group",
        )
        st.plotly_chart(fig, use_container_width=True)

    row4_col1, row4_col2 = st.columns(2)
    with row4_col1:
        followup_counts = df["Recommended_Follow_Up_Time"].value_counts().head(10)
        fig = px.bar(
            x=followup_counts.values, y=followup_counts.index, orientation="h",
            title="Follow-up Priority by Recommended Time",
        )
        fig.update_layout(xaxis_title="Count", yaxis_title="Recommended Follow-up Time")
        st.plotly_chart(fig, use_container_width=True)
    with row4_col2:
        duration_col = columns_map.get("call_duration")
        if duration_col and duration_col in df.columns:
            plot_df = df.copy()
            plot_df["_duration_seconds"] = pd.to_numeric(plot_df[duration_col], errors="coerce")
            plot_df = plot_df.dropna(subset=["_duration_seconds"])
            if not plot_df.empty:
                fig = px.scatter(
                    plot_df, x="_duration_seconds", y="Lead_Score", color="Lead_Priority",
                    title="Call Duration vs Lead Score",
                    labels={"_duration_seconds": "Call Duration (seconds)"},
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Call duration data is not numeric; skipping duration vs score chart.")
        else:
            st.info("No call-duration column detected; skipping duration vs score chart.")

    render_funnel(df, columns_map)


def render_funnel(df: pd.DataFrame, columns_map: Dict[str, str]) -> None:
    total = len(df)
    unreachable = int(
        (
            df["Primary_Objection"].isin(["Wrong or Invalid Number", "Unresponsive", "Early Disconnect"])
            | df["Lead_Exclusion_Reason"].astype(str).str.contains("Wrong number", na=False)
        ).sum()
    )
    connected = max(total - unreachable, 0)

    not_interested = int((df["Primary_Objection"] == "Not Interested").sum())
    engaged = max(connected - not_interested, 0)

    interested = int(df["Lead_Priority"].isin(["P1 - Hot", "P2 - Warm", "P3 - Nurture"]).sum())

    callback_col = columns_map.get("callback_request")
    if callback_col and callback_col in df.columns:
        callback_requested = int(
            df[callback_col].astype(str).str.lower().isin(["true", "yes", "y", "1", "requested"]).sum()
        )
    else:
        callback_requested = int(df["Conversion_Evidence"].str.contains("callback", case=False, na=False).sum())

    booking_col = columns_map.get("booking_date")
    if booking_col and booking_col in df.columns:
        booking_intent = int((df[booking_col].astype(str).str.strip() != "").sum())
    else:
        booking_intent = int(df["Conversion_Evidence"].str.contains("book", case=False, na=False).sum())

    actionable = int(df["Lead_Priority"].isin(["P1 - Hot", "P2 - Warm"]).sum())

    stage_values = [total, connected, engaged, interested, callback_requested, booking_intent, actionable]
    running_cap = total
    capped_values = []
    for v in stage_values:
        running_cap = min(running_cap, v)
        capped_values.append(running_cap)

    fig = px.funnel(
        x=capped_values,
        y=["Total Uploaded", "Connected", "Engaged", "Interested", "Callback Requested", "Booking Intent", "P1/P2 Actionable"],
        title="Conversion Funnel",
    )
    st.plotly_chart(fig, use_container_width=True)


def render_lead_table(df: pd.DataFrame, columns_map: Dict[str, str], reveal_full_numbers: bool = False) -> None:
    st.subheader("Prioritised Lead Table")
    display_df = df.copy()

    mobile_col = columns_map.get("mobile") or columns_map.get("phone_number")
    if mobile_col and mobile_col in display_df.columns and not reveal_full_numbers:
        display_df[mobile_col] = display_df[mobile_col].apply(mask_mobile)

    priority_order_map = {p: i for i, p in enumerate(PRIORITY_ORDER)}
    display_df["_sort_priority"] = display_df["Lead_Priority"].map(priority_order_map).fillna(99)
    display_df = display_df.sort_values(["_sort_priority", "Lead_Score"], ascending=[True, False])
    display_df = display_df.drop(columns=["_sort_priority"])

    st.dataframe(display_df, use_container_width=True, height=420)
