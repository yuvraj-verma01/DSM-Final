from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "outputs" / "analysis"

HIGH_ST_PATH = ANALYSIS / "state_analysis_dataset_high_st_states.csv"
ALL_STATES_PATH = ANALYSIS / "state_analysis_dataset_all_states.csv"

THEME = {
    "blue": "#2f5d7c",
    "brown": "#8a4f28",
    "green": "#4d7c8a",
    "red": "#8f3f3f",
    "gold": "#b7832f",
    "gray": "#6d7475",
    "paper": "#f8f5ec",
    "panel": "#fffdf8",
    "ink": "#202426",
    "muted": "#68716f",
    "line": "#ddd4c3",
}

LABELS = {
    "state": "State",
    "st_literacy_rate_pct": "ST literacy rate (%)",
    "literacy_gap_pct": "Literacy gap (percentage points)",
    "female_literacy_gap_pct": "Female literacy gap within STs (pp)",
    "dropout_primary_pct": "Primary dropout (%)",
    "dropout_upper_primary_pct": "Upper-primary dropout (%)",
    "dropout_secondary_pct": "Secondary dropout (%)",
    "ger_classes_i_viii_clean": "GER I-VIII",
    "ger_classes_ix_x_clean": "Secondary GER IX-X",
    "ger_classes_ix_x_girls_clean": "Girls' secondary GER IX-X",
    "ger_classes_ix_x_gpi_clean": "Secondary GER IX-X GPI",
    "ger_latest_secondary_total_clean": "Latest secondary GER IX-X",
    "ger_latest_secondary_girls_clean": "Latest girls' secondary GER IX-X",
    "gpi_secondary_clean": "Official secondary GPI",
    "gpi_higher_secondary_clean": "Official higher-secondary GPI",
    "st_bpl_mean_pct": "Mean ST poverty rate (%)",
    "employment_wpr_person_per_1000": "WPR (NDAP ratio-style value)",
    "employment_pu_person_per_1000": "Unemployment indicator",
    "employment_wpr_female_per_1000": "Female WPR (NDAP ratio-style value)",
    "mgnreg_sought_not_received_per_1000": "MGNREG sought-not-received per 1000",
    "mgnreg_work_100_plus_days_per_1000": "MGNREG 100-plus-days per 1000",
    "mgnreg_average_days_worked": "Average MGNREG days worked",
    "scholarship_total_release_2023_24_lakh_per_100k_st_pop": "2023-24 scholarship release per 100k ST pop",
    "scholarship_utilization_2023_24_pct": "2023-24 scholarship utilization (%)",
    "scholarship_cumulative_release_lakh_per_100k_st_pop": "2019-24 scholarship release per 100k ST pop",
    "st_share_state_population_pct": "ST share of state population (%)",
    "villages_gt50_per_100k_st_pop": "Villages >50% ST per 100k ST population",
    "villages_gt75_per_100k_st_pop": "Villages >75% ST per 100k ST population",
    "villages_gt90_per_100k_st_pop": "Villages >90% ST per 100k ST population",
    "villages_all_st_per_100k_st_pop": "100% ST villages per 100k ST population",
    "tribe_weighted_literacy_female_pct": "Tribe-weighted female literacy (%)",
    "low_literacy_district_count": "Low female-literacy district count",
}


@st.cache_data
def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    high_st = pd.read_csv(HIGH_ST_PATH)
    all_states = pd.read_csv(ALL_STATES_PATH)
    return add_derived_columns(high_st), add_derived_columns(all_states)


def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in ["ger_classes_i_viii", "ger_classes_ix_x", "ger_classes_xi_xii"]:
        if col in df:
            df[f"{col}_clean"] = df[col].where(df[col].between(0, 500))

    for col in [
        "ger_classes_ix_x_boys",
        "ger_classes_ix_x_girls",
        "ger_classes_xi_xii_boys",
        "ger_classes_xi_xii_girls",
        "ger_classes_i_viii_girls",
    ]:
        if col in df:
            df[f"{col}_clean"] = df[col].where(df[col].between(0, 500))

    for col in ["ger_classes_ix_x_gpi", "ger_classes_xi_xii_gpi", "ger_classes_i_viii_gpi"]:
        if col in df:
            df[f"{col}_clean"] = df[col].where(df[col].between(0, 3))

    for col in [
        "ger_latest_secondary_total",
        "ger_latest_secondary_girls",
        "ger_latest_higher_secondary_total",
        "ger_latest_higher_secondary_girls",
    ]:
        if col in df:
            df[f"{col}_clean"] = df[col].where(df[col].between(0, 500))

    for col in ["gpi_secondary", "gpi_higher_secondary"]:
        if col in df:
            df[f"{col}_clean"] = df[col].where(df[col].between(0, 3))

    if {"tribe_weighted_literacy_male_pct", "tribe_weighted_literacy_female_pct"}.issubset(df.columns):
        df["female_literacy_gap_pct"] = (
            df["tribe_weighted_literacy_male_pct"] - df["tribe_weighted_literacy_female_pct"]
        )

    for label, col in [
        ("gt50", "tribal_villages_gt_50_count"),
        ("gt75", "tribal_villages_gt_75_count"),
        ("gt90", "tribal_villages_gt_90_count"),
        ("all_st", "tribal_villages_100_pct_count"),
    ]:
        if {col, "st_population"}.issubset(df.columns):
            df[f"villages_{label}_per_100k_st_pop"] = df[col] / df["st_population"] * 100000

    return df


def label(col: str) -> str:
    return LABELS.get(col, col.replace("_", " ").title())


def corr_text(df: pd.DataFrame, x: str, y: str) -> str:
    subset = df[[x, y]].dropna()
    if len(subset) < 4 or subset[x].nunique() <= 1 or subset[y].nunique() <= 1:
        return f"Pearson r not shown; n = {len(subset)}"
    r, p = stats.pearsonr(subset[x], subset[y])
    p_text = "p < 0.001" if p < 0.001 else f"p = {p:.4f}"
    return f"Pearson r = {r:.4f}; {p_text}; n = {len(subset)}"


def scatter(
    df: pd.DataFrame,
    x: str,
    y: str,
    title: str,
    color_col: str | None = "dropout_secondary_pct",
    trendline: bool = True,
    height: int = 560,
    chart_key: str | None = None,
) -> None:
    cols = ["state", x, y]
    if color_col in df.columns:
        cols.append(color_col)
    cols = list(dict.fromkeys(cols))
    plot_df = df[cols].dropna(subset=[x, y]).copy()
    if plot_df.empty:
        st.warning(f"No data available for {label(x)} vs {label(y)}.")
        return

    fig = px.scatter(
        plot_df,
        x=x,
        y=y,
        color=color_col if color_col in plot_df.columns else None,
        hover_name="state",
        hover_data={x: ":.2f", y: ":.2f"},
        color_continuous_scale=["#2f5d7c", "#75a99f", "#d8c06c", "#8a4f28"],
        labels={x: label(x), y: label(y), **({color_col: label(color_col)} if color_col else {})},
        title=title,
        height=height,
    )
    if color_col is None:
        fig.update_traces(marker=dict(color="#496f79", size=11, line=dict(color="#ffffff", width=1.2)))
    if trendline and len(plot_df) >= 4 and plot_df[x].nunique() > 1:
        slope, intercept, *_ = stats.linregress(plot_df[x], plot_df[y])
        x_min = float(plot_df[x].min())
        x_max = float(plot_df[x].max())
        fig.add_trace(
            go.Scatter(
                x=[x_min, x_max],
                y=[slope * x_min + intercept, slope * x_max + intercept],
                mode="lines",
                line={"color": "#1f2426", "width": 2.3},
                name="Linear trend",
                hoverinfo="skip",
            )
        )
    fig.update_traces(marker={"size": 12, "line": {"width": 1.4, "color": "white"}, "opacity": 0.92})
    fig.update_layout(
        margin={"l": 30, "r": 25, "t": 72, "b": 58},
        paper_bgcolor=THEME["panel"],
        plot_bgcolor="#fbf8f0",
        font={"size": 16, "color": "#1f2426", "family": "Inter, Segoe UI, sans-serif"},
        title={"font": {"size": 24, "color": THEME["ink"], "family": "Inter, Segoe UI, sans-serif"}},
        legend={"font": {"size": 15}},
        xaxis={"title": {"font": {"size": 17, "color": "#1f2426"}}, "tickfont": {"size": 14, "color": "#1f2426"}},
        yaxis={"title": {"font": {"size": 17, "color": "#1f2426"}}, "tickfont": {"size": 14, "color": "#1f2426"}},
        coloraxis={"colorbar": {"tickfont": {"size": 14, "color": "#1f2426"}, "title": {"font": {"size": 15, "color": "#1f2426"}}}},
        hoverlabel={"bgcolor": "white", "font_size": 14, "font_color": THEME["ink"]},
    )
    fig.update_xaxes(showgrid=True, gridcolor="#e2dacb", zeroline=False, showline=True, linecolor="#9b927f")
    fig.update_yaxes(showgrid=True, gridcolor="#e2dacb", zeroline=False, showline=True, linecolor="#9b927f")
    st.plotly_chart(fig, use_container_width=True, key=chart_key or f"scatter_{title}_{x}_{y}_{color_col}")
    st.caption(corr_text(plot_df, x, y))


def bar_rank(df: pd.DataFrame, value: str, title: str, top_n: int = 10, ascending: bool = False) -> None:
    plot_df = df[["state", value]].dropna().sort_values(value, ascending=ascending).head(top_n)
    fig = px.bar(
        plot_df,
        x=value,
        y="state",
        orientation="h",
        labels={value: label(value), "state": "State"},
        title=title,
        height=520,
        color=value,
        color_continuous_scale=["#2f5d7c", "#6f9b92", "#d7c17a", "#8a4f28"],
    )
    fig.update_layout(
        yaxis={"categoryorder": "total ascending", "tickfont": {"size": 15}, "title": {"font": {"size": 17}}},
        xaxis={"tickfont": {"size": 14}, "title": {"font": {"size": 17}}},
        title={"font": {"size": 24}},
        font={"size": 16, "color": THEME["ink"], "family": "Inter, Segoe UI, sans-serif"},
        paper_bgcolor=THEME["panel"],
        plot_bgcolor="#fbf8f0",
        margin={"l": 25, "r": 20, "t": 72, "b": 45},
    )
    fig.update_xaxes(showgrid=True, gridcolor="#e2dacb", zeroline=False)
    fig.update_yaxes(showgrid=False)
    st.plotly_chart(fig, use_container_width=True, key=f"bar_{title}_{value}")


def correlation_table(df: pd.DataFrame, pairs: list[tuple[str, str]]) -> pd.DataFrame:
    rows = []
    for x, y in pairs:
        subset = df[[x, y]].dropna()
        if len(subset) < 4 or subset[x].nunique() <= 1 or subset[y].nunique() <= 1:
            continue
        r, p = stats.pearsonr(subset[x], subset[y])
        rows.append(
            {
                "x": label(x),
                "y": label(y),
                "n": len(subset),
                "pearson_r": round(float(r), 4),
                "pearson_p": round(float(p), 4),
                "abs_r": round(abs(float(r)), 4),
            }
        )
    return pd.DataFrame(rows).sort_values("abs_r", ascending=False).reset_index(drop=True)


def metric_row(df: pd.DataFrame) -> None:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("High-ST states", f"{len(df)}")
    c2.metric("Mean ST literacy", f"{df['st_literacy_rate_pct'].mean():.1f}%")
    c3.metric("Mean secondary dropout", f"{df['dropout_secondary_pct'].mean():.1f}%")
    c4.metric("Mean MGNREG unmet demand", f"{df['mgnreg_sought_not_received_per_1000'].mean():.1f}")


def state_profile(df: pd.DataFrame, state: str) -> None:
    row = df.loc[df["state"].eq(state)].iloc[0]
    fields = [
        "st_literacy_rate_pct",
        "literacy_gap_pct",
        "dropout_secondary_pct",
        "st_bpl_mean_pct",
        "employment_wpr_person_per_1000",
        "mgnreg_sought_not_received_per_1000",
    ]
    rows = []
    for field in fields:
        value = row.get(field)
        median = df[field].median(skipna=True) if field in df else None
        value_text = "NA" if pd.isna(value) else f"{value:.2f}"
        median_text = "NA" if pd.isna(median) else f"{median:.2f}"
        rows.append(
            '<div class="profile-row">'
            f'<div class="profile-label">{label(field)}</div>'
            f'<div class="profile-value">{value_text}</div>'
            f'<div class="profile-median">Median: {median_text}</div>'
            "</div>"
        )
    profile_html = f'<div class="profile-card"><h3>{state}</h3>{"".join(rows)}</div>'
    st.markdown(profile_html, unsafe_allow_html=True)


def question_tab(title: str, pairs: list[tuple[str, str]], df: pd.DataFrame, charts: list[tuple[str, str, str]], prefix: str) -> None:
    st.subheader(title)
    table = correlation_table(df, pairs)
    if not table.empty:
        st.dataframe(table, use_container_width=True, hide_index=True)
    for index, (chart_title, x, y) in enumerate(charts):
        scatter(df, x, y, chart_title, chart_key=f"{prefix}_{index}_{x}_{y}")


def q5_mismatch(df: pd.DataFrame) -> None:
    st.subheader("Q5. Education-livelihood mismatch states")
    med_lit = df["st_literacy_rate_pct"].median()
    med_ger = df["ger_latest_secondary_total_clean"].median()
    med_wpr = df["employment_wpr_person_per_1000"].median()
    med_unemp = df["employment_pu_person_per_1000"].median()
    med_mgnreg = df["mgnreg_sought_not_received_per_1000"].median()
    med_poverty = df["st_bpl_mean_pct"].median()

    rows = []
    for _, row in df.iterrows():
        education_ok = row["st_literacy_rate_pct"] >= med_lit or row["ger_latest_secondary_total_clean"] >= med_ger
        flags = []
        if pd.notna(row["employment_wpr_person_per_1000"]) and row["employment_wpr_person_per_1000"] <= med_wpr:
            flags.append("low WPR")
        if pd.notna(row["employment_pu_person_per_1000"]) and row["employment_pu_person_per_1000"] >= med_unemp:
            flags.append("high unemployment")
        if pd.notna(row["mgnreg_sought_not_received_per_1000"]) and row["mgnreg_sought_not_received_per_1000"] >= med_mgnreg:
            flags.append("high MGNREG unmet demand")
        if pd.notna(row["st_bpl_mean_pct"]) and row["st_bpl_mean_pct"] >= med_poverty:
            flags.append("high poverty")
        if education_ok and flags:
            rows.append(
                {
                    "state": row["state"],
                    "distress_flags": len(flags),
                    "flags": ", ".join(flags),
                    "st_literacy_rate_pct": row["st_literacy_rate_pct"],
                    "ger_latest_secondary_total_clean": row["ger_latest_secondary_total_clean"],
                    "employment_wpr_person_per_1000": row["employment_wpr_person_per_1000"],
                    "st_bpl_mean_pct": row["st_bpl_mean_pct"],
                }
            )
    result = pd.DataFrame(rows).sort_values("distress_flags", ascending=False)
    st.dataframe(result, use_container_width=True, hide_index=True)
    if not result.empty:
        fig = px.bar(result, x="distress_flags", y="state", orientation="h", color="distress_flags", title="Mismatch flags by state")
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)


def q10_compare(df: pd.DataFrame) -> None:
    st.subheader("Q10. Is secondary dropout a stronger warning signal than literacy alone?")
    outcomes = [
        "st_bpl_mean_pct",
        "employment_wpr_person_per_1000",
        "employment_pu_person_per_1000",
        "mgnreg_sought_not_received_per_1000",
        "mgnreg_work_100_plus_days_per_1000",
    ]
    rows = []
    for outcome in outcomes:
        for predictor in ["st_literacy_rate_pct", "dropout_secondary_pct"]:
            subset = df[[predictor, outcome]].dropna()
            if len(subset) < 4:
                continue
            r, p = stats.pearsonr(subset[predictor], subset[outcome])
            rows.append(
                {
                    "Outcome": label(outcome),
                    "Predictor": label(predictor),
                    "Absolute r": abs(r),
                    "Pearson r": r,
                    "p-value": p,
                    "n": len(subset),
                }
            )
    comp = pd.DataFrame(rows)
    st.dataframe(comp.round(4), use_container_width=True, hide_index=True)
    fig = px.bar(
        comp,
        x="Absolute r",
        y="Outcome",
        color="Predictor",
        barmode="group",
        orientation="h",
        title="Literacy vs secondary dropout as warning signals",
        height=520,
    )
    fig.update_layout(
        yaxis={"categoryorder": "total ascending", "tickfont": {"size": 15}},
        xaxis={"tickfont": {"size": 14}},
        title={"font": {"size": 24}},
        font={"size": 16},
    )
    st.plotly_chart(fig, use_container_width=True)


def render_overview(high_st: pd.DataFrame, all_states: pd.DataFrame) -> None:
    st.markdown(
        """
        <div class="hero-card">
          <div class="eyebrow">DSM Final Project Dashboard</div>
          <h1>Scheduled Tribe education and livelihood outcomes</h1>
          <p>
            Explore how ST schooling indicators relate to dropout, poverty, labour outcomes,
            MGNREG distress, and spatial concentration across high-ST states in India.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    metric_row(high_st)

    variable_options = [
        "st_literacy_rate_pct",
        "literacy_gap_pct",
        "female_literacy_gap_pct",
        "dropout_primary_pct",
        "dropout_upper_primary_pct",
        "dropout_secondary_pct",
        "ger_classes_i_viii_clean",
        "ger_classes_ix_x_clean",
        "ger_classes_ix_x_girls_clean",
        "ger_classes_ix_x_gpi_clean",
        "ger_latest_secondary_total_clean",
        "ger_latest_secondary_girls_clean",
        "gpi_secondary_clean",
        "gpi_higher_secondary_clean",
        "st_bpl_mean_pct",
        "employment_wpr_person_per_1000",
        "employment_pu_person_per_1000",
        "employment_wpr_female_per_1000",
        "mgnreg_sought_not_received_per_1000",
        "mgnreg_work_100_plus_days_per_1000",
        "mgnreg_average_days_worked",
        "scholarship_total_release_2023_24_lakh_per_100k_st_pop",
        "scholarship_utilization_2023_24_pct",
        "scholarship_cumulative_release_lakh_per_100k_st_pop",
        "st_share_state_population_pct",
        "villages_gt50_per_100k_st_pop",
        "villages_gt75_per_100k_st_pop",
        "villages_gt90_per_100k_st_pop",
        "villages_all_st_per_100k_st_pop",
        "tribe_weighted_literacy_female_pct",
        "low_literacy_district_count",
    ]
    variable_options = [col for col in variable_options if col in high_st.columns]

    left, right = st.columns([1.45, 1])
    with left:
        st.markdown("### Build your own relationship graph")
        x = st.selectbox(
            "Choose the X-axis variable",
            variable_options,
            index=variable_options.index("dropout_secondary_pct"),
            format_func=label,
            help="This is the horizontal axis. Use it for the possible explanatory or comparison variable.",
        )
        y = st.selectbox(
            "Choose the Y-axis variable",
            variable_options,
            index=variable_options.index("mgnreg_sought_not_received_per_1000"),
            format_func=label,
            help="This is the vertical axis. Use it for the outcome or variable you want to compare against X.",
        )
        scatter(high_st, x, y, "Custom relationship explorer", color_col=None, chart_key="overview_custom_scatter")

    with right:
        state = st.selectbox("State profile", sorted(high_st["state"].dropna().unique()))
        state_profile(high_st, state)

    st.subheader("State comparison table")
    sort_col = st.selectbox(
        "Sort state table by",
        [
            "dropout_secondary_pct",
            "mgnreg_sought_not_received_per_1000",
            "st_bpl_mean_pct",
            "st_literacy_rate_pct",
            "literacy_gap_pct",
            "employment_wpr_person_per_1000",
        ],
        format_func=label,
    )
    table_cols = [
        "state",
        "st_literacy_rate_pct",
        "literacy_gap_pct",
        "dropout_secondary_pct",
        "st_bpl_mean_pct",
        "employment_wpr_person_per_1000",
        "employment_pu_person_per_1000",
        "mgnreg_sought_not_received_per_1000",
    ]
    display = high_st[table_cols].sort_values(sort_col, ascending=False)
    st.dataframe(display.rename(columns={c: label(c) for c in table_cols}), use_container_width=True, hide_index=True)

    st.subheader("All-state context")
    scatter(
        all_states,
        "st_share_state_population_pct",
        "st_literacy_rate_pct",
        "All states: ST share and ST literacy",
        color_col="st_share_state_population_pct",
    )


def render_questions(high_st: pd.DataFrame, all_states: pd.DataFrame) -> None:
    tabs = st.tabs([f"Q{i}" for i in range(1, 14)])

    with tabs[0]:
        question_tab(
            "Q1. How do ST literacy and literacy gaps describe long-term educational exclusion?",
            [
                ("st_literacy_rate_pct", "employment_wpr_person_per_1000"),
                ("literacy_gap_pct", "employment_wpr_person_per_1000"),
                ("st_literacy_rate_pct", "st_bpl_mean_pct"),
                ("literacy_gap_pct", "st_bpl_mean_pct"),
            ],
            high_st,
            [
                ("ST literacy and work participation", "st_literacy_rate_pct", "employment_wpr_person_per_1000"),
                ("Literacy gap and ST poverty", "literacy_gap_pct", "st_bpl_mean_pct"),
            ],
            "q1",
        )

    with tabs[1]:
        question_tab(
            "Q2. Do stronger GER and schooling participation correspond to lower dropout and poverty?",
            [
                ("ger_classes_i_viii_clean", "st_bpl_mean_pct"),
                ("ger_classes_ix_x_clean", "dropout_upper_primary_pct"),
                ("ger_classes_ix_x_girls_clean", "dropout_upper_primary_pct"),
                ("ger_classes_ix_x_gpi_clean", "dropout_upper_primary_pct"),
                ("dropout_secondary_pct", "st_bpl_mean_pct"),
            ],
            high_st,
            [
                ("GER I-VIII and ST poverty", "ger_classes_i_viii_clean", "st_bpl_mean_pct"),
                ("Girls' secondary GER IX-X and upper-primary dropout", "ger_classes_ix_x_girls_clean", "dropout_upper_primary_pct"),
                ("Secondary dropout and ST poverty", "dropout_secondary_pct", "st_bpl_mean_pct"),
            ],
            "q2",
        )

    with tabs[2]:
        question_tab(
            "Q3. Are enrolment and GER enough, or do dropout rates tell a different story?",
            [
                ("ger_classes_ix_x_clean", "dropout_upper_primary_pct"),
                ("ger_classes_ix_x_clean", "dropout_secondary_pct"),
                ("ger_latest_secondary_total_clean", "dropout_secondary_pct"),
                ("gpi_secondary_clean", "dropout_secondary_pct"),
                ("ger_classes_ix_x_clean", "employment_wpr_person_per_1000"),
            ],
            high_st,
            [
                ("Secondary GER IX-X and upper-primary dropout", "ger_classes_ix_x_clean", "dropout_upper_primary_pct"),
                ("Latest secondary GER IX-X and secondary dropout", "ger_latest_secondary_total_clean", "dropout_secondary_pct"),
                ("Secondary GER IX-X and work participation", "ger_classes_ix_x_clean", "employment_wpr_person_per_1000"),
            ],
            "q3",
        )

    with tabs[3]:
        question_tab(
            "Q4. Is literacy gap more informative than ST literacy alone?",
            [
                ("st_literacy_rate_pct", "st_bpl_mean_pct"),
                ("literacy_gap_pct", "st_bpl_mean_pct"),
                ("st_literacy_rate_pct", "employment_wpr_person_per_1000"),
                ("literacy_gap_pct", "employment_wpr_person_per_1000"),
                ("literacy_gap_pct", "dropout_secondary_pct"),
            ],
            high_st,
            [
                ("Literacy gap and secondary dropout", "literacy_gap_pct", "dropout_secondary_pct"),
                ("ST literacy and ST poverty", "st_literacy_rate_pct", "st_bpl_mean_pct"),
            ],
            "q4",
        )

    with tabs[4]:
        q5_mismatch(high_st)

    with tabs[5]:
        question_tab(
            "Q6. Does MGNREG unmet demand reflect deeper livelihood distress?",
            [
                ("mgnreg_sought_not_received_per_1000", "st_bpl_mean_pct"),
                ("mgnreg_sought_not_received_per_1000", "dropout_secondary_pct"),
                ("mgnreg_sought_not_received_per_1000", "st_literacy_rate_pct"),
                ("mgnreg_work_100_plus_days_per_1000", "dropout_secondary_pct"),
            ],
            high_st,
            [
                ("MGNREG unmet demand and ST poverty", "mgnreg_sought_not_received_per_1000", "st_bpl_mean_pct"),
                ("MGNREG unmet demand and secondary dropout", "mgnreg_sought_not_received_per_1000", "dropout_secondary_pct"),
                ("MGNREG unmet demand and ST literacy", "mgnreg_sought_not_received_per_1000", "st_literacy_rate_pct"),
            ],
            "q6",
        )
        bar_rank(high_st, "mgnreg_sought_not_received_per_1000", "Highest MGNREG unmet demand")

    with tabs[6]:
        question_tab(
            "Q7. Are scholarship-supported states seeing lower secondary dropout?",
            [
                ("scholarship_total_release_2023_24_lakh_per_100k_st_pop", "dropout_secondary_pct"),
                ("scholarship_utilization_2023_24_pct", "dropout_secondary_pct"),
                ("scholarship_cumulative_release_lakh_per_100k_st_pop", "st_bpl_mean_pct"),
            ],
            high_st,
            [
                ("Scholarship release and secondary dropout", "scholarship_total_release_2023_24_lakh_per_100k_st_pop", "dropout_secondary_pct"),
                ("Scholarship utilization and secondary dropout", "scholarship_utilization_2023_24_pct", "dropout_secondary_pct"),
            ],
            "q7",
        )

    with tabs[7]:
        question_tab(
            "Q8. Do high-schooling states still depend heavily on MGNREG?",
            [
                ("ger_latest_secondary_total_clean", "mgnreg_work_100_plus_days_per_1000"),
                ("ger_latest_secondary_total_clean", "mgnreg_sought_not_received_per_1000"),
                ("dropout_secondary_pct", "mgnreg_sought_not_received_per_1000"),
                ("ger_latest_secondary_total_clean", "mgnreg_average_days_worked"),
            ],
            high_st,
            [
                ("Secondary dropout and MGNREG unmet demand", "dropout_secondary_pct", "mgnreg_sought_not_received_per_1000"),
                ("Latest secondary GER and MGNREG unmet demand", "ger_latest_secondary_total_clean", "mgnreg_sought_not_received_per_1000"),
                ("Latest secondary GER and average MGNREG days", "ger_latest_secondary_total_clean", "mgnreg_average_days_worked"),
            ],
            "q8",
        )

    with tabs[8]:
        question_tab(
            "Q9. Does gender parity in enrolment translate into female literacy and work outcomes?",
            [
                ("gpi_secondary_clean", "tribe_weighted_literacy_female_pct"),
                ("gpi_secondary_clean", "employment_wpr_female_per_1000"),
                ("ger_latest_secondary_girls_clean", "employment_wpr_female_per_1000"),
                ("female_literacy_gap_pct", "employment_wpr_female_per_1000"),
            ],
            high_st,
            [
                ("Secondary GPI and female ST literacy", "gpi_secondary_clean", "tribe_weighted_literacy_female_pct"),
                ("Secondary GPI and female work participation", "gpi_secondary_clean", "employment_wpr_female_per_1000"),
                ("Female literacy gap and female work participation", "female_literacy_gap_pct", "employment_wpr_female_per_1000"),
            ],
            "q9",
        )

    with tabs[9]:
        q10_compare(high_st)

    with tabs[10]:
        question_tab(
            "Q11. Do high-ST-share states systematically perform worse?",
            [
                ("st_share_state_population_pct", "st_literacy_rate_pct"),
                ("st_share_state_population_pct", "dropout_secondary_pct"),
                ("st_share_state_population_pct", "st_bpl_mean_pct"),
                ("st_share_state_population_pct", "employment_wpr_person_per_1000"),
            ],
            all_states,
            [
                ("ST share and ST literacy across all states", "st_share_state_population_pct", "st_literacy_rate_pct"),
                ("ST share and ST poverty across all states", "st_share_state_population_pct", "st_bpl_mean_pct"),
            ],
            "q11",
        )

    with tabs[11]:
        question_tab(
            "Q12. Do high concentrations of ST villages show different outcomes?",
            [
                ("villages_gt50_per_100k_st_pop", "mgnreg_sought_not_received_per_1000"),
                ("villages_gt75_per_100k_st_pop", "mgnreg_sought_not_received_per_1000"),
                ("villages_gt50_per_100k_st_pop", "st_literacy_rate_pct"),
                ("villages_gt50_per_100k_st_pop", "dropout_secondary_pct"),
            ],
            high_st,
            [
                ("ST village concentration and MGNREG unmet demand", "villages_gt50_per_100k_st_pop", "mgnreg_sought_not_received_per_1000"),
                ("Higher ST village concentration and MGNREG unmet demand", "villages_gt75_per_100k_st_pop", "mgnreg_sought_not_received_per_1000"),
                ("ST village concentration and secondary dropout", "villages_gt50_per_100k_st_pop", "dropout_secondary_pct"),
            ],
            "q12",
        )
        bar_rank(high_st, "villages_gt50_per_100k_st_pop", "Highest normalized ST village concentration")

    with tabs[12]:
        st.subheader("Q13. Are gender-specific disadvantages hidden behind state averages?")
        bar_rank(high_st, "female_literacy_gap_pct", "Largest female literacy gaps within ST populations")
        scatter(
            high_st,
            "female_literacy_gap_pct",
            "employment_wpr_female_per_1000",
            "Female literacy gap and female work participation",
            chart_key="q13_female_gap_wpr",
        )
        scatter(
            high_st,
            "low_literacy_district_count",
            "female_literacy_gap_pct",
            "Low female-literacy districts and female literacy gap",
            chart_key="q13_low_lit_gap",
        )


def main() -> None:
    st.set_page_config(
        page_title="ST Education and Livelihood Dashboard",
        page_icon="DSM",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@500;600;700;800;900&display=swap');
        header[data-testid="stHeader"],
        div[data-testid="stToolbar"],
        div[data-testid="stDecoration"],
        div[data-testid="stStatusWidget"] {
            display: none !important;
            visibility: hidden !important;
            height: 0 !important;
        }
        .stApp {
            margin-top: 0 !important;
        }
        html, body, [class*="css"] {
            font-size: 18px !important;
            font-family: Inter, Segoe UI, sans-serif !important;
        }
        .stApp {
            background:
                linear-gradient(180deg, rgba(248,245,236,0.96), rgba(240,235,224,0.98)),
                radial-gradient(circle at top left, rgba(47,93,124,0.14), transparent 36%);
            color: #202426;
        }
        .block-container {
            max-width: 1500px;
            padding-top: 1.2rem;
            padding-left: 3rem;
            padding-right: 3rem;
            padding-bottom: 4rem;
        }
        section[data-testid="stSidebar"] {
            background: #fffdf8;
            border-right: 1px solid #ddd4c3;
            box-shadow: 8px 0 24px rgba(49, 43, 32, 0.06);
        }
        section[data-testid="stSidebar"] * {
            color: #202426 !important;
            font-size: 17px !important;
        }
        section[data-testid="stSidebar"] h1,
        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3 {
            color: #2f5d7c !important;
            font-weight: 900 !important;
        }
        .hero-card {
            padding: 30px 34px;
            margin-bottom: 22px;
            border: 1px solid #ddd4c3;
            border-radius: 10px;
            background:
                linear-gradient(135deg, rgba(255,253,248,0.97), rgba(239,232,218,0.94)),
                radial-gradient(circle at 92% 8%, rgba(138,79,40,0.14), transparent 30%);
            box-shadow: 0 18px 45px rgba(49, 43, 32, 0.10);
        }
        div[data-testid="column"] {
            padding: 0.15rem 0.25rem;
        }
        .hero-card .eyebrow {
            margin-bottom: 8px;
            color: #8a4f28;
            font-size: 0.82rem !important;
            font-weight: 900;
            letter-spacing: 0.09em;
            text-transform: uppercase;
        }
        .hero-card h1 {
            margin: 0 0 10px;
            max-width: 980px;
        }
        .hero-card p {
            max-width: 1000px;
            margin: 0;
            color: #4f5a59;
            font-size: 1.12rem !important;
        }
        h1 {
            font-size: 3rem !important;
            line-height: 1.08 !important;
            letter-spacing: 0 !important;
            color: #202426 !important;
        }
        h2 {
            font-size: 2.25rem !important;
            letter-spacing: 0 !important;
            color: #202426 !important;
        }
        h3 {
            font-size: 1.65rem !important;
            letter-spacing: 0 !important;
            color: #202426 !important;
        }
        p, li, label, div, span {
            font-size: 1.04rem;
        }
        div[data-testid="stMetric"] {
            background: #fffdf8;
            border: 1px solid #d8d0c0;
            padding: 22px;
            border-radius: 10px;
            min-height: 128px;
            box-shadow: 0 12px 28px rgba(49, 43, 32, 0.08);
        }
        div[data-testid="stMetric"] * {
            color: #202426 !important;
        }
        div[data-testid="stMetricValue"] {
            color: #2f5d7c !important;
            font-size: 2.45rem !important;
            font-weight: 800 !important;
        }
        div[data-testid="stMetricLabel"] {
            font-size: 1.05rem !important;
            font-weight: 700 !important;
        }
        div[data-testid="stVerticalBlock"] > div:has(> div[data-testid="stPlotlyChart"]),
        div[data-testid="stDataFrame"],
        div[data-testid="stSelectbox"] > div {
            border-radius: 10px;
        }
        div[data-testid="stPlotlyChart"] {
            padding: 12px;
            border: 1px solid #ddd4c3;
            border-radius: 10px;
            background: #fffdf8;
            box-shadow: 0 12px 30px rgba(49, 43, 32, 0.08);
        }
        .stSelectbox [data-baseweb="select"],
        .stSelectbox div[data-baseweb="select"] > div,
        .stSelectbox div[data-baseweb="select"] span,
        .stSelectbox div[data-baseweb="select"] input,
        .stTextInput input {
            border-radius: 8px !important;
            background-color: #fffdf8 !important;
            border-color: #d8d0c0 !important;
            color: #202426 !important;
            -webkit-text-fill-color: #202426 !important;
            box-shadow: 0 8px 18px rgba(49, 43, 32, 0.05);
        }
        .stSelectbox svg {
            fill: #202426 !important;
        }
        div[data-baseweb="popover"],
        div[data-baseweb="menu"],
        ul[role="listbox"] {
            background: #fffdf8 !important;
            color: #202426 !important;
        }
        li[role="option"],
        li[role="option"] * {
            color: #202426 !important;
            background: #fffdf8 !important;
        }
        li[role="option"]:hover,
        li[role="option"][aria-selected="true"] {
            background: #e9dfcf !important;
        }
        .stSelectbox [data-baseweb="select"] * {
            color: #202426 !important;
        }
        div[data-testid="stSelectbox"] label,
        div[data-testid="stDataFrame"] {
            color: #202426 !important;
            font-size: 1.05rem !important;
            font-weight: 800 !important;
        }
        div[data-testid="stDataFrame"] {
            border: 1px solid #ddd4c3;
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0 10px 24px rgba(49, 43, 32, 0.06);
        }
        .profile-card {
            padding: 18px;
            border: 1px solid #ddd4c3;
            border-radius: 10px;
            background: #fffdf8;
            box-shadow: 0 12px 28px rgba(49, 43, 32, 0.08);
        }
        .profile-card h3 {
            margin: 0 0 12px !important;
            color: #2f5d7c !important;
        }
        .profile-row {
            display: grid;
            grid-template-columns: minmax(0, 1fr) auto;
            gap: 4px 14px;
            padding: 11px 0;
            border-bottom: 1px solid #e6dece;
        }
        .profile-row:last-child {
            border-bottom: 0;
        }
        .profile-label {
            color: #202426;
            font-weight: 800;
            font-size: 0.98rem !important;
        }
        .profile-value {
            color: #202426;
            font-weight: 900;
            font-size: 1.08rem !important;
        }
        .profile-median {
            grid-column: 1 / -1;
            color: #68716f;
            font-size: 0.9rem !important;
            font-weight: 700;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }
        .stTabs [data-baseweb="tab"] {
            height: 42px;
            padding: 0 14px;
            border: 1px solid #d8d0c0;
            border-radius: 999px;
            background: #fffdf8;
            color: #2f5d7c;
            font-weight: 800;
        }
        .stTabs [aria-selected="true"] {
            background: #2f5d7c !important;
            color: #ffffff !important;
        }
        div[data-testid="stCaptionContainer"] {
            padding: 8px 2px 16px;
            color: #5c6462;
            font-size: 1rem !important;
            font-weight: 700;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    high_st, all_states = load_data()

    st.sidebar.title("Dashboard")
    page = st.sidebar.radio("View", ["Overview", "Question Visuals"], index=0)
    st.sidebar.caption("Separate Streamlit app. The GitHub Pages article is unchanged.")

    if page == "Overview":
        render_overview(high_st, all_states)
    else:
        render_questions(high_st, all_states)


if __name__ == "__main__":
    main()
