from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy import stats
from shapely.geometry import mapping, shape


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "outputs" / "analysis"

HIGH_ST_PATH = ANALYSIS / "state_analysis_dataset_high_st_states.csv"
ALL_STATES_PATH = ANALYSIS / "state_analysis_dataset_all_states.csv"
GEOJSON_PATH = ROOT / "dashboard_app" / "india_state.geojson"

THEME = {
    "blue": "#66b8e8",
    "brown": "#d79b6b",
    "green": "#69c2aa",
    "red": "#f27d7d",
    "gold": "#d8b45c",
    "gray": "#9aa6ad",
    "paper": "#0d1117",
    "panel": "#171c24",
    "plot": "#11161d",
    "ink": "#f4f7fb",
    "muted": "#aeb8c2",
    "line": "#303947",
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

MAP_FEATURE_KEY = "map_name"
MAP_SIMPLIFY_TOLERANCE = 0.005
MAP_STATE_NAME_FIXES = {
    "Andaman and Nicobar": "Andaman and Nicobar Islands",
    "Andaman and Nicobar Islands": "Andaman and Nicobar Islands",
    "Orissa": "Odisha",
    "Odisha": "Odisha",
    "Uttaranchal": "Uttarakhand",
    "Uttarakhand": "Uttarakhand",
}
MAP_STATE_EXPANSIONS = {
    "Dadra and Nagar Haveli and Daman and Diu": ["Dadra and Nagar Haveli", "Daman and Diu"],
}
RISK_HIGH_COLUMNS = {
    "literacy_gap_pct",
    "female_literacy_gap_pct",
    "dropout_primary_pct",
    "dropout_upper_primary_pct",
    "dropout_secondary_pct",
    "st_bpl_mean_pct",
    "employment_pu_person_per_1000",
    "mgnreg_sought_not_received_per_1000",
    "low_literacy_district_count",
}
MAP_GOOD_SCALE = ["#1d2a34", "#2f6f8f", "#58b6a5", "#d8b45c"]
MAP_RISK_SCALE = ["#211f2d", "#5e4265", "#bd6c4f", "#f27d7d"]
CHART_COLOR_SCALE = ["#66b8e8", "#69c2aa", "#d8b45c", "#d79b6b", "#f27d7d"]


@st.cache_data
def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    high_st = pd.read_csv(HIGH_ST_PATH)
    all_states = pd.read_csv(ALL_STATES_PATH)
    return add_derived_columns(high_st), add_derived_columns(all_states)


@st.cache_data
def load_geojson() -> dict:
    with GEOJSON_PATH.open("r", encoding="utf-8") as handle:
        raw_geojson = json.load(handle)

    features = []
    for feature in raw_geojson["features"]:
        props = feature.get("properties", {})
        raw_name = props.get("NAME_1")
        geometry = feature.get("geometry")
        simplified_geometry = geometry
        if geometry:
            simplified_geometry = mapping(shape(geometry).simplify(MAP_SIMPLIFY_TOLERANCE, preserve_topology=True))
        features.append(
            {
                "type": feature.get("type", "Feature"),
                "properties": {MAP_FEATURE_KEY: normalize_map_state(raw_name)},
                "geometry": simplified_geometry,
            }
        )

    return {"type": "FeatureCollection", "features": features}


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


def normalize_map_state(value: str | None) -> str | None:
    if pd.isna(value):
        return value
    return MAP_STATE_NAME_FIXES.get(str(value).strip(), str(value).strip())


def expand_map_states(value: str | None) -> list[str]:
    normalized = normalize_map_state(value)
    if normalized is None:
        return []
    return MAP_STATE_EXPANSIONS.get(normalized, [normalized])


def corr_text(df: pd.DataFrame, x: str, y: str) -> str:
    subset = df[[x, y]].dropna()
    if len(subset) < 4 or subset[x].nunique() <= 1 or subset[y].nunique() <= 1:
        return f"Pearson r not shown; n = {len(subset)}"
    r, p = stats.pearsonr(subset[x], subset[y])
    p_text = "p < 0.001" if p < 0.001 else f"p = {p:.4f}"
    return f"Pearson r = {r:.4f}; {p_text}; n = {len(subset)}"


def map_metric_options(df: pd.DataFrame) -> list[str]:
    options = [
        "st_literacy_rate_pct",
        "literacy_gap_pct",
        "dropout_upper_primary_pct",
        "dropout_secondary_pct",
        "ger_classes_i_viii_clean",
        "ger_classes_ix_x_clean",
        "ger_latest_secondary_total_clean",
        "st_bpl_mean_pct",
        "employment_wpr_person_per_1000",
        "employment_pu_person_per_1000",
        "mgnreg_sought_not_received_per_1000",
        "mgnreg_average_days_worked",
        "scholarship_utilization_2023_24_pct",
        "st_share_state_population_pct",
        "villages_gt50_per_100k_st_pop",
        "low_literacy_district_count",
    ]
    return [col for col in options if col in df.columns]


def choropleth(df: pd.DataFrame, metric: str, title: str, chart_key: str, height: int = 680) -> None:
    geojson = load_geojson()
    plot_df = df[["state", metric]].dropna(subset=[metric]).copy()
    plot_df["map_state"] = plot_df["state"].map(expand_map_states)
    plot_df = plot_df.explode("map_state").dropna(subset=["map_state"])

    if plot_df.empty:
        st.warning(f"No map data available for {label(metric)}.")
        return

    color_scale = MAP_RISK_SCALE if metric in RISK_HIGH_COLUMNS else MAP_GOOD_SCALE
    fig = px.choropleth(
        plot_df,
        geojson=geojson,
        featureidkey=f"properties.{MAP_FEATURE_KEY}",
        locations="map_state",
        color=metric,
        hover_name="state",
        hover_data={metric: ":.2f"},
        color_continuous_scale=color_scale,
        labels={metric: label(metric), "map_state": "State"},
        title=title,
        height=height,
    )
    fig.update_traces(marker_line_color=THEME["panel"], marker_line_width=0.85)
    fig.update_geos(fitbounds="locations", visible=False)
    fig.update_layout(
        margin={"l": 8, "r": 8, "t": 56, "b": 8},
        paper_bgcolor=THEME["panel"],
        plot_bgcolor=THEME["panel"],
        font={"size": 16, "color": THEME["ink"], "family": "Inter, Segoe UI, sans-serif"},
        title={"font": {"size": 23, "color": THEME["ink"], "family": "Inter, Segoe UI, sans-serif"}},
        coloraxis={"colorbar": {"tickfont": {"size": 13, "color": THEME["ink"]}, "title": {"font": {"size": 14, "color": THEME["ink"]}}}},
        hoverlabel={"bgcolor": THEME["plot"], "font_size": 14, "font_color": THEME["ink"]},
    )
    st.plotly_chart(fig, use_container_width=True, key=chart_key)


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
        color_continuous_scale=CHART_COLOR_SCALE,
        labels={x: label(x), y: label(y), **({color_col: label(color_col)} if color_col else {})},
        title=title,
        height=height,
    )
    if color_col is None:
        fig.update_traces(marker=dict(color=THEME["blue"], size=11, line=dict(color=THEME["paper"], width=1.2)))
    if trendline and len(plot_df) >= 4 and plot_df[x].nunique() > 1:
        slope, intercept, *_ = stats.linregress(plot_df[x], plot_df[y])
        x_min = float(plot_df[x].min())
        x_max = float(plot_df[x].max())
        fig.add_trace(
            go.Scatter(
                x=[x_min, x_max],
                y=[slope * x_min + intercept, slope * x_max + intercept],
                mode="lines",
                line={"color": THEME["ink"], "width": 2.3},
                name="Linear trend",
                hoverinfo="skip",
            )
        )
    fig.update_traces(marker={"size": 12, "line": {"width": 1.4, "color": THEME["paper"]}, "opacity": 0.94})
    fig.update_layout(
        margin={"l": 30, "r": 25, "t": 72, "b": 58},
        paper_bgcolor=THEME["panel"],
        plot_bgcolor=THEME["plot"],
        font={"size": 16, "color": THEME["ink"], "family": "Inter, Segoe UI, sans-serif"},
        title={"font": {"size": 24, "color": THEME["ink"], "family": "Inter, Segoe UI, sans-serif"}},
        legend={"font": {"size": 15, "color": THEME["ink"]}},
        xaxis={"title": {"font": {"size": 17, "color": THEME["ink"]}}, "tickfont": {"size": 14, "color": THEME["ink"]}},
        yaxis={"title": {"font": {"size": 17, "color": THEME["ink"]}}, "tickfont": {"size": 14, "color": THEME["ink"]}},
        coloraxis={"colorbar": {"tickfont": {"size": 14, "color": THEME["ink"]}, "title": {"font": {"size": 15, "color": THEME["ink"]}}}},
        hoverlabel={"bgcolor": THEME["plot"], "font_size": 14, "font_color": THEME["ink"]},
    )
    fig.update_xaxes(showgrid=True, gridcolor=THEME["line"], zeroline=False, showline=True, linecolor="#596473")
    fig.update_yaxes(showgrid=True, gridcolor=THEME["line"], zeroline=False, showline=True, linecolor="#596473")
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
        color_continuous_scale=CHART_COLOR_SCALE,
    )
    fig.update_layout(
        yaxis={"categoryorder": "total ascending", "tickfont": {"size": 15, "color": THEME["ink"]}, "title": {"font": {"size": 17, "color": THEME["ink"]}}},
        xaxis={"tickfont": {"size": 14, "color": THEME["ink"]}, "title": {"font": {"size": 17, "color": THEME["ink"]}}},
        title={"font": {"size": 24, "color": THEME["ink"]}},
        font={"size": 16, "color": THEME["ink"], "family": "Inter, Segoe UI, sans-serif"},
        paper_bgcolor=THEME["panel"],
        plot_bgcolor=THEME["plot"],
        margin={"l": 25, "r": 20, "t": 72, "b": 45},
        hoverlabel={"bgcolor": THEME["plot"], "font_color": THEME["ink"]},
    )
    fig.update_xaxes(showgrid=True, gridcolor=THEME["line"], zeroline=False)
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


def section_header(title: str, eyebrow: str | None = None) -> None:
    eyebrow_html = f'<div class="section-eyebrow">{eyebrow}</div>' if eyebrow else ""
    st.markdown(
        f"""
        <div class="section-heading">
            {eyebrow_html}
            <h2>{title}</h2>
        </div>
        """,
        unsafe_allow_html=True,
    )


def metric_row(df: pd.DataFrame) -> None:
    cards = [
        ("High-ST states", f"{len(df)}", "state cohort"),
        ("Mean ST literacy", f"{df['st_literacy_rate_pct'].mean():.1f}%", "education baseline"),
        ("Mean secondary dropout", f"{df['dropout_secondary_pct'].mean():.1f}%", "retention pressure"),
        ("Mean MGNREG unmet demand", f"{df['mgnreg_sought_not_received_per_1000'].mean():.1f}", "livelihood stress"),
    ]
    card_html = "".join(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">{name}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-note">{note}</div>
        </div>
        """
        for name, value, note in cards
    )
    st.markdown(f'<div class="kpi-grid">{card_html}</div>', unsafe_allow_html=True)


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
        fig = px.bar(
            result,
            x="distress_flags",
            y="state",
            orientation="h",
            color="distress_flags",
            color_continuous_scale=CHART_COLOR_SCALE,
            title="Mismatch flags by state",
        )
        fig.update_layout(
            yaxis={"categoryorder": "total ascending", "tickfont": {"color": THEME["ink"]}},
            xaxis={"tickfont": {"color": THEME["ink"]}},
            paper_bgcolor=THEME["panel"],
            plot_bgcolor=THEME["plot"],
            font={"size": 16, "color": THEME["ink"], "family": "Inter, Segoe UI, sans-serif"},
            title={"font": {"size": 24, "color": THEME["ink"]}},
            hoverlabel={"bgcolor": THEME["plot"], "font_color": THEME["ink"]},
        )
        fig.update_xaxes(showgrid=True, gridcolor=THEME["line"], zeroline=False)
        fig.update_yaxes(showgrid=False)
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
        color_discrete_sequence=[THEME["blue"], THEME["gold"]],
    )
    fig.update_layout(
        yaxis={"categoryorder": "total ascending", "tickfont": {"size": 15, "color": THEME["ink"]}},
        xaxis={"tickfont": {"size": 14, "color": THEME["ink"]}},
        title={"font": {"size": 24, "color": THEME["ink"]}},
        font={"size": 16, "color": THEME["ink"], "family": "Inter, Segoe UI, sans-serif"},
        legend={"font": {"color": THEME["ink"]}},
        paper_bgcolor=THEME["panel"],
        plot_bgcolor=THEME["plot"],
        hoverlabel={"bgcolor": THEME["plot"], "font_color": THEME["ink"]},
    )
    fig.update_xaxes(showgrid=True, gridcolor=THEME["line"], zeroline=False)
    fig.update_yaxes(showgrid=False)
    st.plotly_chart(fig, use_container_width=True)


def render_overview(high_st: pd.DataFrame, all_states: pd.DataFrame) -> None:
    st.markdown(
        """
        <div class="hero-card">
          <div class="eyebrow">DSM Final Project</div>
          <h1>Scheduled Tribe education and livelihood outcomes</h1>
          <p>
            Explore how ST schooling indicators relate to dropout, poverty, labour outcomes,
            MGNREG distress, and spatial concentration across high-ST states in India.
          </p>
          <div class="hero-meta">
            <span>State-level evidence</span>
            <span>Education</span>
            <span>Livelihoods</span>
            <span>Policy prioritization</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    metric_row(high_st)

    map_options = map_metric_options(all_states)
    if map_options:
        section_header("India Map Explorer", "Geography")
        overview_metric = st.selectbox(
            "Choose the indicator for the overall India map",
            map_options,
            index=map_options.index("st_literacy_rate_pct") if "st_literacy_rate_pct" in map_options else 0,
            format_func=label,
            help="This full-width map gives a single national view of one key indicator across all states and union territories in the analysis table.",
        )
        choropleth(
            all_states,
            overview_metric,
            f"Overall India map: {label(overview_metric)}",
            chart_key="overview_india_map",
        )

        section_header("Compare Two India Maps", "Spatial comparison")
        default_left = "dropout_secondary_pct" if "dropout_secondary_pct" in map_options else map_options[0]
        default_right = "st_bpl_mean_pct" if "st_bpl_mean_pct" in map_options else map_options[min(1, len(map_options) - 1)]
        compare_left, compare_right = st.columns(2, gap="small")
        with compare_left:
            left_metric = st.selectbox(
                "Left map indicator",
                map_options,
                index=map_options.index(default_left),
                format_func=label,
                help="Pick the first indicator to compare geographically.",
            )
            choropleth(
                all_states,
                left_metric,
                f"India map: {label(left_metric)}",
                chart_key="overview_india_map_left",
                height=760,
            )
        with compare_right:
            right_metric = st.selectbox(
                "Right map indicator",
                map_options,
                index=map_options.index(default_right),
                format_func=label,
                help="Pick the second indicator to compare side by side with the left map.",
            )
            choropleth(
                all_states,
                right_metric,
                f"India map: {label(right_metric)}",
                chart_key="overview_india_map_right",
                height=760,
            )

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
        section_header("Build Your Own Relationship Graph", "Interactive analysis")
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
        section_header("State Profile", "Focused view")
        state = st.selectbox("State profile", sorted(high_st["state"].dropna().unique()))
        state_profile(high_st, state)

    section_header("State Comparison Table", "Ranked indicators")
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

    section_header("All-State Context", "Benchmark")
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
        initial_sidebar_state="collapsed",
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
            background: linear-gradient(180deg, #0d1117 0%, #10151d 54%, #0d1117 100%);
            color: #f4f7fb;
        }
        .block-container {
            max-width: 1900px;
            padding-top: 1.2rem;
            padding-left: 1.4rem;
            padding-right: 1.4rem;
            padding-bottom: 4rem;
        }
        section[data-testid="stSidebar"] {
            background: #11161d;
            border-right: 1px solid #303947;
            box-shadow: 8px 0 24px rgba(0, 0, 0, 0.25);
        }
        section[data-testid="stSidebar"] * {
            color: #f4f7fb !important;
            font-size: 17px !important;
        }
        section[data-testid="stSidebar"] h1,
        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3 {
            color: #66b8e8 !important;
            font-weight: 900 !important;
        }
        .hero-card {
            padding: 28px 32px;
            margin-bottom: 18px;
            border: 1px solid #303947;
            border-radius: 8px;
            background: linear-gradient(135deg, rgba(23,28,36,0.98), rgba(18,24,32,0.96));
            box-shadow: 0 18px 45px rgba(0, 0, 0, 0.30);
        }
        div[data-testid="column"] {
            padding: 0.08rem 0.12rem;
        }
        .hero-card .eyebrow {
            margin-bottom: 8px;
            color: #d8b45c;
            font-size: 0.82rem !important;
            font-weight: 900;
            letter-spacing: 0.09em;
            text-transform: uppercase;
        }
        .hero-card h1 {
            margin: 0 0 10px;
            max-width: 1060px;
        }
        .hero-card p {
            max-width: 1000px;
            margin: 0;
            color: #aeb8c2;
            font-size: 1.12rem !important;
        }
        .hero-meta {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            margin-top: 18px;
        }
        .hero-meta span {
            padding: 6px 10px;
            border: 1px solid #303947;
            border-radius: 999px;
            background: #11161d;
            color: #cbd4dd;
            font-size: 0.86rem !important;
            font-weight: 800;
        }
        .kpi-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 14px;
            margin: 14px 0 26px;
        }
        .kpi-card {
            position: relative;
            min-height: 132px;
            padding: 20px 20px 18px;
            border: 1px solid #303947;
            border-radius: 8px;
            background: #171c24;
            box-shadow: 0 12px 28px rgba(0, 0, 0, 0.24);
            overflow: hidden;
        }
        .kpi-card::before {
            content: "";
            position: absolute;
            inset: 0 0 auto 0;
            height: 3px;
            background: linear-gradient(90deg, #66b8e8, #69c2aa, #d8b45c);
        }
        .kpi-label {
            color: #aeb8c2;
            font-size: 0.9rem !important;
            font-weight: 900;
            text-transform: uppercase;
        }
        .kpi-value {
            margin-top: 14px;
            color: #f4f7fb;
            font-size: 2.35rem !important;
            line-height: 1;
            font-weight: 900;
        }
        .kpi-note {
            margin-top: 12px;
            color: #66b8e8;
            font-size: 0.88rem !important;
            font-weight: 800;
        }
        .section-heading {
            margin: 34px 0 14px;
            padding-top: 4px;
            border-top: 1px solid #303947;
        }
        .section-eyebrow {
            color: #d8b45c;
            font-size: 0.78rem !important;
            font-weight: 900;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            margin-bottom: 6px;
        }
        .section-heading h2 {
            margin: 0 !important;
            color: #f4f7fb !important;
            font-size: 1.85rem !important;
            line-height: 1.2 !important;
        }
        h1 {
            font-size: 3rem !important;
            line-height: 1.08 !important;
            letter-spacing: 0 !important;
            color: #f4f7fb !important;
        }
        h2 {
            font-size: 2.25rem !important;
            letter-spacing: 0 !important;
            color: #f4f7fb !important;
        }
        h3 {
            font-size: 1.65rem !important;
            letter-spacing: 0 !important;
            color: #f4f7fb !important;
        }
        p, li, label, div, span {
            font-size: 1.04rem;
        }
        div[data-testid="stMetric"] {
            background: #171c24;
            border: 1px solid #303947;
            padding: 22px;
            border-radius: 8px;
            min-height: 128px;
            box-shadow: 0 12px 28px rgba(0, 0, 0, 0.28);
        }
        div[data-testid="stMetric"] * {
            color: #f4f7fb !important;
        }
        div[data-testid="stMetricValue"] {
            color: #66b8e8 !important;
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
            border-radius: 8px;
        }
        div[data-testid="stPlotlyChart"] {
            padding: 8px;
            border: 1px solid #303947;
            border-radius: 8px;
            background: #171c24;
            box-shadow: 0 12px 30px rgba(0, 0, 0, 0.30);
        }
        .stSelectbox [data-baseweb="select"],
        .stSelectbox div[data-baseweb="select"] > div,
        .stSelectbox div[data-baseweb="select"] span,
        .stSelectbox div[data-baseweb="select"] input,
        .stTextInput input {
            border-radius: 8px !important;
            background-color: #11161d !important;
            border-color: #303947 !important;
            color: #f4f7fb !important;
            -webkit-text-fill-color: #f4f7fb !important;
            box-shadow: 0 8px 18px rgba(0, 0, 0, 0.18);
        }
        .stSelectbox svg {
            fill: #f4f7fb !important;
        }
        div[data-baseweb="popover"],
        div[data-baseweb="menu"],
        ul[role="listbox"] {
            background: #171c24 !important;
            color: #f4f7fb !important;
        }
        li[role="option"],
        li[role="option"] * {
            color: #f4f7fb !important;
            background: #171c24 !important;
        }
        li[role="option"]:hover,
        li[role="option"][aria-selected="true"] {
            background: #263140 !important;
        }
        .stSelectbox [data-baseweb="select"] * {
            color: #f4f7fb !important;
        }
        div[data-testid="stSelectbox"] label,
        div[data-testid="stDataFrame"] {
            color: #f4f7fb !important;
            font-size: 1.05rem !important;
            font-weight: 800 !important;
        }
        div[data-testid="stDataFrame"] {
            border: 1px solid #303947;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 10px 24px rgba(0, 0, 0, 0.24);
        }
        .profile-card {
            padding: 18px;
            border: 1px solid #303947;
            border-radius: 8px;
            background: #171c24;
            box-shadow: 0 12px 28px rgba(0, 0, 0, 0.28);
        }
        .profile-card h3 {
            margin: 0 0 12px !important;
            color: #66b8e8 !important;
        }
        .profile-row {
            display: grid;
            grid-template-columns: minmax(0, 1fr) auto;
            gap: 4px 14px;
            padding: 11px 0;
            border-bottom: 1px solid #303947;
        }
        .profile-row:last-child {
            border-bottom: 0;
        }
        .profile-label {
            color: #f4f7fb;
            font-weight: 800;
            font-size: 0.98rem !important;
        }
        .profile-value {
            color: #f4f7fb;
            font-weight: 900;
            font-size: 1.08rem !important;
        }
        .profile-median {
            grid-column: 1 / -1;
            color: #aeb8c2;
            font-size: 0.9rem !important;
            font-weight: 700;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }
        .stTabs [data-baseweb="tab"] {
            height: 42px;
            padding: 0 14px;
            border: 1px solid #303947;
            border-radius: 999px;
            background: #171c24;
            color: #66b8e8;
            font-weight: 800;
        }
        .stTabs [aria-selected="true"] {
            background: #315f7d !important;
            color: #ffffff !important;
        }
        div[data-testid="stRadio"] {
            margin-bottom: 18px;
            display: flex;
            justify-content: center;
        }
        div[data-testid="stRadio"] div[role="radiogroup"] {
            gap: 10px;
            justify-content: center;
            flex-wrap: wrap;
            width: fit-content;
            margin-left: auto;
            margin-right: auto;
        }
        div[data-testid="stRadio"] label[data-baseweb="radio"] {
            min-height: 46px;
            margin: 0 8px 0 0;
            padding: 0 18px;
            min-width: 250px;
            justify-content: center;
            border: 1px solid #303947;
            border-radius: 999px;
            background: #171c24;
            box-shadow: 0 8px 18px rgba(0, 0, 0, 0.22);
        }
        div[data-testid="stRadio"] label[data-baseweb="radio"] > div:first-child {
            display: none;
        }
        div[data-testid="stRadio"] label[data-baseweb="radio"]:has(input:checked) {
            background: #315f7d !important;
            border-color: #66b8e8 !important;
        }
        div[data-testid="stRadio"] label[data-baseweb="radio"] p {
            color: #66b8e8 !important;
            font-weight: 900 !important;
            text-align: center !important;
            width: 100%;
        }
        div[data-testid="stRadio"] label[data-baseweb="radio"]:has(input:checked) p {
            color: #ffffff !important;
        }
        div[data-testid="stCaptionContainer"] {
            padding: 8px 2px 16px;
            color: #aeb8c2;
            font-size: 1rem !important;
            font-weight: 700;
        }
        @media (max-width: 1200px) {
            .kpi-grid {
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }
            .block-container {
                padding-left: 1rem;
                padding-right: 1rem;
            }
        }
        @media (max-width: 720px) {
            .kpi-grid {
                grid-template-columns: 1fr;
            }
            .hero-card {
                padding: 22px;
            }
            h1 {
                font-size: 2.25rem !important;
            }
            div[data-testid="stRadio"] label[data-baseweb="radio"] {
                min-width: 190px;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    high_st, all_states = load_data()

    page = st.radio(
        "Dashboard view",
        ["Overview", "Question Visuals"],
        horizontal=True,
        label_visibility="collapsed",
        key="top_dashboard_view",
    )

    if page == "Overview":
        render_overview(high_st, all_states)
    else:
        render_questions(high_st, all_states)


if __name__ == "__main__":
    main()
