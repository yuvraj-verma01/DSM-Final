from __future__ import annotations

import math
import os
import sqlite3
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "2")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs"
ANALYSIS_DIR = OUTPUT_DIR / "analysis"
CLEAN_DIR = OUTPUT_DIR / "cleaned"
EDA_DIR = OUTPUT_DIR / "eda"
TABLE_DIR = EDA_DIR / "tables"
FIGURE_DIR = EDA_DIR / "figures"
DB_PATH = OUTPUT_DIR / "st_education_project.sqlite"


HIGH_ST_PATH = ANALYSIS_DIR / "state_analysis_dataset_high_st_states.csv"
ALL_STATES_PATH = ANALYSIS_DIR / "state_analysis_dataset_all_states.csv"
INVENTORY_PATH = ANALYSIS_DIR / "data_inventory.csv"


CORE_POLICY_COLUMNS = [
    "st_share_state_population_pct",
    "st_literacy_rate_pct",
    "literacy_gap_pct",
    "tribe_weighted_literacy_female_pct",
    "dropout_primary_pct",
    "dropout_upper_primary_pct",
    "dropout_secondary_pct",
    "ger_classes_i_viii",
    "ger_classes_ix_x",
    "ger_classes_xi_xii",
    "ger_latest_secondary_total",
    "ger_latest_secondary_girls",
    "gpi_secondary",
    "scholarship_total_release_2023_24_lakh_per_100k_st_pop",
    "scholarship_utilization_2023_24_pct",
    "st_bpl_rural_pct",
    "st_bpl_urban_pct",
    "st_bpl_mean_pct",
    "employment_lfpr_person_per_1000",
    "employment_wpr_person_per_1000",
    "employment_pu_person_per_1000",
    "mgnreg_average_days_worked",
    "mgnreg_sought_not_received_per_1000",
    "mgnreg_work_100_plus_days_per_1000",
    "mgnreg_100_plus_share_of_work_received_pct",
    "low_literacy_district_count",
    "tribal_villages_gt_50_count",
    "education_disadvantage_score",
    "economic_vulnerability_score",
    "overall_priority_score",
]


RELATIONSHIP_PAIRS = [
    ("st_literacy_rate_pct", "st_bpl_mean_pct", "How does ST literacy relate to ST poverty?"),
    ("st_literacy_rate_pct", "employment_wpr_person_per_1000", "How does ST literacy relate to work participation?"),
    ("st_literacy_rate_pct", "employment_pu_person_per_1000", "How does ST literacy relate to unemployment?"),
    ("dropout_secondary_pct", "st_bpl_mean_pct", "How does secondary dropout relate to ST poverty?"),
    ("dropout_secondary_pct", "employment_wpr_person_per_1000", "How does secondary dropout relate to work participation?"),
    ("ger_latest_secondary_total", "dropout_secondary_pct", "How does latest secondary ST GER relate to secondary dropout?"),
    ("gpi_secondary", "employment_wpr_female_per_1000", "Does ST secondary gender parity in enrolment relate to female work participation?"),
    (
        "scholarship_total_release_2023_24_lakh_per_100k_st_pop",
        "dropout_secondary_pct",
        "Does ST scholarship release per ST population relate to secondary dropout?",
    ),
    (
        "ger_latest_secondary_total",
        "mgnreg_work_100_plus_days_per_1000",
        "Do higher-secondary-GER states still show MGNREG 100-plus-days dependence?",
    ),
    ("tribe_weighted_literacy_female_pct", "employment_wpr_female_per_1000", "How does female literacy relate to female work participation?"),
    ("literacy_gap_pct", "st_bpl_mean_pct", "How does literacy inequality relate to ST poverty?"),
    ("mgnreg_sought_not_received_per_1000", "st_bpl_mean_pct", "How does unmet MGNREG demand relate to ST poverty?"),
]


def ensure_dirs() -> None:
    for path in [EDA_DIR, TABLE_DIR, FIGURE_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def read_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    high = pd.read_csv(HIGH_ST_PATH)
    all_states = pd.read_csv(ALL_STATES_PATH)
    inventory = pd.read_csv(INVENTORY_PATH)
    return high, all_states, inventory


def write_csv(df: pd.DataFrame, name: str) -> Path:
    path = TABLE_DIR / name
    df.to_csv(path, index=False)
    return path


def fmt(value: float | int | str | None, digits: int = 2) -> str:
    if value is None or pd.isna(value):
        return "NA"
    if isinstance(value, str):
        return value
    return f"{value:.{digits}f}"


def get_existing(df: pd.DataFrame, columns: list[str]) -> list[str]:
    return [col for col in columns if col in df.columns]


def data_quality_tables(high: pd.DataFrame, all_states: pd.DataFrame, inventory: pd.DataFrame) -> dict[str, pd.DataFrame]:
    coverage_rows = []
    for name, df in {"all_states_master": all_states, "high_st_master": high}.items():
        for col in CORE_POLICY_COLUMNS:
            if col not in df.columns:
                continue
            coverage_rows.append(
                {
                    "table": name,
                    "column": col,
                    "non_missing": int(df[col].notna().sum()),
                    "missing": int(df[col].isna().sum()),
                    "missing_pct": round(df[col].isna().mean() * 100, 2),
                    "min": df[col].min(skipna=True),
                    "median": df[col].median(skipna=True),
                    "max": df[col].max(skipna=True),
                }
            )
    coverage = pd.DataFrame(coverage_rows)
    write_csv(coverage, "data_quality_core_variable_coverage.csv")

    missing = (
        high.isna()
        .sum()
        .reset_index()
        .rename(columns={"index": "column", 0: "missing_count"})
        .assign(missing_pct=lambda x: round(x["missing_count"] / len(high) * 100, 2))
        .sort_values(["missing_pct", "column"], ascending=[False, True])
    )
    write_csv(missing, "high_st_master_missingness.csv")

    dataset_quality = inventory[
        [
            "short_name",
            "level",
            "raw_rows",
            "cleaned_rows",
            "state_count",
            "years",
            "key_variables",
            "notes",
        ]
    ].copy()
    write_csv(dataset_quality, "dataset_inventory_for_report.csv")

    return {
        "coverage": coverage,
        "missing": missing,
        "dataset_quality": dataset_quality,
    }


def build_rankings(high: pd.DataFrame) -> dict[str, pd.DataFrame]:
    specs = {
        "lowest_st_literacy.csv": ("st_literacy_rate_pct", True, ["state", "st_literacy_rate_pct", "literacy_gap_pct"]),
        "highest_literacy_gap.csv": ("literacy_gap_pct", False, ["state", "literacy_gap_pct", "st_literacy_rate_pct"]),
        "highest_secondary_dropout.csv": (
            "dropout_secondary_pct",
            False,
            ["state", "dropout_primary_pct", "dropout_upper_primary_pct", "dropout_secondary_pct"],
        ),
        "highest_st_poverty.csv": ("st_bpl_mean_pct", False, ["state", "st_bpl_rural_pct", "st_bpl_urban_pct", "st_bpl_mean_pct"]),
        "lowest_worker_population_ratio.csv": (
            "employment_wpr_person_per_1000",
            True,
            ["state", "employment_lfpr_person_per_1000", "employment_wpr_person_per_1000", "employment_pu_person_per_1000"],
        ),
        "highest_unemployment.csv": (
            "employment_pu_person_per_1000",
            False,
            ["state", "employment_pu_person_per_1000", "employment_lfpr_person_per_1000", "employment_wpr_person_per_1000"],
        ),
        "highest_mgnreg_unmet_work.csv": (
            "mgnreg_sought_not_received_per_1000",
            False,
            ["state", "mgnreg_sought_not_received_per_1000", "mgnreg_average_days_worked"],
        ),
        "lowest_female_literacy.csv": (
            "tribe_weighted_literacy_female_pct",
            True,
            ["state", "tribe_weighted_literacy_female_pct", "tribe_weighted_literacy_male_pct", "tribe_weighted_wpr_pct"],
        ),
        "highest_priority_scores.csv": (
            "overall_priority_score",
            False,
            [
                "state",
                "education_disadvantage_score",
                "economic_vulnerability_score",
                "overall_priority_score",
                "policy_priority_category",
            ],
        ),
        "most_low_literacy_districts.csv": (
            "low_literacy_district_count",
            False,
            ["state", "low_literacy_district_count", "low_literacy_female_min_pct", "low_literacy_female_mean_pct"],
        ),
    }
    rankings = {}
    for filename, (sort_col, ascending, cols) in specs.items():
        if sort_col not in high.columns:
            continue
        ranking = high[get_existing(high, cols)].dropna(subset=[sort_col]).sort_values(sort_col, ascending=ascending)
        ranking = ranking.head(19).reset_index(drop=True)
        write_csv(ranking, filename)
        rankings[filename] = ranking
    return rankings


def relationship_tests(df: pd.DataFrame, label: str) -> pd.DataFrame:
    rows = []
    for x_col, y_col, question in RELATIONSHIP_PAIRS:
        if x_col not in df.columns or y_col not in df.columns:
            continue
        subset = df[[x_col, y_col, "state"]].dropna()
        if len(subset) < 4:
            rows.append(
                {
                    "sample": label,
                    "question": question,
                    "x": x_col,
                    "y": y_col,
                    "n": len(subset),
                    "pearson_r": np.nan,
                    "pearson_p": np.nan,
                    "spearman_r": np.nan,
                    "spearman_p": np.nan,
                    "interpretation": "Insufficient non-missing observations.",
                }
            )
            continue
        pearson = stats.pearsonr(subset[x_col], subset[y_col])
        spearman = stats.spearmanr(subset[x_col], subset[y_col])
        strength = "weak"
        if abs(pearson.statistic) >= 0.7:
            strength = "strong"
        elif abs(pearson.statistic) >= 0.4:
            strength = "moderate"
        direction = "positive" if pearson.statistic > 0 else "negative"
        rows.append(
            {
                "sample": label,
                "question": question,
                "x": x_col,
                "y": y_col,
                "n": len(subset),
                "pearson_r": round(float(pearson.statistic), 4),
                "pearson_p": round(float(pearson.pvalue), 4),
                "spearman_r": round(float(spearman.statistic), 4),
                "spearman_p": round(float(spearman.pvalue), 4),
                "interpretation": f"{strength.title()} {direction} association; exploratory, not causal.",
            }
        )
    return pd.DataFrame(rows)


def run_regression(df: pd.DataFrame, label: str, y_col: str, x_cols: list[str]) -> pd.DataFrame:
    cols = [y_col] + x_cols
    if any(col not in df.columns for col in cols):
        return pd.DataFrame()
    subset = df[["state"] + cols].dropna().copy()
    n = len(subset)
    k = len(x_cols)
    if n <= k + 2:
        return pd.DataFrame(
            [
                {
                    "sample": label,
                    "model": f"{y_col} ~ {' + '.join(x_cols)}",
                    "term": "MODEL_NOT_RUN",
                    "n": n,
                    "r_squared": np.nan,
                    "coefficient": np.nan,
                    "std_error": np.nan,
                    "t_stat": np.nan,
                    "p_value": np.nan,
                    "note": "Too few complete observations for this model.",
                }
            ]
        )

    y = subset[y_col].to_numpy(dtype=float)
    x = subset[x_cols].to_numpy(dtype=float)
    x_design = np.column_stack([np.ones(n), x])
    beta, *_ = np.linalg.lstsq(x_design, y, rcond=None)
    fitted = x_design @ beta
    residuals = y - fitted
    sse = float(np.sum(residuals**2))
    sst = float(np.sum((y - y.mean()) ** 2))
    r_squared = 1 - sse / sst if sst else np.nan
    df_resid = n - k - 1
    sigma2 = sse / df_resid
    xtx_inv = np.linalg.pinv(x_design.T @ x_design)
    se = np.sqrt(np.diag(sigma2 * xtx_inv))
    t_stats = beta / se
    p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), df_resid))
    rows = []
    for term, coef, err, t_stat, p_value in zip(["intercept"] + x_cols, beta, se, t_stats, p_values):
        rows.append(
            {
                "sample": label,
                "model": f"{y_col} ~ {' + '.join(x_cols)}",
                "term": term,
                "n": n,
                "r_squared": round(r_squared, 4),
                "coefficient": round(float(coef), 4),
                "std_error": round(float(err), 4),
                "t_stat": round(float(t_stat), 4),
                "p_value": round(float(p_value), 4),
                "note": "Exploratory OLS; mixed years and small samples mean this is not causal evidence.",
            }
        )
    return pd.DataFrame(rows)


def regression_tables(high: pd.DataFrame, all_states: pd.DataFrame) -> pd.DataFrame:
    model_specs = [
        (
            "st_bpl_mean_pct",
            ["st_literacy_rate_pct", "dropout_secondary_pct", "employment_wpr_person_per_1000"],
        ),
        (
            "employment_wpr_person_per_1000",
            ["st_literacy_rate_pct", "tribe_weighted_literacy_female_pct", "dropout_secondary_pct"],
        ),
        (
            "overall_priority_score",
            ["st_literacy_rate_pct", "st_bpl_mean_pct", "employment_wpr_person_per_1000"],
        ),
    ]
    frames = []
    for label, df in [("high_st_states", high), ("all_states", all_states)]:
        for y_col, x_cols in model_specs:
            frames.append(run_regression(df, label, y_col, x_cols))
    out = pd.concat([frame for frame in frames if not frame.empty], ignore_index=True)
    write_csv(out, "regression_results.csv")
    return out


def build_typology(high: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    features = [
        "education_disadvantage_score",
        "economic_vulnerability_score",
        "st_literacy_rate_pct",
        "literacy_gap_pct",
        "dropout_secondary_pct",
        "st_bpl_mean_pct",
        "employment_wpr_person_per_1000",
        "employment_pu_person_per_1000",
        "mgnreg_sought_not_received_per_1000",
    ]
    features = get_existing(high, features)
    cluster_input = high[["state"] + features].copy()
    values = cluster_input[features].copy()
    values = values.fillna(values.median(numeric_only=True))
    scaler = StandardScaler()
    scaled = scaler.fit_transform(values)
    n_clusters = min(4, len(high))
    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=30)
    labels = model.fit_predict(scaled)
    typology_cols = list(
        dict.fromkeys(
            [
                "state",
                "policy_priority_category",
                "overall_priority_rank",
                "education_disadvantage_score",
                "economic_vulnerability_score",
                "overall_priority_score",
            ]
            + features
        )
    )
    typology = high[typology_cols].copy()
    typology["cluster_id"] = labels + 1

    centers = typology.groupby("cluster_id")[features].mean(numeric_only=True).reset_index()
    center_names = {}
    for _, row in centers.iterrows():
        cluster_id = int(row["cluster_id"])
        edu = row.get("education_disadvantage_score", np.nan)
        econ = row.get("economic_vulnerability_score", np.nan)
        dropout = row.get("dropout_secondary_pct", np.nan)
        poverty = row.get("st_bpl_mean_pct", np.nan)
        if edu >= centers["education_disadvantage_score"].median() and econ >= centers["economic_vulnerability_score"].median():
            name = "Compound education and economic disadvantage"
        elif edu >= centers["education_disadvantage_score"].median():
            name = "Education-system disadvantage"
        elif econ >= centers["economic_vulnerability_score"].median():
            name = "Economic and labour vulnerability"
        elif pd.notna(dropout) and dropout >= centers["dropout_secondary_pct"].median():
            name = "Lower composite risk with retention watchlist"
        elif pd.notna(poverty) and poverty >= centers["st_bpl_mean_pct"].median():
            name = "Lower composite risk with poverty watchlist"
        else:
            name = "Comparatively stronger / lower composite risk"
        center_names[cluster_id] = name
    typology["cluster_name"] = typology["cluster_id"].map(center_names)
    typology = typology.sort_values(["cluster_id", "overall_priority_rank", "state"]).reset_index(drop=True)
    write_csv(typology, "state_typology_clusters.csv")

    cluster_summary = (
        typology.groupby(["cluster_id", "cluster_name"])
        .agg(
            state_count=("state", "count"),
            states=("state", lambda x: ", ".join(sorted(x))),
            avg_education_disadvantage=("education_disadvantage_score", "mean"),
            avg_economic_vulnerability=("economic_vulnerability_score", "mean"),
            avg_overall_priority=("overall_priority_score", "mean"),
            avg_st_literacy=("st_literacy_rate_pct", "mean"),
            avg_secondary_dropout=("dropout_secondary_pct", "mean"),
            avg_st_poverty=("st_bpl_mean_pct", "mean"),
            avg_wpr=("employment_wpr_person_per_1000", "mean"),
        )
        .reset_index()
    )
    write_csv(cluster_summary, "state_typology_cluster_summary.csv")

    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(scaled)
    pca_df = typology[["state", "cluster_id", "cluster_name"]].copy()
    pca_df["pca_1"] = coords[:, 0]
    pca_df["pca_2"] = coords[:, 1]
    write_csv(pca_df, "state_typology_pca_coordinates.csv")
    return typology, cluster_summary, pca_df


def recommendation_for_state(row: pd.Series, thresholds: dict[str, float]) -> tuple[str, str]:
    evidence = []
    recs = []

    if pd.notna(row.get("st_literacy_rate_pct")) and row["st_literacy_rate_pct"] <= thresholds["low_literacy"]:
        evidence.append(f"low ST literacy ({fmt(row['st_literacy_rate_pct'])}%)")
        recs.append("Prioritise foundational literacy, remedial learning, and adult literacy in ST communities")
    if pd.notna(row.get("literacy_gap_pct")) and row["literacy_gap_pct"] >= thresholds["high_gap"]:
        evidence.append(f"large literacy gap ({fmt(row['literacy_gap_pct'])} pp)")
        recs.append("Target the ST-general population literacy gap with ST-focused school support and tracking")
    if pd.notna(row.get("dropout_secondary_pct")) and row["dropout_secondary_pct"] >= thresholds["high_dropout"]:
        evidence.append(f"high secondary dropout ({fmt(row['dropout_secondary_pct'])}%)")
        recs.append("Strengthen secondary retention through hostels, transport, scholarships, and transition support")
    if pd.notna(row.get("tribe_weighted_literacy_female_pct")) and row["tribe_weighted_literacy_female_pct"] <= thresholds["low_female_literacy"]:
        evidence.append(f"low female literacy ({fmt(row['tribe_weighted_literacy_female_pct'])}%)")
        recs.append("Add female literacy interventions: residential schooling, women teachers, safety, sanitation, and community outreach")
    if pd.notna(row.get("st_bpl_mean_pct")) and row["st_bpl_mean_pct"] >= thresholds["high_poverty"]:
        evidence.append(f"high ST poverty ({fmt(row['st_bpl_mean_pct'])}%)")
        recs.append("Bundle schooling support with livelihood, nutrition, and scholarship protection for poor ST households")
    if pd.notna(row.get("employment_wpr_person_per_1000")) and row["employment_wpr_person_per_1000"] <= thresholds["low_wpr"]:
        evidence.append(f"weak WPR ({fmt(row['employment_wpr_person_per_1000'], 0)} per 1000)")
        recs.append("Connect upper-secondary schooling with local skills, placement, and public employment pathways")
    if pd.notna(row.get("mgnreg_sought_not_received_per_1000")) and row["mgnreg_sought_not_received_per_1000"] >= thresholds["high_mgnreg_unmet"]:
        evidence.append(f"high unmet MGNREG demand ({fmt(row['mgnreg_sought_not_received_per_1000'], 0)} per 1000)")
        recs.append("Improve MGNREG work availability while reducing education costs for vulnerable households")
    if pd.notna(row.get("low_literacy_district_count")) and row["low_literacy_district_count"] > 0:
        evidence.append(f"{int(row['low_literacy_district_count'])} low-female-literacy district(s)")
        recs.append("Use district-level targeting instead of only state-wide averages")
    if pd.notna(row.get("tribal_villages_gt_50_count")) and row["tribal_villages_gt_50_count"] >= thresholds["high_village_concentration"]:
        evidence.append(f"many >50% ST villages ({fmt(row['tribal_villages_gt_50_count'], 0)})")
        recs.append("Use geographically concentrated service delivery: school clusters, mobile academic support, and local monitoring")

    if not recs:
        evidence.append("comparatively stronger or incomplete-risk profile in available indicators")
        recs.append("Maintain monitoring and protect gains, with targeted support for weaker districts or groups")

    deduped_recs = list(dict.fromkeys(recs))
    return "; ".join(evidence), "; ".join(deduped_recs[:4])


def recommendation_table(high: pd.DataFrame) -> pd.DataFrame:
    thresholds = {
        "low_literacy": high["st_literacy_rate_pct"].quantile(0.33),
        "high_gap": high["literacy_gap_pct"].quantile(0.67),
        "high_dropout": high["dropout_secondary_pct"].quantile(0.67),
        "low_female_literacy": high["tribe_weighted_literacy_female_pct"].quantile(0.33),
        "high_poverty": high["st_bpl_mean_pct"].quantile(0.67),
        "low_wpr": high["employment_wpr_person_per_1000"].quantile(0.33),
        "high_mgnreg_unmet": high["mgnreg_sought_not_received_per_1000"].quantile(0.67),
        "high_village_concentration": high["tribal_villages_gt_50_count"].quantile(0.67),
    }
    rows = []
    for _, row in high.sort_values("overall_priority_score", ascending=False).iterrows():
        evidence, recommendation = recommendation_for_state(row, thresholds)
        rows.append(
            {
                "state": row["state"],
                "overall_priority_rank": row.get("overall_priority_rank"),
                "policy_priority_category": row.get("policy_priority_category"),
                "cluster_name": row.get("cluster_name", ""),
                "evidence_flags": evidence,
                "recommended_policy_focus": recommendation,
            }
        )
    out = pd.DataFrame(rows)
    write_csv(out, "state_policy_recommendations.csv")
    return out


def sql_demo_tables() -> pd.DataFrame:
    if not DB_PATH.exists():
        return pd.DataFrame()
    queries = {
        "top_priority_high_st_states": """
            select state, overall_priority_rank, overall_priority_score, policy_priority_category
            from state_analysis_dataset
            where is_high_st_share_state = 1
            order by overall_priority_score desc
            limit 10
        """,
        "worst_literacy_high_st_states": """
            select state, st_literacy_rate_pct, literacy_gap_pct
            from state_analysis_dataset
            where is_high_st_share_state = 1
              and st_literacy_rate_pct is not null
            order by st_literacy_rate_pct asc
            limit 10
        """,
        "poverty_employment_high_st_states": """
            select state, st_bpl_mean_pct, employment_wpr_person_per_1000, employment_pu_person_per_1000
            from state_analysis_dataset
            where is_high_st_share_state = 1
              and st_bpl_mean_pct is not null
            order by st_bpl_mean_pct desc
            limit 10
        """,
    }
    rows = []
    with sqlite3.connect(DB_PATH) as conn:
        for name, query in queries.items():
            result = pd.read_sql_query(query, conn)
            result.to_csv(TABLE_DIR / f"sql_{name}.csv", index=False)
            rows.append({"query_name": name, "rows_returned": len(result), "output_file": f"tables/sql_{name}.csv"})
    out = pd.DataFrame(rows)
    write_csv(out, "sql_query_outputs.csv")
    return out


def clean_table(name: str) -> pd.DataFrame:
    return pd.read_csv(CLEAN_DIR / f"{name}.csv")


def merge_state_year(base: pd.DataFrame | None, other: pd.DataFrame) -> pd.DataFrame:
    if other.empty:
        return base if base is not None else other
    if base is None or base.empty:
        return other
    return base.merge(other, on=["state", "year"], how="outer")


def build_sparse_state_year_fact() -> pd.DataFrame:
    """Build a sparse state-year fact table without forcing mismatched years together."""
    fact: pd.DataFrame | None = None

    direct_tables = [
        "literacy",
        "demographics_st",
        "enrolment_st",
        "mgnreg_st",
        "high_st_share",
        "tribal_villages",
        "household_type_rural",
    ]
    for table in direct_tables:
        df = clean_table(table)
        if "year" not in df.columns:
            continue
        keep = [col for col in df.columns if col not in {"additional_info", "additional_information", "social_group", "residence_type"}]
        fact = merge_state_year(fact, df[keep])

    ger = clean_table("ger_st")
    ger_total = ger[ger["gender_key"].eq("total")]
    ger_pivot = ger_total.pivot_table(index=["state", "year"], columns="class_group_key", values="ger_ratio", aggfunc="mean")
    ger_pivot = ger_pivot.rename(columns={col: f"ger_{col}" for col in ger_pivot.columns}).reset_index()
    fact = merge_state_year(fact, ger_pivot)

    employment = clean_table("employment_st")
    emp_pivot = employment.pivot_table(
        index=["state", "year"],
        columns=["employment_indicator_key", "gender_key"],
        values="employment_indicator_value_per_1000",
        aggfunc="mean",
    )
    emp_pivot.columns = [f"employment_{indicator}_{gender}_per_1000" for indicator, gender in emp_pivot.columns]
    fact = merge_state_year(fact, emp_pivot.reset_index())

    poverty = clean_table("poverty_st")
    poverty_pivot = poverty.pivot_table(index=["state", "year"], columns="residence_type", values="st_bpl_pct", aggfunc="mean")
    poverty_pivot = poverty_pivot.rename(columns={col: f"st_bpl_{col}_pct" for col in poverty_pivot.columns})
    poverty_pivot["st_bpl_mean_pct"] = poverty_pivot.mean(axis=1, skipna=True)
    fact = merge_state_year(fact, poverty_pivot.reset_index())

    scst = clean_table("sc_st_residence")
    scst_pivot = scst.pivot_table(index=["state", "year"], columns="residence_type", values="st_population", aggfunc="sum")
    scst_pivot = scst_pivot.rename(columns={col: f"st_population_{col}" for col in scst_pivot.columns})
    fact = merge_state_year(fact, scst_pivot.reset_index())

    low_lit = clean_table("low_literacy_districts")
    low_summary = (
        low_lit.groupby(["state", "year"])
        .agg(
            low_literacy_district_count=("district", "nunique"),
            low_literacy_female_min_pct=("literacy_female_pct", "min"),
            low_literacy_female_mean_pct=("literacy_female_pct", "mean"),
        )
        .reset_index()
    )
    fact = merge_state_year(fact, low_summary)

    dropout = clean_table("dropout_st")
    dropout["year"] = 2022
    dropout = dropout.drop(columns=["academic_year"], errors="ignore")
    fact = merge_state_year(fact, dropout)

    if fact is None:
        fact = pd.DataFrame()
    fact = fact.sort_values(["state", "year"]).reset_index(drop=True)
    write_csv(fact, "fact_state_year_sparse.csv")

    if DB_PATH.exists() and not fact.empty:
        with sqlite3.connect(DB_PATH) as conn:
            fact.to_sql("fact_state_year_sparse", conn, index=False, if_exists="replace")
    return fact


def simple_bar(
    df: pd.DataFrame,
    x_col: str,
    title: str,
    filename: str,
    xlabel: str,
    ascending: bool = True,
    color: str = "#4d7c8a",
    top_n: int | None = None,
) -> None:
    if x_col not in df.columns:
        return
    plot_df = df[["state", x_col]].dropna().sort_values(x_col, ascending=ascending)
    if top_n:
        plot_df = plot_df.head(top_n)
    if plot_df.empty:
        return
    fig_height = max(5.5, len(plot_df) * 0.32)
    fig, ax = plt.subplots(figsize=(10.5, fig_height))
    ax.barh(plot_df["state"], plot_df[x_col], color=color)
    if ascending:
        ax.invert_yaxis()
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / filename, dpi=170)
    plt.close(fig)


def scatter_with_labels(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    filename: str,
    xlabel: str,
    ylabel: str,
    color: str = "#7f4f24",
) -> None:
    if x_col not in df.columns or y_col not in df.columns:
        return
    plot_df = df[["state", x_col, y_col]].dropna()
    if len(plot_df) < 3:
        return
    fig, ax = plt.subplots(figsize=(9, 6.5))
    ax.scatter(plot_df[x_col], plot_df[y_col], color=color, s=46)
    for _, row in plot_df.iterrows():
        ax.annotate(row["state"], (row[x_col], row[y_col]), fontsize=7, xytext=(3, 2), textcoords="offset points")
    if len(plot_df) >= 4:
        m, b = np.polyfit(plot_df[x_col], plot_df[y_col], 1)
        xs = np.linspace(plot_df[x_col].min(), plot_df[x_col].max(), 100)
        ax.plot(xs, m * xs + b, color="#333333", linewidth=1, alpha=0.65)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / filename, dpi=170)
    plt.close(fig)


def correlation_heatmap(df: pd.DataFrame) -> None:
    cols = get_existing(
        df,
        [
            "st_literacy_rate_pct",
            "literacy_gap_pct",
            "dropout_secondary_pct",
            "tribe_weighted_literacy_female_pct",
            "st_bpl_mean_pct",
            "employment_wpr_person_per_1000",
            "employment_pu_person_per_1000",
            "mgnreg_sought_not_received_per_1000",
            "overall_priority_score",
        ],
    )
    corr = df[cols].corr(numeric_only=True)
    corr.to_csv(TABLE_DIR / "high_st_correlation_matrix.csv")
    fig, ax = plt.subplots(figsize=(10, 8))
    image = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)), corr.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(corr.index)), corr.index, fontsize=8)
    for i in range(len(corr.index)):
        for j in range(len(corr.columns)):
            value = corr.iloc[i, j]
            if not pd.isna(value):
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=7)
    ax.set_title("Correlation Matrix: High-ST States")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "correlation_heatmap_high_st_states.png", dpi=170)
    plt.close(fig)


def cluster_plot(pca_df: pd.DataFrame) -> None:
    if pca_df.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 6.5))
    colors = plt.cm.Set2(np.linspace(0, 1, pca_df["cluster_id"].nunique()))
    color_map = {cluster: colors[i] for i, cluster in enumerate(sorted(pca_df["cluster_id"].unique()))}
    for cluster_id, group in pca_df.groupby("cluster_id"):
        ax.scatter(group["pca_1"], group["pca_2"], s=58, label=f"Cluster {cluster_id}", color=color_map[cluster_id])
        for _, row in group.iterrows():
            ax.annotate(row["state"], (row["pca_1"], row["pca_2"]), fontsize=7, xytext=(3, 2), textcoords="offset points")
    ax.axhline(0, color="#888888", linewidth=0.8, alpha=0.5)
    ax.axvline(0, color="#888888", linewidth=0.8, alpha=0.5)
    ax.set_title("State Typology Clusters")
    ax.set_xlabel("PCA dimension 1")
    ax.set_ylabel("PCA dimension 2")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "state_typology_clusters.png", dpi=170)
    plt.close(fig)


def make_figures(high: pd.DataFrame, pca_df: pd.DataFrame) -> None:
    simple_bar(
        high,
        "st_literacy_rate_pct",
        "ST Literacy Rate in High-ST States",
        "st_literacy_rate_high_st_states.png",
        "ST literacy rate (%)",
        ascending=True,
        color="#697a21",
    )
    simple_bar(
        high,
        "literacy_gap_pct",
        "Literacy Gap in High-ST States",
        "literacy_gap_high_st_states.png",
        "Literacy gap (percentage points)",
        ascending=False,
        color="#a65d03",
    )
    simple_bar(
        high,
        "dropout_secondary_pct",
        "Secondary Dropout Rate in High-ST States",
        "secondary_dropout_high_st_states.png",
        "Secondary dropout rate (%)",
        ascending=False,
        color="#8f3f3f",
    )
    simple_bar(
        high,
        "st_bpl_mean_pct",
        "ST Poverty in High-ST States",
        "st_poverty_high_st_states.png",
        "Mean rural/urban ST BPL rate (%)",
        ascending=False,
        color="#7f4f24",
    )
    simple_bar(
        high,
        "employment_wpr_person_per_1000",
        "ST Worker Population Ratio in High-ST States",
        "st_worker_population_ratio_high_st_states.png",
        "WPR per 1000",
        ascending=True,
        color="#4d7c8a",
    )
    simple_bar(
        high,
        "overall_priority_score",
        "Overall Policy Priority Score in High-ST States",
        "overall_priority_score_high_st_states.png",
        "Overall priority score",
        ascending=False,
        color="#5f5b8f",
    )
    scatter_with_labels(
        high,
        "st_literacy_rate_pct",
        "st_bpl_mean_pct",
        "ST Poverty vs ST Literacy",
        "poverty_vs_literacy_high_st_states.png",
        "ST literacy rate (%)",
        "Mean ST BPL rate (%)",
    )
    scatter_with_labels(
        high,
        "dropout_secondary_pct",
        "st_literacy_rate_pct",
        "ST Literacy vs Secondary Dropout",
        "literacy_vs_secondary_dropout_high_st_states.png",
        "Secondary dropout rate (%)",
        "ST literacy rate (%)",
        color="#8f3f3f",
    )
    scatter_with_labels(
        high,
        "tribe_weighted_literacy_female_pct",
        "employment_wpr_female_per_1000",
        "Female Work Participation vs Female Literacy",
        "female_wpr_vs_female_literacy_high_st_states.png",
        "Weighted female ST literacy (%)",
        "Female WPR per 1000",
        color="#4d7c8a",
    )
    scatter_with_labels(
        high,
        "education_disadvantage_score",
        "economic_vulnerability_score",
        "Policy Typology: Education vs Economic Vulnerability",
        "education_vs_economic_vulnerability_high_st_states.png",
        "Education disadvantage score",
        "Economic vulnerability score",
        color="#5f5b8f",
    )
    correlation_heatmap(high)
    cluster_plot(pca_df)


def state_profile_table(high: pd.DataFrame, typology: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "state",
        "st_share_state_population_pct",
        "st_population",
        "st_literacy_rate_pct",
        "literacy_gap_pct",
        "tribe_weighted_literacy_female_pct",
        "dropout_secondary_pct",
        "st_bpl_mean_pct",
        "employment_wpr_person_per_1000",
        "employment_pu_person_per_1000",
        "mgnreg_sought_not_received_per_1000",
        "low_literacy_district_count",
        "overall_priority_score",
        "overall_priority_rank",
        "policy_priority_category",
    ]
    profile = high[get_existing(high, cols)].copy()
    profile = profile.merge(typology[["state", "cluster_id", "cluster_name"]], on="state", how="left")
    profile = profile.sort_values("overall_priority_rank")
    write_csv(profile, "high_st_state_policy_profiles.csv")
    return profile


def markdown_table(df: pd.DataFrame, columns: list[str], n: int = 8) -> list[str]:
    if df.empty:
        return ["No rows available."]
    frame = df[columns].head(n).copy()
    lines = []
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("|" + "|".join(["---"] * len(columns)) + "|")
    for _, row in frame.iterrows():
        values = []
        for col in columns:
            value = row[col]
            if isinstance(value, float):
                values.append(fmt(value, 3 if "score" in col else 2))
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return lines


def key_findings(
    high: pd.DataFrame,
    rankings: dict[str, pd.DataFrame],
    relationships: pd.DataFrame,
    cluster_summary: pd.DataFrame,
) -> list[str]:
    findings = []
    lowest_lit = rankings.get("lowest_st_literacy.csv", pd.DataFrame()).head(3)
    if not lowest_lit.empty:
        findings.append(
            "The weakest ST literacy outcomes among high-ST states are concentrated in "
            + ", ".join(
                f"{row.state} ({fmt(row.st_literacy_rate_pct)}%)" for row in lowest_lit.itertuples(index=False)
            )
            + "."
        )
    dropout = rankings.get("highest_secondary_dropout.csv", pd.DataFrame()).head(3)
    if not dropout.empty:
        findings.append(
            "Secondary dropout is highest in "
            + ", ".join(
                f"{row.state} ({fmt(row.dropout_secondary_pct)}%)" for row in dropout.itertuples(index=False)
            )
            + ", making retention beyond elementary schooling a central intervention area."
        )
    poverty = rankings.get("highest_st_poverty.csv", pd.DataFrame()).head(3)
    if not poverty.empty:
        findings.append(
            "ST poverty is highest in "
            + ", ".join(f"{row.state} ({fmt(row.st_bpl_mean_pct)}%)" for row in poverty.itertuples(index=False))
            + " among states with available poverty data."
        )
    priority = rankings.get("highest_priority_scores.csv", pd.DataFrame()).head(5)
    if not priority.empty:
        findings.append(
            "The first-pass policy priority ranking places "
            + ", ".join(priority["state"].tolist())
            + " at the top because they combine education disadvantage with economic vulnerability."
        )

    rel = relationships.dropna(subset=["pearson_r"]).copy()
    if not rel.empty:
        rel["abs_r"] = rel["pearson_r"].abs()
        strongest = rel.sort_values("abs_r", ascending=False).head(3)
        findings.append(
            "The strongest exploratory relationships are: "
            + "; ".join(
                f"{row.question} (r={fmt(row.pearson_r, 2)}, n={int(row.n)})" for row in strongest.itertuples(index=False)
            )
            + "."
        )

    if not cluster_summary.empty:
        findings.append(
            "The typology separates the high-ST states into "
            + str(cluster_summary["cluster_id"].nunique())
            + " policy profiles, supporting differentiated interventions rather than one uniform ST education policy."
        )
    return findings


def write_report(
    high: pd.DataFrame,
    all_states: pd.DataFrame,
    inventory: pd.DataFrame,
    quality: dict[str, pd.DataFrame],
    rankings: dict[str, pd.DataFrame],
    relationships: pd.DataFrame,
    regressions: pd.DataFrame,
    typology: pd.DataFrame,
    cluster_summary: pd.DataFrame,
    recommendations: pd.DataFrame,
    fact_state_year: pd.DataFrame,
) -> None:
    findings = key_findings(high, rankings, relationships, cluster_summary)
    top_priority = rankings.get("highest_priority_scores.csv", pd.DataFrame())
    report_lines = [
        "# Exploratory Data Analysis and Policy Findings",
        "",
        "## North Star",
        "",
        "The project asks how educational interventions for Scheduled Tribes should be prioritised across high-ST states, given differences in education outcomes and their association with employment and poverty.",
        "",
        "This EDA therefore focuses on four outputs: identifying weak education outcomes, testing education-poverty-labour relationships, classifying state disadvantage profiles, and translating evidence into policy recommendations.",
        "",
        "## Data and Unit of Analysis",
        "",
        f"- Master all-state table: {len(all_states)} states.",
        f"- High-ST-state analytical subset: {len(high)} states.",
        f"- Cleaned datasets used: {len(inventory)}.",
        "- Main unit of analysis: one state profile assembled from the most relevant available state-level observations.",
        f"- Sparse state-year fact table also generated: {len(fact_state_year)} state-year rows.",
        "- Time periods are mixed across sources: most structural education/demography measures are around 2011, labour/MGNREG/GER are around 2013, dropout is 2021-22, and the extra household-type table is later. Treat relationships as exploratory, not causal.",
        "",
        "## Data Quality Summary",
        "",
        "The cleaned data are analysis-ready, but coverage is uneven because the source datasets differ in year, geography, and filtering.",
        "",
    ]
    coverage = quality["coverage"]
    core_missing = coverage[(coverage["table"] == "high_st_master") & (coverage["missing"] > 0)].sort_values(
        "missing_pct", ascending=False
    )
    if not core_missing.empty:
        report_lines.extend(markdown_table(core_missing, ["column", "non_missing", "missing", "missing_pct"], n=10))
    else:
        report_lines.append("No missingness in tracked core variables.")

    report_lines.extend(
        [
            "",
            "## Key Findings",
            "",
        ]
    )
    for finding in findings:
        report_lines.append(f"- {finding}")

    report_lines.extend(
        [
            "",
            "## Priority Ranking",
            "",
        ]
    )
    if not top_priority.empty:
        report_lines.extend(
            markdown_table(
                top_priority,
                [
                    "state",
                    "education_disadvantage_score",
                    "economic_vulnerability_score",
                    "overall_priority_score",
                    "policy_priority_category",
                ],
                n=10,
            )
        )

    report_lines.extend(
        [
            "",
            "## Relationship Analysis",
            "",
            "The following associations use Pearson and Spearman correlations. Because indicators are drawn from mixed years and small samples, they should be interpreted as pattern evidence rather than causal estimates.",
            "",
        ]
    )
    if not relationships.empty:
        rel_view = relationships[["sample", "question", "n", "pearson_r", "pearson_p", "spearman_r", "interpretation"]]
        report_lines.extend(markdown_table(rel_view.sort_values("pearson_r", key=lambda s: s.abs(), ascending=False), rel_view.columns.tolist(), n=10))

    report_lines.extend(
        [
            "",
            "## Regression Checks",
            "",
            "Exploratory OLS models were run only where enough complete observations existed. These are diagnostic checks, not causal models.",
            "",
        ]
    )
    if not regressions.empty:
        model_summary = regressions[regressions["term"].ne("intercept")].copy()
        report_lines.extend(
            markdown_table(
                model_summary,
                ["sample", "model", "term", "n", "r_squared", "coefficient", "p_value"],
                n=14,
            )
        )

    report_lines.extend(
        [
            "",
            "## State Typology",
            "",
            "K-means clustering was used on standardized education, poverty, employment, MGNREG, and priority-score indicators. The clusters are a policy typology, not a definitive classification.",
            "",
        ]
    )
    if not cluster_summary.empty:
        report_lines.extend(
            markdown_table(
                cluster_summary,
                [
                    "cluster_id",
                    "cluster_name",
                    "state_count",
                    "states",
                    "avg_education_disadvantage",
                    "avg_economic_vulnerability",
                    "avg_overall_priority",
                ],
                n=10,
            )
        )

    report_lines.extend(
        [
            "",
            "## Policy Recommendations",
            "",
            "Recommendations are state-specific and evidence-triggered. They should be read as priorities for intervention design, not as claims that one variable caused another.",
            "",
        ]
    )
    if not recommendations.empty:
        report_lines.extend(
            markdown_table(
                recommendations,
                [
                    "state",
                    "overall_priority_rank",
                    "policy_priority_category",
                    "evidence_flags",
                    "recommended_policy_focus",
                ],
                n=12,
            )
        )

    report_lines.extend(
        [
            "",
            "## Figures Generated",
            "",
            "- `figures/st_literacy_rate_high_st_states.png`",
            "- `figures/literacy_gap_high_st_states.png`",
            "- `figures/secondary_dropout_high_st_states.png`",
            "- `figures/st_poverty_high_st_states.png`",
            "- `figures/st_worker_population_ratio_high_st_states.png`",
            "- `figures/overall_priority_score_high_st_states.png`",
            "- `figures/poverty_vs_literacy_high_st_states.png`",
            "- `figures/literacy_vs_secondary_dropout_high_st_states.png`",
            "- `figures/female_wpr_vs_female_literacy_high_st_states.png`",
            "- `figures/education_vs_economic_vulnerability_high_st_states.png`",
            "- `figures/correlation_heatmap_high_st_states.png`",
            "- `figures/state_typology_clusters.png`",
            "",
            "## Database / Fact Table Output",
            "",
            "- `tables/fact_state_year_sparse.csv` preserves the state-year structure without forcing unmatched years into a false common-year table.",
            "- `state_analysis_dataset_high_st_states.csv` remains the main policy profile table because it combines the best available indicators for cross-state prioritisation.",
            "- The SQLite database was updated with a `fact_state_year_sparse` table for database demonstration and query support.",
            "",
            "## Methodological Limitations",
            "",
            "- The analysis combines indicators from different years, so results are cross-sectional and exploratory.",
            "- Poverty data are available for fewer states than education or dropout data.",
            "- The labour dataset does not include the rural/urban split mentioned in the proposal.",
            "- State averages can hide district and tribe-level variation; support tables should be used for case-study depth.",
            "- Composite scores depend on variable choice and equal-weight normalization; they are useful for prioritisation, not absolute measurement.",
            "",
            "## Next Use In Final Report",
            "",
            "Use this EDA report as the source for the report's Data Exploration, Analysis, Findings, and Recommendations sections. The database section should reference the SQLite database and the raw-clean-final structure already created.",
        ]
    )
    (EDA_DIR / "eda_policy_report.md").write_text("\n".join(report_lines), encoding="utf-8")


def main() -> None:
    ensure_dirs()
    high, all_states, inventory = read_data()
    quality = data_quality_tables(high, all_states, inventory)
    rankings = build_rankings(high)

    relationships = pd.concat(
        [
            relationship_tests(high, "high_st_states"),
            relationship_tests(all_states, "all_states"),
        ],
        ignore_index=True,
    )
    write_csv(relationships, "relationship_tests.csv")

    regressions = regression_tables(high, all_states)
    typology, cluster_summary, pca_df = build_typology(high)
    high_with_clusters = high.merge(typology[["state", "cluster_id", "cluster_name"]], on="state", how="left")
    recommendations = recommendation_table(high_with_clusters)
    state_profile_table(high_with_clusters, typology)
    fact_state_year = build_sparse_state_year_fact()
    sql_demo_tables()
    make_figures(high_with_clusters, pca_df)
    write_report(
        high_with_clusters,
        all_states,
        inventory,
        quality,
        rankings,
        relationships,
        regressions,
        typology,
        cluster_summary,
        recommendations,
        fact_state_year,
    )

    print(f"Wrote EDA tables to {TABLE_DIR.relative_to(ROOT)}")
    print(f"Wrote EDA figures to {FIGURE_DIR.relative_to(ROOT)}")
    print(f"Wrote EDA report to {(EDA_DIR / 'eda_policy_report.md').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
