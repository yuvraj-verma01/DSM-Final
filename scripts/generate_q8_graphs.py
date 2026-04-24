from pathlib import Path
from textwrap import fill

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "outputs" / "analysis" / "state_analysis_dataset_high_st_states.csv"
OUT_DIR = ROOT / "graphs" / "q8"

X_COLS = [
    "ger_latest_secondary_total_clean",
    "ger_latest_secondary_girls_clean",
    "gpi_secondary_clean",
    "dropout_secondary_pct",
]

Y_COLS = [
    "mgnreg_job_card_households_per_1000",
    "mgnreg_work_100_plus_days_per_1000",
    "mgnreg_100_plus_share_of_work_received_pct",
    "mgnreg_sought_not_received_per_1000",
    "mgnreg_average_days_worked",
    "mgnreg_work_received_any_days_per_1000",
]


def add_clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in [
        "ger_latest_secondary_total",
        "ger_latest_secondary_girls",
        "ger_latest_higher_secondary_total",
        "ger_latest_higher_secondary_girls",
    ]:
        if col in df.columns:
            df[f"{col}_clean"] = df[col].where(df[col].between(0, 500))
    for col in ["gpi_secondary", "gpi_higher_secondary"]:
        if col in df.columns:
            df[f"{col}_clean"] = df[col].where(df[col].between(0, 3))
    return df


def correlation_table(df: pd.DataFrame) -> None:
    rows = []
    for x in X_COLS:
        for y in Y_COLS:
            if x not in df.columns or y not in df.columns:
                continue
            subset = df[["state", x, y]].dropna()
            if len(subset) < 4 or subset[x].nunique() <= 1 or subset[y].nunique() <= 1:
                continue
            pearson_r, pearson_p = pearsonr(subset[x], subset[y])
            spearman_r, spearman_p = spearmanr(subset[x], subset[y])
            rows.append(
                {
                    "sample": "high_st_states",
                    "x": x,
                    "y": y,
                    "n": len(subset),
                    "pearson_r": round(float(pearson_r), 4),
                    "pearson_p": round(float(pearson_p), 4),
                    "spearman_r": round(float(spearman_r), 4),
                    "spearman_p": round(float(spearman_p), 4),
                    "abs_pearson_r": round(abs(float(pearson_r)), 4),
                }
            )
    pd.DataFrame(rows).sort_values("abs_pearson_r", ascending=False).to_csv(OUT_DIR / "q8_correlations.csv", index=False)


def scatter(
    df: pd.DataFrame,
    x: str,
    y: str,
    title: str,
    x_label: str,
    y_label: str,
    filename: str,
    interpretation: str,
    color: str = "#34699a",
    medians: bool = False,
) -> None:
    plot_df = df[["state", x, y]].dropna().copy()
    r, p = pearsonr(plot_df[x], plot_df[y])
    n = len(plot_df)

    fig, ax = plt.subplots(figsize=(11, 7))
    ax.scatter(plot_df[x], plot_df[y], s=72, color=color, alpha=0.9)

    if len(plot_df) >= 4 and plot_df[x].nunique() > 1:
        slope, intercept = np.polyfit(plot_df[x], plot_df[y], 1)
        xs = np.linspace(plot_df[x].min(), plot_df[x].max(), 100)
        ax.plot(xs, slope * xs + intercept, color="#333333", linewidth=1.8, alpha=0.75)

    if medians:
        ax.axvline(plot_df[x].median(), color="#777777", linestyle="--", linewidth=1.2)
        ax.axhline(plot_df[y].median(), color="#777777", linestyle="--", linewidth=1.2)

    for _, row in plot_df.iterrows():
        ax.annotate(row["state"], (row[x], row[y]), xytext=(5, 4), textcoords="offset points", fontsize=9)

    ax.text(
        0.02,
        0.98,
        f"Pearson r = {r:.4f} | p = {p:.4f} | n = {n}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=11,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.92},
    )
    ax.set_title(title, fontsize=17, pad=12)
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    ax.grid(True, alpha=0.25)
    fig.text(0.08, 0.02, fill(interpretation, width=120), ha="left", va="bottom", fontsize=9.5, color="#444444")
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    fig.savefig(OUT_DIR / filename, dpi=180)
    plt.close(fig)


def rank_bar(df: pd.DataFrame, column: str, title: str, x_label: str, filename: str, color: str) -> None:
    plot_df = df[["state", column]].dropna().sort_values(column, ascending=False).head(10)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(plot_df["state"], plot_df[column], color=color)
    ax.invert_yaxis()
    ax.set_title(title, fontsize=16, pad=12)
    ax.set_xlabel(x_label, fontsize=12)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT_DIR / filename, dpi=180)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for old_file in OUT_DIR.glob("q8_*"):
        if old_file.is_file():
            old_file.unlink()

    high_st = add_clean_columns(pd.read_csv(DATA_PATH))
    correlation_table(high_st)

    scatter(
        high_st,
        "dropout_secondary_pct",
        "mgnreg_sought_not_received_per_1000",
        "Q8: Secondary Dropout and MGNREG Unmet Demand",
        "Secondary dropout (%)",
        "MGNREG sought-not-received per 1000",
        "q8_secondary_dropout_vs_mgnreg_unmet_demand.png",
        "This is the clearest Q8 signal: states with higher secondary dropout also tend to report more unmet MGNREG demand.",
        "#8f3f3f",
    )
    scatter(
        high_st,
        "ger_latest_secondary_total_clean",
        "mgnreg_sought_not_received_per_1000",
        "Q8: Latest Secondary GER IX-X and MGNREG Unmet Demand",
        "Latest secondary GER IX-X",
        "MGNREG sought-not-received per 1000",
        "q8_latest_secondary_ger_vs_mgnreg_unmet_demand.png",
        "This tests whether higher schooling participation is enough to reduce livelihood distress. The relationship is weak, which supports the mismatch argument.",
        "#a65d03",
        medians=True,
    )
    scatter(
        high_st,
        "ger_latest_secondary_total_clean",
        "mgnreg_work_100_plus_days_per_1000",
        "Q8: Latest Secondary GER IX-X and MGNREG 100-Plus-Days",
        "Latest secondary GER IX-X",
        "MGNREG 100-plus-days per 1000",
        "q8_latest_secondary_ger_vs_mgnreg_100_plus_days.png",
        "This captures heavier public-work dependence. It is useful mainly to identify high-GER states that still rely on MGNREG work.",
        "#7f4f24",
        medians=True,
    )
    scatter(
        high_st,
        "ger_latest_secondary_total_clean",
        "mgnreg_average_days_worked",
        "Q8: Latest Secondary GER IX-X and Average MGNREG Days",
        "Latest secondary GER IX-X",
        "Average MGNREG days worked",
        "q8_latest_secondary_ger_vs_mgnreg_average_days.png",
        "Average days worked gives a broader dependence measure than 100-plus-days alone. The relationship is not strong enough for causal claims.",
        "#4d7c8a",
    )
    rank_bar(
        high_st,
        "mgnreg_sought_not_received_per_1000",
        "Q8: Highest MGNREG Unmet Demand",
        "Sought-not-received per 1000",
        "q8_highest_mgnreg_unmet_demand.png",
        "#7f4f24",
    )
    rank_bar(
        high_st,
        "mgnreg_average_days_worked",
        "Q8: Highest Average MGNREG Days Worked",
        "Average days worked",
        "q8_highest_mgnreg_average_days.png",
        "#4d7c8a",
    )
    print(f"Wrote Q8 graphs and correlation table to {OUT_DIR}")


if __name__ == "__main__":
    main()
