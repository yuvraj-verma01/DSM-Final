from pathlib import Path
from textwrap import fill

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "outputs" / "analysis" / "state_analysis_dataset_high_st_states.csv"
OUT_DIR = ROOT / "graphs" / "q12"

CONCENTRATION_COLS = [
    "villages_gt50_per_100k_st_pop",
    "villages_gt75_per_100k_st_pop",
    "villages_gt90_per_100k_st_pop",
    "villages_all_st_per_100k_st_pop",
]

OUTCOME_COLS = [
    "st_literacy_rate_pct",
    "literacy_gap_pct",
    "dropout_secondary_pct",
    "st_bpl_mean_pct",
    "mgnreg_sought_not_received_per_1000",
    "employment_wpr_person_per_1000",
    "employment_pu_person_per_1000",
]


def add_concentration_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for label, col in [
        ("gt50", "tribal_villages_gt_50_count"),
        ("gt75", "tribal_villages_gt_75_count"),
        ("gt90", "tribal_villages_gt_90_count"),
        ("all_st", "tribal_villages_100_pct_count"),
    ]:
        df[f"villages_{label}_per_100k_st_pop"] = df[col] / df["st_population"] * 100000
    return df


def correlation_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for x in CONCENTRATION_COLS:
        for y in OUTCOME_COLS:
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
    table = pd.DataFrame(rows).sort_values("abs_pearson_r", ascending=False)
    table.to_csv(OUT_DIR / "q12_correlations.csv", index=False)
    return table


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
) -> None:
    plot_df = df[["state", x, y]].dropna().copy()
    r, p = pearsonr(plot_df[x], plot_df[y])
    sr, sp = spearmanr(plot_df[x], plot_df[y])
    n = len(plot_df)

    fig, ax = plt.subplots(figsize=(11, 7))
    ax.scatter(plot_df[x], plot_df[y], s=72, color=color, alpha=0.9)

    if len(plot_df) >= 4 and plot_df[x].nunique() > 1:
        slope, intercept = np.polyfit(plot_df[x], plot_df[y], 1)
        xs = np.linspace(plot_df[x].min(), plot_df[x].max(), 100)
        ax.plot(xs, slope * xs + intercept, color="#333333", linewidth=1.8, alpha=0.75)

    for _, row in plot_df.iterrows():
        ax.annotate(row["state"], (row[x], row[y]), xytext=(5, 4), textcoords="offset points", fontsize=9)

    ax.text(
        0.02,
        0.98,
        f"Pearson r = {r:.4f} | p = {p:.4f} | n = {n}\nSpearman r = {sr:.4f} | p = {sp:.4f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10.5,
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
    for old_file in OUT_DIR.glob("q12_*"):
        if old_file.is_file():
            old_file.unlink()

    high_st = add_concentration_columns(pd.read_csv(DATA_PATH))
    correlation_table(high_st)

    scatter(
        high_st,
        "villages_gt50_per_100k_st_pop",
        "mgnreg_sought_not_received_per_1000",
        "Q12: ST Village Concentration and MGNREG Unmet Demand",
        "Villages >50% ST per 100k ST population",
        "MGNREG sought-not-received per 1000",
        "q12_gt50_village_concentration_vs_mgnreg_unmet_demand.png",
        "This is the strongest normalized Q12 relationship: more concentrated ST settlement is associated with higher unmet MGNREG demand.",
        "#34699a",
    )
    scatter(
        high_st,
        "villages_gt75_per_100k_st_pop",
        "mgnreg_sought_not_received_per_1000",
        "Q12: Higher ST Village Concentration and MGNREG Unmet Demand",
        "Villages >75% ST per 100k ST population",
        "MGNREG sought-not-received per 1000",
        "q12_gt75_village_concentration_vs_mgnreg_unmet_demand.png",
        "The >75% threshold is a robustness check: the Spearman relationship remains meaningful even when village concentration is defined more strictly.",
        "#4d7c8a",
    )
    scatter(
        high_st,
        "villages_gt50_per_100k_st_pop",
        "st_literacy_rate_pct",
        "Q12: ST Village Concentration and ST Literacy",
        "Villages >50% ST per 100k ST population",
        "ST literacy rate (%)",
        "q12_gt50_village_concentration_vs_st_literacy.png",
        "After normalizing by ST population, village concentration is not strongly related to ST literacy. This keeps the finding specific to livelihood distress.",
        "#a65d03",
    )
    scatter(
        high_st,
        "villages_gt50_per_100k_st_pop",
        "dropout_secondary_pct",
        "Q12: ST Village Concentration and Secondary Dropout",
        "Villages >50% ST per 100k ST population",
        "Secondary dropout (%)",
        "q12_gt50_village_concentration_vs_secondary_dropout.png",
        "This null-style check matters: normalized concentration does not appear to explain secondary dropout by itself.",
        "#8f3f3f",
    )
    rank_bar(
        high_st,
        "villages_gt50_per_100k_st_pop",
        "Q12: Highest Normalized ST Village Concentration",
        "Villages >50% ST per 100k ST population",
        "q12_highest_gt50_village_concentration.png",
        "#4d7c8a",
    )
    print(f"Wrote Q12 graphs and correlation table to {OUT_DIR}")


if __name__ == "__main__":
    main()
