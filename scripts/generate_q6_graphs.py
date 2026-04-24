from pathlib import Path
from textwrap import fill

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "outputs" / "analysis" / "state_analysis_dataset_high_st_states.csv"
OUT_DIR = ROOT / "graphs" / "q6"

Q6_RELATIONSHIPS = [
    ("mgnreg_sought_not_received_per_1000", "st_bpl_mean_pct"),
    ("mgnreg_sought_not_received_per_1000", "dropout_secondary_pct"),
    ("mgnreg_sought_not_received_per_1000", "st_literacy_rate_pct"),
    ("mgnreg_work_100_plus_days_per_1000", "dropout_secondary_pct"),
]


def correlation_table(df: pd.DataFrame) -> None:
    rows = []
    for x, y in Q6_RELATIONSHIPS:
        subset = df[["state", x, y]].dropna()
        if len(subset) < 4:
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
    pd.DataFrame(rows).sort_values("abs_pearson_r", ascending=False).to_csv(OUT_DIR / "q6_correlations.csv", index=False)


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
    n = len(plot_df)

    fig, ax = plt.subplots(figsize=(11, 7))
    ax.scatter(plot_df[x], plot_df[y], s=72, color=color, alpha=0.9)

    slope, intercept = np.polyfit(plot_df[x], plot_df[y], 1)
    xs = np.linspace(plot_df[x].min(), plot_df[x].max(), 100)
    ax.plot(xs, slope * xs + intercept, color="#333333", linewidth=1.8, alpha=0.75)

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
    for old_file in OUT_DIR.glob("q6_*"):
        if old_file.is_file():
            old_file.unlink()

    high_st = pd.read_csv(DATA_PATH)
    correlation_table(high_st)

    scatter(
        high_st,
        "mgnreg_sought_not_received_per_1000",
        "st_bpl_mean_pct",
        "Q6: MGNREG Unmet Demand and ST Poverty",
        "MGNREG sought-not-received per 1000",
        "Mean ST poverty rate (%)",
        "q6_mgnreg_unmet_demand_vs_st_poverty.png",
        "Unmet MGNREG demand is the clearest Q6 distress signal: where poverty is higher, more households report seeking but not receiving work.",
        "#34699a",
    )
    scatter(
        high_st,
        "mgnreg_sought_not_received_per_1000",
        "dropout_secondary_pct",
        "Q6: MGNREG Unmet Demand and Secondary Dropout",
        "MGNREG sought-not-received per 1000",
        "Secondary dropout (%)",
        "q6_mgnreg_unmet_demand_vs_secondary_dropout.png",
        "This tests whether livelihood distress and school retention problems cluster in the same states.",
        "#8f3f3f",
    )
    scatter(
        high_st,
        "mgnreg_work_100_plus_days_per_1000",
        "dropout_secondary_pct",
        "Q6: MGNREG 100-Plus-Days and Secondary Dropout",
        "MGNREG 100-plus-days per 1000",
        "Secondary dropout (%)",
        "q6_mgnreg_100_plus_days_vs_secondary_dropout.png",
        "This is a weaker signal than unmet demand, but it helps check whether heavier public-work dependence coexists with dropout.",
        "#5f5b8f",
    )
    scatter(
        high_st,
        "mgnreg_sought_not_received_per_1000",
        "st_literacy_rate_pct",
        "Q6: MGNREG Unmet Demand and ST Literacy",
        "MGNREG sought-not-received per 1000",
        "ST literacy rate (%)",
        "q6_mgnreg_unmet_demand_vs_st_literacy.png",
        "The negative slope suggests that unmet work demand tends to be higher where ST literacy is lower.",
        "#a65d03",
    )
    rank_bar(
        high_st,
        "mgnreg_sought_not_received_per_1000",
        "Q6: Highest MGNREG Unmet Demand",
        "Sought-not-received per 1000",
        "q6_highest_mgnreg_unmet_demand.png",
        "#7f4f24",
    )
    rank_bar(
        high_st,
        "mgnreg_work_100_plus_days_per_1000",
        "Q6: Highest MGNREG 100-Plus-Days Among ST Households",
        "100-plus-days per 1000",
        "q6_highest_mgnreg_100_plus_days.png",
        "#4d7c8a",
    )
    print(f"Wrote Q6 graphs and correlation table to {OUT_DIR}")


if __name__ == "__main__":
    main()
