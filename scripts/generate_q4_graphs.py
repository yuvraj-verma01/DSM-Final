from pathlib import Path
from textwrap import fill

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "outputs" / "analysis" / "state_analysis_dataset_high_st_states.csv"
OUT_DIR = ROOT / "graphs" / "q4"


def correlation_row(df: pd.DataFrame, x: str, y: str) -> dict:
    subset = df[["state", x, y]].dropna()
    pearson_r, pearson_p = pearsonr(subset[x], subset[y])
    spearman_r, spearman_p = spearmanr(subset[x], subset[y])
    return {
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


def scatter_with_labels(
    df: pd.DataFrame,
    x: str,
    y: str,
    title: str,
    x_label: str,
    y_label: str,
    filename: str,
    color: str = "#7f4f24",
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
    fig.tight_layout()
    fig.savefig(OUT_DIR / filename, dpi=180)
    plt.close(fig)


def literacy_quadrant(df: pd.DataFrame) -> None:
    plot_df = df[["state", "st_literacy_rate_pct", "literacy_gap_pct"]].dropna().copy()
    x = "st_literacy_rate_pct"
    y = "literacy_gap_pct"
    x_med = plot_df[x].median()
    y_med = plot_df[y].median()

    fig, ax = plt.subplots(figsize=(12, 7.5))
    ax.scatter(plot_df[x], plot_df[y], s=72, color="#4d7c8a", alpha=0.9)
    ax.axvline(x_med, color="#333333", linestyle="--", linewidth=1.2, alpha=0.65)
    ax.axhline(y_med, color="#333333", linestyle="--", linewidth=1.2, alpha=0.65)

    label_offsets = {
        "Assam": (-46, -18),
        "Meghalaya": (8, -20),
        "Manipur": (-42, 8),
        "Sikkim": (12, 10),
        "Nagaland": (8, -18),
        "Lakshadweep": (8, 10),
        "Mizoram": (8, -18),
        "Tripura": (6, 8),
        "Goa": (6, 8),
        "Jharkhand": (6, 8),
        "Chhattisgarh": (6, 8),
        "Odisha": (6, 8),
    }
    for _, row in plot_df.iterrows():
        offset = label_offsets.get(row["state"], (5, 4))
        ax.annotate(row["state"], (row[x], row[y]), xytext=offset, textcoords="offset points", fontsize=9)

    ax.set_xlim(plot_df[x].min() - 2, plot_df[x].max() + 2)
    ax.set_ylim(plot_df[y].min() - 1.3, plot_df[y].max() + 1.1)
    ax.set_title("Q4: ST Literacy Level and Literacy Gap", fontsize=17, pad=12)
    ax.set_xlabel("ST literacy rate (%)", fontsize=12)
    ax.set_ylabel("Literacy gap (percentage points)", fontsize=12)
    ax.grid(True, alpha=0.25)
    caption = fill(
        "Quadrants: lower literacy + larger gap = compound education concern; higher literacy + larger gap = hidden exclusion; higher literacy + smaller gap = comparatively better.",
        width=125,
    )
    fig.text(
        0.08,
        0.02,
        caption,
        ha="left",
        va="bottom",
        fontsize=9.5,
        color="#444444",
    )
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    fig.savefig(OUT_DIR / "q4_literacy_level_vs_literacy_gap_quadrants.png", dpi=180)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for old_file in OUT_DIR.glob("q4_*"):
        if old_file.is_file():
            old_file.unlink()

    high_st = pd.read_csv(DATA_PATH)

    relationships = [
        ("literacy_gap_pct", "dropout_secondary_pct"),
        ("st_literacy_rate_pct", "dropout_secondary_pct"),
        ("literacy_gap_pct", "st_bpl_mean_pct"),
        ("st_literacy_rate_pct", "st_bpl_mean_pct"),
    ]
    pd.DataFrame([correlation_row(high_st, x, y) for x, y in relationships]).to_csv(
        OUT_DIR / "q4_correlations.csv", index=False
    )

    scatter_with_labels(
        high_st,
        "literacy_gap_pct",
        "dropout_secondary_pct",
        "Q4: Literacy Gap and Secondary Dropout",
        "Literacy gap (percentage points)",
        "Secondary dropout (%)",
        "q4_literacy_gap_vs_secondary_dropout.png",
    )
    literacy_quadrant(high_st)
    print(f"Wrote Q4 graphs and correlation table to {OUT_DIR}")


if __name__ == "__main__":
    main()
