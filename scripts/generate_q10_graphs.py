from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import pearsonr, spearmanr


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "outputs" / "analysis" / "state_analysis_dataset_high_st_states.csv"
OUT_DIR = ROOT / "graphs" / "q10"

PREDICTORS = ["st_literacy_rate_pct", "dropout_secondary_pct"]
OUTCOMES = [
    "st_bpl_mean_pct",
    "employment_wpr_person_per_1000",
    "employment_pu_person_per_1000",
    "mgnreg_sought_not_received_per_1000",
    "mgnreg_work_100_plus_days_per_1000",
]

LABELS = {
    "st_literacy_rate_pct": "ST literacy",
    "dropout_secondary_pct": "Secondary dropout",
    "st_bpl_mean_pct": "ST poverty",
    "employment_wpr_person_per_1000": "WPR",
    "employment_pu_person_per_1000": "Unemployment",
    "mgnreg_sought_not_received_per_1000": "MGNREG unmet demand",
    "mgnreg_work_100_plus_days_per_1000": "MGNREG 100-plus-days",
}


def correlation_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for outcome in OUTCOMES:
        for predictor in PREDICTORS:
            subset = df[["state", predictor, outcome]].dropna()
            if len(subset) < 4 or subset[predictor].nunique() <= 1 or subset[outcome].nunique() <= 1:
                continue
            pearson_r, pearson_p = pearsonr(subset[predictor], subset[outcome])
            spearman_r, spearman_p = spearmanr(subset[predictor], subset[outcome])
            rows.append(
                {
                    "sample": "high_st_states",
                    "predictor": predictor,
                    "outcome": outcome,
                    "n": len(subset),
                    "pearson_r": round(float(pearson_r), 4),
                    "pearson_p": round(float(pearson_p), 4),
                    "spearman_r": round(float(spearman_r), 4),
                    "spearman_p": round(float(spearman_p), 4),
                    "abs_pearson_r": round(abs(float(pearson_r)), 4),
                }
            )
    table = pd.DataFrame(rows)
    table["predictor_label"] = table["predictor"].map(LABELS)
    table["outcome_label"] = table["outcome"].map(LABELS)
    table = table.sort_values(["outcome", "abs_pearson_r"], ascending=[True, False])
    table.to_csv(OUT_DIR / "q10_correlations.csv", index=False)
    return table


def comparison_plot(table: pd.DataFrame) -> None:
    pivot = table.pivot(index="outcome_label", columns="predictor_label", values="abs_pearson_r").loc[
        [LABELS[outcome] for outcome in OUTCOMES]
    ]

    fig, ax = plt.subplots(figsize=(10, 6))
    pivot.plot(kind="barh", ax=ax, color=["#8f3f3f", "#34699a"])
    ax.set_title("Q10: Literacy vs Secondary Dropout as Warning Signals", fontsize=16, pad=12)
    ax.set_xlabel("Absolute Pearson correlation")
    ax.set_ylabel("")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(frameon=False, title="")

    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", padding=3, fontsize=9)

    fig.text(
        0.08,
        0.02,
        "Dropout is stronger for ST poverty and MGNREG unmet demand. ST literacy is stronger for the unemployment indicator. "
        "This means Q10 should be read as a comparison of warning signals, not as a claim that dropout dominates every outcome.",
        ha="left",
        va="bottom",
        fontsize=9.5,
        color="#444444",
    )
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    fig.savefig(OUT_DIR / "q10_literacy_vs_dropout_warning_signals.png", dpi=180)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for old_file in OUT_DIR.glob("q10_*"):
        if old_file.is_file():
            old_file.unlink()

    high_st = pd.read_csv(DATA_PATH)
    table = correlation_table(high_st)
    comparison_plot(table)
    print(f"Wrote Q10 graph and correlation table to {OUT_DIR}")


if __name__ == "__main__":
    main()
