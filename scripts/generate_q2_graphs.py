from pathlib import Path
from textwrap import fill

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "outputs" / "analysis" / "state_analysis_dataset_high_st_states.csv"
OUT_DIR = ROOT / "graphs" / "q2"

Q2_RELATIONSHIPS = [
    ("ger_classes_i_viii_clean", "st_bpl_mean_pct"),
    ("ger_classes_ix_x_clean", "dropout_upper_primary_pct"),
    ("ger_classes_ix_x_girls_clean", "dropout_upper_primary_pct"),
    ("ger_classes_ix_x_gpi_clean", "dropout_upper_primary_pct"),
    ("gpi_secondary_clean", "st_bpl_mean_pct"),
    ("dropout_secondary_pct", "st_bpl_mean_pct"),
]


def add_clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in [
        "ger_classes_i_viii",
        "ger_classes_ix_x",
        "ger_classes_ix_x_girls",
        "ger_classes_xi_xii",
        "ger_classes_xi_xii_girls",
        "ger_latest_secondary_total",
        "ger_latest_secondary_girls",
    ]:
        clean_col = f"{col}_clean"
        df[clean_col] = df[col].where(df[col].between(0, 500))

    for col in ["ger_classes_ix_x_gpi", "ger_classes_xi_xii_gpi", "gpi_secondary", "gpi_higher_secondary"]:
        clean_col = f"{col}_clean"
        df[clean_col] = df[col].where(df[col].between(0, 3))

    return df


def make_scatter(
    df: pd.DataFrame,
    x: str,
    y: str,
    title: str,
    x_label: str,
    y_label: str,
    filename: str,
    interpretation: str,
    color: str,
) -> dict:
    plot_df = df[["state", x, y]].dropna().copy()
    r, p = pearsonr(plot_df[x], plot_df[y])
    n = len(plot_df)

    fig, ax = plt.subplots(figsize=(11, 7))
    ax.scatter(plot_df[x], plot_df[y], s=72, color=color, alpha=0.9)

    slope, intercept = np.polyfit(plot_df[x], plot_df[y], 1)
    xs = np.linspace(plot_df[x].min(), plot_df[x].max(), 100)
    ax.plot(xs, slope * xs + intercept, color="#333333", linewidth=1.8, alpha=0.75)

    for _, row in plot_df.iterrows():
        ax.annotate(
            row["state"],
            (row[x], row[y]),
            xytext=(5, 4),
            textcoords="offset points",
            fontsize=9,
            color="#2a2a2a",
        )

    ax.set_title(title, fontsize=17, pad=12)
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    ax.grid(True, alpha=0.25)

    stats_text = f"Pearson r = {r:.4f} | p = {p:.4f} | n = {n}"
    ax.text(
        0.02,
        0.98,
        stats_text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=11,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.92},
    )

    wrapped_interpretation = fill(interpretation, width=120)
    fig.text(0.08, 0.018, wrapped_interpretation, ha="left", va="bottom", fontsize=9.5, color="#444444")
    fig.tight_layout(rect=[0, 0.09, 1, 1])

    out_path = OUT_DIR / filename
    fig.savefig(out_path, dpi=180)
    plt.close(fig)

    return {
        "graph_file": str(out_path.relative_to(ROOT)),
        "x": x,
        "y": y,
        "n": n,
        "pearson_r": round(r, 4),
        "pearson_p": round(p, 4),
        "interpretation": interpretation,
    }


def make_correlation_tables(df: pd.DataFrame) -> None:
    rows = []
    for x, y in Q2_RELATIONSHIPS:
        subset = df[[x, y]].dropna()
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

    corr = pd.DataFrame(rows).sort_values("abs_pearson_r", ascending=False).reset_index(drop=True)
    corr.to_csv(OUT_DIR / "q2_correlations.csv", index=False)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for old_file in OUT_DIR.glob("q2_*"):
        if old_file.is_file():
            old_file.unlink()

    high_st = add_clean_columns(pd.read_csv(DATA_PATH))

    specs = [
        {
            "x": "ger_classes_i_viii_clean",
            "y": "st_bpl_mean_pct",
            "title": "Q2: GER I-VIII and ST Poverty",
            "x_label": "GER I-VIII (cleaned)",
            "y_label": "Mean ST poverty rate (%)",
            "filename": "q2_ger_i_viii_vs_st_poverty.png",
            "color": "#7f4f24",
            "interpretation": (
                "High GER does not necessarily mean better welfare; poorer states may have over-age enrolment, "
                "delayed progression, repeat enrolment, or targeted schooling expansion."
            ),
        },
        {
            "x": "ger_classes_ix_x_girls_clean",
            "y": "dropout_upper_primary_pct",
            "title": "Q2: Girls Secondary GER IX-X and Upper-Primary Dropout",
            "x_label": "Girls secondary GER IX-X (cleaned)",
            "y_label": "Upper-primary dropout (%)",
            "filename": "q2_girls_secondary_ger_ix_x_vs_upper_primary_dropout.png",
            "color": "#5f5b8f",
            "interpretation": (
                "States with stronger girls' secondary participation tend to have lower dropout around the "
                "upper-primary to secondary transition."
            ),
        },
        {
            "x": "ger_classes_ix_x_clean",
            "y": "dropout_upper_primary_pct",
            "title": "Q2: Secondary GER IX-X and Upper-Primary Dropout",
            "x_label": "Secondary GER IX-X (cleaned)",
            "y_label": "Upper-primary dropout (%)",
            "filename": "q2_secondary_ger_ix_x_vs_upper_primary_dropout.png",
            "color": "#8f3f3f",
            "interpretation": (
                "Stronger secondary schooling participation is associated with lower dropout before the "
                "transition into secondary schooling."
            ),
        },
        {
            "x": "ger_classes_ix_x_gpi_clean",
            "y": "dropout_upper_primary_pct",
            "title": "Q2: Secondary GER IX-X GPI and Upper-Primary Dropout",
            "x_label": "Secondary GER IX-X GPI (girls/boys)",
            "y_label": "Upper-primary dropout (%)",
            "filename": "q2_secondary_ger_ix_x_gpi_vs_upper_primary_dropout.png",
            "color": "#4d7c8a",
            "interpretation": (
                "More gender-balanced secondary schooling systems may also be better at retaining students "
                "through the transition stage."
            ),
        },
        {
            "x": "gpi_secondary_clean",
            "y": "st_bpl_mean_pct",
            "title": "Q2: Official Secondary GPI and ST Poverty",
            "x_label": "Official secondary GPI",
            "y_label": "Mean ST poverty rate (%)",
            "filename": "q2_secondary_gpi_vs_st_poverty.png",
            "color": "#a65d03",
            "interpretation": (
                "GPI alone is not enough; parity in enrolment can coexist with high ST poverty and does not "
                "automatically imply economic empowerment."
            ),
        },
        {
            "x": "dropout_secondary_pct",
            "y": "st_bpl_mean_pct",
            "title": "Q2: Secondary Dropout and ST Poverty",
            "x_label": "Secondary dropout (%)",
            "y_label": "Mean ST poverty rate (%)",
            "filename": "q2_secondary_dropout_vs_st_poverty.png",
            "color": "#a65d03",
            "interpretation": (
                "Secondary dropout is a retention indicator. Higher dropout lining up with poverty supports "
                "the argument that progression matters more than enrolment alone."
            ),
        },
    ]

    make_correlation_tables(high_st)
    summary = [make_scatter(high_st, **spec) for spec in specs]
    print(f"Wrote {len(summary)} Q2 graphs and one correlation table to {OUT_DIR}")


if __name__ == "__main__":
    main()
