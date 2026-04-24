from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "notebooks" / "eda_policy_analysis.ipynb"


def md(text: str):
    return nbf.v4.new_markdown_cell(dedent(text).strip())


def code(text: str):
    return nbf.v4.new_code_cell(dedent(text).strip())


cells = [
    md(
        """
        # EDA: ST Education, Employment, and Poverty Across High-ST States

        ## Main Question

        **How do differences in educational outcomes among Scheduled Tribe populations relate to employment and poverty across high-ST states in India, and which states require different policy priorities?**

        ## Core Supporting Questions

        1. How do ST literacy and literacy gaps describe long-term educational exclusion?
        2. Do states with stronger ST schooling participation, measured by GER, also have lower dropout and lower ST poverty?
        3. Are enrolment and GER enough to indicate educational progress, or do dropout rates tell a different story?
        4. Is the literacy gap more informative than ST literacy alone in explaining disadvantage?
        5. Are there states where educational indicators look reasonable, but employment or poverty outcomes remain weak?
        6. Does MGNREG unmet demand reflect deeper livelihood distress among ST households?

        ## Additional Questions

        7. Are scholarship-supported states seeing lower secondary dropout among ST students?
        8. Do states with high ST schooling participation still depend heavily on MGNREG?
        9. Does gender parity in enrolment translate into better female literacy and work outcomes?
        10. Does secondary dropout appear to be a stronger warning signal of poverty and weak labour outcomes than literacy alone?
        11. Do high-ST-share states systematically perform worse, or do outcomes vary substantially?
        12. Do states with high concentrations of ST villages show different education and livelihood outcomes?
        13. Are there gender-specific disadvantages hidden behind state averages?
        """
    ),
    code(
        """
        from pathlib import Path
        import math
        import sqlite3
        import sys

        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        from IPython.display import Markdown, display
        from scipy import stats

        ROOT = Path.cwd()
        if not (ROOT / "outputs").exists():
            ROOT = ROOT.parent
        sys.path.insert(0, str(ROOT))

        pd.set_option("display.max_columns", 120)
        pd.set_option("display.max_colwidth", 180)

        OUTPUTS = ROOT / "outputs"
        DB_PATH = OUTPUTS / "st_education_project.sqlite"

        plt.rcParams["figure.figsize"] = (8.5, 5.5)
        plt.rcParams["axes.grid"] = True
        plt.rcParams["grid.alpha"] = 0.2

        print("Project root:", ROOT)
        """
    ),
    md(
        """
        ## Rebuild Data and EDA Outputs

        The raw files remain unchanged. This cell rebuilds the cleaned tables and derived outputs before the notebook reads them.
        """
    ),
    code(
        """
        import scripts.build_project_data as build_project_data
        import scripts.run_policy_eda as run_policy_eda

        build_project_data.main()
        run_policy_eda.main()
        """
    ),
    md("## Load Main Tables"),
    code(
        """
        high_st = pd.read_csv(OUTPUTS / "analysis" / "state_analysis_dataset_high_st_states.csv").copy()
        all_states = pd.read_csv(OUTPUTS / "analysis" / "state_analysis_dataset_all_states.csv").copy()
        inventory = pd.read_csv(OUTPUTS / "analysis" / "data_inventory.csv")

        # Keep notebook logic conservative. Clean only what is needed for readable plots.
        for col in ["ger_classes_i_viii", "ger_classes_ix_xii"]:
            clean_col = f"{col}_clean"
            high_st[clean_col] = high_st[col].where(high_st[col].between(0, 500))
            all_states[clean_col] = all_states[col].where(all_states[col].between(0, 500))

        for col in ["ger_classes_ix_xii_boys", "ger_classes_ix_xii_girls", "ger_classes_i_viii_girls"]:
            if col in high_st.columns:
                clean_col = f"{col}_clean"
                high_st[clean_col] = high_st[col].where(high_st[col].between(0, 500))
                all_states[clean_col] = all_states[col].where(all_states[col].between(0, 500))

        for col in ["ger_classes_ix_xii_gpi", "ger_classes_i_viii_gpi"]:
            if col in high_st.columns:
                clean_col = f"{col}_clean"
                high_st[clean_col] = high_st[col].where(high_st[col].between(0, 3))
                all_states[clean_col] = all_states[col].where(all_states[col].between(0, 3))

        for col in ["ger_latest_ix_xii_avg", "ger_latest_ix_xii_avg_girls", "ger_latest_secondary_total", "ger_latest_higher_secondary_total"]:
            if col in high_st.columns:
                clean_col = f"{col}_clean"
                high_st[clean_col] = high_st[col].where(high_st[col].between(0, 500))
                all_states[clean_col] = all_states[col].where(all_states[col].between(0, 500))

        for col in ["ger_latest_ix_xii_avg_gpi", "gpi_ix_xii_avg", "gpi_secondary", "gpi_higher_secondary"]:
            if col in high_st.columns:
                clean_col = f"{col}_clean"
                high_st[clean_col] = high_st[col].where(high_st[col].between(0, 3))
                all_states[clean_col] = all_states[col].where(all_states[col].between(0, 3))

        high_st["female_literacy_gap_pct"] = (
            high_st["tribe_weighted_literacy_male_pct"] - high_st["tribe_weighted_literacy_female_pct"]
        )
        all_states["female_literacy_gap_pct"] = (
            all_states["tribe_weighted_literacy_male_pct"] - all_states["tribe_weighted_literacy_female_pct"]
        )

        high_st["villages_gt50_per_100k_st_pop"] = (
            high_st["tribal_villages_gt_50_count"] / high_st["st_population"] * 100000
        )
        all_states["villages_gt50_per_100k_st_pop"] = (
            all_states["tribal_villages_gt_50_count"] / all_states["st_population"] * 100000
        )

        print(f"High-ST states: {high_st.shape[0]} rows")
        print(f"All-state master table: {all_states.shape[0]} rows")
        """
    ),
    code(
        """
        def corr_frame(df, x_cols, y_cols, sample_name):
            rows = []
            for x in x_cols:
                for y in y_cols:
                    subset = df[[x, y]].dropna()
                    if len(subset) < 4:
                        continue
                    r, p = stats.pearsonr(subset[x], subset[y])
                    rows.append(
                        {
                            "sample": sample_name,
                            "x": x,
                            "y": y,
                            "n": len(subset),
                            "pearson_r": round(float(r), 4),
                            "pearson_p": round(float(p), 4),
                            "abs_r": round(abs(float(r)), 4),
                        }
                    )
            return pd.DataFrame(rows).sort_values("abs_r", ascending=False).reset_index(drop=True)


        def compare_predictors(df, predictors, outcomes):
            rows = []
            for outcome in outcomes:
                scores = []
                for predictor in predictors:
                    subset = df[[predictor, outcome]].dropna()
                    if len(subset) < 4:
                        continue
                    r, p = stats.pearsonr(subset[predictor], subset[outcome])
                    scores.append((predictor, len(subset), r, p, abs(r)))
                if not scores:
                    continue
                scores = sorted(scores, key=lambda x: x[4], reverse=True)
                best = scores[0]
                second = scores[1] if len(scores) > 1 else None
                rows.append(
                    {
                        "outcome": outcome,
                        "stronger_predictor": best[0],
                        "n": best[1],
                        "pearson_r": round(best[2], 4),
                        "pearson_p": round(best[3], 4),
                        "runner_up": second[0] if second else None,
                        "runner_up_abs_r": round(second[4], 4) if second else None,
                    }
                )
            return pd.DataFrame(rows)


        def show_scatter(df, x, y, title, x_label, y_label, color="#34699a"):
            plot_df = df[["state", x, y]].dropna()
            fig, ax = plt.subplots(figsize=(8.2, 5.9))
            ax.scatter(plot_df[x], plot_df[y], color=color, s=52)
            for _, row in plot_df.iterrows():
                ax.annotate(row["state"], (row[x], row[y]), fontsize=7, xytext=(3, 2), textcoords="offset points")
            if len(plot_df) >= 4 and plot_df[x].nunique() > 1 and plot_df[y].nunique() > 1:
                r, p = stats.pearsonr(plot_df[x], plot_df[y])
                if p < 0.001:
                    p_text = "p < 0.001"
                else:
                    p_text = f"p = {p:.3f}"
                corr_text = f"Pearson r = {r:.2f}, {p_text}, n = {len(plot_df)}"
            else:
                corr_text = f"Pearson r not shown; n = {len(plot_df)}"
            if len(plot_df) >= 4 and plot_df[x].nunique() > 1:
                m, b = np.polyfit(plot_df[x], plot_df[y], 1)
                xs = np.linspace(plot_df[x].min(), plot_df[x].max(), 100)
                ax.plot(xs, m * xs + b, color="#444444", linewidth=1)
            ax.set_title(title)
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            ax.text(
                0.5,
                -0.18,
                corr_text,
                transform=ax.transAxes,
                ha="center",
                va="top",
                fontsize=10,
                bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "#d0d0d0", "alpha": 0.95},
            )
            fig.subplots_adjust(bottom=0.22)
            plt.show()


        def show_distribution_grid(df, columns, title):
            cols = [c for c in columns if c in df.columns]
            ncols = 2
            nrows = math.ceil(len(cols) / ncols)
            fig, axes = plt.subplots(nrows, ncols, figsize=(12, 3.8 * nrows))
            axes = np.array(axes).reshape(-1)
            for ax, col in zip(axes, cols):
                series = df[col].dropna()
                ax.hist(series, bins=min(8, max(4, len(series) // 2)), color="#4d7c8a", edgecolor="white")
                if len(series) > 0:
                    ax.axvline(series.median(), color="#8f3f3f", linestyle="--", linewidth=1, label="Median")
                    ax.legend(frameon=False, fontsize=8)
                ax.set_title(col)
            for ax in axes[len(cols):]:
                ax.axis("off")
            fig.suptitle(title, y=1.02, fontsize=14)
            fig.tight_layout()
            plt.show()


        def show_rank_bar(df, value_col, title, xlabel, ascending=True, top_n=10, color="#697a21"):
            plot_df = df[["state", value_col]].dropna().sort_values(value_col, ascending=ascending).head(top_n)
            fig, ax = plt.subplots(figsize=(8.5, 0.45 * len(plot_df) + 2))
            ax.barh(plot_df["state"], plot_df[value_col], color=color)
            if ascending:
                ax.invert_yaxis()
            ax.set_title(title)
            ax.set_xlabel(xlabel)
            plt.show()


        def show_abs_correlation_compare(df, predictors, outcomes, title):
            rows = []
            for outcome in outcomes:
                for predictor in predictors:
                    subset = df[[predictor, outcome]].dropna()
                    if len(subset) < 4:
                        continue
                    r, _ = stats.pearsonr(subset[predictor], subset[outcome])
                    rows.append({"outcome": outcome, "predictor": predictor, "abs_r": abs(r)})
            frame = pd.DataFrame(rows)
            pivot = frame.pivot(index="outcome", columns="predictor", values="abs_r")
            pivot.plot(kind="barh", figsize=(9, 5.5), color=["#34699a", "#a65d03", "#8f3f3f"][: len(pivot.columns)])
            plt.title(title)
            plt.xlabel("Absolute Pearson correlation")
            plt.ylabel("")
            plt.legend(frameon=False)
            plt.show()
        """
    ),
    md(
        """
        ## Data Used, In Plain English

        The analysis uses cleaned NDAP-based datasets for ST literacy, literacy gap, enrolment, GER, dropout, poverty, labour outcomes, MGNREG distress, ST population share, village concentration, district low female literacy, and tribe-level literacy summaries. It also adds official UDISE+/OGD policy-support files for latest ST GER, ST Gender Parity Index, and ST scholarship release/utilization. MGNREG 100-plus-days indicators are derived from the existing MGNREG table.
        """
    ),
    code(
        """
        inventory[["short_name", "level", "years", "key_variables", "notes"]].rename(
            columns={
                "short_name": "dataset",
                "level": "data_level",
                "years": "year_coverage",
                "key_variables": "main_variables",
                "notes": "how_used",
            }
        )
        """
    ),
    md(
        """
        ## Coverage Caveat

        Poverty is the biggest missing-data problem. GER also needs care because some historical GER values are ratio-style and can exceed 100, so the notebook uses cleaned GER fields for visual work.
        """
    ),
    code(
        """
        coverage = pd.read_csv(OUTPUTS / "eda" / "tables" / "data_quality_core_variable_coverage.csv")
        coverage.query(
            "table == 'high_st_master' and column in ['st_literacy_rate_pct','literacy_gap_pct','dropout_secondary_pct','ger_classes_ix_xii','ger_latest_ix_xii_avg','gpi_ix_xii_avg','scholarship_total_release_2023_24_lakh_per_100k_st_pop','employment_lfpr_person_per_1000','employment_wpr_person_per_1000','employment_pu_person_per_1000','st_bpl_mean_pct','mgnreg_sought_not_received_per_1000','mgnreg_work_100_plus_days_per_1000','low_literacy_district_count']"
        )[["column", "non_missing", "missing", "missing_pct", "min", "median", "max"]].sort_values("missing_pct", ascending=False)
        """
    ),
    md(
        """
        ## Distribution EDA

        Before asking relationship questions, it helps to see how the variables are distributed across high-ST states.
        """
    ),
    code(
        """
        high_st[[
            "st_literacy_rate_pct",
            "literacy_gap_pct",
            "dropout_secondary_pct",
            "ger_classes_ix_xii_clean",
            "ger_latest_ix_xii_avg_clean",
            "gpi_ix_xii_avg_clean",
            "employment_wpr_person_per_1000",
            "employment_pu_person_per_1000",
            "st_bpl_mean_pct",
            "mgnreg_sought_not_received_per_1000",
            "mgnreg_work_100_plus_days_per_1000",
            "scholarship_total_release_2023_24_lakh_per_100k_st_pop",
        ]].describe().T
        """
    ),
    code(
        """
        show_distribution_grid(
            high_st,
            ["st_literacy_rate_pct", "literacy_gap_pct", "dropout_secondary_pct", "ger_latest_ix_xii_avg_clean", "gpi_ix_xii_avg_clean"],
            "Education indicator distributions across high-ST states",
        )

        show_distribution_grid(
            high_st,
            ["employment_wpr_person_per_1000", "employment_pu_person_per_1000", "st_bpl_mean_pct", "mgnreg_sought_not_received_per_1000", "mgnreg_work_100_plus_days_per_1000", "scholarship_total_release_2023_24_lakh_per_100k_st_pop"],
            "Livelihood and distress indicator distributions across high-ST states",
        )
        """
    ),
    md(
        """
        ## Q1. How Do ST Literacy And Literacy Gaps Describe Long-Term Educational Exclusion?

        Literacy is not the same as current schooling quality. It mainly tells us whether people have basic reading and writing ability. In this project, literacy is used as a **long-term attainment and exclusion indicator**, while GER and dropout are used as the schooling-pipeline indicators.
        """
    ),
    code(
        """
        q1 = corr_frame(
            high_st,
            ["st_literacy_rate_pct", "literacy_gap_pct"],
            [
                "employment_lfpr_person_per_1000",
                "employment_wpr_person_per_1000",
                "employment_pu_person_per_1000",
                "st_bpl_mean_pct",
            ],
            "high_st_states",
        )
        q1
        """
    ),
    code(
        """
        show_scatter(high_st, "st_literacy_rate_pct", "employment_wpr_person_per_1000", "Q1: ST literacy and work participation", "ST literacy rate (%)", "WPR (NDAP ratio-style value)")
        show_scatter(high_st, "literacy_gap_pct", "employment_wpr_person_per_1000", "Q1: Literacy gap and work participation", "Literacy gap (percentage points)", "WPR (NDAP ratio-style value)", color="#a65d03")
        show_scatter(high_st, "st_literacy_rate_pct", "st_bpl_mean_pct", "Q1: ST literacy and ST poverty", "ST literacy rate (%)", "Mean ST poverty rate (%)")
        show_scatter(high_st, "literacy_gap_pct", "st_bpl_mean_pct", "Q1: Literacy gap and ST poverty", "Literacy gap (percentage points)", "Mean ST poverty rate (%)", color="#a65d03")
        """
    ),
    md(
        """
        **Reading Q1:** literacy and the literacy gap are useful for showing long-term exclusion, but they should not be treated as complete measures of educational progress. The current schooling question needs GER and dropout.
        """
    ),
    md(
        """
        ## Q2. Do States With Stronger ST Schooling Participation Also Have Lower Dropout And Lower ST Poverty?

        This is the more direct schooling question. GER captures participation/enrolment; dropout captures retention; poverty tests whether stronger schooling participation is associated with better economic conditions.
        """
    ),
    code(
        """
        q2 = corr_frame(
            high_st,
            [
                "ger_classes_i_viii_clean",
                "ger_classes_ix_xii_clean",
                "ger_classes_ix_xii_girls_clean",
                "ger_classes_ix_xii_gpi_clean",
                "ger_latest_ix_xii_avg_clean",
                "ger_latest_ix_xii_avg_girls_clean",
                "gpi_ix_xii_avg_clean",
            ],
            ["dropout_primary_pct", "dropout_upper_primary_pct", "dropout_secondary_pct", "st_bpl_mean_pct"],
            "high_st_states",
        )
        q2
        """
    ),
    code(
        """
        show_scatter(high_st, "ger_classes_i_viii_clean", "dropout_primary_pct", "Q2: GER I-VIII and primary dropout", "GER I-VIII (cleaned)", "Primary dropout (%)")
        show_scatter(high_st, "ger_classes_ix_xii_clean", "dropout_secondary_pct", "Q2: GER IX-XII and secondary dropout", "GER IX-XII (cleaned)", "Secondary dropout (%)", color="#8f3f3f")
        show_scatter(high_st, "ger_classes_ix_xii_girls_clean", "dropout_secondary_pct", "Q2: Girls GER IX-XII and secondary dropout", "Girls GER IX-XII (cleaned)", "Secondary dropout (%)", color="#5f5b8f")
        show_scatter(high_st, "ger_latest_ix_xii_avg_clean", "dropout_secondary_pct", "Q2: Latest GER IX-XII average and secondary dropout", "Latest GER IX-XII average", "Secondary dropout (%)", color="#4d7c8a")
        show_scatter(high_st, "ger_latest_ix_xii_avg_girls_clean", "dropout_secondary_pct", "Q2: Latest girls GER IX-XII average and secondary dropout", "Latest girls GER IX-XII average", "Secondary dropout (%)", color="#5f5b8f")
        show_scatter(high_st, "ger_classes_ix_xii_clean", "st_bpl_mean_pct", "Q2: GER IX-XII and ST poverty", "GER IX-XII (cleaned)", "Mean ST poverty rate (%)", color="#7f4f24")
        show_scatter(high_st, "dropout_secondary_pct", "st_bpl_mean_pct", "Q2: Secondary dropout and ST poverty", "Secondary dropout (%)", "Mean ST poverty rate (%)", color="#a65d03")
        """
    ),
    code(
        """
        high_dropout_high_poverty = high_st[
            high_st["dropout_secondary_pct"].notna() & high_st["st_bpl_mean_pct"].notna()
        ][[
            "state",
            "ger_classes_ix_xii_clean",
            "ger_latest_ix_xii_avg_clean",
            "ger_classes_ix_xii_girls_clean",
            "ger_latest_ix_xii_avg_girls_clean",
            "gpi_ix_xii_avg_clean",
            "ger_classes_ix_xii_gpi_clean",
            "dropout_secondary_pct",
            "st_bpl_mean_pct",
            "employment_wpr_person_per_1000",
            "mgnreg_sought_not_received_per_1000",
        ]].sort_values(["dropout_secondary_pct", "st_bpl_mean_pct"], ascending=[False, False])

        high_dropout_high_poverty
        """
    ),
    md(
        """
        **Reading Q2:** this is stronger than using literacy alone. If GER is high but dropout or poverty is also high, schooling access is not translating into stable progression or economic security.
        """
    ),
    md(
        """
        ## Q3. Are Enrolment And GER Enough To Indicate Educational Progress, Or Do Dropout Rates Tell A Different Story?
        """
    ),
    code(
        """
        q2 = corr_frame(
            high_st,
            ["ger_classes_i_viii_clean", "ger_classes_ix_xii_clean", "ger_latest_ix_xii_avg_clean", "ger_latest_ix_xii_avg_girls_clean", "gpi_ix_xii_avg_clean", "primary_total"],
            ["dropout_secondary_pct", "employment_wpr_person_per_1000", "st_bpl_mean_pct"],
            "high_st_states",
        )
        q2
        """
    ),
    code(
        """
        show_scatter(high_st, "ger_classes_ix_xii_clean", "dropout_secondary_pct", "Q3: GER IX-XII and secondary dropout", "GER IX-XII (cleaned)", "Secondary dropout (%)")
        show_scatter(high_st, "ger_latest_ix_xii_avg_clean", "dropout_secondary_pct", "Q3: Latest GER IX-XII average and secondary dropout", "Latest GER IX-XII average", "Secondary dropout (%)", color="#4d7c8a")
        show_scatter(high_st, "ger_classes_ix_xii_gpi_clean", "dropout_secondary_pct", "Q3: GER gender parity and secondary dropout", "GER IX-XII GPI (girls/boys)", "Secondary dropout (%)", color="#5f5b8f")
        show_scatter(high_st, "gpi_ix_xii_avg_clean", "dropout_secondary_pct", "Q3: Official GPI IX-XII and secondary dropout", "Official GPI IX-XII average", "Secondary dropout (%)", color="#5f5b8f")
        show_scatter(high_st, "ger_classes_ix_xii_clean", "employment_wpr_person_per_1000", "Q3: GER IX-XII and work participation", "GER IX-XII (cleaned)", "WPR (NDAP ratio-style value)", color="#7f4f24")
        show_scatter(high_st, "primary_total", "dropout_secondary_pct", "Q3: Primary enrolment and secondary dropout", "Primary ST enrolment total", "Secondary dropout (%)", color="#8f3f3f")
        """
    ),
    code(
        """
        med_ger = high_st["ger_latest_ix_xii_avg_clean"].median()
        med_dropout = high_st["dropout_secondary_pct"].median()
        med_wpr = high_st["employment_wpr_person_per_1000"].median()

        high_ger_high_dropout = high_st[
            (high_st["ger_latest_ix_xii_avg_clean"] >= med_ger)
            & (high_st["dropout_secondary_pct"] >= med_dropout)
        ][["state", "ger_latest_ix_xii_avg_clean", "gpi_ix_xii_avg_clean", "dropout_secondary_pct", "employment_wpr_person_per_1000", "st_bpl_mean_pct"]].sort_values(["dropout_secondary_pct", "ger_latest_ix_xii_avg_clean"], ascending=[False, False])

        high_ger_low_wpr = high_st[
            (high_st["ger_latest_ix_xii_avg_clean"] >= med_ger)
            & (high_st["employment_wpr_person_per_1000"] <= med_wpr)
        ][["state", "ger_latest_ix_xii_avg_clean", "gpi_ix_xii_avg_clean", "dropout_secondary_pct", "employment_wpr_person_per_1000", "st_bpl_mean_pct"]].sort_values("employment_wpr_person_per_1000")

        display(Markdown("### States with relatively high GER IX-XII but also high dropout"))
        display(high_ger_high_dropout)
        display(Markdown("### States with relatively high GER IX-XII but weak work participation"))
        display(high_ger_low_wpr)
        """
    ),
    md(
        """
        **Reading Q3:** access-style indicators are not enough. Some states show reasonable GER but still struggle with retention or weak work participation.
        """
    ),
    md(
        """
        ## Q4. Is The Literacy Gap More Informative Than ST Literacy Alone In Explaining Disadvantage?
        """
    ),
    code(
        """
        q3 = compare_predictors(
            high_st,
            ["st_literacy_rate_pct", "literacy_gap_pct"],
            ["st_bpl_mean_pct", "employment_wpr_person_per_1000", "employment_pu_person_per_1000", "dropout_secondary_pct"],
        )
        q3
        """
    ),
    code(
        """
        show_abs_correlation_compare(
            high_st,
            ["st_literacy_rate_pct", "literacy_gap_pct"],
            ["st_bpl_mean_pct", "employment_wpr_person_per_1000", "employment_pu_person_per_1000", "dropout_secondary_pct"],
            "Q4: Which matters more - ST literacy or literacy gap?",
        )
        """
    ),
    md(
        """
        **Reading Q4:** the literacy gap is often at least as informative as absolute ST literacy, especially for work participation, unemployment, and dropout. It captures exclusion, not just low level.
        """
    ),
    md(
        """
        ## Q5. Are There States Where Educational Indicators Look Reasonable, But Employment Or Poverty Outcomes Remain Weak?

        We label these as **states with an education-livelihood mismatch**.
        """
    ),
    code(
        """
        med_lit = high_st["st_literacy_rate_pct"].median()
        med_ger = high_st["ger_latest_ix_xii_avg_clean"].median()
        med_wpr = high_st["employment_wpr_person_per_1000"].median()
        med_unemp = high_st["employment_pu_person_per_1000"].median()
        med_pov = high_st["st_bpl_mean_pct"].median()
        med_mgnreg = high_st["mgnreg_sought_not_received_per_1000"].median()
        med_mgnreg_100 = high_st["mgnreg_work_100_plus_days_per_1000"].median()

        mismatch_rows = []
        for _, row in high_st.iterrows():
            education_ok = (
                (pd.notna(row["st_literacy_rate_pct"]) and row["st_literacy_rate_pct"] >= med_lit)
                or (pd.notna(row["ger_latest_ix_xii_avg_clean"]) and row["ger_latest_ix_xii_avg_clean"] >= med_ger)
            )
            distress = []
            if pd.notna(row["employment_wpr_person_per_1000"]) and row["employment_wpr_person_per_1000"] <= med_wpr:
                distress.append("low_wpr")
            if pd.notna(row["employment_pu_person_per_1000"]) and row["employment_pu_person_per_1000"] >= med_unemp:
                distress.append("high_unemployment")
            if pd.notna(row["st_bpl_mean_pct"]) and row["st_bpl_mean_pct"] >= med_pov:
                distress.append("high_poverty")
            if pd.notna(row["mgnreg_sought_not_received_per_1000"]) and row["mgnreg_sought_not_received_per_1000"] >= med_mgnreg:
                distress.append("high_mgnreg_unmet")
            if pd.notna(row["mgnreg_work_100_plus_days_per_1000"]) and row["mgnreg_work_100_plus_days_per_1000"] >= med_mgnreg_100:
                distress.append("high_mgnreg_100_plus")

            if education_ok and len(distress) >= 2:
                mismatch_rows.append(
                    {
                        "state": row["state"],
                        "st_literacy_rate_pct": row["st_literacy_rate_pct"],
                        "ger_ix_xii": row["ger_classes_ix_xii_clean"],
                        "ger_latest_ix_xii_avg": row["ger_latest_ix_xii_avg_clean"],
                        "employment_wpr_person_per_1000": row["employment_wpr_person_per_1000"],
                        "employment_pu_person_per_1000": row["employment_pu_person_per_1000"],
                        "st_bpl_mean_pct": row["st_bpl_mean_pct"],
                        "mgnreg_sought_not_received_per_1000": row["mgnreg_sought_not_received_per_1000"],
                        "mgnreg_work_100_plus_days_per_1000": row["mgnreg_work_100_plus_days_per_1000"],
                        "distress_count": len(distress),
                        "distress_flags": ", ".join(distress),
                    }
                )

        mismatch_df = pd.DataFrame(mismatch_rows).sort_values(["distress_count", "state"], ascending=[False, True])
        mismatch_df
        """
    ),
    code(
        """
        if not mismatch_df.empty:
            fig, ax = plt.subplots(figsize=(8.5, 0.45 * len(mismatch_df) + 2))
            ax.barh(mismatch_df["state"], mismatch_df["distress_count"], color="#5f5b8f")
            ax.invert_yaxis()
            ax.set_title("Q5: Education-livelihood mismatch states")
            ax.set_xlabel("Number of distress flags")
            plt.show()
        """
    ),
    md(
        """
        **Reading Q5:** this is one of the most policy-useful outputs. These states do not fit a simple low-education-equals-bad-outcomes story.
        """
    ),
    md(
        """
        ## Q6. Does MGNREG Unmet Demand Reflect Deeper Livelihood Distress Among ST Households?
        """
    ),
    code(
        """
        q5 = corr_frame(
            high_st,
            [
                "mgnreg_sought_not_received_per_1000",
                "mgnreg_work_100_plus_days_per_1000",
                "mgnreg_100_plus_share_of_work_received_pct",
                "mgnreg_average_days_worked",
                "mgnreg_job_card_households_per_1000",
            ],
            ["st_bpl_mean_pct", "dropout_secondary_pct", "employment_wpr_person_per_1000", "st_literacy_rate_pct"],
            "high_st_states",
        )
        q5
        """
    ),
    code(
        """
        show_scatter(high_st, "mgnreg_sought_not_received_per_1000", "st_bpl_mean_pct", "Q6: MGNREG unmet demand and ST poverty", "MGNREG sought-not-received per 1000", "Mean ST poverty rate (%)")
        show_scatter(high_st, "mgnreg_sought_not_received_per_1000", "dropout_secondary_pct", "Q6: MGNREG unmet demand and secondary dropout", "MGNREG sought-not-received per 1000", "Secondary dropout (%)", color="#8f3f3f")
        show_scatter(high_st, "mgnreg_work_100_plus_days_per_1000", "dropout_secondary_pct", "Q6: MGNREG 100-plus-days and secondary dropout", "MGNREG 100-plus-days per 1000", "Secondary dropout (%)", color="#5f5b8f")
        show_scatter(high_st, "mgnreg_sought_not_received_per_1000", "st_literacy_rate_pct", "Q6: MGNREG unmet demand and ST literacy", "MGNREG sought-not-received per 1000", "ST literacy rate (%)", color="#a65d03")
        show_rank_bar(high_st, "mgnreg_sought_not_received_per_1000", "Q6: Highest MGNREG unmet demand", "Sought-not-received per 1000", ascending=False, top_n=10, color="#7f4f24")
        show_rank_bar(high_st, "mgnreg_work_100_plus_days_per_1000", "Q6: Highest MGNREG 100-plus-days among ST households", "100-plus-days per 1000", ascending=False, top_n=10, color="#4d7c8a")
        """
    ),
    md(
        """
        **Reading Q6:** unmet MGNREG demand behaves like a distress indicator more clearly than simple job-card counts do.
        """
    ),
    md(
        """
        ## Q7. Are Scholarship-Supported States Seeing Lower Secondary Dropout Among ST Students?

        This is a policy-response question. The scholarship dataset measures pre-matric and post-matric ST scholarship release and utilization in Rs lakh. Because larger states naturally receive larger totals, the notebook also uses release per 100,000 ST population.
        """
    ),
    code(
        """
        q7 = corr_frame(
            high_st,
            [
                "scholarship_total_release_2023_24_lakh",
                "scholarship_total_utilized_2023_24_lakh",
                "scholarship_utilization_2023_24_pct",
                "scholarship_total_release_2023_24_lakh_per_100k_st_pop",
                "scholarship_cumulative_release_lakh_per_100k_st_pop",
            ],
            ["dropout_secondary_pct", "st_bpl_mean_pct", "ger_latest_ix_xii_avg_clean"],
            "high_st_states",
        )
        q7
        """
    ),
    code(
        """
        show_scatter(high_st, "scholarship_total_release_2023_24_lakh_per_100k_st_pop", "dropout_secondary_pct", "Q7: ST scholarship release per ST population and secondary dropout", "2023-24 scholarship release lakh per 100k ST pop", "Secondary dropout (%)", color="#7f4f24")
        show_scatter(high_st, "scholarship_utilization_2023_24_pct", "dropout_secondary_pct", "Q7: Scholarship utilization and secondary dropout", "2023-24 scholarship utilization (%)", "Secondary dropout (%)", color="#5f5b8f")
        show_scatter(high_st, "scholarship_cumulative_release_lakh_per_100k_st_pop", "st_bpl_mean_pct", "Q7: Cumulative scholarship release and ST poverty", "2019-24 release lakh per 100k ST pop", "Mean ST poverty rate (%)", color="#a65d03")
        """
    ),
    code(
        """
        high_st[[
            "state",
            "dropout_secondary_pct",
            "st_bpl_mean_pct",
            "scholarship_total_release_2023_24_lakh",
            "scholarship_total_utilized_2023_24_lakh",
            "scholarship_utilization_2023_24_pct",
            "scholarship_total_release_2023_24_lakh_per_100k_st_pop",
            "scholarship_cumulative_release_lakh_per_100k_st_pop",
        ]].sort_values("scholarship_total_release_2023_24_lakh_per_100k_st_pop", ascending=False)
        """
    ),
    md(
        """
        **Reading Q7:** if scholarship release is higher in high-dropout states, that does not mean scholarships cause dropout. It means support may be targeted toward need, or that money alone is not enough to solve retention.
        """
    ),
    md(
        """
        ## Q8. Do States With High ST Schooling Participation Still Depend Heavily On MGNREG?

        This directly tests the education-livelihood mismatch idea. If GER is high but MGNREG dependence or unmet demand is also high, schooling participation is not yet translating into secure livelihoods.
        """
    ),
    code(
        """
        q8 = corr_frame(
            high_st,
            ["ger_latest_ix_xii_avg_clean", "ger_latest_ix_xii_avg_girls_clean", "gpi_ix_xii_avg_clean", "dropout_secondary_pct"],
            [
                "mgnreg_job_card_households_per_1000",
                "mgnreg_work_100_plus_days_per_1000",
                "mgnreg_100_plus_share_of_work_received_pct",
                "mgnreg_sought_not_received_per_1000",
            ],
            "high_st_states",
        )
        q8
        """
    ),
    code(
        """
        show_scatter(high_st, "ger_latest_ix_xii_avg_clean", "mgnreg_work_100_plus_days_per_1000", "Q8: Latest GER IX-XII average and MGNREG 100-plus-days", "Latest GER IX-XII average", "MGNREG 100-plus-days per 1000", color="#7f4f24")
        show_scatter(high_st, "ger_latest_ix_xii_avg_clean", "mgnreg_sought_not_received_per_1000", "Q8: Latest GER IX-XII average and MGNREG unmet demand", "Latest GER IX-XII average", "MGNREG sought-not-received per 1000", color="#a65d03")
        show_scatter(high_st, "dropout_secondary_pct", "mgnreg_work_100_plus_days_per_1000", "Q8: Secondary dropout and MGNREG 100-plus-days", "Secondary dropout (%)", "MGNREG 100-plus-days per 1000", color="#5f5b8f")
        """
    ),
    code(
        """
        high_ger_mgnreg = high_st[
            (high_st["ger_latest_ix_xii_avg_clean"] >= high_st["ger_latest_ix_xii_avg_clean"].median())
            & (
                (high_st["mgnreg_work_100_plus_days_per_1000"] >= high_st["mgnreg_work_100_plus_days_per_1000"].median())
                | (high_st["mgnreg_sought_not_received_per_1000"] >= high_st["mgnreg_sought_not_received_per_1000"].median())
            )
        ][[
            "state",
            "ger_latest_ix_xii_avg_clean",
            "ger_latest_ix_xii_avg_girls_clean",
            "gpi_ix_xii_avg_clean",
            "dropout_secondary_pct",
            "mgnreg_work_100_plus_days_per_1000",
            "mgnreg_sought_not_received_per_1000",
            "employment_wpr_person_per_1000",
            "st_bpl_mean_pct",
        ]].sort_values(["mgnreg_work_100_plus_days_per_1000", "mgnreg_sought_not_received_per_1000"], ascending=[False, False])

        high_ger_mgnreg
        """
    ),
    md(
        """
        **Reading Q8:** the correlations are not strong enough to claim a direct relationship, but the state table is useful. It identifies places where reasonable ST schooling participation coexists with MGNREG dependence or unmet demand.
        """
    ),
    md(
        """
        ## Q9. Does Gender Parity In Enrolment Translate Into Better Female Literacy And Work Outcomes?

        A GPI value near 1 means parity in enrolment. This section uses the official ST Gender Parity Index dataset and asks whether enrolment parity lines up with later female literacy and work outcomes.
        """
    ),
    code(
        """
        q9 = corr_frame(
            high_st,
            ["gpi_ix_xii_avg_clean", "gpi_secondary_clean", "gpi_higher_secondary_clean", "ger_latest_ix_xii_avg_girls_clean", "female_literacy_gap_pct"],
            ["tribe_weighted_literacy_female_pct", "employment_wpr_female_per_1000", "low_literacy_district_count"],
            "high_st_states",
        )
        q9
        """
    ),
    code(
        """
        show_scatter(high_st, "gpi_ix_xii_avg_clean", "tribe_weighted_literacy_female_pct", "Q9: Official GPI IX-XII and female ST literacy", "Official GPI IX-XII average", "Female ST literacy (%)", color="#5f5b8f")
        show_scatter(high_st, "gpi_ix_xii_avg_clean", "employment_wpr_female_per_1000", "Q9: Official GPI IX-XII and female work participation", "Official GPI IX-XII average", "Female WPR (NDAP ratio-style value)", color="#7f4f24")
        show_scatter(high_st, "ger_latest_ix_xii_avg_girls_clean", "employment_wpr_female_per_1000", "Q9: Latest girls GER IX-XII and female work participation", "Latest girls GER IX-XII average", "Female WPR (NDAP ratio-style value)", color="#a65d03")
        """
    ),
    code(
        """
        high_st[[
            "state",
            "gpi_secondary",
            "gpi_higher_secondary",
            "gpi_ix_xii_avg",
            "ger_latest_ix_xii_avg_girls",
            "tribe_weighted_literacy_female_pct",
            "female_literacy_gap_pct",
            "employment_wpr_female_per_1000",
            "low_literacy_district_count",
        ]].sort_values("gpi_ix_xii_avg", ascending=False)
        """
    ),
    md(
        """
        **Reading Q9:** official enrolment parity does not automatically translate into female work participation. If GPI is near parity but female WPR remains weak, the state has a gendered education-livelihood mismatch rather than only an enrolment problem.
        """
    ),
    md(
        """
        ## Q10. Does Secondary Dropout Appear To Be A Stronger Warning Signal Of Poverty And Weak Labour Outcomes Than Literacy Alone?
        """
    ),
    code(
        """
        q6 = compare_predictors(
            high_st,
            ["st_literacy_rate_pct", "dropout_secondary_pct"],
            [
                "st_bpl_mean_pct",
                "employment_wpr_person_per_1000",
                "employment_pu_person_per_1000",
                "mgnreg_sought_not_received_per_1000",
                "mgnreg_work_100_plus_days_per_1000",
            ],
        )
        q6
        """
    ),
    code(
        """
        show_abs_correlation_compare(
            high_st,
            ["st_literacy_rate_pct", "dropout_secondary_pct"],
            ["st_bpl_mean_pct", "employment_wpr_person_per_1000", "employment_pu_person_per_1000", "mgnreg_sought_not_received_per_1000", "mgnreg_work_100_plus_days_per_1000"],
            "Q10: Literacy vs dropout as warning signals",
        )
        """
    ),
    md(
        """
        **Reading Q10:** secondary dropout is often the stronger warning signal for poverty, weak work participation, and MGNREG distress.
        """
    ),
    md(
        """
        ## Q11. Do High-ST-Share States Systematically Perform Worse, Or Do Outcomes Vary Substantially?
        """
    ),
    code(
        """
        q7 = corr_frame(
            all_states,
            ["st_share_state_population_pct"],
            ["st_literacy_rate_pct", "dropout_secondary_pct", "st_bpl_mean_pct", "employment_wpr_person_per_1000"],
            "all_states",
        )
        q7
        """
    ),
    code(
        """
        show_scatter(all_states, "st_share_state_population_pct", "st_literacy_rate_pct", "Q11: ST share and ST literacy", "ST share in state population (%)", "ST literacy rate (%)")
        show_scatter(all_states, "st_share_state_population_pct", "st_bpl_mean_pct", "Q11: ST share and ST poverty", "ST share in state population (%)", "Mean ST poverty rate (%)", color="#7f4f24")
        """
    ),
    md(
        """
        **Reading Q11:** concentration alone does not explain everything. High-ST states differ sharply in education and livelihood outcomes.
        """
    ),
    md(
        """
        ## Q12. Do States With High Concentrations Of ST Villages Show Different Education And Livelihood Outcomes?

        Raw village counts are not directly comparable across states, so the notebook uses villages per 100,000 ST population.
        """
    ),
    code(
        """
        q8 = corr_frame(
            high_st,
            ["villages_gt50_per_100k_st_pop"],
            ["st_literacy_rate_pct", "dropout_secondary_pct", "st_bpl_mean_pct", "mgnreg_sought_not_received_per_1000"],
            "high_st_states",
        )
        q8
        """
    ),
    code(
        """
        show_scatter(high_st, "villages_gt50_per_100k_st_pop", "mgnreg_sought_not_received_per_1000", "Q12: ST village concentration and MGNREG distress", "Villages >50% ST per 100k ST population", "MGNREG sought-not-received per 1000")
        show_scatter(high_st, "villages_gt50_per_100k_st_pop", "st_literacy_rate_pct", "Q12: ST village concentration and ST literacy", "Villages >50% ST per 100k ST population", "ST literacy rate (%)", color="#a65d03")
        show_rank_bar(high_st, "villages_gt50_per_100k_st_pop", "Q12: Highest normalized ST village concentration", "Villages >50% ST per 100k ST population", ascending=False, top_n=10, color="#4d7c8a")
        """
    ),
    md(
        """
        **Reading Q12:** normalized village concentration looks more connected to MGNREG distress than to literacy alone. It is useful, but secondary.
        """
    ),
    md(
        """
        ## Q13. Are There Gender-Specific Disadvantages Hidden Behind State Averages?
        """
    ),
    code(
        """
        q9 = corr_frame(
            high_st,
            ["female_literacy_gap_pct", "tribe_weighted_literacy_female_pct"],
            ["employment_wpr_female_per_1000", "st_bpl_mean_pct", "low_literacy_district_count"],
            "high_st_states",
        )
        q9
        """
    ),
    code(
        """
        show_rank_bar(high_st, "female_literacy_gap_pct", "Q13: Largest female literacy gaps within ST populations", "Male-female literacy gap (percentage points)", ascending=False, top_n=10, color="#8f3f3f")
        show_scatter(high_st, "female_literacy_gap_pct", "employment_wpr_female_per_1000", "Q13: Female literacy gap and female work participation", "Female literacy gap (percentage points)", "Female WPR (NDAP ratio-style value)", color="#5f5b8f")
        """
    ),
    code(
        """
        high_st[[
            "state",
            "tribe_weighted_literacy_male_pct",
            "tribe_weighted_literacy_female_pct",
            "female_literacy_gap_pct",
            "low_literacy_district_count",
            "employment_wpr_female_per_1000",
        ]].sort_values("female_literacy_gap_pct", ascending=False).head(10)
        """
    ),
    md(
        """
        **Reading Q13:** the strongest gender result is descriptive: Rajasthan, Odisha, Jharkhand, Jammu and Kashmir, and Madhya Pradesh show particularly large female literacy disadvantages.
        """
    ),
    md(
        """
        ## Policy Synthesis

        The most useful policy story from this notebook is:

        - **Odisha and Madhya Pradesh:** compound education and livelihood distress.
        - **Jharkhand and Chhattisgarh:** livelihood distress and MGNREG-linked vulnerability remain central.
        - **Maharashtra and Gujarat:** access or GER alone is not enough because retention and poverty concerns remain.
        - **Rajasthan and Jammu and Kashmir:** literacy inequality and gender disadvantage deserve special targeting.
        - **Assam, Nagaland, Arunachal Pradesh, Tripura:** education-livelihood mismatch patterns suggest labour-side or household-distress constraints even when education indicators are not the weakest.
        """
    ),
    code(
        """
        policy_profiles = pd.read_csv(OUTPUTS / "eda" / "tables" / "state_policy_recommendations.csv")
        policy_profiles[[
            "state",
            "overall_priority_rank",
            "policy_priority_category",
            "evidence_flags",
            "recommended_policy_focus",
        ]].head(12)
        """
    ),
    md(
        """
        ## Database Requirement

        The project guideline requires database storage and querying. The project uses SQLite with cleaned tables, a final state analysis table, and a sparse state-year fact table.
        """
    ),
    code(
        """
        with sqlite3.connect(DB_PATH) as conn:
            tables = pd.read_sql_query(
                "select name from sqlite_master where type='table' order by name", conn
            )
            top_priority_query = pd.read_sql_query(
                '''
                select state, overall_priority_rank, overall_priority_score, policy_priority_category
                from state_analysis_dataset
                where is_high_st_share_state = 1
                order by overall_priority_score desc
                limit 10
                ''',
                conn,
            )

        display(tables)
        display(top_priority_query)
        """
    ),
    md(
        """
        ## Final Takeaway

        This project is strongest when framed as:

        **How do differences in ST educational outcomes relate to livelihood distress, and what policy package does each state type need?**

        That keeps the analysis relationship-based, policy-relevant, and honest about where the data are strongest.
        """
    ),
]


def main() -> None:
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    nb = nbf.v4.new_notebook()
    nb["cells"] = cells
    nb["metadata"] = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "pygments_lexer": "ipython3",
        },
    }
    nbf.write(nb, NOTEBOOK_PATH)
    print(f"Wrote {NOTEBOOK_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
