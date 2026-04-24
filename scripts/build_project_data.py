from __future__ import annotations

import re
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs"
CLEAN_DIR = OUTPUT_DIR / "cleaned"
ANALYSIS_DIR = OUTPUT_DIR / "analysis"
FIGURE_DIR = OUTPUT_DIR / "figures"
DB_PATH = OUTPUT_DIR / "st_education_project.sqlite"


ALL_INDIA_NAMES = {"all india", "india", "total"}


def ensure_dirs() -> None:
    for path in [OUTPUT_DIR, CLEAN_DIR, ANALYSIS_DIR, FIGURE_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def parse_year(value) -> pd.NA | int:
    if pd.isna(value):
        return pd.NA
    matches = re.findall(r"\b(?:19|20)\d{2}\b", str(value))
    if not matches:
        return pd.NA
    return int(matches[-1])


def normalize_state(value) -> str | float:
    if pd.isna(value):
        return np.nan
    text = re.sub(r"\s+", " ", str(value).strip())
    key = re.sub(r"[^A-Z0-9]+", " ", text.upper()).strip()
    mapping = {
        "ANDAMAN NICOBAR ISLANDS": "Andaman and Nicobar Islands",
        "ANDAMAN AND NICOBAR ISLANDS": "Andaman and Nicobar Islands",
        "ANDAMAN NICOBAR": "Andaman and Nicobar Islands",
        "ANDAMAN AND NICOBAR": "Andaman and Nicobar Islands",
        "ANDAMAN NICOBAR ISLAND": "Andaman and Nicobar Islands",
        "A N ISLANDS": "Andaman and Nicobar Islands",
        "DADRA NAGAR HAVELI": "Dadra and Nagar Haveli",
        "DADRA AND NAGAR HAVELI": "Dadra and Nagar Haveli",
        "DAMAN DIU": "Daman and Diu",
        "DAMAN AND DIU": "Daman and Diu",
        "DADRA NAGAR HAVELI DAMAN DIU": "Dadra and Nagar Haveli and Daman and Diu",
        "DADRA AND NAGAR HAVELI AND DAMAN AND DIU": "Dadra and Nagar Haveli and Daman and Diu",
        "THE DADRA AND NAGAR HAVELI AND DAMAN AND DIU": "Dadra and Nagar Haveli and Daman and Diu",
        "JAMMU KASHMIR": "Jammu and Kashmir",
        "JAMMU AND KASHMIR": "Jammu and Kashmir",
        "NCT OF DELHI": "Delhi",
        "DELHI": "Delhi",
        "ORISSA": "Odisha",
        "PONDICHERRY": "Puducherry",
        "PUDUCHERRY": "Puducherry",
        "CHHATTIGARH": "Chhattisgarh",
        "CHHATTISGARH": "Chhattisgarh",
        "UTTARANCHAL": "Uttarakhand",
        "UTTARAKHAND": "Uttarakhand",
        "ALL INDIA": "All India",
        "INDIA": "India",
    }
    if key in mapping:
        return mapping[key]

    name = text.title()
    replacements = {
        " And ": " and ",
        " Of ": " of ",
        " The ": " the ",
        " Nct ": " NCT ",
    }
    for old, new in replacements.items():
        name = name.replace(old, new)
    return name


def numeric(series: pd.Series) -> pd.Series:
    if series.dtype == "object":
        cleaned = (
            series.astype(str)
            .str.replace(",", "", regex=False)
            .str.replace("%", "", regex=False)
            .replace({"nan": np.nan, "": np.nan, "NA": np.nan, "N/A": np.nan})
        )
        return pd.to_numeric(cleaned, errors="coerce")
    return pd.to_numeric(series, errors="coerce")


def col_like(df: pd.DataFrame, *tokens: str) -> str:
    lowered = {col: col.lower() for col in df.columns}
    for col, low in lowered.items():
        if all(token.lower() in low for token in tokens):
            return col
    raise KeyError(f"No column found with tokens {tokens}. Columns: {list(df.columns)}")


def clean_base(df: pd.DataFrame, state_col: str, year_col: str | None = "Year") -> pd.DataFrame:
    out = pd.DataFrame()
    out["state"] = df[state_col].map(normalize_state)
    if year_col and year_col in df.columns:
        out["year"] = df[year_col].map(parse_year).astype("Int64")
    return out


def drop_aggregate_rows(df: pd.DataFrame) -> pd.DataFrame:
    if "state" not in df.columns:
        return df
    mask = ~df["state"].astype(str).str.strip().str.lower().isin(ALL_INDIA_NAMES)
    return df.loc[mask].reset_index(drop=True)


def slug(text) -> str:
    value = str(text).lower()
    value = value.replace("%", "pct")
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return re.sub(r"_+", "_", value).strip("_")


def class_slug(value: str) -> str:
    text = str(value).lower()
    if "i-viii" in text or "i viii" in text:
        return "classes_i_viii"
    if "ix-xii" in text or "ix xii" in text:
        return "classes_ix_xii"
    if "xi-xii" in text or "xi xii" in text:
        return "classes_xi_xii"
    if "ix-x" in text or "ix x" in text:
        return "classes_ix_x"
    if "i-xii" in text or "i xii" in text:
        return "classes_i_xii"
    if "i-x" in text or "i x" in text:
        return "classes_i_x"
    return slug(value)


def gender_slug(value: str) -> str:
    text = str(value).strip().lower()
    return {"person": "person", "total": "total", "boys": "boys", "girls": "girls"}.get(text, slug(text))


def indicator_slug(value: str) -> str:
    text = str(value).lower()
    if "labour force" in text or "lfpr" in text:
        return "lfpr"
    if "worker population" in text or "wpr" in text:
        return "wpr"
    if "unemployed" in text or "pu" in text:
        return "pu"
    return slug(value)


def clean_literacy(df: pd.DataFrame) -> pd.DataFrame:
    out = clean_base(df, "State Name")
    if "Additional Info" in df.columns:
        out["additional_info"] = df["Additional Info"].replace({np.nan: ""})
    out["total_literacy_rate_pct"] = numeric(col_series(df, "literacy rate of total population"))
    out["st_literacy_rate_pct"] = numeric(col_series(df, "literacy rate of scheduled tribes"))
    out["literacy_gap_pct"] = numeric(col_series(df, "gap in literacy rate"))
    return drop_aggregate_rows(out)


def col_series(df: pd.DataFrame, *tokens: str) -> pd.Series:
    return df[col_like(df, *tokens)]


def clean_enrolment(df: pd.DataFrame) -> pd.DataFrame:
    out = clean_base(df, "State")
    columns = list(df.columns)
    rename = [
        "pre_primary_boys",
        "pre_primary_girls",
        "primary_boys",
        "primary_girls",
        "upper_primary_boys",
        "upper_primary_girls",
    ]
    for source, target in zip(columns[3:], rename):
        out[target] = numeric(df[source])
    out["pre_primary_total"] = out["pre_primary_boys"] + out["pre_primary_girls"]
    out["primary_total"] = out["primary_boys"] + out["primary_girls"]
    out["upper_primary_total"] = out["upper_primary_boys"] + out["upper_primary_girls"]
    out["pre_primary_female_share_pct"] = pct_share(out["pre_primary_girls"], out["pre_primary_total"])
    out["primary_female_share_pct"] = pct_share(out["primary_girls"], out["primary_total"])
    out["upper_primary_female_share_pct"] = pct_share(out["upper_primary_girls"], out["upper_primary_total"])
    return drop_aggregate_rows(out)


def pct_share(part: pd.Series, whole: pd.Series) -> pd.Series:
    return np.where(whole.replace(0, np.nan).notna(), part / whole.replace(0, np.nan) * 100, np.nan)


def valid_percent(series: pd.Series) -> pd.Series:
    values = numeric(series)
    return values.where(values.between(0, 100))


def clean_high_st_share(df: pd.DataFrame) -> pd.DataFrame:
    out = clean_base(df, "State")
    if "Additional Info" in df.columns:
        out["additional_info"] = df["Additional Info"].replace({np.nan: ""})
    out["st_share_state_population_pct"] = numeric(col_series(df, "scheduled tribes", "state to total population"))
    out["state_share_india_st_population_pct"] = numeric(col_series(df, "total st population in india"))
    return drop_aggregate_rows(out)


def clean_sc_st_residence(df: pd.DataFrame) -> pd.DataFrame:
    out = clean_base(df, "State")
    out["residence_type"] = df["Name Of The Region"].astype(str).str.strip().str.lower()
    if "Additional Info" in df.columns:
        out["additional_info"] = df["Additional Info"].replace({np.nan: ""})
    out["sc_population"] = numeric(col_series(df, "scheduled caste"))
    out["st_population"] = numeric(col_series(df, "scheduled tribe"))
    return drop_aggregate_rows(out)


def clean_tribal_villages(df: pd.DataFrame) -> pd.DataFrame:
    out = clean_base(df, "State")
    columns = list(df.columns)
    names = [
        "tribal_villages_gt_25_count",
        "tribal_villages_gt_50_count",
        "tribal_villages_gt_75_count",
        "tribal_villages_gt_90_count",
        "tribal_villages_100_pct_count",
    ]
    for source, target in zip(columns[3:], names):
        out[target] = numeric(df[source])
    return drop_aggregate_rows(out)


def clean_mgnreg(df: pd.DataFrame) -> pd.DataFrame:
    out = clean_base(df, "State")
    out["social_group"] = df["Social Group"].astype(str).str.strip()
    out["residence_type"] = df["Type Of Residence"].astype(str).str.strip().str.lower()
    columns = list(df.columns)
    names = [
        "mgnreg_job_card_households_per_1000",
        "mgnreg_work_less_20_days_per_1000",
        "mgnreg_work_20_50_days_per_1000",
        "mgnreg_work_50_100_days_per_1000",
        "mgnreg_work_100_plus_days_per_1000",
        "mgnreg_sought_not_received_per_1000",
        "mgnreg_not_seeking_work_per_1000",
        "mgnreg_all_status_per_1000",
        "mgnreg_average_days_worked",
    ]
    for source, target in zip(columns[5:], names):
        out[target] = numeric(df[source])
    return drop_aggregate_rows(out)


def clean_ger(df: pd.DataFrame) -> pd.DataFrame:
    out = clean_base(df, "State")
    out["gender"] = df["Gender"].astype(str).str.strip()
    out["gender_key"] = out["gender"].map(gender_slug)
    out["class_group"] = df["Type Of Classes"].astype(str).str.strip()
    out["class_group_key"] = out["class_group"].map(class_slug)
    out["ger_ratio"] = numeric(col_series(df, "gross enrolment ratio"))
    return drop_aggregate_rows(out)


def clean_low_literacy_districts(df: pd.DataFrame) -> pd.DataFrame:
    out = clean_base(df, "State")
    out["district"] = df["District"].astype(str).str.strip()
    out["literacy_total_pct"] = numeric(col_series(df, "literacy rate of persons"))
    out["literacy_male_pct"] = numeric(col_series(df, "literacy rate of male"))
    out["literacy_female_pct"] = numeric(col_series(df, "literacy rate of female"))
    return drop_aggregate_rows(out)


def clean_tribe_socioeconomic(df: pd.DataFrame) -> pd.DataFrame:
    out = clean_base(df, "State")
    out["tribe_name"] = df["Name Of The Scheduled Tribe (St)"].astype(str).str.strip()
    columns = list(df.columns)
    names = [
        "households",
        "male_population",
        "female_population",
        "sex_ratio",
        "child_sex_ratio",
        "literacy_total_pct",
        "literacy_male_pct",
        "literacy_female_pct",
        "wpr_pct",
        "main_worker_pct",
        "marginal_worker_pct",
    ]
    for source, target in zip(columns[4:], names):
        out[target] = numeric(df[source])
    for target in [
        "literacy_total_pct",
        "literacy_male_pct",
        "literacy_female_pct",
        "wpr_pct",
        "main_worker_pct",
        "marginal_worker_pct",
    ]:
        out[target] = out[target].where(out[target].between(0, 100))
    out["tribe_population"] = out["male_population"] + out["female_population"]
    return drop_aggregate_rows(out)


def clean_employment(df: pd.DataFrame) -> pd.DataFrame:
    state_col = col_like(df, "state", "ut")
    out = clean_base(df, state_col)
    out["gender"] = df["Gender"].astype(str).str.strip()
    out["gender_key"] = out["gender"].map(gender_slug)
    indicator_col = col_like(df, "employment indicators")
    out["employment_indicator"] = df[indicator_col].astype(str).str.strip()
    out["employment_indicator_key"] = out["employment_indicator"].map(indicator_slug)
    out["employment_indicator_value_per_1000"] = numeric(col_series(df, "employment indicator value"))
    return drop_aggregate_rows(out)


def clean_demographics(df: pd.DataFrame) -> pd.DataFrame:
    out = clean_base(df, "State")
    if "Additional Information" in df.columns:
        out["additional_information"] = df["Additional Information"].replace({np.nan: ""})
    columns = list(df.columns)
    names = [
        "total_population",
        "state_share_india_population_pct",
        "st_population",
        "state_share_india_st_population_pct",
        "st_share_state_population_pct",
        "decadal_growth_pct",
    ]
    for source, target in zip(columns[4:], names):
        out[target] = numeric(df[source])
    return drop_aggregate_rows(out)


def clean_dropout(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame()
    out["state"] = df["States/UTs"].map(normalize_state)
    out["academic_year"] = "2021-22"
    out["dropout_primary_pct"] = numeric(df["Primary"])
    out["dropout_upper_primary_pct"] = numeric(df["Upper Primary"])
    out["dropout_secondary_pct"] = numeric(df["Secondary"])
    return drop_aggregate_rows(out)


def clean_poverty(df: pd.DataFrame) -> pd.DataFrame:
    out = clean_base(df, "State")
    out["residence_type"] = df["Residence Type"].astype(str).str.strip().str.lower()
    out["st_bpl_pct"] = numeric(col_series(df, "below the poverty line"))
    return drop_aggregate_rows(out)


def clean_household_type(df: pd.DataFrame) -> pd.DataFrame:
    out = clean_base(df, "State")
    columns = list(df.columns)
    names = [
        "hh_self_employed_agriculture_pct",
        "hh_self_employed_non_agriculture_pct",
        "hh_regular_wage_salary_pct",
        "hh_casual_labour_agriculture_pct",
        "hh_casual_labour_non_agriculture_pct",
        "hh_others_pct",
        "estimated_households",
        "sample_households",
    ]
    for source, target in zip(columns[3:], names):
        out[target] = numeric(df[source])
    return drop_aggregate_rows(out)


DATASETS = [
    {
        "name": "literacy",
        "proposal_no": "1",
        "in_proposal": True,
        "level": "state-year",
        "candidates": [
            "data/raw/literacy.csv",
            "Literacy Rate of Total Population and Scheduled Tribes Population and Gap in Literacy Rate.csv",
        ],
        "cleaner": clean_literacy,
        "key_variables": "total literacy, ST literacy, literacy gap",
        "notes": "Core education attainment dataset.",
    },
    {
        "name": "enrolment_st",
        "proposal_no": "2",
        "in_proposal": True,
        "level": "state-year",
        "candidates": [
            "data/raw/enrolment_st.csv",
            "State-Wise Enrolment by Stages of School Education of Scheduled Tribe (ST) Students - Pre-Primary, Primary, Upper-Primary.csv"
        ],
        "cleaner": clean_enrolment,
        "key_variables": "ST enrolment by stage and gender",
        "notes": "Includes derived totals and female shares.",
    },
    {
        "name": "high_st_share",
        "proposal_no": "3",
        "in_proposal": True,
        "level": "state-year",
        "candidates": [
            "data/raw/high_st_share.csv",
            "StateUT-wise with Percentage of Tribal Population More than the Country Average.csv",
        ],
        "cleaner": clean_high_st_share,
        "key_variables": "ST share in state population, state share of national ST population",
        "notes": "Defines high-ST-share states.",
    },
    {
        "name": "sc_st_residence",
        "proposal_no": "4",
        "in_proposal": True,
        "level": "state-year-residence",
        "candidates": [
            "data/raw/sc_st_residence.csv",
            "Scheduled Caste (SC) & Scheduled Tribes (ST) Population as Per Residence.csv",
        ],
        "cleaner": clean_sc_st_residence,
        "key_variables": "SC population, ST population by rural/urban residence",
        "notes": "Useful for demographic scale and rural/urban distribution.",
    },
    {
        "name": "tribal_villages",
        "proposal_no": "5",
        "in_proposal": True,
        "level": "state-year",
        "candidates": [
            "data/raw/tribal_villages.csv",
            "State-Wise Distribution of Tribal Villages by Different Concentration of Groups.csv",
        ],
        "cleaner": clean_tribal_villages,
        "key_variables": "villages above ST concentration thresholds",
        "notes": "Raw metadata labels UOM as percentage, but values behave like counts.",
    },
    {
        "name": "mgnreg_st",
        "proposal_no": "6",
        "in_proposal": True,
        "level": "state-year",
        "candidates": [
            "data/raw/mgnreg_st.csv",
            "NUMBER~1.CSV",
            "Number of households having MGNREG job card per 1000 households, per 1000 distribution of households by status of getting work in MGNREG works and average number of days got work during last 365 days for each household.csv",
        ],
        "cleaner": clean_mgnreg,
        "key_variables": "MGNREG job cards, work received, unmet demand, average days",
        "notes": "Filtered to Scheduled Tribe and rural households.",
    },
    {
        "name": "ger_st",
        "proposal_no": "7",
        "in_proposal": True,
        "level": "state-year-gender-class",
        "candidates": [
            "data/raw/ger_st.csv",
            "GROSSE~1.CSV",
            "Gross Enrolment Ratio (GER) - Scheduled Tribe - Classes I-VIII (6-13 Years), Classes IX-X (14-15 Years), Classes I-X (6-15 Years), Classes XI-XII (16-17 Years), Classes IX-XII (14-17 Years), and Classes I-XII (6-17 Yea.csv",
        ],
        "cleaner": clean_ger,
        "key_variables": "GER by gender and class group",
        "notes": "Raw filename is truncated by Windows path limits.",
    },
    {
        "name": "low_literacy_districts",
        "proposal_no": "8",
        "in_proposal": True,
        "level": "district-year",
        "candidates": [
            "data/raw/low_literacy_districts.csv",
            "State-wise and District-wise Information on Very Low Scheduled Tribe (ST) Literacy Rate (LR) (where Female Literacy Rate (LR) below 35 %).csv"
        ],
        "cleaner": clean_low_literacy_districts,
        "key_variables": "district ST literacy rates where female literacy is below 35 percent",
        "notes": "Supports state case studies and extreme-deprivation counts.",
    },
    {
        "name": "tribe_socioeconomic",
        "proposal_no": "9",
        "in_proposal": True,
        "level": "state-year-tribe",
        "candidates": [
            "data/raw/tribe_socioeconomic.csv",
            "State-Wise List of Scheduled Tribes (STs) with Details in Terms of Households, Population, Sex Ratio, Child Sex Ratio, Literacy, Worker Participation Rate, Main Worker and Marginal Worker.csv"
        ],
        "cleaner": clean_tribe_socioeconomic,
        "key_variables": "tribe-level households, population, literacy, WPR, worker type",
        "notes": "Aggregated to state level with population-weighted averages.",
    },
    {
        "name": "employment_st",
        "proposal_no": "10",
        "in_proposal": True,
        "level": "state-year-gender-indicator",
        "candidates": [
            "data/raw/employment_st.csv",
            "LABOUR~1.CSV",
            "Labour Force Participation Rate (LFPR), Worker Population Ratio (WPR), Proportion Unemployed (PU) according to Usual Status (Principal Status (PS) + Subsidiary Status (SS)) for each State or UT - Scheduled Tribe (ST) -.csv",
        ],
        "cleaner": clean_employment,
        "key_variables": "LFPR, WPR, PU by gender",
        "notes": "Does not include rural/urban despite the proposal wording.",
    },
    {
        "name": "demographics_st",
        "proposal_no": "11",
        "in_proposal": True,
        "level": "state-year",
        "candidates": [
            "data/raw/demographics_st.csv",
            "State-wise Demographic Status of Total Population & Scheduled Tribe (ST) Population.csv",
        ],
        "cleaner": clean_demographics,
        "key_variables": "total population, ST population, ST population share",
        "notes": "Main base table for state-level merging.",
    },
    {
        "name": "dropout_st",
        "proposal_no": "12",
        "in_proposal": True,
        "level": "state-academic-year",
        "candidates": [
            "data/raw/dropout_st.csv",
            "StateUT-wise Dropout Rates of Scheduled Tribes (STs) 2021-22.csv",
        ],
        "cleaner": clean_dropout,
        "key_variables": "ST dropout rates at primary, upper primary, secondary levels",
        "notes": "External 2021-22 dataset from proposal.",
    },
    {
        "name": "poverty_st",
        "proposal_no": "13",
        "in_proposal": True,
        "level": "state-year-residence",
        "candidates": [
            "data/raw/poverty_st.csv",
            "Percentage of ST Population Below Poverty Line (Tendulkar Methodology).csv",
        ],
        "cleaner": clean_poverty,
        "key_variables": "ST population below poverty line by residence",
        "notes": "Economic deprivation indicator.",
    },
    {
        "name": "household_type_rural",
        "proposal_no": "",
        "in_proposal": False,
        "level": "state-year",
        "candidates": [
            "data/raw/household_type_rural.csv",
            "StateUT-wise Percentage Distribution of Households by Household Type for Rural Areas.csv",
        ],
        "cleaner": clean_household_type,
        "key_variables": "rural household type distribution",
        "notes": "Extra dataset not listed in proposal.",
    },
]


def resolve_source(candidates: list[str]) -> Path:
    for candidate in candidates:
        path = ROOT / candidate
        if path.exists():
            return path
    raise FileNotFoundError(f"None of these source files were found: {candidates}")


def read_source(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")


def write_cleaned_data() -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    cleaned: dict[str, pd.DataFrame] = {}
    inventory_rows = []
    for spec in DATASETS:
        source_path = resolve_source(spec["candidates"])
        raw = read_source(source_path)
        clean = spec["cleaner"](raw)
        clean = clean.drop_duplicates().reset_index(drop=True)
        cleaned[spec["name"]] = clean
        out_path = CLEAN_DIR / f"{spec['name']}.csv"
        clean.to_csv(out_path, index=False)

        year_values = []
        if "year" in clean.columns:
            year_values = sorted([int(v) for v in clean["year"].dropna().unique()])
        elif "academic_year" in clean.columns:
            year_values = sorted(clean["academic_year"].dropna().unique())

        inventory_rows.append(
            {
                "short_name": spec["name"],
                "proposal_no": spec["proposal_no"],
                "in_proposal": spec["in_proposal"],
                "source_file": source_path.name,
                "cleaned_file": str(out_path.relative_to(ROOT)),
                "level": spec["level"],
                "raw_rows": len(raw),
                "cleaned_rows": len(clean),
                "raw_columns": len(raw.columns),
                "cleaned_columns": len(clean.columns),
                "state_count": clean["state"].nunique(dropna=True) if "state" in clean.columns else "",
                "years": ", ".join(map(str, year_values)),
                "key_variables": spec["key_variables"],
                "notes": spec["notes"],
            }
        )

    inventory = pd.DataFrame(inventory_rows)
    inventory.to_csv(ANALYSIS_DIR / "data_inventory.csv", index=False)
    return cleaned, inventory


def latest_by_state(df: pd.DataFrame, year_col: str = "year") -> pd.DataFrame:
    if year_col not in df.columns:
        return df.drop_duplicates("state")
    temp = df.dropna(subset=["state"]).copy()
    temp["_sort_year"] = temp[year_col].fillna(-9999)
    return temp.sort_values(["state", "_sort_year"]).groupby("state", as_index=False).tail(1).drop(columns="_sort_year")


def weighted_average(group: pd.DataFrame, value_col: str, weight_col: str) -> float:
    values = numeric(group[value_col])
    weights = numeric(group[weight_col])
    mask = values.notna() & weights.notna() & (weights > 0)
    if not mask.any():
        return np.nan
    return float(np.average(values.loc[mask], weights=weights.loc[mask]))


def aggregate_tribe_socioeconomic(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for state, group in df.groupby("state", dropna=True):
        non_total = group[~group["tribe_name"].str.lower().str.contains("all schedule tribes", na=False)]
        rows.append(
            {
                "state": state,
                "tribe_count": non_total["tribe_name"].nunique(),
                "tribe_population_total": non_total["tribe_population"].sum(min_count=1),
                "tribe_weighted_literacy_total_pct": weighted_average(non_total, "literacy_total_pct", "tribe_population"),
                "tribe_weighted_literacy_male_pct": weighted_average(non_total, "literacy_male_pct", "tribe_population"),
                "tribe_weighted_literacy_female_pct": weighted_average(non_total, "literacy_female_pct", "tribe_population"),
                "tribe_weighted_wpr_pct": weighted_average(non_total, "wpr_pct", "tribe_population"),
                "tribe_weighted_main_worker_pct": weighted_average(non_total, "main_worker_pct", "tribe_population"),
                "tribe_weighted_marginal_worker_pct": weighted_average(non_total, "marginal_worker_pct", "tribe_population"),
            }
        )
    return pd.DataFrame(rows)


def merge_left(base: pd.DataFrame, other: pd.DataFrame) -> pd.DataFrame:
    if other.empty:
        return base
    return base.merge(other, on="state", how="left")


def build_state_analysis_dataset(cleaned: dict[str, pd.DataFrame]) -> pd.DataFrame:
    demographics = latest_by_state(cleaned["demographics_st"])
    base_cols = [
        "state",
        "year",
        "total_population",
        "st_population",
        "st_share_state_population_pct",
        "state_share_india_st_population_pct",
        "decadal_growth_pct",
    ]
    state_df = demographics[[col for col in base_cols if col in demographics.columns]].rename(
        columns={"year": "demographics_year"}
    )

    high_st = latest_by_state(cleaned["high_st_share"])
    high_st = high_st[
        [
            "state",
            "year",
            "st_share_state_population_pct",
            "state_share_india_st_population_pct",
        ]
    ].rename(
        columns={
            "year": "high_st_share_year",
            "st_share_state_population_pct": "high_st_share_state_population_pct",
            "state_share_india_st_population_pct": "high_st_state_share_india_st_population_pct",
        }
    )
    high_st["is_high_st_share_state"] = True
    state_df = merge_left(state_df, high_st)
    state_df["is_high_st_share_state"] = state_df["is_high_st_share_state"].eq(True)

    literacy = latest_by_state(cleaned["literacy"])
    literacy = literacy[
        ["state", "year", "total_literacy_rate_pct", "st_literacy_rate_pct", "literacy_gap_pct"]
    ].rename(columns={"year": "literacy_year"})
    state_df = merge_left(state_df, literacy)

    enrolment = latest_by_state(cleaned["enrolment_st"])
    enrolment_cols = [
        "state",
        "year",
        "pre_primary_total",
        "primary_total",
        "upper_primary_total",
        "pre_primary_female_share_pct",
        "primary_female_share_pct",
        "upper_primary_female_share_pct",
    ]
    enrolment = enrolment[[col for col in enrolment_cols if col in enrolment.columns]].rename(
        columns={"year": "enrolment_year"}
    )
    state_df = merge_left(state_df, enrolment)

    dropout = cleaned["dropout_st"].copy()
    state_df = merge_left(state_df, dropout)

    ger = cleaned["ger_st"].copy()
    ger_total = ger[ger["gender_key"].eq("total")]
    ger_pivot = ger_total.pivot_table(index="state", columns="class_group_key", values="ger_ratio", aggfunc="mean")
    ger_pivot = ger_pivot.rename(columns={col: f"ger_{col}" for col in ger_pivot.columns}).reset_index()
    state_df = merge_left(state_df, ger_pivot)

    employment = cleaned["employment_st"].copy()
    emp_pivot = employment.pivot_table(
        index="state",
        columns=["employment_indicator_key", "gender_key"],
        values="employment_indicator_value_per_1000",
        aggfunc="mean",
    )
    emp_pivot.columns = [f"employment_{indicator}_{gender}_per_1000" for indicator, gender in emp_pivot.columns]
    emp_pivot = emp_pivot.reset_index()
    state_df = merge_left(state_df, emp_pivot)

    poverty = cleaned["poverty_st"].copy()
    poverty_latest = poverty.sort_values("year").groupby(["state", "residence_type"], as_index=False).tail(1)
    poverty_pivot = poverty_latest.pivot_table(index="state", columns="residence_type", values="st_bpl_pct", aggfunc="mean")
    poverty_pivot = poverty_pivot.rename(columns={col: f"st_bpl_{slug(col)}_pct" for col in poverty_pivot.columns})
    poverty_pivot["st_bpl_mean_pct"] = poverty_pivot.mean(axis=1, skipna=True)
    poverty_pivot = poverty_pivot.reset_index()
    state_df = merge_left(state_df, poverty_pivot)

    mgnreg = latest_by_state(cleaned["mgnreg_st"])
    mgnreg_cols = [
        "state",
        "year",
        "mgnreg_job_card_households_per_1000",
        "mgnreg_work_less_20_days_per_1000",
        "mgnreg_work_20_50_days_per_1000",
        "mgnreg_work_50_100_days_per_1000",
        "mgnreg_work_100_plus_days_per_1000",
        "mgnreg_sought_not_received_per_1000",
        "mgnreg_not_seeking_work_per_1000",
        "mgnreg_average_days_worked",
    ]
    mgnreg = mgnreg[[col for col in mgnreg_cols if col in mgnreg.columns]].rename(columns={"year": "mgnreg_year"})
    state_df = merge_left(state_df, mgnreg)

    scst = cleaned["sc_st_residence"].copy()
    scst_latest = scst.sort_values("year").groupby(["state", "residence_type"], as_index=False).tail(1)
    st_pivot = scst_latest.pivot_table(index="state", columns="residence_type", values="st_population", aggfunc="sum")
    st_pivot = st_pivot.rename(columns={col: f"st_population_{slug(col)}" for col in st_pivot.columns})
    st_pivot["st_population_residence_total"] = st_pivot.sum(axis=1, skipna=True)
    if "st_population_urban" in st_pivot.columns:
        st_pivot["st_population_urban_share_pct"] = (
            st_pivot["st_population_urban"] / st_pivot["st_population_residence_total"].replace(0, np.nan) * 100
        )
    state_df = merge_left(state_df, st_pivot.reset_index())

    villages = latest_by_state(cleaned["tribal_villages"])
    villages = villages.drop(columns=["year"], errors="ignore")
    state_df = merge_left(state_df, villages)

    low_lit = cleaned["low_literacy_districts"].copy()
    low_summary = (
        low_lit.groupby("state")
        .agg(
            low_literacy_district_count=("district", "nunique"),
            low_literacy_female_min_pct=("literacy_female_pct", "min"),
            low_literacy_female_mean_pct=("literacy_female_pct", "mean"),
            low_literacy_total_mean_pct=("literacy_total_pct", "mean"),
        )
        .reset_index()
    )
    state_df = merge_left(state_df, low_summary)

    tribe_summary = aggregate_tribe_socioeconomic(cleaned["tribe_socioeconomic"])
    state_df = merge_left(state_df, tribe_summary)

    household = latest_by_state(cleaned["household_type_rural"])
    household = household.drop(columns=["year"], errors="ignore")
    state_df = merge_left(state_df, household)

    return add_scores(state_df)


def minmax(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    lo = values.min(skipna=True)
    hi = values.max(skipna=True)
    if pd.isna(lo) or pd.isna(hi) or hi == lo:
        return pd.Series(np.nan, index=series.index)
    return (values - lo) / (hi - lo)


def add_scores(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["st_literacy_deprivation_pct"] = 100 - out.get("st_literacy_rate_pct")
    if "tribe_weighted_literacy_female_pct" in out.columns:
        out["female_literacy_deprivation_pct"] = 100 - out["tribe_weighted_literacy_female_pct"]

    education_inputs = [
        "st_literacy_deprivation_pct",
        "female_literacy_deprivation_pct",
        "literacy_gap_pct",
        "dropout_secondary_pct",
        "low_literacy_district_count",
    ]
    economic_inputs = [
        "st_bpl_mean_pct",
        "employment_pu_person_per_1000",
        "mgnreg_sought_not_received_per_1000",
    ]

    for col in education_inputs + economic_inputs:
        if col in out.columns:
            out[f"score_component_{col}"] = minmax(out[col])

    education_components = [f"score_component_{col}" for col in education_inputs if f"score_component_{col}" in out.columns]
    economic_components = [f"score_component_{col}" for col in economic_inputs if f"score_component_{col}" in out.columns]
    out["education_disadvantage_score"] = out[education_components].mean(axis=1, skipna=True)
    out["economic_vulnerability_score"] = out[economic_components].mean(axis=1, skipna=True)
    out["overall_priority_score"] = out[
        ["education_disadvantage_score", "economic_vulnerability_score"]
    ].mean(axis=1, skipna=True)

    edu_cut = out["education_disadvantage_score"].quantile(0.67)
    econ_cut = out["economic_vulnerability_score"].quantile(0.67)

    def classify(row: pd.Series) -> str:
        edu_high = row["education_disadvantage_score"] >= edu_cut
        econ_high = row["economic_vulnerability_score"] >= econ_cut
        if edu_high and econ_high:
            return "High education disadvantage and high economic vulnerability"
        if edu_high:
            return "Education disadvantage priority"
        if econ_high:
            return "Economic vulnerability priority"
        return "Monitor / comparatively lower priority"

    out["policy_priority_category"] = out.apply(classify, axis=1)
    out["overall_priority_rank"] = out["overall_priority_score"].rank(ascending=False, method="min")
    return out.sort_values(["overall_priority_rank", "state"]).reset_index(drop=True)


def write_sqlite(cleaned: dict[str, pd.DataFrame], state_df: pd.DataFrame) -> None:
    if DB_PATH.exists():
        DB_PATH.unlink()
    with sqlite3.connect(DB_PATH) as conn:
        for name, df in cleaned.items():
            df.to_sql(name, conn, index=False, if_exists="replace")
        state_df.to_sql("state_analysis_dataset", conn, index=False, if_exists="replace")


def write_analysis_outputs(cleaned: dict[str, pd.DataFrame], state_df: pd.DataFrame) -> None:
    state_df.to_csv(ANALYSIS_DIR / "state_analysis_dataset_all_states.csv", index=False)
    high_state_df = state_df[state_df["is_high_st_share_state"].fillna(False)].copy()
    high_state_df.to_csv(ANALYSIS_DIR / "state_analysis_dataset_high_st_states.csv", index=False)

    score_cols = [
        "state",
        "is_high_st_share_state",
        "education_disadvantage_score",
        "economic_vulnerability_score",
        "overall_priority_score",
        "overall_priority_rank",
        "policy_priority_category",
    ]
    state_df[[col for col in score_cols if col in state_df.columns]].to_csv(
        ANALYSIS_DIR / "state_disadvantage_scores.csv", index=False
    )

    correlation_cols = [
        "st_literacy_rate_pct",
        "literacy_gap_pct",
        "dropout_secondary_pct",
        "st_bpl_mean_pct",
        "employment_lfpr_person_per_1000",
        "employment_wpr_person_per_1000",
        "employment_pu_person_per_1000",
        "mgnreg_sought_not_received_per_1000",
        "overall_priority_score",
    ]
    usable_cols = [col for col in correlation_cols if col in state_df.columns]
    state_df[usable_cols].corr(numeric_only=True).to_csv(ANALYSIS_DIR / "correlations.csv")
    write_summary_markdown(cleaned, state_df, high_state_df)
    write_figures(high_state_df)


def write_summary_markdown(cleaned: dict[str, pd.DataFrame], state_df: pd.DataFrame, high_state_df: pd.DataFrame) -> None:
    top_priority = high_state_df.sort_values("overall_priority_score", ascending=False).head(10)
    lines = [
        "# ST Education Project Data Build Summary",
        "",
        "## Generated Outputs",
        "",
        "- Cleaned CSVs: `outputs/cleaned/`",
        "- Data inventory: `outputs/analysis/data_inventory.csv`",
        "- Combined state-level dataset: `outputs/analysis/state_analysis_dataset_all_states.csv`",
        "- High-ST state subset: `outputs/analysis/state_analysis_dataset_high_st_states.csv`",
        "- Disadvantage scores: `outputs/analysis/state_disadvantage_scores.csv`",
        "- Correlation matrix: `outputs/analysis/correlations.csv`",
        "- SQLite database: `outputs/st_education_project.sqlite`",
        "- Figures: `outputs/figures/`",
        "",
        "## Scope Notes",
        "",
        f"- Cleaned datasets created: {len(cleaned)}",
        f"- State rows in combined table: {len(state_df)}",
        f"- High-ST-share states identified from the proposal dataset: {len(high_state_df)}",
        "- Original raw CSV files were not renamed or edited.",
        "- The employment dataset has LFPR/WPR/PU by state and gender, but no rural/urban field.",
        "- The tribal-villages file has percentage wording in the raw column labels, but the values look like village counts.",
        "",
        "## Top High-ST States By Overall Priority Score",
        "",
    ]
    if top_priority.empty:
        lines.append("No high-ST state priority rows were available.")
    else:
        lines.append("| Rank | State | Overall | Education | Economic | Category |")
        lines.append("|---:|---|---:|---:|---:|---|")
        for _, row in top_priority.iterrows():
            lines.append(
                "| {rank:.0f} | {state} | {overall:.3f} | {education:.3f} | {economic:.3f} | {category} |".format(
                    rank=row.get("overall_priority_rank", np.nan),
                    state=row.get("state", ""),
                    overall=row.get("overall_priority_score", np.nan),
                    education=row.get("education_disadvantage_score", np.nan),
                    economic=row.get("economic_vulnerability_score", np.nan),
                    category=row.get("policy_priority_category", ""),
                )
            )

    lines.extend(
        [
            "",
            "## Suggested Next Analytical Use",
            "",
            "Use the combined high-ST table for the main state comparison, and use the district and tribe-level cleaned files for case-study evidence.",
        ]
    )
    (ANALYSIS_DIR / "build_summary.md").write_text("\n".join(lines), encoding="utf-8")


def write_figures(high_state_df: pd.DataFrame) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    if high_state_df.empty:
        return

    plot_df = high_state_df.dropna(subset=["overall_priority_score"]).sort_values("overall_priority_score").tail(15)
    if not plot_df.empty:
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.barh(plot_df["state"], plot_df["overall_priority_score"], color="#4d7c8a")
        ax.set_xlabel("Overall priority score")
        ax.set_title("High-ST states by overall priority score")
        fig.tight_layout()
        fig.savefig(FIGURE_DIR / "priority_score_high_st_states.png", dpi=160)
        plt.close(fig)

    lit_df = high_state_df.dropna(subset=["st_literacy_rate_pct"]).sort_values("st_literacy_rate_pct").head(15)
    if not lit_df.empty:
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.barh(lit_df["state"], lit_df["st_literacy_rate_pct"], color="#697a21")
        ax.invert_yaxis()
        ax.set_xlabel("ST literacy rate (%)")
        ax.set_title("Lowest ST literacy rates among high-ST states")
        fig.tight_layout()
        fig.savefig(FIGURE_DIR / "lowest_st_literacy_high_st_states.png", dpi=160)
        plt.close(fig)

    scatter_df = high_state_df.dropna(subset=["st_literacy_rate_pct", "st_bpl_mean_pct"])
    if not scatter_df.empty:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(scatter_df["st_literacy_rate_pct"], scatter_df["st_bpl_mean_pct"], color="#7f4f24")
        for _, row in scatter_df.iterrows():
            ax.annotate(row["state"], (row["st_literacy_rate_pct"], row["st_bpl_mean_pct"]), fontsize=7, alpha=0.8)
        ax.set_xlabel("ST literacy rate (%)")
        ax.set_ylabel("ST BPL rate, mean rural/urban (%)")
        ax.set_title("ST poverty vs literacy")
        fig.tight_layout()
        fig.savefig(FIGURE_DIR / "poverty_vs_literacy_high_st_states.png", dpi=160)
        plt.close(fig)


def main() -> None:
    ensure_dirs()
    cleaned, _inventory = write_cleaned_data()
    state_df = build_state_analysis_dataset(cleaned)
    write_analysis_outputs(cleaned, state_df)
    write_sqlite(cleaned, state_df)
    print(f"Processed {len(cleaned)} datasets")
    print(f"Wrote cleaned CSVs to {CLEAN_DIR.relative_to(ROOT)}")
    print(f"Wrote analysis outputs to {ANALYSIS_DIR.relative_to(ROOT)}")
    print(f"Wrote SQLite database to {DB_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
