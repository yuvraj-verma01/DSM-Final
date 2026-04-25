# ST Education and Livelihood Dashboard

Separate interactive dashboard app for the DSM final project.

## Run

From the project root:

```powershell
streamlit run dashboard_app/streamlit_app.py
```

The app reads the existing project outputs:

- `outputs/analysis/state_analysis_dataset_high_st_states.csv`
- `outputs/analysis/state_analysis_dataset_all_states.csv`
- `outputs/analysis/state_disadvantage_scores.csv`

It does not change the GitHub Pages article site.

## What It Shows

- Project-level KPI cards
- State explorer and sortable priority table
- Interactive relationship explorer
- Question tabs for Q1-Q13 with the most important visuals from the notebook logic
- Correlation values under each interactive plot
