# ST Education Project Data Build Summary

## Generated Outputs

- Cleaned CSVs: `outputs/cleaned/`
- Data inventory: `outputs/analysis/data_inventory.csv`
- Combined state-level dataset: `outputs/analysis/state_analysis_dataset_all_states.csv`
- High-ST state subset: `outputs/analysis/state_analysis_dataset_high_st_states.csv`
- Disadvantage scores: `outputs/analysis/state_disadvantage_scores.csv`
- Correlation matrix: `outputs/analysis/correlations.csv`
- SQLite database: `outputs/st_education_project.sqlite`
- Figures: `outputs/figures/`

## Scope Notes

- Cleaned datasets created: 14
- State rows in combined table: 34
- High-ST-share states identified from the proposal dataset: 19
- Original raw CSV files were not renamed or edited.
- The employment dataset has LFPR/WPR/PU by state and gender, but no rural/urban field.
- The tribal-villages file has percentage wording in the raw column labels, but the values look like village counts.

## Top High-ST States By Overall Priority Score

| Rank | State | Overall | Education | Economic | Category |
|---:|---|---:|---:|---:|---|
| 1 | Odisha | 0.634 | 0.800 | 0.468 | High education disadvantage and high economic vulnerability |
| 3 | Madhya Pradesh | 0.574 | 0.766 | 0.382 | High education disadvantage and high economic vulnerability |
| 6 | Jharkhand | 0.507 | 0.549 | 0.465 | Economic vulnerability priority |
| 7 | Maharashtra | 0.505 | 0.645 | 0.366 | High education disadvantage and high economic vulnerability |
| 9 | Rajasthan | 0.454 | 0.676 | 0.233 | Education disadvantage priority |
| 10 | Nagaland | 0.435 | 0.293 | 0.577 | Economic vulnerability priority |
| 11 | Chhattisgarh | 0.424 | 0.494 | 0.354 | Economic vulnerability priority |
| 12 | Jammu and Kashmir | 0.410 | 0.788 | 0.032 | Education disadvantage priority |
| 13 | Gujarat | 0.409 | 0.531 | 0.286 | Monitor / comparatively lower priority |
| 15 | Tripura | 0.380 | 0.337 | 0.424 | Economic vulnerability priority |

## Suggested Next Analytical Use

Use the combined high-ST table for the main state comparison, and use the district and tribe-level cleaned files for case-study evidence.