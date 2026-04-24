# Exploratory Data Analysis and Policy Findings

## North Star

The project asks how educational interventions for Scheduled Tribes should be prioritised across high-ST states, given differences in education outcomes and their association with employment and poverty.

This EDA therefore focuses on four outputs: identifying weak education outcomes, testing education-poverty-labour relationships, classifying state disadvantage profiles, and translating evidence into policy recommendations.

## Data and Unit of Analysis

- Master all-state table: 34 states.
- High-ST-state analytical subset: 19 states.
- Cleaned datasets used: 17.
- Main unit of analysis: one state profile assembled from the most relevant available state-level observations.
- Sparse state-year fact table also generated: 455 state-year rows.
- Time periods are mixed across sources: most structural education/demography measures are around 2011, labour/MGNREG/GER are around 2013, dropout is 2021-22, and the extra household-type table is later. Treat relationships as exploratory, not causal.

## Data Quality Summary

The cleaned data are analysis-ready, but coverage is uneven because the source datasets differ in year, geography, and filtering.

| column | non_missing | missing | missing_pct |
|---|---|---|---|
| low_literacy_district_count | 7 | 12 | 63.16 |
| st_bpl_rural_pct | 9 | 10 | 52.63 |
| st_bpl_urban_pct | 9 | 10 | 52.63 |
| st_bpl_mean_pct | 9 | 10 | 52.63 |
| scholarship_utilization_2023_24_pct | 17 | 2 | 10.53 |
| st_literacy_rate_pct | 18 | 1 | 5.26 |
| literacy_gap_pct | 18 | 1 | 5.26 |
| scholarship_total_release_2023_24_lakh_per_100k_st_pop | 18 | 1 | 5.26 |
| employment_lfpr_person_per_1000 | 18 | 1 | 5.26 |
| employment_wpr_person_per_1000 | 18 | 1 | 5.26 |

## Key Findings

- The weakest ST literacy outcomes among high-ST states are concentrated in Madhya Pradesh (50.60%), Jammu and Kashmir (50.60%), Odisha (52.20%).
- Secondary dropout is highest in Odisha (33.12%), Meghalaya (21.99%), Maharashtra (21.04%), making retention beyond elementary schooling a central intervention area.
- ST poverty is highest in Odisha (51.60%), Chhattisgarh (43.90%), Madhya Pradesh (43.80%) among states with available poverty data.
- The first-pass policy priority ranking places Odisha, Madhya Pradesh, Jharkhand, Maharashtra, Rajasthan at the top because they combine education disadvantage with economic vulnerability.
- The strongest exploratory relationships are: How does secondary dropout relate to ST poverty? (r=0.66, n=17); How does unmet MGNREG demand relate to ST poverty? (r=0.65, n=9); How does ST literacy relate to unemployment? (r=0.63, n=27).
- The typology separates the high-ST states into 4 policy profiles, supporting differentiated interventions rather than one uniform ST education policy.

## Priority Ranking

| state | education_disadvantage_score | economic_vulnerability_score | overall_priority_score | policy_priority_category |
|---|---|---|---|---|
| Odisha | 0.800 | 0.468 | 0.634 | High education disadvantage and high economic vulnerability |
| Madhya Pradesh | 0.766 | 0.382 | 0.574 | High education disadvantage and high economic vulnerability |
| Jharkhand | 0.549 | 0.465 | 0.507 | Economic vulnerability priority |
| Maharashtra | 0.645 | 0.366 | 0.505 | High education disadvantage and high economic vulnerability |
| Rajasthan | 0.676 | 0.233 | 0.454 | Education disadvantage priority |
| Nagaland | 0.293 | 0.577 | 0.435 | Economic vulnerability priority |
| Chhattisgarh | 0.494 | 0.354 | 0.424 | Economic vulnerability priority |
| Jammu and Kashmir | 0.788 | 0.032 | 0.410 | Education disadvantage priority |
| Gujarat | 0.531 | 0.286 | 0.409 | Monitor / comparatively lower priority |
| Tripura | 0.337 | 0.424 | 0.380 | Economic vulnerability priority |

## Relationship Analysis

The following associations use Pearson and Spearman correlations. Because indicators are drawn from mixed years and small samples, they should be interpreted as pattern evidence rather than causal estimates.

| sample | question | n | pearson_r | pearson_p | spearman_r | interpretation |
|---|---|---|---|---|---|---|
| all_states | How does secondary dropout relate to ST poverty? | 17 | 0.66 | 0.00 | 0.62 | Moderate positive association; exploratory, not causal. |
| high_st_states | How does unmet MGNREG demand relate to ST poverty? | 9 | 0.65 | 0.06 | 0.42 | Moderate positive association; exploratory, not causal. |
| all_states | How does ST literacy relate to unemployment? | 27 | 0.63 | 0.00 | 0.64 | Moderate positive association; exploratory, not causal. |
| high_st_states | How does ST literacy relate to unemployment? | 18 | 0.62 | 0.01 | 0.73 | Moderate positive association; exploratory, not causal. |
| high_st_states | How does secondary dropout relate to ST poverty? | 9 | 0.53 | 0.15 | 0.55 | Moderate positive association; exploratory, not causal. |
| high_st_states | How does secondary dropout relate to work participation? | 18 | 0.51 | 0.03 | 0.62 | Moderate positive association; exploratory, not causal. |
| high_st_states | How does ST literacy relate to work participation? | 18 | -0.48 | 0.05 | -0.35 | Moderate negative association; exploratory, not causal. |
| high_st_states | Do higher-secondary-GER states still show MGNREG 100-plus-days dependence? | 19 | 0.47 | 0.04 | 0.32 | Moderate positive association; exploratory, not causal. |
| high_st_states | How does female literacy relate to female work participation? | 18 | -0.46 | 0.05 | -0.47 | Moderate negative association; exploratory, not causal. |
| high_st_states | How does latest secondary ST GER relate to secondary dropout? | 19 | 0.45 | 0.05 | 0.36 | Moderate positive association; exploratory, not causal. |

## Regression Checks

Exploratory OLS models were run only where enough complete observations existed. These are diagnostic checks, not causal models.

| sample | model | term | n | r_squared | coefficient | p_value |
|---|---|---|---|---|---|---|
| high_st_states | st_bpl_mean_pct ~ st_literacy_rate_pct + dropout_secondary_pct + employment_wpr_person_per_1000 | st_literacy_rate_pct | 9 | 0.67 | -0.53 | 0.32 |
| high_st_states | st_bpl_mean_pct ~ st_literacy_rate_pct + dropout_secondary_pct + employment_wpr_person_per_1000 | dropout_secondary_pct | 9 | 0.67 | 1.23 | 0.05 |
| high_st_states | st_bpl_mean_pct ~ st_literacy_rate_pct + dropout_secondary_pct + employment_wpr_person_per_1000 | employment_wpr_person_per_1000 | 9 | 0.67 | -0.13 | 0.06 |
| high_st_states | employment_wpr_person_per_1000 ~ st_literacy_rate_pct + tribe_weighted_literacy_female_pct + dropout_secondary_pct | st_literacy_rate_pct | 18 | 0.33 | -3.75 | 0.53 |
| high_st_states | employment_wpr_person_per_1000 ~ st_literacy_rate_pct + tribe_weighted_literacy_female_pct + dropout_secondary_pct | tribe_weighted_literacy_female_pct | 18 | 0.33 | 1.77 | 0.69 |
| high_st_states | employment_wpr_person_per_1000 ~ st_literacy_rate_pct + tribe_weighted_literacy_female_pct + dropout_secondary_pct | dropout_secondary_pct | 18 | 0.33 | 2.77 | 0.21 |
| high_st_states | overall_priority_score ~ st_literacy_rate_pct + st_bpl_mean_pct + employment_wpr_person_per_1000 | st_literacy_rate_pct | 9 | 0.80 | -0.01 | 0.13 |
| high_st_states | overall_priority_score ~ st_literacy_rate_pct + st_bpl_mean_pct + employment_wpr_person_per_1000 | st_bpl_mean_pct | 9 | 0.80 | 0.01 | 0.02 |
| high_st_states | overall_priority_score ~ st_literacy_rate_pct + st_bpl_mean_pct + employment_wpr_person_per_1000 | employment_wpr_person_per_1000 | 9 | 0.80 | 0.00 | 0.52 |
| all_states | st_bpl_mean_pct ~ st_literacy_rate_pct + dropout_secondary_pct + employment_wpr_person_per_1000 | st_literacy_rate_pct | 16 | 0.52 | -0.07 | 0.82 |
| all_states | st_bpl_mean_pct ~ st_literacy_rate_pct + dropout_secondary_pct + employment_wpr_person_per_1000 | dropout_secondary_pct | 16 | 0.52 | 1.19 | 0.01 |
| all_states | st_bpl_mean_pct ~ st_literacy_rate_pct + dropout_secondary_pct + employment_wpr_person_per_1000 | employment_wpr_person_per_1000 | 16 | 0.52 | -0.07 | 0.04 |
| all_states | employment_wpr_person_per_1000 ~ st_literacy_rate_pct + tribe_weighted_literacy_female_pct + dropout_secondary_pct | st_literacy_rate_pct | 26 | 0.16 | -2.67 | 0.66 |
| all_states | employment_wpr_person_per_1000 ~ st_literacy_rate_pct + tribe_weighted_literacy_female_pct + dropout_secondary_pct | tribe_weighted_literacy_female_pct | 26 | 0.16 | 1.14 | 0.80 |

## State Typology

K-means clustering was used on standardized education, poverty, employment, MGNREG, and priority-score indicators. The clusters are a policy typology, not a definitive classification.

| cluster_id | cluster_name | state_count | states | avg_education_disadvantage | avg_economic_vulnerability | avg_overall_priority |
|---|---|---|---|---|---|---|
| 1 | Economic and labour vulnerability | 3 | Lakshadweep, Nagaland, Tripura | 0.21 | 0.47 | 0.34 |
| 2 | Compound education and economic disadvantage | 8 | Chhattisgarh, Dadra and Nagar Haveli and Daman and Diu, Gujarat, Jharkhand, Madhya Pradesh, Maharashtra, Odisha, Rajasthan | 0.62 | 0.35 | 0.49 |
| 3 | Lower composite risk with poverty watchlist | 7 | Arunachal Pradesh, Assam, Goa, Manipur, Meghalaya, Mizoram, Sikkim | 0.30 | 0.19 | 0.24 |
| 4 | Education-system disadvantage | 1 | Jammu and Kashmir | 0.79 | 0.03 | 0.41 |

## Policy Recommendations

Recommendations are state-specific and evidence-triggered. They should be read as priorities for intervention design, not as claims that one variable caused another.

| state | overall_priority_rank | policy_priority_category | evidence_flags | recommended_policy_focus |
|---|---|---|---|---|
| Odisha | 1.00 | High education disadvantage and high economic vulnerability | low ST literacy (52.20%); large literacy gap (20.60 pp); high secondary dropout (33.12%); low female literacy (41.04%); high ST poverty (51.60%); high unmet MGNREG demand (255 per 1000); 6 low-female-literacy district(s); many >50% ST villages (17798) | Prioritise foundational literacy, remedial learning, and adult literacy in ST communities; Target the ST-general population literacy gap with ST-focused school support and tracking; Strengthen secondary retention through hostels, transport, scholarships, and transition support; Add female literacy interventions: residential schooling, women teachers, safety, sanitation, and community outreach |
| Madhya Pradesh | 3.00 | High education disadvantage and high economic vulnerability | low ST literacy (50.60%); large literacy gap (18.80 pp); high secondary dropout (17.60%); low female literacy (41.20%); high ST poverty (43.80%); high unmet MGNREG demand (236 per 1000); 11 low-female-literacy district(s); many >50% ST villages (15022) | Prioritise foundational literacy, remedial learning, and adult literacy in ST communities; Target the ST-general population literacy gap with ST-focused school support and tracking; Strengthen secondary retention through hostels, transport, scholarships, and transition support; Add female literacy interventions: residential schooling, women teachers, safety, sanitation, and community outreach |
| Jharkhand | 6.00 | Economic vulnerability priority | low ST literacy (57.10%); low female literacy (46.16%); weak WPR (375 per 1000); high unmet MGNREG demand (245 per 1000); 6 low-female-literacy district(s); many >50% ST villages (12239) | Prioritise foundational literacy, remedial learning, and adult literacy in ST communities; Add female literacy interventions: residential schooling, women teachers, safety, sanitation, and community outreach; Connect upper-secondary schooling with local skills, placement, and public employment pathways; Improve MGNREG work availability while reducing education costs for vulnerable households |
| Maharashtra | 7.00 | High education disadvantage and high economic vulnerability | large literacy gap (16.60 pp); high secondary dropout (21.04%); many >50% ST villages (6738) | Target the ST-general population literacy gap with ST-focused school support and tracking; Strengthen secondary retention through hostels, transport, scholarships, and transition support; Use geographically concentrated service delivery: school clusters, mobile academic support, and local monitoring |
| Rajasthan | 9.00 | Education disadvantage priority | low ST literacy (52.80%); large literacy gap (13.30 pp); low female literacy (37.13%); 11 low-female-literacy district(s); many >50% ST villages (7763) | Prioritise foundational literacy, remedial learning, and adult literacy in ST communities; Target the ST-general population literacy gap with ST-focused school support and tracking; Add female literacy interventions: residential schooling, women teachers, safety, sanitation, and community outreach; Use district-level targeting instead of only state-wide averages |
| Nagaland | 10.00 | Economic vulnerability priority | high secondary dropout (16.79%); weak WPR (385 per 1000) | Strengthen secondary retention through hostels, transport, scholarships, and transition support; Connect upper-secondary schooling with local skills, placement, and public employment pathways |
| Chhattisgarh | 11.00 | Economic vulnerability priority | low ST literacy (59.10%); low female literacy (48.62%); high ST poverty (43.90%); high unmet MGNREG demand (233 per 1000); 2 low-female-literacy district(s) | Prioritise foundational literacy, remedial learning, and adult literacy in ST communities; Add female literacy interventions: residential schooling, women teachers, safety, sanitation, and community outreach; Bundle schooling support with livelihood, nutrition, and scholarship protection for poor ST households; Improve MGNREG work availability while reducing education costs for vulnerable households |
| Jammu and Kashmir | 12.00 | Education disadvantage priority | low ST literacy (50.60%); large literacy gap (16.60 pp); low female literacy (39.17%); 15 low-female-literacy district(s) | Prioritise foundational literacy, remedial learning, and adult literacy in ST communities; Target the ST-general population literacy gap with ST-focused school support and tracking; Add female literacy interventions: residential schooling, women teachers, safety, sanitation, and community outreach; Use district-level targeting instead of only state-wide averages |
| Gujarat | 13.00 | Monitor / comparatively lower priority | large literacy gap (15.60 pp); high secondary dropout (20.35%); 1 low-female-literacy district(s) | Target the ST-general population literacy gap with ST-focused school support and tracking; Strengthen secondary retention through hostels, transport, scholarships, and transition support; Use district-level targeting instead of only state-wide averages |
| Tripura | 15.00 | Economic vulnerability priority | weak WPR (372 per 1000) | Connect upper-secondary schooling with local skills, placement, and public employment pathways |
| Dadra and Nagar Haveli and Daman and Diu | 17.00 | Monitor / comparatively lower priority | comparatively stronger or incomplete-risk profile in available indicators | Maintain monitoring and protect gains, with targeted support for weaker districts or groups |
| Assam | 19.00 | Monitor / comparatively lower priority | weak WPR (386 per 1000); high unmet MGNREG demand (282 per 1000); many >50% ST villages (6626) | Connect upper-secondary schooling with local skills, placement, and public employment pathways; Improve MGNREG work availability while reducing education costs for vulnerable households; Use geographically concentrated service delivery: school clusters, mobile academic support, and local monitoring |

## Figures Generated

- `figures/st_literacy_rate_high_st_states.png`
- `figures/literacy_gap_high_st_states.png`
- `figures/secondary_dropout_high_st_states.png`
- `figures/st_poverty_high_st_states.png`
- `figures/st_worker_population_ratio_high_st_states.png`
- `figures/overall_priority_score_high_st_states.png`
- `figures/poverty_vs_literacy_high_st_states.png`
- `figures/literacy_vs_secondary_dropout_high_st_states.png`
- `figures/female_wpr_vs_female_literacy_high_st_states.png`
- `figures/education_vs_economic_vulnerability_high_st_states.png`
- `figures/correlation_heatmap_high_st_states.png`
- `figures/state_typology_clusters.png`

## Database / Fact Table Output

- `tables/fact_state_year_sparse.csv` preserves the state-year structure without forcing unmatched years into a false common-year table.
- `state_analysis_dataset_high_st_states.csv` remains the main policy profile table because it combines the best available indicators for cross-state prioritisation.
- The SQLite database was updated with a `fact_state_year_sparse` table for database demonstration and query support.

## Methodological Limitations

- The analysis combines indicators from different years, so results are cross-sectional and exploratory.
- Poverty data are available for fewer states than education or dropout data.
- The labour dataset does not include the rural/urban split mentioned in the proposal.
- State averages can hide district and tribe-level variation; support tables should be used for case-study depth.
- Composite scores depend on variable choice and equal-weight normalization; they are useful for prioritisation, not absolute measurement.

## Next Use In Final Report

Use this EDA report as the source for the report's Data Exploration, Analysis, Findings, and Recommendations sections. The database section should reference the SQLite database and the raw-clean-final structure already created.