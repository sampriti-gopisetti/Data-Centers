# Figures

Exported from the midterm report's Data section (built as a Claude Doc from
`cumulative_modeling.csv` and `dc_registry.csv`; see `src/analysis/eda.py`
for the underlying numbers). Static PNGs here so the repo is self-contained
without the doc link. Numbering matches the doc's Fig 1-15.

**Dataset overview**
- `fig01_class_balance.png` — 1,727 case tracts vs. 7,840 matched controls.

**Where data centers are**
- `fig02_map_by_era.png` — tract locations colored by construction era (442
  known-year tracts).
- `fig03_map_by_poverty.png` — tract locations colored by poverty rate
  (400-tract sample).
- `fig04_top_states_by_case_count.png` — top 10 states by data-center tract
  count; these 10 states hold ~45% of all 1,727 data-center tracts.

**What distinguishes a data-center tract**
- `fig05_feature_heatmap.png` — share of tracts that are data centers, by %
  developed land and poverty rate. Concentration rises with developed land
  across poverty levels.
- `fig06_case_control_feature_diff.png` — median feature values, case vs.
  control, percent difference. Confirms the matching (density, RUCA class)
  worked: income and density land close to zero; the real gaps are land use
  and poverty rate.
- `fig07_correlation_with_is_case.png` — strongest 10 of 22 features checked
  by Pearson correlation with `is_case`. Headline finding: no single feature
  correlates strongly (all |r| < 0.11) — expected for a siting decision
  driven by feature combinations, not one factor, which is why Step 3 uses
  tree-based models rather than linear ones.

**Equity and environmental dimensions**
- `fig08_ej_burden.png` — EJScreen indices, case vs. control, percent
  difference. Traffic proximity is the largest gap (+44.8%).
- `fig09_race_composition.png` — race/ethnicity composition, case vs.
  control.
- `fig10_ruca_composition.png` — rural/urban classification, case vs.
  control. Nearly identical, confirming the RUCA matching worked.
- `fig11_climate_diff.png` — climate normals, case vs. control, percent
  difference. Data-center tracts are marginally cooler and drier.

**Change over time**
- `fig12_construction_timeline.png` — site count by construction year,
  2011-2025. Rising trend, peaking in 2022.
- `fig13_before_after_diff.png` — average change since construction,
  data-center tracts vs. matched controls (44 before/after cases). Less
  population growth, no improvement in poverty rate, more income/home-value
  growth than controls.

**Context**
- `fig14_top_operators.png` — top 10 operators by site count.
- `fig15_site_status.png` — registry sites by status.
