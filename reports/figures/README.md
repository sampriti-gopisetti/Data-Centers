# Figures

Exported from the midterm report's Data section (built as a Claude Doc from
`cumulative_modeling.csv`; see `src/analysis/eda.py` for the underlying
numbers). Static PNGs here so the repo is self-contained without the doc link.

- `01_class_balance.png` — case vs. control row counts.
- `02_case_control_feature_diff.png` — median feature values, case vs.
  control, percent difference. Confirms the matching (density, RUCA class)
  worked: income and density land close to zero; the real gaps are land use
  and poverty rate.
- `03_top_states_by_case_count.png` — top 10 states by data-center tract
  count.
- `04_correlation_with_is_case.png` — strongest 10 of 22 features checked by
  Pearson correlation with `is_case`. Headline finding: no single feature
  correlates strongly (all |r| < 0.11) — expected for a siting decision
  driven by feature combinations, not one factor, which is why Step 3 uses
  tree-based models rather than linear ones.
