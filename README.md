# Predictive Analytics Modelling: A Case Study of the Financial Times European Business Schools Ranking

**Python implementation accompanying the ESMT Master's Thesis, Berlin, 2026**
Arshia Gupta and Aditi Sharma
Advisors: Prof. Harald Hungenberg, Margarita Kriger


## Overview

This notebook reconstructs the Financial Times (FT) European Business Schools Ranking
formula in Python, verifies it exactly against ESMT Berlin's internal Excel models, and
tests whether machine learning can improve on that reconstruction as a forecasting tool.
It runs end-to-end on 45 Excel workbooks (embargoed FT indicator tables + ESMT's internal
reconstruction models) and produces every table, figure, and statistic reported in the
thesis.

**Headline results reproduced by this notebook**

| Finding | Result |
|---|---|
| Formula reconstruction accuracy | Exact match to ESMT's Excel model (score diff = 0.0000000000, all 3 verification years) |
| MiM cross-sectional prediction | OLS/Ridge, MAE = 1.8 (n = 590), beats Proudlove (2011b) benchmark of 2.52 |
| EMBA cross-sectional prediction | Best model (Random Forest), MAE = 20.5 (n = 500) ” fails due to a governance structural break |
| Dominant driver of composite rank | Programme participation (r = 0.954, p = 0.0008 after BH correction), not salary (r = 0.014, p = 0.976) |
| 2026 authoritative estimate | Rank 12 Or 13 (Stage 1, anchored to verified 2025 values) |

## Notebook Structure

The analysis is a single Jupyter notebook, organised into 9 cells that run sequentially
(each cell depends on variables created by the ones before it, always run top to bottom):

| Cell | Name | What it does |
|---|---|---|
| 1 | Environment Setup | All imports; checks for XGBoost/LightGBM; sets `DATA_PATH` and confirms the 45 Excel files are found |
| 2 | Data Loading | Builds the 2014-2025 composite ranking history; defines `clean_numeric()` and `gender_parity()`; loads the FT embargoed ranking tables via `load_embargoed()` and `get_val()` |
| 3 | Variable Definitions and Helpers | Re-defines preprocessing helpers for the reconstruction stage; `read_recon_model()` parses ESMT's Excel reconstruction workbooks (2023-2025); quick verification of loaded values |
| 4 | Stage 1 â€” Formula Reconstruction | Reconstructs the FT composite score for all European schools per year, ranks them, validates ESMT's score/rank against Excel to 10 decimal places, produces the Top-20 table and `4_stage1_reconstruction.png` |
| 5 | Exploratory Data Analysis & Feature Matrix | Engineers features (`sal_roll2`, `sal_ewm`, `board_roll2`, `sal_trend`, `mba_x_salary`, `rank_lag1`); produces indicator trend charts (`5_eda.png`) and Benjaminiâ€“Hochberg-corrected Pearson correlations |
| 6 | Stage 2A - ML Sensitivity Analysis | Trains 10 models (OLS, XGBoost) on ESMT's 7-year panel (n=5 train / n=2 test); rolling-origin CV; Lasso feature selection (`run_model()`) **for feature selection and bias-variance demonstration only, not model comparison** |
| 7 | Statistical Testing | Manual OLS via `numpy.linalg.lstsq` (`run_ols()`, `print_ols()`); ANOVA; MBA-active vs. MBA-absent t-test; Shapiro-Wilk/Jarque-Bera normality checks; VIF computation; 2022-imputation sensitivity check |
| 8 | Stage 2C (Primary ML) + Stage 3 Scenario Simulation | `build_cross_section()` pools ~100 schools/year into MiM (n=590) and EMBA (n=500) panels; `run_cs_model()` runs leave-one-year-out validation across all 10 models; `simulate()` runs the N=10,000-draw OLS scenario simulation; produces `8_scenario_simulation.png` |
| 9 | Final Summary and Conclusions | Assembles the complete results table, ranking history, hypothesis outcomes, leverage ranking, 2026 prediction, limitations, and contributions |

---

## Data Requirements

The notebook expects **all 45 source Excel files in a single flat folder** (no
subdirectories) pointed to by `DATA_PATH` in Cell 1:

```python
DATA_PATH = os.path.expanduser("~/Desktop/esmt_thesis/")
```

That folder contain:

| File group | Years | Role |
|---|---|---|
| FT export ranking files | 2014-2025 | composite rank history (hardcoded into `composite_df` in Cell 2, not read from Excel) |
| FT embargoed ranking tables | MBA (2020-22, 2025); EMBA (2019-24); MiM (2019â€“24); ExEd (2020, 2022-25) | raw indicator values, loaded via `load_embargoed()` |
| ESMT Excel reconstruction models | 2023-2025 | ground-truth verification for Stage 1, parsed via `read_recon_model()` |
| ESMT filled survey submissions | MBA 2025; MiM 2023-24; EMBA 2023-24 | preprocessing cross-validation |
| FT blank survey templates | all 5 programmes, 2019-2025 | variable definitions only, not loaded programmatically |

Cell 1 will report `Data path found` and the count of `.xlsx` files found; if it reports
`Data path NOT found`, update `DATA_PATH` and re-run before continuing.

> **Data access note:** the embargoed FT tables and ESMT's internal Excel models are provided under embargo to participating schools and are not publicly redistributable. This notebook assumes you already have local access to those files.


## Setup

```bash
# create environment
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# install dependencies
pip install -r requirements.txt
```

**requirements.txt** (everything imported in Cell 1):

```
numpy
pandas
openpyxl
matplotlib
seaborn
scipy
scikit-learn
xgboost
lightgbm
```

XGBoost and LightGBM are wrapped in `try/except` in Cell 1” the notebook will still run
without them, but Stage 2A/2C model comparisons will be missing those two models.

Note: OLS/regression statistics in Cell 7 are computed manually via `numpy.linalg.lstsq`
rather than `statsmodels`, so `statsmodels` is not a dependency.


## Running the Notebook

1. Place all 45 source Excel files in one folder.
2. Open the notebook and set `DATA_PATH` in **Cell 1** to that folder's path.
3. Run all cells **in order, top to bottom** â€” later cells depend on DataFrames and
   dictionaries built in earlier ones (e.g. Cell 4's `reconstructed_scores` feeds Cell 9's
   summary; Cell 5's engineered features feed Cell 6).
4. Three PNG figures are written to the working directory during the run:
   - `4_stage1_reconstruction.png` â€” Python vs. Excel score verification (Stage 1)
   - `5_eda.png`  ESMT ranking analytics / indicator trends (EDA)
   - `8_scenario_simulation.png` Stage 3 scenario distributions
5. Cell 9 prints the complete results summary, matching Table 7.18 in the thesis.

Re-running the notebook each year with newly released embargoed data (updated `DATA_PATH`
contents) re-verifies the formula, retrains the cross-sectional models, and regenerates
the forecast â€” the notebook is designed to be re-run annually, not used once.


## Key Functions

| Function | Cell | Purpose |
|---|---|---|
| `clean_numeric(val)` | 2 | Extracts a numeric value from FT's human-formatted cells (e.g. `"83 (100)"`, `83.0`, `"17="` `17.0`) |
| `gender_parity(value)` | 2 | Applies the FT's gender-parity fold: values above 50% become `100 - value` |
| `load_embargoed(filepath)` | 2 | Loads and parses an FT embargoed ranking table |
| `get_val(prog, year, col_keyword)` | 2 | Pulls a specific indicator value for ESMT from a loaded embargoed table |
| `read_recon_model(wb, year)` | 3 | Parses ESMT's internal Excel reconstruction workbook, reading the stored `SUMPRODUCT` weighted average (column AE) and programme count (column AG) |
| `run_model(model, X_tr, y_tr, X_te)` | 6 | Fits a given sklearn model and returns rank predictions, used across all 10 models in Stage 2A |
| `run_ols(X, y, var_names)` / `print_ols(res, title)` | 7 | Manual OLS via `numpy.linalg.lstsq`; returns/prints coefficients, SEs, t-stats, p-values, RÂ², F-test |
| `build_cross_section(prog, keyword_map)` | 8 | Pools all European schools' indicator data across years into one panel (MiM: 590 obs, EMBA: 500 obs) |
| `run_cs_model(df, features, model, train_years, test_year)` | 8 | Runs one leave-one-year-out fold for a given model on the cross-sectional panel |
| `simulate(mba, sal_mu, sal_sigma=2.0, n=N_SIM)` | 8 | Runs the Stage 3 Monte Carlo scenario simulation (`N_SIM = 10,000` draws) |


## Methodology Summary

- **Formula:** S(i,j) = 70 × [(Z(i,j) − MinZ(i)) / (MaxZ(i) − MinZ(i))] + 30, aggregated across five
  weighted programme sub-rankings (MBA 25%, EMBA 25%, MiM 25%, ExEd Custom 12.5%,
  ExEd Open 12.5%), following Proudlove (2011b), with the gender-parity fold applied to
  three diversity indicators before standardisation.
- **Stage 1 verification:** reads ESMT's stored `SUMPRODUCT`-weighted average directly
  from the Excel workbook rather than recomputing from individual S(i,j) values, since the
  Excel model applies the FT's official programme weights internally.
- **Validation:** rolling-origin cross-validation in Stage 2A (train on past years, test
  on the next) and leave-one-year-out validation in Stage 2C, never standard k-fold, to
  avoid temporal leakage.
- **Models (10, both stages):** OLS, Bayesian Ridge, Ridge, Lasso, ElasticNet, Random
  Forest, Extra Trees, Gradient Boosting, LightGBM, XGBoost.
- **Regularisation:** Lasso/ElasticNet use alpha = 0.5 in Stage 2A (n=5) and alpha = 0.05 in
  Stage 2C (400-500), reflecting the theoretical decrease in optimal regularisation
  strength as sample size grows.
- **Primary metric:** Mean Absolute Error in rank positions, matching Proudlove (2011b)
  for direct comparability.
- **Statistical testing:** Benjamin-Hochberg corrected Pearson correlations, OLS with
  F-test as primary evidence (individual t-statistics unreliable once VIF hits the
  implementation ceiling of 999), independent-samples t-test, Lasso feature selection as a
  robustness check.

---

## Key Limitations

- **Small N at the ESMT-specific level (n = 7 years).** Stage 2A is feature-selection and
  bias-variance evidence only and its MAE values are explicitly excluded from the Cell 9
  results table, since selecting a "best" model from 10 candidates on 2 test points has an
  expected false-positive rate near 50%.
- **VIF hits the implementation ceiling (999)** in the Stage 3 OLS model at n=6 with an
  interaction term and” individual coefficients are not used as standalone evidence; the
  F-test and group t-test are primary.
- **Stage 3 scenario bounds are circular** coefficient draw bounds are derived from the
  same 6-year dataset being analysed (disclosed explicitly in Cell 8's output). Stage 1
  remains the authoritative point estimate.
- **Single-institution design.** All Stage 2A/Stage 3 results are ESMT-specific; Stage 2C
  pools other European schools but only to predict ESMT's position.
- **Formula verification scope.** Python reproduces ESMT's Excel model exactly (diff =
  0.0000000000); whether that Excel model itself matches the FT's internal calculation
  cannot be independently verified.


## Citation

> Gupta, A., and Sharma, A. (2026). *Predictive Analytics Modelling: A Case Study of the Financial Times European Business Schools Ranking* [Master's thesis, ESMT European School of Management and Technology, Berlin].

## Acknowledgements

Supervised by Prof. Harald Hungenberg and Margarita Kriger, ESMT Berlin. Built on FT
embargoed ranking data and ESMT's internal reconstruction models, provided for academic
use with ESMT Berlin's clearance. AI-assisted tools (Claude, ChatGPT, Copilot) supported
brainstorming, coding, debugging, and language refinement during development; all
analysis, interpretation, and conclusions are the authors' own.

## Data Ethics Note

The FT embargoed tables and ESMT's internal Excel models are used here only in aggregate,
cross-sectional analyses. No individual competitor school's raw indicator data is
reproduced or reported. FT's published ranking files (2014-2025) are publicly available at
rankings.ft.com.
