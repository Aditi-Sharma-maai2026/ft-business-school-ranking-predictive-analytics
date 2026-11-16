# Predicting the FT European Business Schools Ranking

Master's thesis project (M.Sc. Analytics and AI, 2026). Python reconstruction of the Financial Times European Business Schools ranking formula, followed by statistical testing, machine-learning models and scenario simulation to identify the main drivers of a school's composite rank.

## Data availability

The underlying data is **confidential and not included** in this repository. It was provided by the institution for thesis research only. This repository contains the code and methodology; no raw records, school-level scores or internal model files are published.

The analysis used three types of source data:

| Source | Coverage | Used for |
|---|---|---|
| FT ranking export files | 2014–2025 | Composite rank history of the case-study school |
| FT ranking tables by programme (EMBA, MiM, MBA, Executive Education Custom and Open) | 2019–2025, about 1,964 records from 100+ schools | Raw indicators for all ranked schools |
| Institutional Excel reconstruction models | 2023–2025 | Ground truth for the formula reconstruction |

## Repository structure

| File | Purpose |
|---|---|
| `config.py` | Data folder, file names, sheet names and column positions |
| `make_synthetic_data.py` | Creates random, made-up Excel files so the pipeline can run without the confidential data |
| `run_all.py` | Runs all stages in order and saves charts to `outputs/` |
| `src/stage0_setup.py` | Imports and settings |
| `src/stage1_data_loading.py` | Loads the FT programme ranking tables and reconstruction workbooks |
| `src/stage2_variables_and_helpers.py` | FT indicator weights, cleaning helpers, gender parity adjustment |
| `src/stage3_formula_reconstruction.py` | Rebuilds the composite formula and verifies it against the institutional model |
| `src/stage4_eda.py` | Feature matrix, charts and correlation tests with Benjamini-Hochberg correction |
| `src/stage5_ml_sensitivity.py` | 10 models, rolling-origin cross-validation and Lasso feature selection |
| `src/stage6_statistical_tests.py` | OLS regression, t-test, VIF and sensitivity checks |
| `src/stage7_cross_sectional_ml_and_scenarios.py` | Cross-sectional models, structural break test and Monte Carlo scenario simulation |
| `src/stage8_summary.py` | Results summary and limitations |

## How to run

```bash
pip install -r requirements.txt
python make_synthetic_data.py   # only if you do not have the real data
python run_all.py
```

With the real data, place the Excel files in `data/` and adjust `config.py` to match their file names, sheet names and column positions. With synthetic data the code runs end to end, but the printed results are meaningless, and some summary text in stage 8 refers to the original thesis results.

## Variables

### Target variables

| Variable | Description |
|---|---|
| `composite_rank` | Published position in the FT European Business Schools ranking (lower = better) |
| `sub_rank` | A school's position in an individual programme ranking (e.g. MiM, EMBA) |
| `pct_rank` | Percentile rank = `sub_rank / n_schools`; normalises for the number of schools ranked each year |
| `final_score` | Composite score computed in Python from the reconstructed formula |
| `excel_score` | Composite score from the institutional Excel model, used to verify `final_score` |

### Formula reconstruction variables

| Variable | Description |
|---|---|
| `Sij` | Index score of school *j* in programme ranking *i*, rescaled to a 30–100 range: `S = 70 × (Z − MinZ) / (MaxZ − MinZ) + 30` |
| `sij_mba`, `sij_emba`, `sij_mim`, `sij_custom`, `sij_open` | Index score for each of the five programme rankings (0 when the school is not ranked in that programme) |
| `sum_sij` | Sum of a school's programme index scores |
| `n_active` | Number of programme rankings in which the school appears |

The composite score combines the programme index scores, adjusting for how many programmes a school takes part in.

### FT ranking indicators (model inputs)

Each programme ranking combines 17–21 weighted indicators (weights sum to 1.0 per programme). Indicators are either values where higher is better (e.g. percentages, salaries) or ranks where a lower number is better.

| Group | Indicators |
|---|---|
| Career outcomes | Weighted salary (US$), salary increase (%), career progress rank, aims achieved (%), employed at three months (%), value for money rank, careers service rank, alumni network rank, sector diversity rank |
| Gender diversity | Female faculty (%), female students (%), women on board (%) |
| Internationalisation | International faculty (%), international students (%), international board (%), international mobility / work mobility rank, international course experience rank |
| Academic quality | Faculty with doctorates (%), FT research rank |
| Sustainability | ESG and net zero teaching rank, carbon footprint rank |

### Engineered features (longitudinal model, one school, 2019–2025)

| Variable | Description |
|---|---|
| `emba_salary_inc`, `mim_salary_inc`, `mba_salary_inc` | Salary increase (%) reported in each programme ranking |
| `sal_avg` / `sal_combined` | Average salary increase across available programmes |
| `sal_roll2`, `sal_ewm` | Two-year rolling mean and exponentially weighted mean of salary increase |
| `sal_trend` | Year-on-year change in salary increase |
| `emba_intl_board`, `mim_intl_board`, `board_avg` | International board share (%) by programme and averaged |
| `board_roll2` | Two-year rolling mean of international board share |
| `women_board` | Women on board (%) |
| `emba_intl_fac`, `faculty_doc`, `ft_research` | International faculty (%), faculty with doctorates (%), FT research rank |
| `mba_active` | 1 if the MBA programme was ranked that year, else 0 (programme participation) |
| `n_programmes` | Number of programme rankings the school took part in |
| `mba_x_salary` | Interaction term: `mba_active × sal_combined` |
| `rank_lag1` | Previous year's composite rank |

### Cross-sectional features (all ranked schools, EMBA and MiM)

| Variable | Description |
|---|---|
| `school`, `year`, `prog` | Identifiers for school, ranking year and programme |
| `salary_inc` | Salary increase (%) |
| `intl_board`, `intl_faculty`, `intl_students` | International board, faculty and student shares (%) |
| `faculty_doc` | Faculty with doctorates (%) |
| `aims` | Aims achieved (%) |
| `ft_research` | FT research rank (EMBA only) |
| `women_board` | Women on board (%) |
| `n_schools` | Number of schools ranked in that programme and year |
| `is_esmt` | 1 for the case-study school, else 0 |

## Preprocessing

- **Numeric cleaning:** FT table cells such as `83 (100)` or `17=` (tied rank) are converted to plain numbers.
- **Gender parity fold:** for female faculty, female students and women on board, values above 50% are replaced by `100 − value`, so that distance from parity is scored. This step is applied in the institutional model but is not documented in the published literature.
- **Missing values:** where one programme's table was unusable for a year, salary features fall back to the available programme; a sensitivity check confirms the main coefficients are robust to this choice.

## Methods

Formula reconstruction verified against Excel (score difference of 0 to 10 decimal places); exploratory analysis with Pearson correlations and Benjamini-Hochberg false discovery rate correction; OLS regression with F-tests and VIF; 10 models (OLS, Bayesian Ridge, Ridge, Lasso, ElasticNet, Random Forest, Extra Trees, Gradient Boosting, LightGBM, XGBoost) with rolling-origin and leave-one-year-out cross-validation; Lasso feature selection; structural break test; and an OLS-based scenario simulation (10,000 runs).

## Key results

- Formula reconstruction matched the institutional Excel model exactly in all 3 verification years.
- Models trained on 590 Master in Management records from 100+ schools predicted the case-study school's Master in Management rank with a mean absolute error of 1.8 rank positions across five test years, below the published benchmark of 2.52 (Proudlove, 2011b).
- Programme participation is the dominant driver of the composite rank (r = −0.954).