"""
Settings for the project.
The thesis data is confidential and is not part of this repository.
"""
import os

# Where the Excel files would be kept (never uploaded)
DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# Which FT programme ranking tables exist, by year
RANKING_YEARS = {
    "emba":        [2019, 2020, 2021, 2022, 2023, 2024],
    "mim":         [2019, 2020, 2021, 2022, 2023, 2024],
    "mba":         [2020, 2021, 2022],
    "exed_custom": [2020, 2022, 2023, 2024, 2025],
    "exed_open":   [2020, 2022, 2023, 2024, 2025],
}
RANKING_FILE_PATTERN = "{prog}_{year}.xlsx"

# The workbooks used to check the rebuilt formula
RECON_FILE_PATTERN = "reconstruction_{year}.xlsx"
RECON_YEARS = [2023, 2024, 2025]
RECON_SHEETS = {2023: "Model 2023", 2024: "Model 2024", 2025: "Model 2025"}

# Which column holds which value in those workbooks (counting from 0)
RECON_COLS = {
    "school": 1,
    "sij_mba": 18, "sij_emba": 19, "sij_mim": 20,
    "sij_custom": 21, "sij_open": 22,
    "sum_sij": 30, "n_active": 32, "final_score": 35,
}
RECON_FIRST_DATA_ROW = 2

# Inputs for the scenario simulation. None means "take them from the data".
SCENARIO_SALARY_EMBA = None
SCENARIO_SALARY_MIM = None
SCENARIO_SALARY_PEAK = None


def ranking_files():
    """Builds the list of ranking table file names."""
    return {prog: {yr: RANKING_FILE_PATTERN.format(prog=prog, year=yr) for yr in yrs}
            for prog, yrs in RANKING_YEARS.items()}


def recon_files():
    """Builds the list of reconstruction workbook file names."""
    return {yr: RECON_FILE_PATTERN.format(year=yr) for yr in RECON_YEARS}