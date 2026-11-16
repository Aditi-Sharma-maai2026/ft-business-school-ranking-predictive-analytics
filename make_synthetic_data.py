"""
Create a synthetic, randomly generated dataset so the pipeline can be
run without the confidential thesis data. The numbers are made up and
carry no information about any real school.

Usage:  python make_synthetic_data.py   (writes Excel files to config.DATA_PATH)
"""
import os
import numpy as np
import openpyxl
import config

rng = np.random.default_rng(7)
N_SCHOOLS = 100
CASE_NAME = "ESMT Berlin"

COLUMNS = {
    "emba": ["Salary today (US$)", "Salary increase (%)", "International board (%)",
             "Women on board (%)", "International faculty (%)", "International students (%)",
             "Faculty with doctorates (%)", "Aims achieved (%)", "FT research rank"],
    "mim": ["Weighted salary (US$)", "Salary percentage increase", "International board (%)",
            "Women on board (%)", "International faculty (%)", "International students (%)",
            "Faculty with doctorates (%)", "Aims achieved (%)"],
    "mba": ["Weighted salary (US$)", "Salary percentage increase", "International board (%)",
            "Women on board (%)", "International faculty (%)", "FT research rank"],
    "exed_custom": ["Preparation", "Course design", "Teaching methods", "International clients"],
    "exed_open": ["Preparation", "Course design", "Teaching methods", "International participants"],
}


def school_names():
    return [CASE_NAME] + [f"School {i:03d}" for i in range(1, N_SCHOOLS)]


def ranking_table(prog, year, path):
    names = school_names()
    quality = rng.normal(0, 1, len(names))
    order = np.argsort(-quality)
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append([f"FT {prog.upper()} table {year} (synthetic data)"])
    ws.append([])
    ws.append([])
    ws.append([f"Rank in {year}", "Rank prev 1", "Rank prev 2", "3-year average", "School name",
               "Country"] + COLUMNS[prog])
    for rank, idx in enumerate(order, 1):
        q = quality[idx]
        vals = []
        for col in COLUMNS[prog]:
            if "rank" in col.lower():
                vals.append(int(np.clip(round(50 - 15 * q + rng.normal(0, 10)), 1, 100)))
            elif "US$" in col:
                vals.append(int(120000 + 25000 * q + rng.normal(0, 8000)))
            else:
                vals.append(round(float(np.clip(45 + 10 * q + rng.normal(0, 8), 5, 95)), 1))
        ws.append([rank, None, None, None, names[idx], "Country"] + vals)
    wb.save(path)


def recon_workbook(year, path):
    C = config.RECON_COLS
    width = max(C.values()) + 1
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = config.RECON_SHEETS[year]
    ws.append(["Synthetic reconstruction model"] + [None] * (width - 1))
    ws.append(["header"] + [None] * (width - 1))
    for name in school_names()[:35]:
        row = [None] * width
        row[C["school"]] = name
        sij = []
        for key in ["sij_mba", "sij_emba", "sij_mim", "sij_custom", "sij_open"]:
            v = float(np.clip(rng.normal(65, 15), 30, 100)) if rng.random() > 0.25 else 0.0
            row[C[key]] = v
            sij.append(v)
        n = sum(1 for v in sij if v > 0) or 1
        s = sum(sij)
        row[C["sum_sij"]] = s
        row[C["n_active"]] = n
        row[C["final_score"]] = s * (n + 2.0) / (3.0 * n)
        ws.append(row)
    wb.save(path)


if __name__ == "__main__":
    os.makedirs(config.DATA_PATH, exist_ok=True)
    count = 0
    for prog, files in config.ranking_files().items():
        for year, fname in files.items():
            ranking_table(prog, year, os.path.join(config.DATA_PATH, fname))
            count += 1
    for year, fname in config.recon_files().items():
        recon_workbook(year, os.path.join(config.DATA_PATH, fname))
        count += 1
    print(f"Wrote {count} synthetic Excel files to {config.DATA_PATH}")