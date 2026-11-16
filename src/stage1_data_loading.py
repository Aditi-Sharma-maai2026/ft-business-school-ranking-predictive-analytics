# ═══════════════════════════════════════════════════════════════
# CELL 2: DATA LOADING
# Sources:
#   1. FT export ranking files (2014-2025) — composite history
#   2. FT programme ranking tables (2019-2025) — raw indicators
#   3. Institutional reconstruction workbooks (2023-2025) — Sij values
# ═══════════════════════════════════════════════════════════════
print("=" * 65)
print("CELL 2: DATA LOADING")
print("=" * 65)
# 2.1 COMPOSITE RANKING HISTORY
print()
print("2.1 ESMT Composite Ranking History (2014-2025)")
print("-" * 55)
composite_df = pd.DataFrame({
    'composite_rank': [21,16,21,18,24,9,9,7,8,19,19,12],
    'mba_active':     [1,1,1,1,1,1,1,1,1,0,0,1],
    'mim_active':     [0,0,0,0,0,1,1,1,1,1,1,1],
    'exed_active':    [0,0,0,0,0,0,0,1,1,1,1,1],
    'n_programmes':   [2,2,2,2,2,3,3,5,5,4,4,5],
}, index=[2014,2015,2016,2017,2018,
          2019,2020,2021,2022,2023,2024,2025])
composite_df.index.name = 'year'
display(composite_df)
# 2.2 CLEANING HELPERS
def clean_numeric(val):
    """
    Extract first numeric from messy FT table cells.
    Examples: '83 (100)' →  83.0 | '17=' →  17.0
    The trailing '=' appears in FT tables to denote tied ranks.
    """
    if val is None:
        return None
    s = str(val).strip().rstrip('=')
    if s in ['nan','None','','-']:
        return None
    m = re.match(r'^([0-9]+\.?[0-9]*)', s.replace(',',''))
    if m:
        try:
            return float(m.group(1))
        except:
            return None
    return None
def gender_parity(value):
    """
    FT parity fold: values above 50% →  (100 - value).
    Confirmed against the institutional reconstruction models.
    Applies to: female faculty %, female students %,
                women on board %.
    """
    if value is None or (isinstance(value, float)
                         and np.isnan(value)):
        return value
    return (100.0 - value) if value > 50.0 else value
# 2.3 LOAD PROGRAMME RANKING TABLES
print()
print("2.3 Loading Programme Ranking Tables")
print("-" * 55)
ranking_files = config.ranking_files()
def load_ranking_table(filepath):
    wb   = openpyxl.load_workbook(filepath, data_only=True)
    ws   = wb.active
    data = list(ws.iter_rows(values_only=True))
    header_idx = None
    for i, row in enumerate(data):
        if row[0] and 'rank' in str(row[0]).lower():
            header_idx = i; break
    if header_idx is None:
        header_idx = 3
    header = list(data[header_idx])
    seen   = {}
    uniq   = []
    for h in header:
        s = str(h).strip() if h else 'unnamed'
        if s in seen:
            seen[s] += 1; uniq.append(f"{s}_{seen[s]}")
        else:
            seen[s] = 0; uniq.append(s)
    df = pd.DataFrame(data[header_idx+1:], columns=uniq)
    # School name is at column index 4 in all FT tables
    school_col = df.columns[4]
    df = df.rename(columns={school_col: 'School'})
    df = df[df['School'].apply(
        lambda x: isinstance(x,str) and len(x.strip()) > 1
    )]
    df = df.drop_duplicates(subset='School', keep='first')
    df = df.set_index('School')
    return df
loaded = {}
print(f"  {'Programme':<14} {'Year':<6} {'Schools':>8} {'ESMT':>6}")
print("  " + "-" * 38)
for prog, year_files in ranking_files.items():
    loaded[prog] = {}
    for year, fname in year_files.items():
        fpath = os.path.join(DATA_PATH, fname)
        if not os.path.exists(fpath):
            continue
        try:
            df = load_ranking_table(fpath)
            if len(df) < 10:
                continue
            loaded[prog][year] = df
            esmt_found = df.index.str.contains(
                'ESMT', case=False, na=False
            ).any()
            print(f"  {prog:<14} {year:<6} {len(df):>8} "
                  f"{'✓ ' if esmt_found else '✗':>6}")
        except Exception as e:
            print(f"  {prog:<14} {year:<6} {'ERROR':>8} "
                  f"{str(e)[:30]}")
total_obs = sum(len(d) for p in loaded for d in loaded[p].values())
print(f"\n  Total ranking-table observations: {total_obs}")
# 2.4 HELPER: GET ESMT VALUE FROM A RANKING TABLE
def get_val(prog, year, col_keyword):
    """
    Retrieve ESMT's value for a given indicator from
    the ranking tables. Returns None if not found.
    Uses keyword matching on column names — robust to
    minor FT column name changes across years.
    """
    if prog not in loaded or year not in loaded[prog]:
        return None
    df   = loaded[prog][year]
    esmt = df[df.index.str.contains('ESMT', case=False,
                                     na=False)]
    if esmt.empty:
        return None
    col = next((c for c in df.columns
                if col_keyword.lower() in str(c).lower()), None)
    if col is None:
        return None
    return clean_numeric(esmt.iloc[0].get(col))
# 2.5 LOAD RECONSTRUCTION WORKBOOKS
print()
print("2.5 Loading Reconstruction Model Workbooks")
print("-" * 55)
recon_files = config.recon_files()
recon_wbs = {}
for year, fname in recon_files.items():
    fpath = os.path.join(DATA_PATH, fname)
    if os.path.exists(fpath):
        wb = openpyxl.load_workbook(fpath, data_only=True)
        recon_wbs[year] = wb
        print(f"  {year}:  sheets={wb.sheetnames[:3]}")
    else:
        print(f"  {year}:  NOT FOUND — {fname}")
print()
print("=" * 65)
print("CELL 2 COMPLETE")
print("=" * 65)
print(f"\n  Composite history: 12 years (2014-2025)")
print(f"  Ranking-table obs: {total_obs}")
print(f"  Recon models:      {len(recon_wbs)} years")
print("\nReady for Cell 3: Variable Definitions")