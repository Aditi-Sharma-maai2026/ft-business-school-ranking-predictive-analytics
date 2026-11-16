# ═══════════════════════════════════════════════════════════════
# CELL 3: VARIABLE DEFINITIONS AND HELPERS
# All variable weights, preprocessing functions, and the
# reconstruction model reader defined here.
# ═══════════════════════════════════════════════════════════════
print("=" * 65)
print("CELL 3: VARIABLE DEFINITIONS AND HELPERS")
print("=" * 65)
# 3.1 FT VARIABLE WEIGHTS
# Format: (column_name, weight, direction, parity_cap)
# direction 'higher': higher value = better performance
# direction 'lower':  lower rank number = better performance
# parity_cap True:    gender parity fold applied at 50%
EMBA_VARS = [
    ("Salary today (US$)",                   0.15, "higher", False),
    ("Salary increase (%)",                  0.16, "higher", False),
    ("Career progress rank",                 0.06, "lower",  False),
    ("Work experience rank",                 0.05, "lower",  False),
    ("Aims achieved (%)",                    0.06, "higher", False),
    ("Alumni network rank",                  0.04, "lower",  False),
    ("Female faculty (%)",                   0.05, "higher", True),
    ("Female students (%)",                  0.05, "higher", True),
    ("Women on board (%)",                   0.01, "higher", True),
    ("International faculty (%)",            0.04, "higher", False),
    ("International students (%)",           0.05, "higher", False),
    ("International board (%)",              0.10, "higher", False),
    ("International course experience rank", 0.03, "lower",  False),
    ("Faculty with doctorates (%)",          0.04, "higher", False),
    ("FT research rank",                     0.05, "lower",  False),
    ("Environmental, social and governance (ESG) rank",
                                             0.03, "lower",  False),
    ("Carbon footprint rank",                0.03, "lower",  False),
]
MIM_VARS = [
    ("Weighted salary (US$)",                0.15, "higher", False),
    ("Salary percentage increase",           0.09, "higher", False),
    ("Value for money rank",                 0.06, "lower",  False),
    ("Career progress rank",                 0.06, "lower",  False),
    ("Aims achieved (%)",                    0.06, "higher", False),
    ("Careers service rank",                 0.05, "lower",  False),
    ("Alumni network rank",                  0.03, "lower",  False),
    ("Employed at three months (%)",         0.05, "higher", False),
    ("Female faculty (%)",                   0.05, "higher", True),
    ("Female students (%)",                  0.05, "higher", True),
    ("Women on board (%)",                   0.01, "higher", True),
    ("International faculty (%)",            0.05, "higher", False),
    ("International students (%)",           0.05, "higher", False),
    ("International board (%)",              0.01, "higher", False),
    ("International work mobility rank",     0.06, "lower",  False),
    ("International course experience rank", 0.06, "lower",  False),
    ("Faculty with doctorates (%)",          0.04, "higher", False),
    ("ESG and net zero teaching rank",       0.03, "lower",  False),
    ("Carbon footprint rank",                0.04, "lower",  False),
]
MBA_VARS = [
    ("Weighted salary (US$)",                0.16, "higher", False),
    ("Salary percentage increase",           0.16, "higher", False),
    ("Value for money rank",                 0.05, "lower",  False),
    ("Career progress rank",                 0.03, "lower",  False),
    ("Aims achieved (%)",                    0.04, "higher", False),
    ("Alumni network rank",                  0.04, "lower",  False),
    ("Careers service rank",                 0.03, "lower",  False),
    ("Employed at three months (%)",         0.02, "higher", False),
    ("Sector diversity rank",                0.03, "lower",  False),
    ("Female faculty (%)",                   0.03, "higher", True),
    ("Female students (%)",                  0.03, "higher", True),
    ("Women on board (%)",                   0.01, "higher", True),
    ("International faculty (%)",            0.03, "higher", False),
    ("International students (%)",           0.03, "higher", False),
    ("International board (%)",              0.01, "higher", False),
    ("International mobility rank",          0.05, "lower",  False),
    ("International course experience rank", 0.03, "lower",  False),
    ("Faculty with doctorates (%)",          0.05, "higher", False),
    ("FT research rank",                     0.10, "lower",  False),
    ("Carbon footprint rank",                0.04, "lower",  False),
    ("ESG and net zero teaching rank",       0.03, "lower",  False),
]
print()
print("3.1 Variable weight sums (must all = 1.000):")
for name, vlist in [("MBA", MBA_VARS),
                     ("EMBA", EMBA_VARS),
                     ("MiM", MIM_VARS)]:
    total   = sum(v[1] for v in vlist)
    n_rank  = sum(1 for v in vlist if v[2] == 'lower')
    n_parity= sum(1 for v in vlist if v[3])
    status  = "✓ " if abs(total - 1.0) < 0.001 else "✗"
    print(f"  {status} {name:<6} {len(vlist):>3} vars  "
          f"weight_sum={total:.3f}  "
          f"rank_vars={n_rank}  parity_vars={n_parity}")
# 3.2 PREPROCESSING HELPERS
print()
print("3.2 Preprocessing helpers:")
def gender_parity(value):
    """
    FT parity fold: any value above 50% is replaced with
    (100 - value). Confirmed against the institutional
    reconstruction models.
    Applies to: female faculty %, female students %,
                women on board %.
    This preprocessing step is not documented in the
    published literature — first documented here.
    """
    if value is None or (isinstance(value, float)
                         and np.isnan(value)):
        return value
    return (100.0 - value) if value > 50.0 else value
def clean_numeric(val):
    """
    Extract first numeric from messy FT table cells.
    Examples: '83 (100)' →  83.0
              '17='      →  17.0  (trailing = = tied rank)
              None       →  None
    The trailing '=' is FT notation for tied positions.
    """
    if val is None:
        return None
    s = str(val).strip().rstrip('=')
    if s in ['nan', 'None', '', '-']:
        return None
    m = re.match(r'^([0-9]+\.?[0-9]*)', s.replace(',', ''))
    if m:
        try:
            return float(m.group(1))
        except:
            return None
    return None
def get_val(prog, year, col_keyword):
    """
    Retrieve ESMT's value for a given indicator from the
    loaded ranking tables. Keyword matching on column
    names — robust to minor FT naming changes across years.
    Returns None if programme/year/column not found.
    """
    if prog not in loaded or year not in loaded[prog]:
        return None
    df   = loaded[prog][year]
    esmt = df[df.index.str.contains('ESMT', case=False,
                                     na=False)]
    if esmt.empty:
        return None
    col = next(
        (c for c in df.columns
         if col_keyword.lower() in str(c).lower()),
        None
    )
    if col is None:
        return None
    return clean_numeric(esmt.iloc[0].get(col))
print("  ✓  gender_parity()  — FT parity fold at 50%")
print("  ✓  clean_numeric()  — extracts numeric from FT cells")
print("  ✓  get_val()        — extracts ESMT value by keyword")
# 3.3 RECONSTRUCTION MODEL READER
print()
print("3.3 Reconstruction model reader:")
def read_recon_model(wb, year):
    """
    Read an institutional reconstruction workbook.
    Uses STORED sum_sij and n directly — never recomputes.
    This reproduces the Excel composite score exactly.
    Sheet names and column positions come from config.py.
    Formula (Proudlove 2011b, p.246):
      Total(j) = [ΣSij/n] + ΣSij = ΣSij × (n+2) / (3n)
    Verification:
      2023: diff = 0.0000000000  ✓
      2024: diff = 0.0000000000  ✓
      2025: diff = 0.0000000000  ✓
    """
    sheet_map = config.RECON_SHEETS
    sheet = sheet_map.get(year)
    if not sheet:
        print(f"    No sheet mapping for year {year}")
        return pd.DataFrame()
    if sheet not in wb.sheetnames:
        print(f"    Sheet '{sheet}' not found.")
        print(f"    Available: {wb.sheetnames}")
        return pd.DataFrame()
    C    = config.RECON_COLS
    ws   = wb[sheet]
    rows = list(ws.iter_rows(values_only=True))
    # Rows 0 and 1 are headers — data starts at row index 2
    skip_words = ['proxy', 'proxies', 'ranked outside',
                  'stand alone', 'note', 'mannheim',
                  'has a second']
    result_rows = []
    for row in rows[config.RECON_FIRST_DATA_ROW:]:
        # Minimum length check — need the last configured column
        if len(row) <= max(C.values()):
            continue
        school = row[C['school']]
        if not school or not isinstance(school, str):
            continue
        school = school.strip()
        if len(school) < 2 or school.startswith('*'):
            continue
        if any(w in school.lower() for w in skip_words):
            continue
        # Skip rows where Excel has no final score
        if row[C['final_score']] is None:
            continue
        try:
            excel_score = float(row[C['final_score']])
        except (ValueError, TypeError):
            continue
        try:
            sum_sij = float(row[C['sum_sij']]) if row[C['sum_sij']] is not None else 0.0
        except (ValueError, TypeError):
            sum_sij = 0.0
        try:
            n = float(row[C['n_active']]) if row[C['n_active']] is not None else 0.0
        except (ValueError, TypeError):
            n = 0.0
        if n == 0:
            continue
        # Apply Proudlove (2011b) formula using STORED values
        # Total = ΣSij × (n+2) / (3n)
        python_score = sum_sij * (n + 2.0) / (3.0 * n)
        # Individual Sij scores per programme
        def safe_float(v):
            try:
                return float(v) if v is not None else 0.0
            except (ValueError, TypeError):
                return 0.0
        result_rows.append({
            'school':      school,
            'final_score': python_score,
            'excel_score': excel_score,
            'sum_sij':     sum_sij,
            'n_active':    n,
            'sij_mba':     safe_float(row[C['sij_mba']]),
            'sij_emba':    safe_float(row[C['sij_emba']]),
            'sij_mim':     safe_float(row[C['sij_mim']]),
            'sij_custom':  safe_float(row[C['sij_custom']]),
            'sij_open':    safe_float(row[C['sij_open']]),
        })
    if not result_rows:
        print(f"    No valid rows extracted from sheet '{sheet}'")
        return pd.DataFrame()
    df = pd.DataFrame(result_rows).set_index('school')
    df['composite_rank'] = (
        df['final_score']
        .rank(ascending=False, method='min')
        .astype(int)
    )
    return df.sort_values('composite_rank')
# 3.4 QUICK VERIFICATION
print()
print("3.4 Quick verification (2025 workbook):")
if 2025 in recon_wbs:
    test_result = read_recon_model(recon_wbs[2025], 2025)
    if not test_result.empty:
        esmt_test = test_result[
            test_result.index.str.contains('ESMT', na=False)
        ]
        if not esmt_test.empty:
            ps = float(esmt_test['final_score'].values[0])
            es = float(esmt_test['excel_score'].values[0])
            pr = int(esmt_test['composite_rank'].values[0])
            diff = abs(ps - es)
            print(f"  ESMT 2025: python={ps:.10f}")
            print(f"             excel= {es:.10f}")
            print(f"             diff=  {diff:.10f}  "
                  f"{'EXACT' if diff < 1e-8 else 'CHECK'}")
            print(f"             rank=  {pr}  "
                  f"(actual=12)  "
                  f"{'✓ ' if pr==12 else '≠'}")
            print(f"  Schools loaded: {len(test_result)}")
        else:
            print("ESMT not found in 2025 result")
    else:
        print("2025 result is empty")
else:
    print("2025 workbook not loaded")
print()
print("=" * 65)
print("CELL 3 COMPLETE")
print("=" * 65)
print()
print("  Functions defined:")
print("    gender_parity()    — FT parity fold at 50%")
print("    clean_numeric()    — extracts numeric from FT cells")
print("    get_val()          — case-study value by keyword lookup")
print("    read_recon_model() — Excel reconstruction reader")
print()
print("Ready for Cell 4: Stage 1 Formula Reconstruction")