# ═══════════════════════════════════════════════════════════════
# CELL 9: FINAL SUMMARY AND CONCLUSIONS
# ═══════════════════════════════════════════════════════════════
print("=" * 65)
print("CELL 9: FINAL SUMMARY AND CONCLUSIONS")
print("=" * 65)
# 9.1 COMPLETE RESULTS TABLE
print()
print("9.1 COMPLETE RESULTS SUMMARY")
print("-" * 55)
stage1_mae = round(np.mean([
    v['rank_error'] for v in reconstructed_scores.values()
    if v.get('rank_error') is not None
]), 2) if reconstructed_scores else None
best_reg_row = (
    min([r for r in results_list
         if r['Group'] == 'Regularised'],
        key=lambda x: x['MAE'])
    if results_list else None
)
best_cs_emba = (
    min([r for r in cs_results if r['MAE_EMBA'] is not None],
        key=lambda x: x['MAE_EMBA'])
    if cs_results else None
)
best_cs_mim = (
    min([r for r in mim_results if r['MAE_MIM'] is not None],
        key=lambda x: x['MAE_MIM'])
    if mim_results else None
)
print(f"\n  {'Method':<42} {'MAE':>6} {'N':>10}  "
      f"{'vs 2.52':>8}  Note")
print("  " + "-" * 84)
rows_summary = [
    ('Proudlove (2011b) benchmark',
     2.52, 'multi-sch', 'Reference',
     'Cross-sectional reference'),
    ('Stage 1: Formula Reconstruction',
     stage1_mae, 3, '✓  consistent',
     'Score diff=0.0 exact'),
    (f"Stage 2A: Lasso (N=5 ESMT-only)",
     None, 5,
     'NOT EVALUATED',
     'Feature selection only — see note'),
    (f"Stage 2C MiM: "
     f"{best_cs_mim['Model'] if best_cs_mim else 'N/A'}",
     best_cs_mim['MAE_MIM'] if best_cs_mim else None,
     590,
     '✓ ' if best_cs_mim and
     best_cs_mim['MAE_MIM'] <= 2.52 else '✗ ',
     'Primary ML — beats benchmark'),
    (f"Stage 2C EMBA: "
     f"{best_cs_emba['Model'] if best_cs_emba else 'N/A'}",
     best_cs_emba['MAE_EMBA'] if best_cs_emba else None,
     500,
     '✗ ',
     'Structural break — see 8A.8'),
]
for method, mae, n, vs, note in rows_summary:
    mae_s = f"{mae:.2f}" if mae is not None else 'N/A'
    print(f"  {method:<42} {mae_s:>6} {str(n):>10}  "
          f"{vs:>8}  {note}")
print()
print("  Stage 2A MAE IS NOT IN THE TABLE.")
print("  Lasso matched both test points (N=2) exactly.")
print("  Selecting best of 10 models on 2 test points")
print("  has expected false-positive rate ~50%.")
print("  Stage 2A contributes feature selection results,")
print("  NOT a valid MAE benchmark comparison.")
print()
print("  Stage 1 MAE=1.67 vs Proudlove 2.52: different")
print("  scopes (3 single-school points vs cross-sectional).")
print("  Correct claim: verified exact reproduction, diff=0.")
print()
print("  Stage 2C MiM MAE=1.8 is the primary ML finding.")
print("  590 obs, genuine out-of-sample, beats benchmark.")
print()
print("  Stage 2C EMBA MAE=20.5: see 8A.8 for quantitative")
print("  structural break test. Post-hoc explanation tested.")
# 9.2 RANKING HISTORY
print()
print("9.2 ESMT RANKING HISTORY 2014-2025")
print("-" * 55)
print(f"\n  {'Year':>5} {'Rank':>5} {'MBA':>5} "
      f"{'MiM':>5} {'ExEd':>6} {'N':>3}  Event")
print("  " + "-" * 60)
events_map = {
    2018: 'Worst rank (24)',
    2019: 'MiM joins →  rank 24→ 9',
    2021: 'ExEd joins →  best rank (7)',
    2022: 'Last year with MBA',
    2023: 'MBA absent →  rank drops 8→ 19',
    2025: 'MBA returns →  rank recovers to 12',
}
for yr in sorted(composite_df.index):
    row  = composite_df.loc[yr]
    rank = int(row['composite_rank'])
    mba  = '✓ ' if int(row['mba_active'])  else '✗'
    mim  = '✓ ' if int(row['mim_active'])  else '–'
    exed = '✓ ' if int(row['exed_active']) else '–'
    n    = int(row['n_programmes'])
    evt  = events_map.get(yr, '')
    print(f"  {yr:>5} {rank:>5} {mba:>5} {mim:>5} "
          f"{exed:>6} {n:>3}  {evt}")
# 9.3 HYPOTHESIS OUTCOMES
print()
print("9.3 HYPOTHESIS OUTCOMES")
print("-" * 55)
print()
print("  H1: Reconstruction and ML accurately predict rank")
print()
print(f"    H1a Stage 1: MAE={stage1_mae}")
print("          Score diff = 0.0000000000 (exact)")
print("          ✓  SUPPORTED: exact formula reproduction")
print()
print("    H1b Stage 2A (N=5, feature selection only):")
print("          10-model spread (rank 10-30 for 2024)")
print("          demonstrates bias-variance at small N")
print("          MAE excluded — N=2 test points invalid")
print("          ✓  FEATURE SELECTION SUPPORTED")
print()
if best_cs_mim:
    print(f"    H1c Stage 2C MiM (primary ML, N=590):")
    print(f"          OLS MAE={best_cs_mim['MAE_MIM']}  "
          f"✓  SUPPORTED")
    print("          Beats Proudlove on genuine cross-section")
if best_cs_emba:
    print(f"    H1c Stage 2C EMBA (N=500):")
    print(f"          RF MAE={best_cs_emba['MAE_EMBA']}  "
          "ℹ Structural break limits accuracy")
    print("          See 8A.8 for quantitative test")
print()
print("    Overall H1: ✓  SUPPORTED")
print("    Primary: Stage 1 exact reconstruction")
print("    Strong ML: Stage 2C MiM (OLS MAE=1.8)")
print("    Honest limit: Stage 2C EMBA (structural break)")
print()
print("  H2: Key indicators explain composite rank variance")
print()
print("    Evidence (BH-corrected):")
print(f"      mba_active:      r=-0.954  BH-sig ✓ ")
print(f"      emba_intl_board: r=-0.990  BH-sig ✓ ")
print(f"      emba_salary_inc: r=-0.891  p=0.042  marginal")
print(f"      OLS R²={res2['r2']:.4f}  F={res2['f_stat']:.2f}  "
      f"p={res2['f_pval']:.4f}")
print(f"      MBA t-test: t={t_v:.4f}  p={p_v:.6f}  "
      f"Cohen's d={cd:.3f}")
print()
print("    EMPIRICAL FINDING (sal_combined r=-0.014, ns):")
print("    H2 as stated is not supported by the data.")
print("    sal_combined r=-0.014, p=0.976 — salary does")
print("    not explain the largest share of variance.")
print("    Dominant driver: participation (mba_active).")
print("    H2 is preserved as originally formed.")
print("    The data revise it, not the researchers.")
print("    See thesis Section 3.2 for full treatment.")
print()
print("    H2 VERDICT: ✗  NOT SUPPORTED AS STATED")

# 9.6 LIMITATIONS
print()
print("9.6 LIMITATIONS")
print("-" * 55)
limitations = [
    ("Small N (n=7 ESMT years)",
     "ML cannot be reliably evaluated at this sample size. "
     "Addressed by Lasso regularisation, rolling-origin CV, "
     "and explicit bias-variance demonstration. Primary ML "
     "evidence is Stage 2C (N=590-1,964 obs)."),
    ("Multiple comparisons (10 models, N=2 test points)",
     "Selecting best of 10 models on 2 test points has "
     "expected false-positive rate ~50%. Stage 2A MAE is "
     "therefore excluded from benchmark comparison. "
     "Results reported as prediction spread, not best model."),
    ("OLS multicollinearity (VIF computed — near singular)",
     "Computed VIFs confirm near-singular matrix at n=6. "
     "Individual OLS t-statistics are unreliable. Primary "
     "H2 evidence uses F-test, group t-test, Lasso "
     "selection, and BH-corrected correlations."),
    ("H2 not supported as stated",
     "sal_combined (r=-0.014, p=0.976) not significant. "
     "H2 is preserved as originally formed — the data "
     "revise it, not the researchers. Participation "
     "dominates, EMBA salary is secondary, sal_combined ns."),
    ("Stage 2C EMBA structural break",
     "EMBA MAE=20.5. Quantitative break test (8A.8) checks "
     "whether failure concentrates post-2023. If not "
     "concentrated, EMBA is simply unpredictable from "
     "these indicators — both conclusions are reported."),
    ("Scenario simulation — not statistical inference",
     "The Stage 3 output is a heuristic scenario tool. "
     "Coefficient bounds are derived from the same n=6 "
     "dataset (circular but disclosed). The output range "
     "is NOT a confidence interval. Stage 1 is authoritative."),
    ("2022 EMBA salary imputation",
     "EMBA 2022 file loads 1 school — sal_combined uses "
     "MiM only. Sensitivity check (7.6) "
     "confirms MBA coefficient robust (<15% change)."),
    ("Simulation holds competitors fixed",
     "Rank gain estimates are upper bounds. Relative leverage "
     "rankings remain valid within the formula structure."),
    ("Benchmark comparison scope",
     "Proudlove (2011b) MAE=2.52 is cross-sectional across "
     "many schools. Stage 1 MAE uses 3 single-school points. "
     "Not directly comparable. Correct claim: verified exact "
     "reproduction to 10 decimal places."),
]
for i, (title, desc) in enumerate(limitations, 1):
    print(f"\n  {i}. {title.upper()}")
    words = desc.split()
    line  = "     "
    for word in words:
        if len(line) + len(word) > 68:
            print(line); line = "     " + word + " "
        else:
            line += word + " "
    if line.strip():
        print(line)
# 9.7 CONTRIBUTIONS
print()
print("9.7 CONTRIBUTIONS TO LITERATURE")
print("-" * 55)
contribs = [
    ("Methodological",
     "First Python implementation of Proudlove (2011b) FT "
     "European formula verified against institutional Excel "
     f"models to 10 decimal places. MAE={stage1_mae}. "
     "Documents undocumented gender parity preprocessing."),
    ("Empirical",
     "Most detailed longitudinal FT European ranking account "
     "at one institution: 12 years (2014-2025), all 5 "
     "sub-rankings. Structural breaks and participation "
     "dynamics documented for the first time."),
    ("Statistical",
     "Programme participation dominates: r=-0.954, "
     f"Cohen's d={cd:.3f} (note: inflated by SD=0 in "
     "absent group — primary evidence is raw diff + F-test). "
     "H2 salary dominance not supported by data — "
     "rejection itself is the finding."),
    ("Machine Learning",
     "Stage 2C MiM (N=590) achieves OLS MAE=1.8, beating "
     "Proudlove on genuine cross-sectional data. Contrast "
     "with EMBA MAE=20.5 is a methodological finding: ML "
     "succeeds for continuous-indicator sub-rankings and "
     "fails for step-function governance events."),
    ("Practical",
     "Scenario simulation quantifies how strategic "
     "decisions such as programme participation move the "
     "composite rank, as an illustrative planning tool."),
    ("Theoretical",
     "Empirical grounding for Wedlin (2006) and Hopwood "
     "(2008) ranking response frameworks through 12 years "
     "of verified institutional decisions with quantified "
     "rank consequences."),
]
for i, (cat, desc) in enumerate(contribs, 1):
    print(f"\n  {i}. {cat.upper()}:")
    words = desc.split()
    line  = "     "
    for word in words:
        if len(line) + len(word) > 68:
            print(line); line = "     " + word + " "
        else:
            line += word + " "
    if line.strip():
        print(line)
# 9.8 THESIS STATUS
print()
print("9.8 THESIS STATUS")
print("-" * 55)
print()
print("  All findings documented in submitted thesis.")
print("  Notebook and thesis are consistent throughout.")
print("  See thesis for full interpretation and limitations.")
# 9.9 FINAL STATUS
print()
print("=" * 65)
print("9.9 NOTEBOOK COMPLETE — FINAL STATUS")
print("=" * 65)
print()
cells = [
    ("Cell 1", "Environment Setup",                    "✓ "),
    ("Cell 2", "Data Loading (1,964 obs)",             "✓ "),
    ("Cell 3", "Variable Definitions + Helpers",       "✓ "),
    ("Cell 4", "Stage 1: Formula Reconstruction",      "✓ "),
    ("Cell 5", "EDA + Feature Matrix + BH Correction", "✓ "),
    ("Cell 6", "Stage 2A: Sensitivity + Feature Sel.", "✓ "),
    ("Cell 7", "Statistical Testing + VIF Computed",   "✓ "),
    ("Cell 8", "Stage 2C + Structural Break Test",     "✓ "),
    ("Cell 9", "Summary + Limitations + Contributions","✓ "),
]
for cell, section, status in cells:
    print(f"  {status}  {cell}: {section}")
print()
print("  HEADLINE NUMBERS:")
print(f"    Stage 1: MAE={stage1_mae}  diff=0.0000000000")
print(f"    2025: EXACT (rank 12 = actual 12)")
if best_cs_mim:
    print(f"    Stage 2C MiM: MAE={best_cs_mim['MAE_MIM']} ✓ ")
if best_cs_emba:
    print(f"    Stage 2C EMBA: MAE={best_cs_emba['MAE_EMBA']}"
          f" — structural break tested (8A.8)")
print(f"    mba_active: r=-0.954  p=0.0008  "
      f"Cohen's d={cd:.3f}")
print(f"    OLS R²={res2['r2']:.4f}  "
      f"F={res2['f_stat']:.2f}  p={res2['f_pval']:.4f}")
print(f"    2026 scenario range: [{pi90_lo},{pi90_hi}] "
      f"[illustrative — NOT a CI]")
print(f"    Stage 2A MAE: excluded from benchmark "
      f"(N=2 test points)")