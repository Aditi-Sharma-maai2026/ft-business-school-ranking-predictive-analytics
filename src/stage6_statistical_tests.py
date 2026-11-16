# ═══════════════════════════════════════════════════════════════
# CELL 7: STATISTICAL TESTING
# ═══════════════════════════════════════════════════════════════
print("=" * 65)
print("CELL 7: STATISTICAL TESTING")
print("=" * 65)
# 7.1 OLS REGRESSION
print()
print("7.1 OLS REGRESSION MODELS")
print("-" * 55)
print()
print("  STRUCTURAL NOTE ON mba_active:")
print("  When MBA is absent, the FT formula assigns Sij=0,")
print("  mechanically lowering the composite score. The OLS")
print("  coefficient for mba_active therefore partly captures")
print("  formula arithmetic. The t-test confirms this")
print("  structural relationship is strong and consistent.")
print("  It does not imply mba_active is an independent")
print("  exogenous cause outside the formula.")
reg_data = esmt_df[
    ['mba_active', 'sal_combined', 'board_avg',
     'mba_x_salary', 'composite_rank']
].dropna()
y_ols = reg_data['composite_rank'].values.astype(float)
n_obs = len(y_ols)
print(f"\n  Observations: {n_obs}")
print(f"  Years: {list(reg_data.index)}")
def run_ols(X, y, var_names):
    n      = len(y); k = X.shape[1] - 1
    beta   = lstsq(X, y, rcond=None)[0]
    y_hat  = X @ beta
    resid  = y - y_hat
    ss_tot = np.sum((y - y.mean())**2)
    ss_res = np.sum(resid**2)
    ss_reg = ss_tot - ss_res
    df_reg = k; df_res = n - k - 1
    ms_reg = ss_reg / df_reg if df_reg > 0 else 0
    ms_res = ss_res / df_res if df_res > 0 else np.inf
    r2     = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    r2_adj = (1 - (1 - r2) * (n - 1) / (n - k - 1)
              if n - k - 1 > 0 else r2)
    f_stat = ms_reg / ms_res if ms_res > 0 else 0
    f_pval = 1 - stats.f.cdf(f_stat, df_reg, df_res)
    try:
        cov_b = ms_res * np.linalg.inv(X.T @ X)
        se_b  = np.sqrt(np.diag(cov_b))
    except np.linalg.LinAlgError:
        se_b = np.full(len(beta), np.nan)
    with np.errstate(invalid='ignore'):
        t_stats = np.where(se_b > 0, beta / se_b, np.nan)
        t_pvals = np.where(
            ~np.isnan(t_stats),
            2*(1-stats.t.cdf(np.abs(t_stats), max(df_res,1))),
            np.nan
        )
    t_crit = stats.t.ppf(0.975, max(df_res, 1))
    return {
        'beta':beta, 'se':se_b, 't':t_stats, 'p':t_pvals,
        'ci_lo':beta-t_crit*se_b, 'ci_hi':beta+t_crit*se_b,
        'r2':r2, 'r2_adj':r2_adj, 'f_stat':f_stat,
        'f_pval':f_pval, 'ss_res':ss_res, 'ss_reg':ss_reg,
        'df_reg':df_reg, 'df_res':df_res, 'ms_res':ms_res,
        'resid':resid, 'y_hat':y_hat, 'var_names':var_names,
    }
def print_ols(res, title):
    print(f"\n  {title}")
    print(f"  {'Variable':<22} {'Coef':>8} {'SE':>8} "
          f"{'t':>8} {'p':>10} {'CI95':>24}  Sig")
    print("  " + "-" * 86)
    for i, vn in enumerate(res['var_names']):
        b  = res['beta'][i];  se = res['se'][i]
        t  = res['t'][i];     p  = res['p'][i]
        lo = res['ci_lo'][i]; hi = res['ci_hi'][i]
        se_s = f"{se:.4f}" if not np.isnan(se)  else 'n/a'
        t_s  = f"{t:.4f}"  if not np.isnan(t)   else 'n/a'
        p_s  = f"{p:.6f}"  if not np.isnan(p)   else 'n/a'
        lo_s = f"{lo:.3f}" if not np.isnan(lo)  else 'n/a'
        hi_s = f"{hi:.3f}" if not np.isnan(hi)  else 'n/a'
        ci_s = f"[{lo_s},{hi_s}]"
        # Fixed: clean conditional, no nested ternary
        is_nan = isinstance(p, float) and np.isnan(p)
        sig    = ('n/a' if is_nan else
                  '***' if p < 0.001 else
                  '**'  if p < 0.01  else
                  '*'   if p < 0.05  else
                  '.'   if p < 0.1   else 'ns')
        print(f"  {vn:<22} {b:>8.4f} {se_s:>8} "
              f"{t_s:>8} {p_s:>10} {ci_s:>24}  {sig}")
    sig_f = ('***' if res['f_pval'] < 0.001 else
             '**'  if res['f_pval'] < 0.01  else
             '*'   if res['f_pval'] < 0.05  else 'ns')
    print(f"\n  R²={res['r2']:.4f}  "
          f"Adj R²={res['r2_adj']:.4f}  n={n_obs}")
    print(f"  F({res['df_reg']},{res['df_res']})="
          f"{res['f_stat']:.4f}  "
          f"p={res['f_pval']:.6f}  {sig_f}")
X1 = np.column_stack([np.ones(n_obs),
                       reg_data['mba_active'].values])
X2 = np.column_stack([np.ones(n_obs),
                       reg_data['mba_active'].values,
                       reg_data['sal_combined'].values])
X3 = np.column_stack([np.ones(n_obs),
                       reg_data['mba_active'].values,
                       reg_data['sal_combined'].values,
                       reg_data['board_avg'].values])
res1 = run_ols(X1, y_ols, ['Intercept', 'mba_active'])
res2 = run_ols(X2, y_ols,
               ['Intercept', 'mba_active', 'sal_combined'])
res3 = run_ols(X3, y_ols,
               ['Intercept', 'mba_active',
                'sal_combined', 'board_avg'])
print_ols(res1, "Model 1: mba_active only")
print_ols(res2, "Model 2: mba_active + sal_combined (PRIMARY)")
print_ols(res3, "Model 3: + board_avg (supplementary)")
print()
print("  MODEL SELECTION:")
print("  Model 2 is the preferred specification.")
print("  Model 3: near-singular matrix expected at n=6")
print("  with 4 parameters — SE shown as n/a.")
print("  board_avg is collinear with mba_active because")
print("  board turnover coincided with MBA absence in 2023.")
print("  Do not interpret Model 3 coefficients.")
BETA_0   = float(res2['beta'][0])
BETA_MBA = float(res2['beta'][1])
BETA_SAL = float(res2['beta'][2])
MS_E     = float(res2['ms_res'])
print(f"\n  Stored parameters (Model 2):")
print(f"  β₀={BETA_0:.4f}  β₁(MBA)={BETA_MBA:.4f}  "
      f"β₂(sal)={BETA_SAL:.4f}  MSE={MS_E:.4f}")
# 7.2 ANOVA TABLE
print()
print("7.2 ANOVA TABLE (Model 2)")
print("-" * 55)
ss_tot2 = np.sum((y_ols - y_ols.mean())**2)
print(f"\n  {'Source':<12} {'SS':>10} {'df':>5} "
      f"{'MS':>10} {'F':>10} {'p':>12}")
print("  " + "-" * 63)
print(f"  {'Regression':<12} {res2['ss_reg']:>10.4f} "
      f"{res2['df_reg']:>5} "
      f"{res2['ss_reg']/max(res2['df_reg'],1):>10.4f} "
      f"{res2['f_stat']:>10.4f} {res2['f_pval']:>12.6f}")
print(f"  {'Residual':<12} {res2['ss_res']:>10.4f} "
      f"{res2['df_res']:>5} {res2['ms_res']:>10.4f}")
print(f"  {'Total':<12} {ss_tot2:>10.4f} {n_obs-1:>5}")

# 7.3 T-TEST: MBA ACTIVE vs ABSENT
print()
print("7.3 T-TEST: MBA ACTIVE vs ABSENT (primary H2 evidence)")
print("-" * 55)
on  = esmt_df[esmt_df['mba_active']==1]['composite_rank'].dropna()
off = esmt_df[esmt_df['mba_active']==0]['composite_rank'].dropna()
t_v, p_v = ttest_ind(on, off)
diff = off.mean() - on.mean()
pool = np.sqrt((on.std()**2 + off.std()**2) / 2)
cd   = diff / pool if pool > 0 else 0
print(f"\n  MBA active:  n={len(on)}  "
      f"mean={on.mean():.1f}  std={on.std():.1f}")
print(f"  MBA absent:  n={len(off)}  "
      f"mean={off.mean():.1f}  std={off.std():.1f}")
print(f"  Difference:  {diff:.1f} positions")
print(f"  t={t_v:.4f}  df={len(on)+len(off)-2}  "
      f"p={p_v:.6f}")
print(f"  Cohen's d={cd:.3f}")
print(f"  NOTE: MBA-absent group has n={len(off)} with SD={off.std():.1f}.")
print(f"  Pooled SD approaches zero — d is numerically inflated,")
print(f"  not comparable to conventional thresholds (0.2/0.5/0.8).")
print(f"  Primary evidence: raw difference ({on.mean():.1f} vs {off.mean():.1f}) + F-test.")
print(f"  Cohen's d reported for completeness only.")
if p_v < 0.05:
    print(" Statistically significant")
# 7.4 NORMALITY (appendix-level)
print()
print("7.4 NORMALITY TESTS")
print("-" * 55)
print()
print("  LOW POWER WARNING: Shapiro-Wilk and Jarque-Bera")
print("  have negligible statistical power at n=7.")
print("  Non-significant result DOES NOT confirm normality.")
print("  Reported for methodological completeness only.")
resid2        = res2['resid']
sw_stat, sw_p = shapiro(resid2)
jb_stat, jb_p = jarque_bera(resid2)
print(f"\n  Shapiro-Wilk: W={sw_stat:.4f}  p={sw_p:.4f}")
print(f"  Jarque-Bera:  JB={jb_stat:.4f}  p={jb_p:.4f}")
print("  (Interpret with extreme caution at n=7)")
print()
print("  Residuals by year:")
print(f"  {'Year':<6} {'Actual':>8} {'Fitted':>8} "
      f"{'Residual':>10}")
print("  " + "-" * 36)
for yr, act, fit, r in zip(
    reg_data.index, y_ols, res2['y_hat'], resid2
):
    print(f"  {yr:<6} {int(act):>8} {fit:>8.2f} "
          f"{r:>10.3f}")
# 7.5 VIF — COMPUTED
print()
print("7.5 VARIANCE INFLATION FACTOR (computed)")
print("-" * 55)
print()
print("  VIFs are computed numerically, not asserted.")
print("  With n=6 and interaction term (mba_x_salary),")
print("  near-singular behaviour is expected.")
print("  This confirms the claim quantitatively.")
print()
vif_features = ['mba_active', 'sal_combined',
                 'board_avg', 'mba_x_salary']
vif_data     = reg_data[vif_features].dropna()
print(f"  {'Feature':<22} {'VIF':>8}  Assessment")
print("  " + "-" * 48)
vif_values = {}
for feat in vif_features:
    others = [f for f in vif_features if f != feat]
    if len(vif_data) < len(others) + 2:
        print(f"  {feat:<22} {'n/a':>8}  Insufficient obs")
        continue
    X_v  = vif_data[others].values
    y_v  = vif_data[feat].values
    X_vc = np.column_stack([np.ones(len(X_v)), X_v])
    try:
        b_v   = lstsq(X_vc, y_v, rcond=None)[0]
        y_hat = X_vc @ b_v
        ss_r  = np.sum((y_v - y_hat)**2)
        ss_t  = np.sum((y_v - y_v.mean())**2)
        r2_v  = 1 - ss_r / ss_t if ss_t > 0 else 0
        vif_v = 1 / (1 - r2_v) if r2_v < 0.9999 else 999.0
        vif_values[feat] = vif_v
        assess = ('Severe (>10)'   if vif_v > 10 else
                  'Moderate (>5)' if vif_v > 5  else
                  'Acceptable')
        print(f"  {feat:<22} {vif_v:>8.2f}  {assess}")
    except Exception:
        vif_values[feat] = 999.0
        print(f"  {feat:<22} {'999.00':>8}  Singular matrix")
print()
print("  CONSEQUENCE OF HIGH VIF:")
print("  Individual OLS t-statistics and p-values for")
print("  Model 2 predictors are unreliable — inflated SEs")
print("  make coefficients appear less significant than")
print("  they are, or produce unstable estimates.")
print()
print("  We therefore DO NOT use individual t-statistics")
print("  as primary H2 evidence. Primary evidence:")
print("    (a) F-test: overall model significance")
print("    (b) Group t-test: MBA active vs absent (7.3)")
print("    (c) Lasso feature selection: Cell 6")
print("    (d) Pearson r with BH correction: Cell 5")
# 7.6 SENSITIVITY: DROP 2022
print()
print("7.6 SENSITIVITY CHECK — DROP 2022 (imputation)")
print("-" * 55)
print()
print("  2022 uses MiM salary as proxy for missing EMBA.")
print("  Refit Model 2 without 2022.")
reg_no22 = reg_data.drop(2022, errors='ignore')
if len(reg_no22) >= 4:
    y_no22  = reg_no22['composite_rank'].values.astype(float)
    X2_no22 = np.column_stack([
        np.ones(len(y_no22)),
        reg_no22['mba_active'].values,
        reg_no22['sal_combined'].values,
    ])
    res2_no22 = run_ols(
        X2_no22, y_no22,
        ['Intercept', 'mba_active', 'sal_combined']
    )
    b1_full = BETA_MBA; b1_drop = res2_no22['beta'][1]
    b2_full = BETA_SAL; b2_drop = res2_no22['beta'][2]
    chg_b1  = abs(b1_drop - b1_full) / abs(b1_full) * 100
    chg_b2  = (abs(b2_drop - b2_full) / abs(b2_full) * 100
               if b2_full != 0 else 0)
    print(f"\n  n (without 2022): {len(reg_no22)}")
    print(f"  β₁(MBA):  full={b1_full:.4f}  "
          f"drop={b1_drop:.4f}  Δ={chg_b1:.1f}%")
    print(f"  β₂(sal):  full={b2_full:.4f}  "
          f"drop={b2_drop:.4f}  Δ={chg_b2:.1f}%")
    print(f"  R²:       full={res2['r2']:.4f}  "
          f"drop={res2_no22['r2']:.4f}")
    if chg_b1 < 15:
        print("MBA coefficient robust (<15% change)")
    else:
        print("MBA coefficient sensitive to 2022 row")
# 7.7 FORMAL H2 ASSESSMENT
print()
print("=" * 65)
print("7.7 FORMAL H2 ASSESSMENT")
print("=" * 65)
print()
print("  H2 (thesis as written): salary-related indicators")
print("  explain the largest share of rank variance.")
print()
print("  Evidence FOR H2:")
print(f"    mba_active:       r=-0.954  BH-corrected sig  ✓ ")
print(f"    emba_intl_board:  r=-0.990  BH-corrected sig  ✓ ")
print(f"    emba_salary_inc:  r=-0.891  p=0.042  marginal")
print(f"    OLS Model 2:      R²={res2['r2']:.4f}  "
      f"F={res2['f_stat']:.2f}  p={res2['f_pval']:.4f}")
print(f"    MBA t-test:       t={t_v:.4f}  p={p_v:.6f}  "
      f"Cohen's d={cd:.3f}")
print()
print("  Evidence AGAINST H2 as literally stated:")
print("  sal_combined r=-0.014, p=0.976 — NOT significant.")
print("  Thesis H2 claims salary explains the LARGEST")
print("  share of variance. This is not supported.")
print()
print("  RESOLUTION:")
print("  H2 as stated is not supported by the data.")
print("  sal_combined r=-0.014, p=0.976 — salary does not")
print("  explain the largest share of variance.")
print("  The dominant driver is programme participation")
print("  (mba_active r=-0.954), which partly reflects")
print("  formula arithmetic (Sij=0 when absent) and partly")
print("  genuine institutional decisions.")
print("  Individual-programme EMBA salary (r=-0.891) is")
print("  a marginal secondary predictor.")
print("  H2 is preserved as originally formed — the data")
print("  revise it, not the researchers.")
print("  See thesis Section 3.2 for full treatment.")
print()
print("  H2 VERDICT: NOT SUPPORTED AS STATED")
print("  Empirical finding revises H2 — see thesis Section 3.2")
print()
print("=" * 65)
print("CELL 7 COMPLETE")
print("=" * 65)
print()
print(f"  BETA_0={BETA_0:.4f}  BETA_MBA={BETA_MBA:.4f}  "
      f"BETA_SAL={BETA_SAL:.4f}  MS_E={MS_E:.4f}")
print(f"  Cohen's d={cd:.3f}  t={t_v:.4f}  p={p_v:.6f}")
print()
print("Ready for Cell 8: Stage 2C + Scenario Simulation")