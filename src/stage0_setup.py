# ═══════════════════════════════════════════════════════════════
# FT EUROPEAN BUSINESS SCHOOLS RANKING — PREDICTIVE ANALYTICS
#
# CELL 1: ENVIRONMENT SETUP
# All imports here.
# ═══════════════════════════════════════════════════════════════
import os, re, warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import openpyxl
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import pearsonr, shapiro, jarque_bera, ttest_ind
from numpy.linalg import lstsq
from sklearn.linear_model import (
    LinearRegression, Ridge, Lasso, ElasticNet, BayesianRidge
)
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor,
    ExtraTreesRegressor
)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from sklearn.utils import resample
try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
    print("✓  XGBoost loaded")
except ImportError:
    XGBOOST_AVAILABLE = False
    print("✗  XGBoost not available")
try:
    from lightgbm import LGBMRegressor
    LIGHTGBM_AVAILABLE = True
    print("✓  LightGBM loaded")
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("✗  LightGBM not available")
pd.set_option('display.max_columns', 50)
pd.set_option('display.max_rows', 100)
pd.set_option('display.float_format', '{:.3f}'.format)
pd.set_option('display.width', 120)
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['axes.titleweight'] = 'bold'
# CONFIG: data location, file names and column layout live in config.py
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from config import DATA_PATH
try:
    from IPython.display import display
except ImportError:
    display = print
if os.path.exists(DATA_PATH):
    xlsx = [f for f in os.listdir(DATA_PATH) if f.endswith('.xlsx')]
    print(f"\n✓  Data path found: {DATA_PATH}")
    print(f"✓  Excel files: {len(xlsx)}")
else:
    print(f"✗  Data path NOT found: {DATA_PATH}")
    print("  →  Update DATA_PATH in config.py, or run make_synthetic_data.py.")
print(f"\n✓  XGBoost:  {XGBOOST_AVAILABLE}")
print(f"✓  LightGBM: {LIGHTGBM_AVAILABLE}")
print("\nReady for Cell 2: Data Loading")