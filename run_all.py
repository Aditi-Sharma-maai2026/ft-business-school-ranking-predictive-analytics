"""
Runs the whole analysis, step by step, in the same order
as the original notebook cells. Charts are saved to the outputs folder.

How to use:  python run_all.py
"""
import os
import runpy

# Save charts to files instead of opening windows
os.environ.setdefault("MPLBACKEND", "Agg")

HERE = os.path.dirname(os.path.abspath(__file__))

# The code files, in the order they should run
STAGES = [
    "stage0_setup.py",
    "stage1_data_loading.py",
    "stage2_variables_and_helpers.py",
    "stage3_formula_reconstruction.py",
    "stage4_eda.py",
    "stage5_ml_sensitivity.py",
    "stage6_statistical_tests.py",
    "stage7_cross_sectional_ml_and_scenarios.py",
    "stage8_summary.py",
]

if __name__ == "__main__":
    out_dir = os.path.join(HERE, "outputs")
    os.makedirs(out_dir, exist_ok=True)
    os.chdir(out_dir)
    namespace = {}
    for stage in STAGES:
        path = os.path.join(HERE, "src", stage)
        # Each step can use what the previous steps created,
        # just like notebook cells share their variables
        namespace = runpy.run_path(path, init_globals={**namespace, "__file__": path})