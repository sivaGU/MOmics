"""Environment + dependency check for MOmics.

Run with: python verify_setup.py
"""

import importlib
import sys
from pathlib import Path

REQUIRED_PACKAGES = [
    "streamlit", "pandas", "numpy", "xgboost", "shap", "plotly", "fpdf", "matplotlib", "joblib",
]

REQUIRED_PATHS = [
    "docs.py",
    "logo.png",
    "momics_app.py",
    "requirements.txt",
    "models/MOmics_v11_locked_pipeline.pkl",
    "data/discovery/all_subtypes.v5.1.tsv",
    "data/discovery/rnaseq_washu_readcount.v4.0.tsv",
    "data/discovery/proteome_mssm_per_gene_imputed.v4.0.tsv",
    "data/discovery/metabolome_pnnl.v4.0.tsv",
]

HERE = Path(__file__).parent


def check_python_version():
    ok = sys.version_info >= (3, 8)
    print(f"[{'OK' if ok else 'FAIL'}] Python {sys.version.split()[0]} (>= 3.8 required)")
    return ok


def check_packages():
    all_ok = True
    for package in REQUIRED_PACKAGES:
        try:
            importlib.import_module(package)
            print(f"[OK] {package}")
        except ImportError:
            print(f"[FAIL] {package} not installed (pip install -r requirements.txt)")
            all_ok = False
    return all_ok


def check_paths():
    all_ok = True
    for rel_path in REQUIRED_PATHS:
        path = HERE / rel_path
        if path.exists():
            print(f"[OK] {rel_path}")
        else:
            print(f"[FAIL] {rel_path} not found")
            all_ok = False
    return all_ok


def main():
    print("=== MOmics setup verification ===\n")
    results = [check_python_version(), check_packages(), check_paths()]
    print()
    if all(results):
        print("All checks passed. Run `streamlit run momics_app.py` to launch MOmics.")
        sys.exit(0)
    else:
        print("Some checks failed. Resolve the issues above before launching the app.")
        sys.exit(1)


if __name__ == "__main__":
    main()
