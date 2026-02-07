#!/usr/bin/env python3
"""Simple data validation script used by GitHub Actions workflow"""
import sys
from pathlib import Path
# Ensure repository root is on sys.path so imports like `import config` work when invoked from scripts/
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
import config
from src.utils import load_csv, validate_data

def main():
    train_path = config.DATA_DIR / config.TRAIN_DATA_FILE
    print(f"Running data validation on: {train_path}")
    df = load_csv(train_path)
    ok = validate_data(df)
    if not ok:
        print("Data validation FAILED")
        sys.exit(1)
    print("Data validation PASSED")

if __name__ == '__main__':
    main()
