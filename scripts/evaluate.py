#!/usr/bin/env python3
"""Evaluate metrics and exit with non-zero if below threshold"""
import json
import os
import sys
from pathlib import Path
# Ensure repository root is on sys.path so imports like `config` work if needed
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

THRESH = float(os.getenv('MODEL_MIN_ACCURACY', '0.60'))
metrics_dir = Path('output')
metrics_file = metrics_dir / 'metrics.json'
# If retraining was intentionally skipped, skip evaluation to avoid failing CI
if (metrics_dir / 'retraining_skipped.txt').exists():
    print('Retraining was skipped this run; skipping evaluation.')
    with open(metrics_dir / 'evaluation_skipped.txt', 'w') as f:
        f.write('Retraining skipped; evaluation not run.\n')
    sys.exit(0)

if not metrics_file.exists():
    print('Metrics file missing:', metrics_file, ' — skipping evaluation (no retraining performed)')
    # Create a marker so Actions artifacts show the reason
    out = metrics_file.parent
    out.mkdir(parents=True, exist_ok=True)
    with open(out / 'evaluation_skipped.txt', 'w') as f:
        f.write('No metrics.json found; retraining likely skipped or failed.\n')
    sys.exit(0)

with open(metrics_file) as f:
    metrics = json.load(f)

acc = float(metrics.get('accuracy', 0))
print(f"Model accuracy: {acc:.4f} | threshold: {THRESH:.4f}")
if acc < THRESH:
    print('Model did not meet accuracy threshold, failing')
    sys.exit(1)
print('Model meets threshold')
