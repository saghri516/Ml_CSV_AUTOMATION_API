#!/usr/bin/env python3
"""Retraining script used by GitHub Actions workflow
Saves a versioned model and outputs metrics to output/metrics.json
"""
import json
import sys
from pathlib import Path
# Ensure repository root is on sys.path so imports like `import config` work when invoked from scripts/
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
import config
from models.autom_model import AutomatedMLModel


def main():
    Path(config.MODELS_DIR).mkdir(parents=True, exist_ok=True)
    Path(config.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    train_csv = str(config.DATA_DIR / config.TRAIN_DATA_FILE)
    print(f"Starting training on {train_csv}")

    # Skip retraining if training data is not present on the runner
    if not (config.DATA_DIR / config.TRAIN_DATA_FILE).exists():
        print(f"Training data not found at {config.DATA_DIR / config.TRAIN_DATA_FILE}. Skipping retraining.")
        out = config.OUTPUT_DIR
        out.mkdir(parents=True, exist_ok=True)
        with open(out / 'retraining_skipped.txt', 'w') as f:
            f.write(f"Training data missing: {config.DATA_DIR / config.TRAIN_DATA_FILE}\n")
        return

    model = AutomatedMLModel()
    res = model.train(train_csv)
    if not res.get('success'):
        print('Training failed:', res.get('error', res))
        sys.exit(1)

    # Save versioned model
    saved = model.save_versioned(base_name=f"model_{model.config.get('model_type','model')}")
    if not saved:
        print('Failed to save model')
        sys.exit(1)

    # write metrics
    metrics = {
        'accuracy': float(res['accuracy']),
        'precision': float(res['precision']),
        'recall': float(res['recall']),
        'f1_score': float(res['f1_score']),
        'metadata': res.get('metadata', {})
    }

    metrics_path = Path(config.OUTPUT_DIR) / 'metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    # Save trained model path for artifact upload
    with open(Path(config.OUTPUT_DIR)/'trained_model_path.txt','w') as f:
        f.write(saved)

    print('Training completed. Model saved to:', saved)
    print('Metrics written to:', metrics_path)


if __name__ == '__main__':
    main()
