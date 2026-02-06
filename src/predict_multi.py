"""
Predict with multiple saved models on a CSV file and combine results.

Usage:
    python -m src.predict_multi --csv data/test.csv --models_dir models
    python -m src.predict_multi --csv data/test.csv --models models/model_random_forest_v1.pkl,models/model_logistic_regression_v1.pkl

This will produce a CSV in the `output/` folder containing the original data plus per-model prediction and confidence columns.
"""
from pathlib import Path
from typing import List
import argparse
import pandas as pd
from datetime import datetime
import logging

from models.autom_model import AutomatedMLModel

logger = logging.getLogger(__name__)


def predict_multiple(models: List[str], csv_path: str, out_dir: str = "output") -> str:
    """Load each model and predict on the CSV, returning the path to the combined output CSV.

    Args:
        models: List of model file paths (pickled AutomatedMLModel artifacts).
        csv_path: Path to CSV to predict on.
        out_dir: Directory to save combined predictions.

    Returns:
        Path string to saved combined predictions CSV.
    """
    csv_file = Path(csv_path)
    if not csv_file.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_file}")

    df = pd.read_csv(csv_file)
    base_df = df.copy()

    for mpath in models:
        mfile = Path(mpath)
        if not mfile.exists():
            logger.warning(f"Model file not found, skipping: {mpath}")
            continue

        # Derive a friendly prefix for columns from filename
        prefix = mfile.stem  # e.g., model_random_forest_v1

        model = AutomatedMLModel()
        ok = model.load_model(str(mfile))
        if not ok:
            logger.warning(f"Failed to load model, skipping: {mpath}")
            continue

        try:
            preds = model.predict(str(csv_file))
        except Exception as e:
            logger.error(f"Prediction failed for {mpath}: {e}")
            continue

        # preds contains original columns plus 'prediction' and optional 'confidence'
        # Rename these to include model prefix
        pred_col = f"prediction_{prefix}"
        base_df[pred_col] = preds['prediction'].values

        if 'confidence' in preds.columns:
            conf_col = f"confidence_{prefix}"
            base_df[conf_col] = preds['confidence'].values

        logger.info(f"Added predictions from {mpath} as {pred_col}")

    # Ensure output directory exists
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_file = out_path / f"predictions_multi_{timestamp}.csv"
    base_df.to_csv(out_file, index=False)

    logger.info(f"Saved combined predictions to: {out_file}")
    return str(out_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict with multiple saved models and combine results')
    parser.add_argument('--csv', required=True, help='CSV file to run predictions on')
    parser.add_argument('--models_dir', help='Directory containing model .pkl files')
    parser.add_argument('--models', help='Comma-separated model file paths')
    parser.add_argument('--out', help='Output directory', default='output')
    args = parser.parse_args()

    models_list = []
    if args.models_dir:
        md = Path(args.models_dir)
        if md.exists() and md.is_dir():
            models_list.extend([str(p) for p in md.glob('*.pkl')])
    if args.models:
        models_list.extend([s.strip() for s in args.models.split(',') if s.strip()])

    if not models_list:
        print('No model files provided or found in directory')
    else:
        out = predict_multiple(models_list, args.csv, args.out)
        print('Combined predictions saved to:', out)
