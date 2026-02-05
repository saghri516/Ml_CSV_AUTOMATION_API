"""
Train multiple model types and save versioned model artifacts.

Usage:
    python -m src.train_multi --csv data/train.csv --models random_forest,logistic_regression,gradient_boosting

This script will train each model type using `models.AutomatedMLModel` and save
models to `config.MODELS_DIR` with filenames like `model_{type}_v1.pkl`.
"""
from pathlib import Path
from typing import List
import argparse
import config
from models.autom_model import AutomatedMLModel
from src.logger import logger

DEFAULT_MODEL_TYPES = [
    'random_forest',
    'gradient_boosting',
    'logistic_regression',
    'svm'
]


def train_multiple(csv_path: str, model_types: List[str] = None) -> List[str]:
    """Train multiple models on a CSV and save versioned artifacts.

    Args:
        csv_path: Path to training CSV file
        model_types: List of model type strings to train

    Returns:
        List of saved model file paths
    """
    saved_paths = []
    model_types = model_types or DEFAULT_MODEL_TYPES

    csv_file = Path(csv_path)
    if not csv_file.exists():
        logger.error(f"Training CSV not found: {csv_file}")
        return saved_paths

    # Ensure models dir exists
    models_dir = Path(config.MODELS_DIR)
    models_dir.mkdir(parents=True, exist_ok=True)

    for mt in model_types:
        logger.info(f"Training model type: {mt}")
        try:
            model = AutomatedMLModel(config={'model_type': mt})
            result = model.train(str(csv_file))
            if not result.get('success'):
                logger.error(f"Training failed for {mt}: {result.get('error', 'unknown')}")
                continue

            saved = model.save_versioned(base_name=f"model_{mt}", models_dir=str(models_dir))
            if saved:
                saved_paths.append(saved)
                logger.info(f"Saved model: {saved}")
            else:
                logger.error(f"Failed to save model for {mt}")
        except Exception as e:
            logger.error(f"Exception training {mt}: {str(e)}")

    return saved_paths


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train multiple models and save versioned files')
    parser.add_argument('--csv', required=True, help='Training CSV path')
    parser.add_argument('--models', help='Comma-separated model types (default: all)', default='')
    args = parser.parse_args()

    models_arg = [m.strip() for m in args.models.split(',') if m.strip()] if args.models else None
    saved = train_multiple(args.csv, models_arg)
    if saved:
        print('Saved models:')
        for p in saved:
            print(' -', p)
    else:
        print('No models saved')