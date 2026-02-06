import unittest
import numpy as np
import pandas as pd
from pathlib import Path

# Ensure project root is on sys.path so local modules like `config` and `src` are importable
import sys, os
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config

from src.train_multi import train_multiple
from src.predict_multi import predict_multiple

class TestPredictMultiple(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data_dir = Path('data')
        cls.data_dir.mkdir(exist_ok=True)
        np.random.seed(1)
        df = pd.DataFrame({
            'f1': np.random.uniform(0, 10, 50),
            'f2': np.random.uniform(0, 5, 50),
            'target': np.random.randint(0, 2, 50)
        })
        cls.train_csv = cls.data_dir / 'train_predict_multi.csv'
        df.to_csv(cls.train_csv, index=False)

        # Train two models for testing
        cls.model_types = ['random_forest', 'logistic_regression']
        cls.saved = train_multiple(str(cls.train_csv), cls.model_types)

        # Create a small test CSV for prediction (without target column)
        pred_df = pd.DataFrame({
            'f1': np.random.uniform(0, 10, 10),
            'f2': np.random.uniform(0, 5, 10)
        })
        cls.pred_csv = cls.data_dir / 'test_predict_multi.csv'
        pred_df.to_csv(cls.pred_csv, index=False)

    def test_predict_multiple_outputs(self):
        # Ensure we saved at least one model
        self.assertTrue(len(self.saved) >= 1)

        out = predict_multiple(self.saved, str(self.pred_csv), out_dir='output')
        self.assertTrue(Path(out).exists())

        df_out = pd.read_csv(out)
        # Check that prediction columns for each saved model are present
        for p in self.saved:
            prefix = Path(p).stem
            self.assertIn(f'prediction_{prefix}', df_out.columns)

    @classmethod
    def tearDownClass(cls):
        # Clean up CSVs
        if cls.train_csv.exists():
            cls.train_csv.unlink()
        if cls.pred_csv.exists():
            cls.pred_csv.unlink()
        # Clean up models saved
        models_dir = Path(config.MODELS_DIR)
        if models_dir.exists():
            for f in list(models_dir.iterdir()):
                if any(f.name.startswith(f'model_{mt}') for mt in cls.model_types):
                    try:
                        f.unlink()
                    except Exception:
                        pass

if __name__ == '__main__':
    unittest.main()
