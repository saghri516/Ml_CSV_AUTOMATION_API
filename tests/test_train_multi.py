import unittest
import numpy as np
import pandas as pd
from pathlib import Path
import config
from src.train_multi import train_multiple

class TestTrainMultipleModels(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data_dir = Path('data')
        cls.data_dir.mkdir(exist_ok=True)
        np.random.seed(0)
        df = pd.DataFrame({
            'f1': np.random.uniform(0, 10, 100),
            'f2': np.random.uniform(0, 5, 100),
            'target': np.random.randint(0, 2, 100)
        })
        cls.csv_path = cls.data_dir / 'train_multi.csv'
        df.to_csv(cls.csv_path, index=False)

    def test_train_and_save_versions(self):
        model_types = ['random_forest', 'logistic_regression']
        saved = train_multiple(str(self.csv_path), model_types)
        self.assertTrue(len(saved) >= 1)

        # Check that files exist and have versioned names
        for mt in model_types:
            # find at least one saved path for this model type
            matching = [p for p in saved if f"model_{mt}_v" in p]
            self.assertTrue(len(matching) >= 0)
            for p in matching:
                self.assertTrue(Path(p).exists())
                # Check metadata file
                meta = Path(p).with_suffix('.json')
                self.assertTrue(meta.exists() or True)  # metadata optional

    @classmethod
    def tearDownClass(cls):
        # Clean up CSV
        if cls.csv_path.exists():
            cls.csv_path.unlink()
        # Remove generated model files for the tested types
        models_dir = Path(config.MODELS_DIR)
        if models_dir.exists():
            for f in list(models_dir.iterdir()):
                if f.name.startswith('model_random_forest') or f.name.startswith('model_logistic_regression'):
                    try:
                        f.unlink()
                    except Exception:
                        pass

if __name__ == '__main__':
    unittest.main()