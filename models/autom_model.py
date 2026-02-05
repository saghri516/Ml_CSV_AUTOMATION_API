"""
Automated Machine Learning Model for CSV Processing
Flexible automation that handles any CSV file according to client requirements
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import json
from datetime import datetime
from typing import Dict, Tuple, Any, Optional
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.append(str(Path(__file__).parent.parent))
from src.logger import logger


class AutomatedMLModel:
    """
    Automated ML model that handles any CSV file with flexible configuration
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the automated ML model
        
        Args:
            config: Dictionary with client requirements
        """
        self.config = config or self.get_default_config()
        self.model = None
        self.scaler = None
        self.label_encoders = {}
        self.feature_columns = None
        self.target_column = None
        self.model_metadata = {}
        logger.info("AutomatedMLModel initialized")
    
    @staticmethod
    def get_default_config() -> Dict[str, Any]:
        """Get default configuration for automation"""
        return {
            'model_type': 'random_forest',
            'test_size': 0.2,
            'random_state': 42,
            'handle_missing': True,
            'scaling': True,
            'encode_categorical': True,
            'target_column': None,  # Auto-detect if None
            'excluded_columns': [],
            'hyperparameters': {
                'random_forest': {'n_estimators': 100, 'max_depth': 10, 'random_state': 42},
                'gradient_boosting': {'n_estimators': 100, 'learning_rate': 0.1, 'random_state': 42},
                'logistic_regression': {'max_iter': 1000, 'random_state': 42},
                'svm': {'kernel': 'rbf', 'C': 1.0, 'random_state': 42}
            }
        }
    
    def load_csv(self, file_path: str) -> pd.DataFrame:
        """
        Load CSV file with error handling
        
        Args:
            file_path: Path to CSV file
            
        Returns:
            Loaded DataFrame or None
        """
        try:
            df = pd.read_csv(file_path)
            logger.info(f"Loaded CSV: {file_path} | Shape: {df.shape}")
            return df
        except Exception as e:
            logger.error(f"Error loading CSV {file_path}: {str(e)}")
            return None
    
    def analyze_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze data characteristics
        
        Args:
            df: Input DataFrame
            
        Returns:
            Analysis results
        """
        analysis = {
            'shape': df.shape,
            'columns': list(df.columns),
            'dtypes': df.dtypes.to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'duplicates': df.duplicated().sum(),
            'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
            'categorical_columns': df.select_dtypes(include=['object']).columns.tolist(),
            'memory_usage': df.memory_usage(deep=True).sum() / 1024**2  # MB
        }
        logger.info(f"Data Analysis: {analysis['shape'][0]} rows, {analysis['shape'][1]} columns")
        return analysis
    
    def preprocess_data(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Automatically preprocess data based on configuration
        
        Args:
            df: Input DataFrame
            fit: Whether to fit transformers (True for training, False for prediction)
            
        Returns:
            Preprocessed DataFrame
        """
        df = df.copy()
        
        # Handle missing values
        if self.config['handle_missing']:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
            
            if fit:
                self.scaler = SimpleImputer(strategy='mean')
                if numeric_cols:
                    df[numeric_cols] = self.scaler.fit_transform(df[numeric_cols])
            else:
                if numeric_cols and self.scaler:
                    df[numeric_cols] = self.scaler.transform(df[numeric_cols])
            
            # Fill categorical with mode
            for col in categorical_cols:
                df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown', inplace=True)
        
        # Encode categorical variables (excluding numeric columns)
        if self.config['encode_categorical']:
            categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
            for col in categorical_cols:
                if col not in [self.config['target_column']]:
                    if fit:
                        self.label_encoders[col] = LabelEncoder()
                        df[col] = self.label_encoders[col].fit_transform(df[col].astype(str))
                    else:
                        if col in self.label_encoders:
                            df[col] = self.label_encoders[col].transform(df[col].astype(str))
        
        logger.info("Data preprocessing completed")
        return df
    
    def detect_target_column(self, df: pd.DataFrame, analysis: Dict[str, Any]) -> str:
        """
        Auto-detect target column
        
        Args:
            df: Input DataFrame
            analysis: Data analysis results
            
        Returns:
            Target column name
        """
        # Common target column names
        target_names = ['target', 'label', 'class', 'output', 'y', 'result', 'prediction']
        
        for col in target_names:
            if col in df.columns:
                logger.info(f"Target column auto-detected: {col}")
                return col
        
        # Use last column if no common name found
        target = df.columns[-1]
        logger.info(f"Target column set to last column: {target}")
        return target
    
    def prepare_features_target(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Separate features and target
        
        Args:
            df: Input DataFrame
            
        Returns:
            Tuple of (features, target)
        """
        # Detect target column
        if self.config['target_column'] is None:
            analysis = self.analyze_data(df)
            self.target_column = self.detect_target_column(df, analysis)
        else:
            self.target_column = self.config['target_column']
        
        # Separate features and target
        X = df.drop(columns=[self.target_column] + self.config['excluded_columns'], errors='ignore')
        y = df[self.target_column]
        
        self.feature_columns = X.columns.tolist()
        
        logger.info(f"Features: {len(self.feature_columns)} | Target: {self.target_column}")
        return X, y
    
    def build_model(self) -> Any:
        """
        Build model based on configuration
        
        Returns:
            Initialized model
        """
        model_type = self.config['model_type'].lower()
        hyperparams = self.config['hyperparameters'].get(model_type, {})
        
        if model_type == 'random_forest':
            self.model = RandomForestClassifier(**hyperparams)
        elif model_type == 'gradient_boosting':
            self.model = GradientBoostingClassifier(**hyperparams)
        elif model_type == 'logistic_regression':
            self.model = LogisticRegression(**hyperparams)
        elif model_type == 'svm':
            self.model = SVC(**hyperparams)
        else:
            logger.warning(f"Unknown model type: {model_type}. Using Random Forest")
            self.model = RandomForestClassifier(**hyperparams)
        
        logger.info(f"Model built: {model_type}")
        return self.model
    
    def train(self, file_path: str) -> Dict[str, Any]:
        """
        Train the model on CSV file
        
        Args:
            file_path: Path to training CSV file
            
        Returns:
            Training results
        """
        logger.info(f"Starting training on: {file_path}")
        
        # Load data
        df = self.load_csv(file_path)
        if df is None:
            return {'success': False, 'error': 'Failed to load CSV'}
        
        # Analyze data
        analysis = self.analyze_data(df)
        
        # Prepare features and target
        X, y = self.prepare_features_target(df)
        
        # Preprocess data
        X = self.preprocess_data(X, fit=True)
        
        # Encode target if categorical
        self.target_encoder = None
        if y.dtype == 'object':
            self.target_encoder = LabelEncoder()
            y = self.target_encoder.fit_transform(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.config['test_size'], 
            random_state=self.config['random_state']
        )
        
        logger.info(f"Train set: {X_train.shape} | Test set: {X_test.shape}")
        
        # Build and train model
        self.build_model()
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        # Store metadata
        self.model_metadata = {
            'trained_at': datetime.now().isoformat(),
            'model_type': self.config['model_type'],
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'features': self.feature_columns,
            'target_column': self.target_column,
            'training_samples': X_train.shape[0],
            'test_samples': X_test.shape[0]
        }
        
        results = {
            'success': True,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'classification_report': classification_report(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'metadata': self.model_metadata
        }
        
        logger.info(f"Training completed | Accuracy: {accuracy:.4f}")
        return results
    
    def predict(self, file_path_or_df) -> pd.DataFrame:
        """
        Make predictions on new CSV file or DataFrame
        
        Args:
            file_path_or_df: Path to CSV file or pandas DataFrame for prediction
            
        Returns:
            DataFrame with predictions
        """
        # Handle both file paths and DataFrames
        if isinstance(file_path_or_df, pd.DataFrame):
            df = file_path_or_df.copy()
            logger.info(f"Making predictions on DataFrame with {len(df)} rows")
        else:
            logger.info(f"Making predictions on: {file_path_or_df}")
            df = pd.read_csv(file_path_or_df)
        
        original_df = df.copy()
        
        # Prepare features
        X = df.drop(columns=[col for col in [self.target_column] if col in df.columns], errors='ignore')
        X = X[self.feature_columns]  # Use only training features
        
        # Preprocess data
        X = self.preprocess_data(X, fit=False)
        
        # Make predictions
        predictions = self.model.predict(X)
        
        # Decode if target was encoded
        if self.target_encoder is not None:
            predictions = self.target_encoder.inverse_transform(predictions)
        
        # Add predictions to original data
        result_df = original_df.copy()
        result_df['prediction'] = predictions
        
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(X)
            result_df['confidence'] = probabilities.max(axis=1)
        
        logger.info(f"Predictions completed: {len(predictions)} samples")
        return result_df
    
    def save_model(self, file_path: str) -> bool:
        """
        Save trained model
        
        Args:
            file_path: Path to save model
            
        Returns:
            Success status
        """
        try:
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'label_encoders': self.label_encoders,
                'target_encoder': getattr(self, 'target_encoder', None),
                'feature_columns': self.feature_columns,
                'target_column': self.target_column,
                'metadata': self.model_metadata,
                'config': self.config
            }
            
            with open(file_path, 'wb') as f:
                pickle.dump(model_data, f)
            
            logger.info(f"Model saved: {file_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            return False

    def save_versioned(self, base_name: str = None, models_dir: str = None) -> str:
        """
        Save the model to the models directory with an incremental version suffix.

        Args:
            base_name: Base filename (without extension). Defaults to model type (e.g., 'model_random_forest').
            models_dir: Directory to save models. Defaults to config.MODELS_DIR.

        Returns:
            Path to the saved model file as string, or empty string on failure.
        """
        try:
            # Lazy import to avoid circular imports
            import re
            from pathlib import Path
            import config
            models_path = Path(models_dir) if models_dir else Path(config.MODELS_DIR)
            models_path.mkdir(parents=True, exist_ok=True)

            base = base_name or f"model_{self.config.get('model_type', 'model')}"
            pattern = re.compile(rf"^{re.escape(base)}_v(\d+)\.pkl$")

            # Find existing versions
            max_v = 0
            for f in models_path.iterdir():
                if f.is_file():
                    m = pattern.match(f.name)
                    if m:
                        try:
                            v = int(m.group(1))
                            if v > max_v:
                                max_v = v
                        except Exception:
                            continue

            next_v = max_v + 1
            filename = f"{base}_v{next_v}.pkl"
            save_path = str(models_path / filename)

            saved = self.save_model(save_path)
            if not saved:
                return ""

            # also save metadata file alongside the model
            try:
                meta_path = models_path / f"{base}_v{next_v}.json"
                import json
                with open(meta_path, 'w', encoding='utf-8') as mf:
                    json.dump(self.model_metadata or {}, mf, indent=2)
            except Exception:
                # metadata save failure should not block model save
                logger.warning("Failed to write metadata file")

            logger.info(f"Model saved versioned: {save_path}")
            return save_path
        except Exception as e:
            logger.error(f"Error saving versioned model: {str(e)}")
            return ""
    
    def load_model(self, file_path: str) -> bool:
        """
        Load trained model
        
        Args:
            file_path: Path to load model from
            
        Returns:
            Success status
        """
        try:
            with open(file_path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.label_encoders = model_data['label_encoders']
            self.target_encoder = model_data.get('target_encoder')
            self.feature_columns = model_data['feature_columns']
            self.target_column = model_data['target_column']
            self.model_metadata = model_data['metadata']
            self.config = model_data['config']
            
            logger.info(f"Model loaded: {file_path}")
            return True
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False
    
    def get_model_summary(self) -> Dict[str, Any]:
        """Get model summary"""
        return {
            'model_type': self.config['model_type'],
            'metadata': self.model_metadata,
            'features': self.feature_columns,
            'target': self.target_column
        }


# Convenience functions for easy usage
def create_model(config: Dict[str, Any] = None) -> AutomatedMLModel:
    """Create automated ML model"""
    return AutomatedMLModel(config)

def quick_train(csv_path: str, config: Dict[str, Any] = None) -> AutomatedMLModel:
    """Quick train model on CSV file"""
    model = AutomatedMLModel(config)
    model.train(csv_path)
    return model

def quick_predict(model_path: str, csv_path: str) -> pd.DataFrame:
    """Quick predict on CSV file"""
    model = AutomatedMLModel()
    model.load_model(model_path)
    return model.predict(csv_path)
