# Machine Learning CSV Automation Project

This project automates machine learning workflows for CSV data processing and prediction.

## Project Structure

```
MACHINE_LEARNING_CSV_AUTOMATION/
├── data/                  # CSV files for training and testing
├── models/                # Saved trained models
├── src/                   # Python source code
│   ├── __init__.py
│   ├── train.py          # Training script
│   ├── predict.py        # Prediction script
│   ├── utils.py          # Utility functions
│   └── logger.py         # Logging setup
├── notebooks/            # Jupyter notebooks for exploration
├── output/               # Prediction results and reports
├── logs/                 # Application logs
├── configs/              # Configuration files
├── requirements.txt      # Python dependencies
├── config.py            # Main configuration
├── main.py              # Entry point
└── README.md            # This file
```

## Setup & Installation

### Step 1: Install Python Dependencies
```powershell
pip install -r requirements.txt
```

This installs:
- pandas, numpy (data processing)
- scikit-learn (machine learning)
- matplotlib, seaborn (visualization)
- jupyter (notebooks)

### Step 2: Prepare Your Data

Place your CSV files in the `data/` folder:
- `train.csv` - For training the model
- `test.csv` - For making predictions
- Or use `demo.csv` - Sample file included

### Step 3: Configure Your Project

Edit `config.py` to customize:
- Model type (random_forest, gradient_boosting, etc.)
- Test/train split ratio
- Other hyperparameters

## How to Run

### Method 1: Using the Main Script (Basic)
```powershell
python main.py
```
This will:
1. Train the model on `data/train.csv`
2. Make predictions on `data/test.csv`
3. Save results to `output/` folder
4. Generate logs in `logs/` folder

### Method 2: Using the Automated Model (Recommended)

Create a Python script or use Python terminal:

```python
from models.autom_model import AutomatedMLModel

# Create model with default settings
model = AutomatedMLModel()

# Train on your CSV
results = model.train('data/demo.csv')

print(f"Accuracy: {results['accuracy']:.4f}")
print(f"F1-Score: {results['f1_score']:.4f}")

# Save the trained model
model.save_model('models/trained_model.pkl')

# Make predictions on new data
predictions = model.predict('data/test.csv')
predictions.to_csv('output/predictions.csv', index=False)
```

### Method 3: Custom Configuration
```python
from models.autom_model import AutomatedMLModel

# Define custom requirements
config = {
    'model_type': 'gradient_boosting',
    'test_size': 0.2,
    'handle_missing': True,
    'scaling': True,
    'encode_categorical': True,
    'hyperparameters': {
        'gradient_boosting': {
            'n_estimators': 150,
            'learning_rate': 0.1,
            'max_depth': 5
        }
    }
}

# Create and train model
model = AutomatedMLModel(config)
results = model.train('data/demo.csv')

# Get model summary
print(model.get_model_summary())
```

### Method 4: Quick Train & Predict
```python
from models.autom_model import quick_train, quick_predict

# Quick train
model = quick_train('data/demo.csv')

# Quick predict
predictions = quick_predict('models/trained_model.pkl', 'data/test.csv')
```

### Method 5: Train multiple models and save versioned artifacts
You can train multiple model types and save them with incremental version numbers (for example, `model_random_forest_v1.pkl`, `model_logistic_regression_v2.pkl`) using the helper script:

```powershell
python -m src.train_multi --csv data/train.csv --models random_forest,logistic_regression,svm
```

This will train each specified model type and save models under the `models/` directory with `_vN` suffixes. A small metadata JSON file is also saved alongside each model containing training metrics and metadata.

## Complete Workflow Example

### Step-by-Step Execution:

```powershell
# 1. Navigate to project directory
cd C:\Users\SAGHRI\OneDrive\Desktop\MACHINE_LEARNING_CSV_AUTOMATION

# 2. Install requirements (first time only)
pip install -r requirements.txt

# 3. Run the main project
python main.py
```

### Or use Python interactively:

```powershell
# Start Python
python

# Then run in Python shell:
from models.autom_model import AutomatedMLModel

# Create model
model = AutomatedMLModel()

# Train
print("Training model...")
results = model.train('data/demo.csv')

# Display results
print(f"✓ Model Accuracy: {results['accuracy']:.4f}")
print(f"✓ F1-Score: {results['f1_score']:.4f}")

# Save model
model.save_model('models/trained_model.pkl')
print("✓ Model saved")

# Predict
predictions = model.predict('data/test.csv')
print(f"✓ Predictions made: {len(predictions)} samples")

# Save predictions
predictions.to_csv('output/predictions.csv', index=False)
print("✓ Predictions saved to output/predictions.csv")

# Exit
exit()
```

## Features

- ✅ Reads CSV files from the data folder
- ✅ Automatically detects target columns
- ✅ Handles missing values and categorical data
- ✅ Trains machine learning models
- ✅ Supports multiple model types (Random Forest, Gradient Boosting, SVM, etc.)
- ✅ Generates detailed predictions with confidence scores
- ✅ Produces comprehensive performance reports
- ✅ Logs all activities and results
- ✅ Exports predictions to output folder
- ✅ Save/Load trained models

## Output Files

After running the project, you'll find:

- **output/predictions.csv** - Predictions with confidence scores
- **output/prediction_summary.txt** - Summary report
- **output/report.txt** - Model performance metrics
- **logs/execution.log** - Detailed execution logs
- **models/trained_model.pkl** - Saved trained model

## Troubleshooting

### Issue: Module not found
```
Solution: pip install -r requirements.txt
```

### Issue: CSV file not found
```
Solution: Place your CSV files in the data/ folder
```

### Issue: Import errors
```
Solution: Make sure you're in the project root directory when running scripts
```

## Model Types Available

1. **random_forest** (default) - Best for most cases
2. **gradient_boosting** - Often more accurate but slower
3. **logistic_regression** - Fast, good for binary classification
4. **svm** - Good for complex patterns
