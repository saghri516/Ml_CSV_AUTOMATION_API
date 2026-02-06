"""
FastAPI for Machine Learning CSV Automation
Enables predictions via REST API with automatic documentation
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from pathlib import Path
from datetime import datetime
from models.autom_model import AutomatedMLModel
from src.logger import logger
import traceback
import uvicorn
from io import BytesIO

app = FastAPI(
    title="ML CSV Automation API",
    description="Automated Machine Learning for CSV files",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model instance
model = None

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    global model
    model_path = Path('models/trained_model.pkl')
    if model_path.exists():
        model = AutomatedMLModel()
        model.load_model(str(model_path))
        logger.info("Model loaded on startup")
    else:
        logger.info("No pre-trained model found")

@app.get("/", tags=["Info"])
async def root():
    """Get API information and available endpoints"""
    return {
        "name": "ML CSV Automation API",
        "version": "1.0",
        "framework": "FastAPI",
        "endpoints": {
            "GET /": "API information",
            "GET /docs": "Swagger UI documentation",
            "GET /redoc": "ReDoc documentation",
            "GET /health": "Health check",
            "POST /validate-csv": "Validate CSV format",
            "POST /train": "Train new model",
            "POST /predict": "Make predictions",
            "GET /model-info": "Get model information"
        },
        "docs_url": "http://localhost:5000/docs",
        "redoc_url": "http://localhost:5000/redoc"
    }

@app.get("/health", tags=["Monitoring"])
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": model is not None
    }

@app.get("/model-info", tags=["Model"])
async def model_info():
    """Get model information"""
    if model is None:
        raise HTTPException(status_code=404, detail="Model not loaded. Train a model first.")
    
    try:
        summary = model.get_model_summary()
        return summary
    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/validate-csv", tags=["Utilities"])
async def validate_csv(file: UploadFile = File(...)):
    """Validate CSV format"""
    try:
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="File must be CSV format")
        
        # Read CSV
        contents = await file.read()
        df = pd.read_csv(BytesIO(contents))
        
        validation_result = {
            "valid": True,
            "shape": list(df.shape),
            "columns": list(df.columns),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "missing_values": df.isnull().sum().to_dict(),
            "duplicates": int(df.duplicated().sum())
        }
        
        logger.info(f"CSV validation successful: {df.shape}")
        return validation_result
        
    except Exception as e:
        logger.error(f"CSV validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Invalid CSV: {str(e)}")

@app.post("/predict", tags=["Prediction"])
async def predict(file: UploadFile = File(...)):
    """Make predictions from uploaded CSV"""
    try:
        if model is None:
            raise HTTPException(status_code=404, detail="Model not loaded. Train a model first.")
        
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="File must be CSV format")
        
        # Read CSV
        contents = await file.read()
        df = pd.read_csv(BytesIO(contents))
        
        # Check if CSV is empty
        if df.empty:
            raise HTTPException(status_code=400, detail="CSV file is empty")
        
        # Validate features only if model has trained features
        if model.feature_columns:
            missing_features = set(model.feature_columns) - set(df.columns)
            if missing_features:
                raise HTTPException(
                    status_code=400,
                    detail=f"Missing features: {list(missing_features)}. Expected: {model.feature_columns}"
                )
        
        # Make predictions with DataFrame (not filename)
        predictions_df = model.predict(df)
        
        # Save predictions
        output_file = Path('output') / f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        predictions_df.to_csv(output_file, index=False)
        
        # Return results
        result = {
            "status": "success",
            "total_predictions": len(predictions_df),
            "predictions": predictions_df.to_dict('records')[:10],
            "saved_to": str(output_file),
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"Predictions made successfully: {len(predictions_df)} samples")
        return result
        
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/train", tags=["Training"])
async def train(file: UploadFile = File(...)):
    """Train new model from uploaded CSV"""
    try:
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="File must be CSV format")
        
        # Save temporary file
        temp_path = Path('data') / 'temp_train.csv'
        contents = await file.read()
        with open(temp_path, 'wb') as f:
            f.write(contents)
        
        # Train model
        global model
        model = AutomatedMLModel()
        results = model.train(str(temp_path))
        
        if not results.get('success'):
            return {"status": "error", "error": results.get('error', 'Training failed')}
        
        # Save model
        model_path = Path('models') / f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        model.save_model(str(model_path))
        
        # Clean temp file
        temp_path.unlink()
        
        # Prepare response
        response = {
            "status": "success",
            "message": "Model trained successfully",
            "accuracy": float(results['accuracy']),
            "precision": float(results['precision']),
            "recall": float(results['recall']),
            "f1_score": float(results['f1_score']),
            "model_saved": str(model_path),
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"Model trained: Accuracy={results['accuracy']:.4f}")
        return response
        
    except Exception as e:
        logger.error(f"Training error: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/download/{filename}", tags=["Utilities"])
async def download_file(filename: str):
    """Download prediction file"""
    try:
        file_path = Path('output') / filename
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="File not found")
        
        return FileResponse(
            path=file_path,
            filename=filename,
            media_type='text/csv'
        )
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Download error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

def ensure_dirs():
    # Create necessary directories used by the app/ui
    Path('data').mkdir(exist_ok=True)
    Path('models').mkdir(exist_ok=True)
    Path('output').mkdir(exist_ok=True)
    Path('logs').mkdir(exist_ok=True)


def run_streamlit_app():
    """Streamlit UI integration for interactive use.

    Run with:
        streamlit run app.py -- --streamlit
    Or set environment variable STREAMLIT=1 and run python app.py
    """
    try:
        import streamlit as st
    except Exception as e:
        logger.error(f"Streamlit not available: {e}")
        raise

    # Local imports for Streamlit UI functionality
    from src.train_multi import DEFAULT_MODEL_TYPES, train_multiple
    from src.predict_multi import predict_multiple
    import json

    ensure_dirs()

    st.set_page_config(page_title="ML CSV Automation", layout="wide")
    st.title("ML CSV Automation - Streamlit UI")

    st.sidebar.header("Actions")
    mode = st.sidebar.selectbox("Mode", ["Explore CSV", "Train Models", "Predict with Models", "Manage Models"])

    if mode == "Explore CSV":
        uploaded = st.file_uploader("Upload CSV", type="csv")
        if uploaded:
            df = pd.read_csv(uploaded)
            st.subheader("Data Preview")
            st.dataframe(df.head())
            st.write("Shape:", df.shape)
            st.write("Dtypes:", df.dtypes.to_dict())
            st.write("Missing values:", df.isnull().sum().to_dict())

    elif mode == "Train Models":
        uploaded = st.file_uploader("Upload training CSV", type="csv")
        models = st.multiselect("Model types to train", DEFAULT_MODEL_TYPES, DEFAULT_MODEL_TYPES)
        test_size = st.slider("Test size", 0.05, 0.5, 0.2)
        if st.button("Start training"):
            if not uploaded:
                st.error("Upload a CSV first")
            else:
                tmp = Path('data') / 'st_temp_train.csv'
                with open(tmp, 'wb') as f:
                    f.write(uploaded.getbuffer())
                saved = train_multiple(str(tmp), models)
                st.success(f"Saved {len(saved)} models")

                # show metadata for each saved model
                rows = []
                for p in saved:
                    meta_file = Path(p).with_suffix('.json')
                    meta = {}
                    if meta_file.exists():
                        try:
                            meta = json.loads(meta_file.read_text())
                        except Exception:
                            meta = {}
                    rows.append({"model_path": p, "metadata": meta})
                st.write(rows)

    elif mode == "Predict with Models":
        uploaded = st.file_uploader("Upload CSV to predict", type="csv")
        models_dir = st.text_input("Models directory", 'models')
        models_list = [str(p) for p in Path(models_dir).glob('*.pkl')] if Path(models_dir).exists() else []
        selected = st.multiselect("Select models", models_list, models_list)
        if st.button("Run prediction"):
            if not selected or not uploaded:
                st.error("Upload CSV and select at least one model")
            else:
                tmp = Path('data') / 'st_temp_pred.csv'
                with open(tmp, 'wb') as f:
                    f.write(uploaded.getbuffer())
                out = predict_multiple(selected, str(tmp), out_dir='output')
                df_out = pd.read_csv(out)
                st.subheader("Combined Predictions")
                st.dataframe(df_out.head())
                csv_bytes = df_out.to_csv(index=False).encode()
                st.download_button("Download predictions", csv_bytes, file_name=Path(out).name)

    elif mode == "Manage Models":
        st.subheader("Available models")
        md = Path('models')
        if md.exists():
            dfm = []
            for p in sorted(md.glob('*.pkl')):
                meta_file = p.with_suffix('.json')
                meta = {}
                if meta_file.exists():
                    try:
                        meta = json.loads(meta_file.read_text())
                    except Exception:
                        meta = {}
                dfm.append({"path": str(p), "metadata": meta})
            st.dataframe(dfm)
        else:
            st.info("No models dir yet")


if __name__ == "__main__":
    ensure_dirs()

    import sys, os

    # Decide whether to run Streamlit UI or FastAPI server
    run_streamlit = False
    if '--streamlit' in sys.argv or os.environ.get('STREAMLIT', '').lower() == '1' or 'streamlit' in sys.modules:
        run_streamlit = True

    if run_streamlit:
        run_streamlit_app()
    else:
        logger.info("Starting FastAPI server...")
        uvicorn.run(
            "app:app",
            host="127.0.0.1",
            port=5000,
            reload=False,
            log_level="info"
        )
