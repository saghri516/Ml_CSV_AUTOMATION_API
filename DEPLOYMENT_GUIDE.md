# Deployment Guide

## Quick Start (Development)

### 1. Install Dependencies
```powershell
pip install -r requirements.txt
pip install fastapi uvicorn
```

### 2. Run Locally
```powershell
python app.py
```

API runs at: `http://localhost:5000`

---

## Production Deployment

### Option 1: Docker (Recommended)

#### Prerequisites
- Docker and Docker Compose installed

#### Deploy with Docker
```bash
# Build image
docker build -t ml-automation .

# Run container
docker run -p 5000:5000 -v $(pwd)/data:/app/data -v $(pwd)/models:/app/models ml-automation
```

#### Deploy with Docker Compose
```bash
# Start services
docker-compose up -d

# View logs
docker-compose logs -f ml-api

# Stop services
docker-compose down
```

#### Verify Deployment
```bash
curl http://localhost:5000/health
```

---

### Option 2: Gunicorn (Linux/Mac)

#### Install Gunicorn
```bash
pip install gunicorn
```

#### Run with Gunicorn
```bash
gunicorn --bind 0.0.0.0:5000 --workers 4 --timeout 120 app:app
```

#### Run in Background
```bash
nohup gunicorn --bind 0.0.0.0:5000 --workers 4 --timeout 120 app:app &
```

---

### Option 3: Systemd Service (Linux)

Create `/etc/systemd/system/ml-automation.service`:
```ini
[Unit]
Description=ML CSV Automation Service
After=network.target

[Service]
Type=notify
User=www-data
WorkingDirectory=/home/user/MACHINE_LEARNING_CSV_AUTOMATION
ExecStart=/usr/bin/gunicorn --bind 0.0.0.0:5000 --workers 4 --timeout 120 app:app
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Then:
```bash
sudo systemctl daemon-reload
sudo systemctl enable ml-automation
sudo systemctl start ml-automation
sudo systemctl status ml-automation
```

---

### Option 4: Cloud Deployment (AWS/Heroku/GCP)

#### Heroku
```bash
# Create Procfile
echo "web: gunicorn app:app" > Procfile

# Deploy
git init
git add .
git commit -m "Initial commit"
heroku create
git push heroku main
```

#### AWS EC2
```bash
# SSH into instance
ssh -i key.pem ec2-user@instance-ip

# Install dependencies
sudo yum update -y
sudo yum install python3 python3-pip -y

# Clone repo and setup
git clone <repo-url>
cd MACHINE_LEARNING_CSV_AUTOMATION
pip install -r requirements.txt

# Run with Gunicorn
gunicorn --bind 0.0.0.0:5000 --workers 4 app:app
```

---

## Configuration

### Environment Variables
Create `.env` file:
```
ENVIRONMENT=production
DEBUG=False
PORT=5000
MODEL_TYPE=random_forest
LOG_LEVEL=INFO
```

Load in your app:
```python
from dotenv import load_dotenv
import os

load_dotenv()
flask_env = os.getenv('FLASK_ENV', 'development')
```

---

## Security Considerations

### 1. HTTPS/SSL
```bash
# Generate self-signed certificate
openssl req -x509 -newkey rsa:4096 -nodes -out cert.pem -keyout key.pem -days 365

# Run with SSL
gunicorn --certfile=cert.pem --keyfile=key.pem --bind 0.0.0.0:5000 app:app
```

### 2. API Key Authentication
```python
from functools import wraps

def require_api_key(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        api_key = request.headers.get('X-API-Key')
        if not api_key or api_key != os.getenv('API_KEY'):
            return jsonify({'error': 'Invalid API key'}), 401
        return f(*args, **kwargs)
    return decorated

@app.route('/predict', methods=['POST'])
@require_api_key
def predict():
    # Protected endpoint
    pass
```

### 3. File Upload Security
```python
from werkzeug.utils import secure_filename
import os

ALLOWED_EXTENSIONS = {'csv'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

if file and allowed_file(file.filename):
    filename = secure_filename(file.filename)
```

---

## Monitoring & Logging

### View Logs
```bash
# Local
tail -f logs/app.log

# Docker
docker logs -f ml-api

# Systemd
journalctl -u ml-automation -f
```

### Health Check
```bash
curl http://localhost:5000/health
```

### Monitoring Tools
- **Prometheus**: For metrics
- **ELK Stack**: For logs
- **Grafana**: For visualization
- **NewRelic**: For APM

---

## Testing

### Run Unit Tests
```bash
python -m unittest tests.py -v
```

### API Testing with Postman
1. Import API_DOCUMENTATION.md
2. Configure base URL: `http://localhost:5000`
3. Test each endpoint

### Load Testing
```bash
# Using Apache Bench
ab -n 1000 -c 10 http://localhost:5000/health

# Using wrk
wrk -t4 -c100 -d30s http://localhost:5000/health
```

---

## Scaling

### Horizontal Scaling (Multiple Instances)
```bash
# Load balancer (Nginx)
upstream ml_api {
    server localhost:5000;
    server localhost:5001;
    server localhost:5002;
}

server {
    listen 80;
    location / {
        proxy_pass http://ml_api;
    }
}
```

### Vertical Scaling
- Increase Gunicorn workers: `--workers 8`
- Increase machine resources

### Caching
```python
from flask_caching import Cache

cache = Cache(app, config={'CACHE_TYPE': 'simple'})

@app.route('/model-info')
@cache.cached(timeout=300)
def model_info():
    return get_model_info()
```

---

## Troubleshooting

### Port Already in Use
```bash
# Find process using port 5000
lsof -i :5000

# Kill process
kill -9 <PID>
```

### Model Not Loading
```bash
# Check model file exists
ls -la models/

# Check permissions
chmod 644 models/*.pkl
```

### High Memory Usage
```bash
# Reduce Gunicorn workers
gunicorn --workers 2 app:app

# Limit data file size in config
MAX_FILE_SIZE=50  # MB
```

---

## Backup & Recovery

### Backup Models
```bash
tar -czf models_backup_$(date +%Y%m%d).tar.gz models/
```

### Backup Data
```bash
tar -czf data_backup_$(date +%Y%m%d).tar.gz data/
```

### Restore
```bash
tar -xzf models_backup_20260204.tar.gz
tar -xzf data_backup_20260204.tar.gz
```

---

## CI/CD Pipeline (GitHub Actions)

A production-ready workflow is provided at `.github/workflows/ci-cd.yml` (no cloud or registry push by default).

**Key points**
- **Triggers:** `push` and `pull_request` to `main`.
- **Jobs:** `lint` (flake8), `test` (pytest with junit & coverage), and `build` (Docker Buildx producing `image.tar`). The `build` job runs only after the `test` job succeeds.
- **Caching:** Uses `actions/cache` to cache pip packages keyed on `requirements.txt`.
- **Artifacts:** Uploaded artifacts include `flake8-report`, `test-reports` (`junit.xml` and `coverage.xml`), and the built `image.tar`.
- **Model Retraining:** A separate workflow `.github/workflows/model_retraining.yml` runs weekly and on push to `main` to retrain models using `models/autom_model.py`. It produces `trained-models` (saved `models/*.pkl`), `output/metrics.json`, and `output/report.txt` artifacts. The job enforces a configurable minimum metric (`MODEL_MIN_ACCURACY`, default `0.60`) and fails if the model doesn't meet the threshold.
- **Best practices:** concurrency cancel-in-progress, minimal permissions, separate jobs, and building with Buildx (no Docker-in-Docker).

**Run checks locally:**
- Lint: `pip install flake8 && flake8 src/ tests/ --max-line-length=88`
- Tests: `pip install pytest pytest-cov && pytest -q --maxfail=1 --junitxml=reports/junit.xml --cov=src --cov-report=xml:reports/coverage.xml`
- Build image artifact: `docker build -t myimage:tag . && docker save myimage:tag -o image.tar`

> Note: If you want to push images to a registry (Docker Hub or GitHub Packages), we can add a secure job that runs only on tagged releases and uses repository secrets for authentication.

---

## Performance Optimization

### 1. Model Caching
```python
# Load model once
import pickle

CACHED_MODEL = None

def get_model():
    global CACHED_MODEL
    if CACHED_MODEL is None:
        with open('models/trained_model.pkl', 'rb') as f:
            CACHED_MODEL = pickle.load(f)
    return CACHED_MODEL
```

### 2. Batch Processing
```python
# Accept multiple CSV files
@app.route('/predict-batch', methods=['POST'])
def predict_batch():
    files = request.files.getlist('files')
    results = []
    for file in files:
        result = process_file(file)
        results.append(result)
    return jsonify(results)
```

### 3. Async Processing
```python
from celery import Celery

celery = Celery(app.name, broker='redis://localhost:6379')

@celery.task
def train_model_async(file_path):
    # Long-running task
    pass
```

---

## Support & Maintenance

- **Version**: 1.0
- **Last Updated**: 2026-02-04
- **Python**: 3.8+
- **Support**: Check logs and API_DOCUMENTATION.md
