# 🛡️ Network Security Detection System

> A comprehensive end-to-end machine learning solution for detecting and classifying network security threats in real-time using advanced ML algorithms, automated pipelines, and production-ready deployment.

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-2.0%2B-green.svg)](https://flask.palletsprojects.com/)
[![MongoDB](https://img.shields.io/badge/MongoDB-4.4%2B-brightgreen.svg)](https://www.mongodb.com/)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://www.docker.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0%2B-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📑 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Technology Stack](#-technology-stack)
- [System Architecture](#-system-architecture)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Configuration](#-configuration)
- [Quick Start](#-quick-start)
- [Detailed Usage](#-detailed-usage)
- [Data Pipeline](#-data-pipeline)
- [Model Training](#-model-training)
- [Making Predictions](#-making-predictions)
- [API Reference](#-api-reference)
- [Docker Deployment](#-docker-deployment)
- [Cloud Deployment](#-cloud-deployment)
- [Monitoring & Logging](#-monitoring--logging)
- [Testing](#-testing)
- [Performance Optimization](#-performance-optimization)
- [Troubleshooting](#-troubleshooting)
- [Best Practices](#-best-practices)
- [Contributing](#-contributing)
- [FAQ](#-faq)
- [Roadmap](#-roadmap)
- [License](#-license)
- [Contact](#-contact)

---

## 🎯 Overview

### Making Predictions

#### Web Interface Prediction

1. **Start the application**
```bash
python app.py
```

2. **Navigate to** `http://localhost:5000`

3. **Upload your data file** (CSV format)

4. **View and download results**

#### API Prediction Examples

**Single Record Prediction:**
```python
import requests

url = "http://localhost:5000/api/predict"
data = {
    "protocol_type": "tcp",
    "service": "http",
    "flag": "SF",
    "src_bytes": 181,
    "dst_bytes": 5450,
    "count": 8,
    "srv_count": 8,
    "serror_rate": 0.0,
    "srv_serror_rate": 0.0,
    "rerror_rate": 0.0,
    "srv_rerror_rate": 0.0,
    "same_srv_rate": 1.0,
    "diff_srv_rate": 0.0,
    "srv_diff_host_rate": 0.0,
    "dst_host_count": 9,
    "dst_host_srv_count": 9
}

response = requests.post(url, json=data)
result = response.json()

print(f"Prediction: {result['label']}")
print(f"Confidence: {result['probability']}")
```

**Batch Prediction:**
```python
import requests
import pandas as pd

# Prepare data
df = pd.read_csv('test_data.csv')

# Upload file
url = "http://localhost:5000/api/predict/batch"
files = {'file': open('test_data.csv', 'rb')}

response = requests.post(url, files=files)
result = response.json()

print(f"Total predictions: {result['total_records']}")
print(f"Normal: {result['summary']['normal_count']}")
print(f"Attacks: {result['summary']['attack_count']}")
print(f"Results saved to: {result['output_file']}")
```

#### Python Script Prediction

```python
from network_security.pipeline.prediction_pipeline import PredictionPipeline
import pandas as pd
import numpy as np

# Initialize predictor
predictor = PredictionPipeline()

# Load test data
test_data = pd.read_csv('network_data/test_data.csv')

# Make predictions
predictions = predictor.predict(test_data)

# Get probability scores
predictions, probabilities = predictor.predict_proba(test_data)

# Create results dataframe
results_df = test_data.copy()
results_df['prediction'] = predictions
results_df['normal_probability'] = probabilities[:, 0]
results_df['attack_probability'] = probabilities[:, 1]
results_df['prediction_label'] = np.where(
    predictions == 0, 'Normal', 'Attack'
)

# Save results
output_path = 'prediction_output/results.csv'
results_df.to_csv(output_path, index=False)
print(f"Results saved to {output_path}")

# Display summary
print("\nPrediction Summary:")
print(results_df['prediction_label'].value_counts())
print(f"\nAverage Attack Probability: {results_df['attack_probability'].mean():.4f}")
```

---

## 🔄 Data Pipeline

### ETL Pipeline

#### Running ETL

```bash
# Basic ETL
python push_data_etl_py

# With custom source
python push_data_etl_py --source network_data/raw_data.csv

# Specify database and collection
python push_data_etl_py \
    --source network_data/raw_data.csv \
    --database network_security \
    --collection train_data

# With data validation
python push_data_etl_py \
    --source network_data/raw_data.csv \
    --validate \
    --schema data_schema/schema.json
```

#### Custom ETL Script

```python
import pandas as pd
from pymongo import MongoClient
from network_security.utils.database_utils import MongoDBClient
from network_security.logger import logging

class DataETL:
    def __init__(self, source_path, mongo_url, database, collection):
        self.source_path = source_path
        self.mongo_client = MongoDBClient(mongo_url)
        self.database = database
        self.collection = collection
    
    def extract(self):
        """Extract data from source"""
        logging.info(f"Extracting data from {self.source_path}")
        df = pd.read_csv(self.source_path)
        return df
    
    def transform(self, df):
        """Transform data"""
        logging.info("Transforming data")
        
        # Remove duplicates
        df = df.drop_duplicates()
        
        # Handle missing values
        df = df.fillna(df.mean())
        
        # Convert data types
        df = df.astype({
            'src_bytes': 'int64',
            'dst_bytes': 'int64',
            'count': 'int32'
        })
        
        # Add timestamp
        df['created_at'] = pd.Timestamp.now()
        
        return df
    
    def load(self, df):
        """Load data to MongoDB"""
        logging.info(f"Loading data to MongoDB: {self.database}.{self.collection}")
        
        # Convert to dict
        records = df.to_dict('records')
        
        # Insert to MongoDB
        collection = self.mongo_client.get_collection(
            self.database, 
            self.collection
        )
        collection.insert_many(records)
        
        logging.info(f"Loaded {len(records)} records")
    
    def run(self):
        """Run ETL pipeline"""
        df = self.extract()
        df = self.transform(df)
        self.load(df)
        logging.info("ETL completed successfully")

# Usage
etl = DataETL(
    source_path='network_data/raw_data.csv',
    mongo_url='mongodb://localhost:27017',
    database='network_security',
    collection='network_data'
)
etl.run()
```

### Data Validation

```python
from network_security.components.data_validation import DataValidation
from network_security.entity.config_entity import DataValidationConfig
import json

# Load schema
with open('data_schema/schema.json', 'r') as f:
    schema = json.load(f)

# Initialize validator
config = DataValidationConfig()
validator = DataValidation(config, data_ingestion_artifact)

# Run validation
validation_artifact = validator.initiate_data_validation()

# Check results
if validation_artifact.validation_status:
    print("✓ Data validation passed")
    print(f"Valid data saved to: {validation_artifact.valid_data_dir}")
else:
    print("✗ Data validation failed")
    print(f"Check report: {validation_artifact.drift_report_file_path}")
```

---

## 🤖 Model Training

### Algorithm Comparison

The system trains and compares multiple algorithms:

| Algorithm | Pros | Cons | Best For |
|-----------|------|------|----------|
| **Random Forest** | High accuracy, handles non-linear data | Can overfit, slower | Balanced datasets |
| **XGBoost** | Excellent performance, fast | Requires tuning | Large datasets |
| **SVM** | Good for high-dimensional data | Slow on large datasets | Small to medium data |
| **Logistic Regression** | Fast, interpretable | Linear relationships only | Simple patterns |
| **Decision Tree** | Easy to interpret | Prone to overfitting | Quick baselines |

### Training Workflow

```
Data → Split → Train Models → Cross-Validation → Select Best → Save
```

### Hyperparameter Tuning

```python
from sklearn.model_selection import GridSearchCV
from network_security.components.model_trainer import ModelTrainer

# Define parameter grid
param_grid = {
    'RandomForest': {
        'n_estimators': [50, 100, 200],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    },
    'XGBoost': {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 6, 9],
        'learning_rate': [0.01, 0.1, 0.3],
        'subsample': [0.8, 0.9, 1.0]
    }
}

# Initialize trainer
trainer = ModelTrainer(model_trainer_config, data_transformation_artifact)

# Run hyperparameter tuning
best_model, best_params = trainer.tune_hyperparameters(
    param_grid=param_grid,
    cv=5,
    scoring='f1_weighted'
)

print(f"Best Model: {best_model}")
print(f"Best Parameters: {best_params}")
```

### Model Evaluation

```python
from network_security.components.model_evaluation import ModelEvaluation
from network_security.utils.ml_utils import calculate_metrics
import matplotlib.pyplot as plt

# Evaluate model
evaluator = ModelEvaluation(model_evaluation_config, model_trainer_artifact)
evaluation_artifact = evaluator.initiate_model_evaluation()

# Print metrics
print("Model Performance:")
print(f"Accuracy:  {evaluation_artifact.accuracy:.4f}")
print(f"Precision: {evaluation_artifact.precision:.4f}")
print(f"Recall:    {evaluation_artifact.recall:.4f}")
print(f"F1-Score:  {evaluation_artifact.f1_score:.4f}")
print(f"ROC-AUC:   {evaluation_artifact.roc_auc:.4f}")

# Plot confusion matrix
from sklearn.metrics import ConfusionMatrixDisplay
ConfusionMatrixDisplay.from_predictions(
    y_true=evaluation_artifact.y_true,
    y_pred=evaluation_artifact.y_pred
)
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')
plt.show()
```

---

## 🌐 API Reference

### Complete API Documentation

#### Authentication (Optional)

If authentication is enabled:

```bash
# Get API token
curl -X POST http://localhost:5000/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username": "admin", "password": "password"}'

# Response
{
  "token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "expires_in": 3600
}

# Use token in requests
curl -X POST http://localhost:5000/api/predict \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{...}'
```

#### Endpoints

##### 1. Health Check

```bash
GET /
GET /health
GET /api/health
```

**Response:**
```json
{
  "status": "healthy",
  "service": "Network Security Detection System",
  "version": "2.0.0",
  "uptime": "3d 5h 23m",
  "timestamp": "2026-01-06T10:30:00Z",
  "database": {
    "status": "connected",
    "type": "MongoDB"
  },
  "model": {
    "loaded": true,
    "version": "1.0.0",
    "last_trained": "2026-01-05T15:30:00Z"
  }
}
```

##### 2. Single Prediction

```bash
POST /api/predict
POST /api/v1/predict
```

**Request:**
```json
{
  "protocol_type": "tcp",
  "service": "http",
  "flag": "SF",
  "src_bytes": 181,
  "dst_bytes": 5450,
  "count": 8,
  "srv_count": 8,
  "serror_rate": 0.0,
  "srv_serror_rate": 0.0,
  "rerror_rate": 0.0,
  "srv_rerror_rate": 0.0,
  "same_srv_rate": 1.0,
  "diff_srv_rate": 0.0,
  "srv_diff_host_rate": 0.0,
  "dst_host_count": 9,
  "dst_host_srv_count": 9
}
```

**Response:**
```json
{
  "success": true,
  "prediction": {
    "class": 0,
    "label": "Normal",
    "probability": {
      "normal": 0.892,
      "attack": 0.108
    },
    "confidence": "high"
  },
  "metadata": {
    "model_version": "1.0.0",
    "prediction_id": "pred_20260106_103045_xyz",
    "timestamp": "2026-01-06T10:30:45Z",
    "processing_time_ms": 12
  }
}
```

##### 3. Batch Prediction

```bash
POST /api/predict/batch
```

**Request:**
- Content-Type: `multipart/form-data`
- Field: `file` (CSV file)

**Response:**
```json
{
  "success": true,
  "total_records": 1000,
  "processed": 1000,
  "failed": 0,
  "summary": {
    "normal_count": 847,
    "attack_count": 153,
    "normal_percentage": 84.7,
    "attack_percentage": 15.3
  },
  "predictions": [
    {
      "record_id": 1,
      "prediction": 0,
      "label": "Normal",
      "confidence": 0.89
    },
    // ... more predictions
  ],
  "output_file": "prediction_output/predictions_20260106_103000.csv",
  "download_url": "/api/download/predictions_20260106_103000.csv",
  "metadata": {
    "model_version": "1.0.0",
    "batch_id": "batch_20260106_103000",
    "timestamp": "2026-01-06T10:30:00Z",
    "processing_time_ms": 523
  }
}
```

##### 4. Model Information

```bash
GET /api/model/info
GET /api/v1/model/info
```

**Response:**
```json
{
  "model": {
    "name": "NetworkSecurityModel",
    "type": "RandomForestClassifier",
    "version": "1.0.0",
    "algorithm": "Random Forest",
    "trained_date": "2026-01-05T15:30:00Z",
    "last_updated": "2026-01-05T15:30:00Z"
  },
  "training": {
    "training_samples": 50000,
    "validation_samples": 10000,
    "test_samples": 12500,
    "features": 20,
    "classes": 2,
    "class_names": ["Normal", "Attack"]
  },
  "performance": {
    "accuracy": 0.9612,
    "precision": 0.9534,
    "recall": 0.9442,
    "f1_score": 0.9488,
    "roc_auc": 0.9856
  },
  "features": [
    "protocol_type",
    "service",
    "flag",
    "src_bytes",
    "dst_bytes",
    // ... more features
  ]
}
```

##### 5. Prediction History

```bash
GET /api/predictions/history?limit=10&offset=0&from_date=2026-01-01&to_date=2026-01-06
```

**Response:**
```json
{
  "success": true,
  "total": 1523,
  "limit": 10,
  "offset": 0,
  "filters": {
    "from_date": "2026-01-01",
    "to_date": "2026-01-06"
  },
  "predictions": [
    {
      "id": "pred_20260106_103045_xyz",
      "timestamp": "2026-01-06T10:30:45Z",
      "input": {
        "protocol_type": "tcp",
        "service": "http",
        // ... more fields
      },
      "output": {
        "prediction": 0,
        "label": "Normal",
        "confidence": 0.89
      },
      "metadata": {
        "model_version": "1.0.0",
        "processing_time_ms": 12
      }
    },
    // ... more predictions
  ],
  "pagination": {
    "current_page": 1,
    "total_pages": 153,
    "has_next": true,
    "has_previous": false,
    "next_offset": 10,
    "previous_offset": null
  }
}
```

##### 6. Statistics

```bash
GET /api/stats
GET /api/v1/stats
```

**Response:**
```json
{
  "success": true,
  "period": {
    "from": "2026-01-01T00:00:00Z",
    "to": "2026-01-06T23:59:59Z"
  },
  "predictions": {
    "total": 15234,
    "today": 523,
    "this_week": 3421,
    "this_month": 15234
  },
  "distribution": {
    "normal": {
      "count": 12987,
      "percentage": 85.3
    },
    "attack": {
      "count": 2247,
      "percentage": 14.7
    }
  },
  "performance": {
    "avg_processing_time_ms": 14.2,
    "max_processing_time_ms": 156,
    "min_processing_time_ms": 8
  },
  "top_attack_types": [
    {"type": "DoS", "count": 1234},
    {"type": "Probe", "count": 678},
    {"type": "R2L", "count": 245},
    {"type": "U2R", "count": 90}
  ]
}
```

##### 7. Download Results

```bash
GET /api/download/{filename}
```

Downloads the specified prediction result file.

### Error Responses

#### 400 Bad Request
```json
{
  "success": false,
  "error": {
    "code": "INVALID_INPUT",
    "message": "Invalid input data",
    "details": {
      "field": "src_bytes",
      "issue": "Must be a positive integer"
    }
  },
  "timestamp": "2026-01-06T10:30:00Z"
}
```

#### 404 Not Found
```json
{
  "success": false,
  "error": {
    "code": "NOT_FOUND",
    "message": "Resource not found",
    "details": "The requested prediction ID does not exist"
  },
  "timestamp": "2026-01-06T10:30:00Z"
}
```

#### 500 Internal Server Error
```json
{
  "success": false,
  "error": {
    "code": "INTERNAL_ERROR",
    "message": "Internal server error",
    "details": "An unexpected error occurred while processing your request"
  },
  "timestamp": "2026-01-06T10:30:00Z",
  "request_id": "req_20260106_103000_abc"
}
```

---

## 🐳 Docker Deployment

### Quick Start with Docker

```bash
# Build image
docker build -t network-security:latest .

# Run container
docker run -d -p 5000:5000 --name network-security-app network-security:latest

# View logs
docker logs -f network-security-app

# Stop container
docker stop network-security-app
```

### Docker Compose Deployment

**docker-compose.yml:**
```yaml
version: '3.8'

services:
  web:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: network-security-web
    ports:
      - "5000:5000"
    environment:
      - FLASK_ENV=production
      - MONGO_DB_URL=mongodb://mongodb:27017
      - MONGO_DB_NAME=network_security
    volumes:
      - ./prediction_output:/app/prediction_output
      - ./final_model:/app/final_model
      - ./logs:/app/logs
    depends_on:
      - mongodb
    restart: unless-stopped
    networks:
      - app-network
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:5000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

  mongodb:
    image: mongo:4.4
    container_name: network-security-mongodb
    ports:
      - "27017:27017"
    volumes:
      - mongodb_data:/data/db
      - ./mongo-init:/docker-entrypoint-initdb.d
    environment:
      - MONGO_INITDB_ROOT_USERNAME=admin
      - MONGO_INITDB_ROOT_PASSWORD=password
      - MONGO_INITDB_DATABASE=network_security
    restart: unless-stopped
    networks:
      - app-network
    healthcheck:
      test: echo 'db.runCommand("ping").ok' | mongo localhost:27017/test --quiet
      interval: 10s
      timeout: 10s
      retries: 5

  nginx:
    image: nginx:alpine
    container_name: network-security-nginx
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf
      - ./nginx/ssl:/etc/nginx/ssl
    depends_on:
      - web
    restart: unless-stopped
    networks:
      - app-network

networks:
  app-network:
    driver: bridge

volumes:
  mongodb_data:
    driver: local
```

**Run with Docker Compose:**
```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Scale web service
docker-compose up -d --scale web=3

# Stop all services
docker-compose down

# Stop and remove volumes
docker-compose down -v
```

### Production Dockerfile

```dockerfile
# Multi-stage build for smaller image
FROM python:3.9-slim as builder

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --user -r requirements.txt

# Final stage
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy Python dependencies from builder
COPY --from=builder /root/.local /root/.local

# Make sure scripts in .local are usable
ENV PATH=/root/.local/bin:$PATH

# Copy application code
COPY . .

# Install package
RUN pip install --no-cache-dir -e .

# Create necessary directories
RUN mkdir -p logs artifacts prediction_output final_model

# Expose port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
  CMD python -c "import requests; requests.get('http://localhost:5000/health')"

# Run application
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "4", "--threads", "2", "--timeout", "120", "app:app"]
```

### Docker Best Practices

1. **Use .dockerignore:**
```
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
venv/
env/
*.log
.git/
.gitignore
README.md
tests/
docs/
notebooks/
.pytest_cache/
.coverage
htmlcov/
.DS_Store
```

2. **Environment Variables in Docker:**
```bash
# Pass environment variables
docker run -d \
  -e MONGO_DB_URL=mongodb://mongodb:27017 \
  -e FLASK_ENV=production \
  -e LOG_LEVEL=INFO \
  network-security:latest

# Use env file
docker run -d --env-file .env network-security:latest
```

3. **Volume Mounting:**
```bash
# Mount specific directories
docker run -d \
  -v $(pwd)/final_model:/app/final_model \
  -v $(pwd)/prediction_output:/app/prediction_output \
  -v $(pwd)/logs:/app/logs \
  network-security:latest
```

---

## What is NetworkSecurity2?

NetworkSecurity2 is an enterprise-grade, machine learning-powered network intrusion detection system (NIDS) that analyzes network traffic patterns to identify potential security threats and anomalies in real-time. Built with scalability and production deployment in mind, this system offers a complete ML pipeline from data ingestion to model deployment.

### Problem Statement

Modern networks face constant threats from various types of attacks including DoS, DDoS, port scanning, malware, phishing, and zero-day exploits. Traditional signature-based detection systems struggle to identify novel attack patterns. This project addresses these challenges by leveraging machine learning to:

- Detect known and unknown attack patterns
- Adapt to evolving threat landscapes
- Minimize false positives
- Provide real-time threat detection
- Scale across enterprise networks

### Key Capabilities

**🔍 Detection & Classification**
- Multi-class attack classification (DoS, Probe, R2L, U2R, Normal)
- Binary classification (Attack vs Normal)
- Anomaly detection for zero-day threats
- Real-time traffic analysis
- 95%+ accuracy on validation datasets

**🏗️ Production-Ready Architecture**
- Modular pipeline design
- RESTful API for integration
- Docker containerization
- Scalable MongoDB backend
- Comprehensive logging and monitoring

**📊 Advanced ML Capabilities**
- Multiple algorithm support (Random Forest, XGBoost, SVM, etc.)
- Automated hyperparameter tuning
- Cross-validation and model selection
- Class imbalance handling (SMOTE)
- Feature engineering and selection

**🌐 User-Friendly Interface**
- Web-based dashboard
- Batch and real-time predictions
- Result visualization
- CSV upload/download
- API documentation

### Use Cases

| Industry | Application |
|----------|-------------|
| **Enterprise IT** | Network security monitoring, intrusion detection |
| **Cloud Providers** | Multi-tenant security, API protection |
| **Financial Services** | Transaction monitoring, fraud detection |
| **Healthcare** | HIPAA compliance, patient data protection |
| **Government** | Critical infrastructure protection |
| **Education** | Campus network security, research data protection |
| **IoT/Manufacturing** | Industrial control system security |
| **Telecommunications** | Network traffic analysis, DDoS mitigation |

---

## ✨ Key Features

### Core Detection Features

#### 🎯 Multi-Class Attack Detection
```
✓ DoS (Denial of Service) attacks
✓ Probe (Port scanning, vulnerability scanning)
✓ R2L (Remote to Local - unauthorized access)
✓ U2R (User to Root - privilege escalation)
✓ Normal traffic classification
```

#### 🔍 Advanced Analysis
- **Behavioral Analysis**: Identifies patterns deviating from normal behavior
- **Protocol Analysis**: Deep packet inspection across TCP/IP layers
- **Statistical Analysis**: Time-series analysis of traffic patterns
- **Signature Matching**: Combines rule-based and ML approaches

### Pipeline Features

#### 📥 Data Ingestion
- Multiple data source support (CSV, JSON, MongoDB, APIs)
- Automated data collection
- Real-time streaming support
- Batch processing capabilities
- Data versioning

#### ✅ Data Validation
- JSON schema validation
- Data type checking
- Range validation
- Missing value detection
- Outlier identification
- Data quality reports

#### 🔄 Data Transformation
- Missing value imputation
- Categorical encoding (One-Hot, Label, Target)
- Numerical scaling (StandardScaler, MinMaxScaler)
- Feature engineering
- Dimensionality reduction
- SMOTE for class balancing

#### 🤖 Model Training
- Multiple algorithm support
- Automated hyperparameter tuning
- Cross-validation (K-Fold, Stratified)
- Ensemble methods
- Model versioning
- Experiment tracking

#### 📊 Model Evaluation
- Comprehensive metrics (Accuracy, Precision, Recall, F1, ROC-AUC)
- Confusion matrices
- ROC curves
- Precision-Recall curves
- Classification reports
- Feature importance analysis

### Web Application Features

#### 🖥️ User Interface
- Clean, intuitive design
- Responsive layout (mobile-friendly)
- File upload with drag-and-drop
- Real-time prediction results
- Download predictions as CSV
- Historical prediction viewing

#### 🔌 API Features
- RESTful API design
- JSON request/response
- Batch prediction endpoints
- Model information endpoints
- Authentication support (optional)
- Rate limiting
- API documentation (Swagger/OpenAPI)

### DevOps & Deployment

#### 🐳 Containerization
- Docker support
- Docker Compose orchestration
- Multi-stage builds
- Image optimization
- Health checks

#### ☁️ Cloud-Ready
- AWS deployment guides
- Azure integration
- GCP support
- Kubernetes manifests
- CI/CD pipeline templates

#### 📝 Logging & Monitoring
- Structured logging
- Error tracking
- Performance monitoring
- Custom metrics
- Alert configurations

---

## 🛠️ Technology Stack

### Programming & Frameworks

| Category | Technology | Version | Purpose |
|----------|------------|---------|---------|
| **Language** | Python | 3.8+ | Core development language |
| **Web Framework** | Flask | 2.0+ | REST API and web interface |
| **WSGI Server** | Gunicorn | 20.1+ | Production web server |

### Machine Learning Stack

| Library | Purpose |
|---------|---------|
| **scikit-learn** | ML algorithms, preprocessing, metrics |
| **XGBoost** | Gradient boosting models |
| **imbalanced-learn** | Handling imbalanced datasets (SMOTE) |
| **pandas** | Data manipulation and analysis |
| **numpy** | Numerical computing |
| **scipy** | Scientific computing |

### Data Storage & Processing

| Technology | Purpose |
|------------|---------|
| **MongoDB** | Primary database for storing training data |
| **pymongo** | Python MongoDB driver |
| **pickle** | Model serialization |
| **joblib** | Efficient model persistence |

### Web & API

| Technology | Purpose |
|------------|---------|
| **Flask-CORS** | Cross-Origin Resource Sharing |
| **Jinja2** | Template engine |
| **WTForms** | Form validation (optional) |

### Validation & Testing

| Technology | Purpose |
|------------|---------|
| **jsonschema** | JSON schema validation |
| **pytest** | Testing framework |
| **unittest** | Unit testing |
| **coverage** | Code coverage analysis |

### DevOps & Deployment

| Technology | Purpose |
|------------|---------|
| **Docker** | Containerization |
| **docker-compose** | Multi-container orchestration |
| **python-dotenv** | Environment variable management |

### Development Tools

| Tool | Purpose |
|------|---------|
| **Git** | Version control |
| **VS Code** | IDE (recommended) |
| **Jupyter** | Interactive development |
| **Postman** | API testing |

### Complete Requirements

```txt
# Core Dependencies
Flask==2.0.3
gunicorn==20.1.0
python-dotenv==0.19.2

# Machine Learning
scikit-learn==1.0.2
xgboost==1.5.2
imbalanced-learn==0.9.0
pandas==1.4.1
numpy==1.22.2
scipy==1.8.0

# Database
pymongo==4.0.2
pymongo[srv]==4.0.2

# Utilities
jsonschema==4.4.0
python-dateutil==2.8.2
pytz==2021.3

# Web
Flask-CORS==3.0.10
Jinja2==3.0.3
MarkupSafe==2.1.0

# Testing
pytest==7.0.1
pytest-cov==3.0.0
coverage==6.3.2

# Logging
colorlog==6.6.0

# AWS (Optional)
boto3==1.21.6
botocore==1.24.6

# Serialization
dill==0.3.4
cloudpickle==2.0.0
```

---

## 🏗️ System Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        CLIENT LAYER                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │ Web Browser │  │   Mobile    │  │  API Client │             │
│  │   (HTML/JS) │  │     App     │  │   (Python)  │             │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘             │
└─────────┼─────────────────┼─────────────────┼───────────────────┘
          │                 │                 │
          └─────────────────┴─────────────────┘
                            │
┌───────────────────────────▼───────────────────────────────────────┐
│                     APPLICATION LAYER                              │
│  ┌──────────────────────────────────────────────────────────┐    │
│  │               Flask Web Application                       │    │
│  │                     (app.py)                              │    │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │    │
│  │  │   Routes     │  │  Middleware  │  │     CORS     │   │    │
│  │  │   Handler    │  │   Security   │  │   Handler    │   │    │
│  │  └──────────────┘  └──────────────┘  └──────────────┘   │    │
│  └────────────────────────┬─────────────────────────────────┘    │
└───────────────────────────┼───────────────────────────────────────┘
                            │
┌───────────────────────────▼───────────────────────────────────────┐
│                     PIPELINE LAYER                                 │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │              Training Pipeline (main.py)                    │  │
│  │  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐        │  │
│  │  │Ingest│→ │Valid │→ │Trans │→ │Train │→ │ Eval │        │  │
│  │  └──────┘  └──────┘  └──────┘  └──────┘  └──────┘        │  │
│  └────────────────────────────────────────────────────────────┘  │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │           Prediction Pipeline (PredictionPipeline)          │  │
│  │  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐                   │  │
│  │  │ Load │→ │Prepr │→ │Predic│→ │Output│                   │  │
│  │  │ Data │  │ocess │  │  t   │  │      │                   │  │
│  │  └──────┘  └──────┘  └──────┘  └──────┘                   │  │
│  └────────────────────────────────────────────────────────────┘  │
└───────────────────────────────┬───────────────────────────────────┘
                                │
┌───────────────────────────────▼───────────────────────────────────┐
│                    COMPONENT LAYER                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           │
│  │     Data     │  │     Data     │  │     Data     │           │
│  │  Ingestion   │  │  Validation  │  │Transformation│           │
│  └──────────────┘  └──────────────┘  └──────────────┘           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           │
│  │    Model     │  │    Model     │  │    Model     │           │
│  │   Trainer    │  │  Evaluation  │  │   Pusher     │           │
│  └──────────────┘  └──────────────┘  └──────────────┘           │
└───────────────────────────────┬───────────────────────────────────┘
                                │
┌───────────────────────────────▼───────────────────────────────────┐
│                     STORAGE LAYER                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           │
│  │   MongoDB    │  │    Local     │  │     Cloud    │           │
│  │   Database   │  │  File System │  │   Storage    │           │
│  │              │  │              │  │   (S3/GCS)   │           │
│  └──────────────┘  └──────────────┘  └──────────────┘           │
└───────────────────────────────────────────────────────────────────┘
```

### Detailed Pipeline Flow

#### Training Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    TRAINING PIPELINE                             │
└─────────────────────────────────────────────────────────────────┘

Step 1: DATA INGESTION
┌────────────────────────────────────────┐
│ Input Sources:                         │
│  • MongoDB Collections                 │
│  • CSV Files                           │
│  • JSON Files                          │
│  • APIs                                │
├────────────────────────────────────────┤
│ Operations:                            │
│  • Load raw data                       │
│  • Train/Test split (80/20)           │
│  • Save split data                     │
└────────────────┬───────────────────────┘
                 │
                 ▼
Step 2: DATA VALIDATION
┌────────────────────────────────────────┐
│ Validation Checks:                     │
│  • Schema validation                   │
│  • Data type verification              │
│  • Range checking                      │
│  • Missing value analysis              │
│  • Duplicate detection                 │
├────────────────────────────────────────┤
│ Output:                                │
│  • Validation report                   │
│  • Clean data                          │
│  • Drift report                        │
└────────────────┬───────────────────────┘
                 │
                 ▼
Step 3: DATA TRANSFORMATION
┌────────────────────────────────────────┐
│ Preprocessing:                         │
│  • Missing value imputation            │
│  • Categorical encoding                │
│  • Feature scaling                     │
│  • Outlier handling                    │
├────────────────────────────────────────┤
│ Feature Engineering:                   │
│  • New feature creation                │
│  • Feature selection                   │
│  • Dimensionality reduction            │
├────────────────────────────────────────┤
│ Class Balancing:                       │
│  • SMOTE oversampling                  │
│  • Undersampling (if needed)           │
├────────────────────────────────────────┤
│ Output:                                │
│  • Transformed data                    │
│  • Preprocessor object                 │
└────────────────┬───────────────────────┘
                 │
                 ▼
Step 4: MODEL TRAINING
┌────────────────────────────────────────┐
│ Algorithms:                            │
│  • Random Forest                       │
│  • XGBoost                             │
│  • SVM                                 │
│  • Logistic Regression                 │
│  • Decision Tree                       │
├────────────────────────────────────────┤
│ Training Process:                      │
│  • K-Fold Cross-validation (k=5)       │
│  • Hyperparameter tuning (Grid/Random) │
│  • Model selection (best CV score)     │
├────────────────────────────────────────┤
│ Output:                                │
│  • Trained models                      │
│  • Best model selection                │
└────────────────┬───────────────────────┘
                 │
                 ▼
Step 5: MODEL EVALUATION
┌────────────────────────────────────────┐
│ Metrics:                               │
│  • Accuracy                            │
│  • Precision, Recall, F1-Score         │
│  • ROC-AUC                             │
│  • Confusion Matrix                    │
├────────────────────────────────────────┤
│ Visualizations:                        │
│  • ROC Curves                          │
│  • Precision-Recall Curves             │
│  • Feature Importance                  │
├────────────────────────────────────────┤
│ Output:                                │
│  • Evaluation report                   │
│  • Performance metrics                 │
└────────────────┬───────────────────────┘
                 │
                 ▼
Step 6: MODEL PERSISTENCE
┌────────────────────────────────────────┐
│ Save Artifacts:                        │
│  • Model file (model.pkl)              │
│  • Preprocessor (preprocessor.pkl)     │
│  • Model metadata (JSON)               │
│  • Training logs                       │
├────────────────────────────────────────┤
│ Optional Cloud Upload:                 │
│  • AWS S3                              │
│  • Google Cloud Storage                │
│  • Azure Blob Storage                  │
└────────────────────────────────────────┘
```

#### Prediction Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                   PREDICTION PIPELINE                            │
└─────────────────────────────────────────────────────────────────┘

Step 1: DATA LOADING
┌────────────────────────────────────────┐
│ Input:                                 │
│  • CSV file upload                     │
│  • JSON API request                    │
│  • Real-time stream                    │
├────────────────────────────────────────┤
│ Operations:                            │
│  • Load input data                     │
│  • Initial validation                  │
└────────────────┬───────────────────────┘
                 │
                 ▼
Step 2: PREPROCESSING
┌────────────────────────────────────────┐
│ Load Artifacts:                        │
│  • Preprocessor object                 │
│  • Feature list                        │
├────────────────────────────────────────┤
│ Transform Data:                        │
│  • Apply same transformations          │
│  • Ensure feature consistency          │
└────────────────┬───────────────────────┘
                 │
                 ▼
Step 3: PREDICTION
┌────────────────────────────────────────┐
│ Load Model:                            │
│  • Trained model file                  │
├────────────────────────────────────────┤
│ Generate Predictions:                  │
│  • Class predictions                   │
│  • Probability scores                  │
│  • Confidence levels                   │
└────────────────┬───────────────────────┘
                 │
                 ▼
Step 4: OUTPUT GENERATION
┌────────────────────────────────────────┐
│ Format Results:                        │
│  • Add predictions to data             │
│  • Include confidence scores           │
│  • Generate labels                     │
├────────────────────────────────────────┤
│ Save Output:                           │
│  • CSV file with timestamp             │
│  • JSON response (API)                 │
│  • Database storage (optional)         │
└────────────────────────────────────────┘
```

---

## 📁 Project Structure

### Complete Directory Tree

```
NetworkSecurity2/
│
├── 📂 network_security/                    # Main Python package
│   │
│   ├── 📂 components/                      # Pipeline components
│   │   ├── __init__.py
│   │   ├── data_ingestion.py              # Data loading & splitting
│   │   ├── data_validation.py             # Schema validation
│   │   ├── data_transformation.py         # Preprocessing & feature engineering
│   │   ├── model_trainer.py               # Model training logic
│   │   ├── model_evaluation.py            # Model evaluation metrics
│   │   └── model_pusher.py                # Model deployment
│   │
│   ├── 📂 pipeline/                        # Pipeline orchestration
│   │   ├── __init__.py
│   │   ├── training_pipeline.py           # End-to-end training workflow
│   │   └── prediction_pipeline.py         # Prediction workflow
│   │
│   ├── 📂 entity/                          # Data structures & configs
│   │   ├── __init__.py
│   │   ├── config_entity.py               # Configuration dataclasses
│   │   └── artifact_entity.py             # Artifact dataclasses
│   │
│   ├── 📂 utils/                           # Utility functions
│   │   ├── __init__.py
│   │   ├── main_utils.py                  # General utilities
│   │   ├── ml_utils.py                    # ML helper functions
│   │   └── database_utils.py              # MongoDB operations
│   │
│   ├── 📂 constants/                       # Project constants
│   │   ├── __init__.py
│   │   └── training_pipeline.py           # Pipeline constants
│   │
│   ├── 📂 exception/                       # Custom exceptions
│   │   ├── __init__.py
│   │   └── exception.py                   # Exception handlers
│   │
│   ├── 📂 logger/                          # Logging configuration
│   │   ├── __init__.py
│   │   └── logger.py                      # Logger setup
│   │
│   ├── 📂 cloud_storage/                   # Cloud storage (optional)
│   │   ├── __init__.py
│   │   ├── s3_operations.py               # AWS S3 operations
│   │   └── gcs_operations.py              # Google Cloud Storage
│   │
│   └── __init__.py
│
├── 📂 config/                              # Configuration files
│   ├── model_config.yaml                  # Model configurations
│   ├── training_config.yaml               # Training parameters
│   └── deployment_config.yaml             # Deployment settings
│
├── 📂 data_schema/                         # Data validation schemas
│   ├── schema.json                        # Main schema definition
│   ├── training_schema.json               # Training data schema
│   └── prediction_schema.json             # Prediction input schema
│
├── 📂 network_data/                        # Raw data storage
│   ├── raw/                               # Unprocessed data
│   ├── processed/                         # Processed data
│   ├── train/                             # Training data
│   ├── test/                              # Test data
│   └── README.md                          # Data documentation
│
├── 📂 valid_data/                          # Validated data
│   ├── train_validated.csv
│   ├── test_validated.csv
│   └── validation_report.json
│
├── 📂 artifacts/                           # Training artifacts
│   ├── <timestamp>/                       # Timestamped runs
│   │   ├── data_ingestion/
│   │   ├── data_validation/
│   │   ├── data_transformation/
│   │   ├── model_trainer/
│   │   └── model_evaluation/
│   └── latest/                            # Latest run artifacts
│
├── 📂 final_model/                         # Production models
│   ├── model.pkl                          # Trained model
│   ├── preprocessor.pkl                   # Preprocessing pipeline
│   ├── model_metadata.json                # Model info
│   ├── feature_names.json                 # Feature list
│   └── model_metrics.json                 # Performance metrics
│
├── 📂 prediction_output/                   # Prediction results
│   ├── predictions_20260106_103000.csv
│   ├── predictions_20260106_120000.csv
│   └── prediction_summary.json
│
├── 📂 logs/                                # Application logs
│   ├── app.log                            # Application logs
│   ├── training.log                       # Training logs
│   ├── prediction.log                     # Prediction logs
│   └── error.log                          # Error logs
│
├── 📂 templates/                           # HTML templates
│   ├── base.html                          # Base template
│   ├── index.html                         # Home page
│   ├── upload.html                        # Upload page
│   ├── predict.html                       # Prediction page
│   ├── results.html                       # Results display
│   └── about.html                         # About page
│
├── 📂 static/                              # Static assets
│   ├── 📂 css/
│   │   ├── style.css
│   │   └── bootstrap.min.css
│   ├── 📂 js/
│   │   ├── main.js
│   │   └── upload.js
│   ├── 📂 images/
│   │   ├── logo.png
│   │   └── background.jpg
│   └── 📂 fonts/
│
├── 📂 notebooks/                           # Jupyter notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_model_evaluation.ipynb
│
├── 📂 tests/                               # Test suite
│   ├── __init__.py
│   ├── 📂 unit/                            # Unit tests
│   │   ├── test_components.py
│   │   ├── test_pipeline.py
│   │   └── test_utils.py
│   ├── 📂 integration/                     # Integration tests
│   │   ├── test_training_pipeline.py
│   │   └── test_prediction_pipeline.py
│   └── 📂 fixtures/                        # Test fixtures
│       ├── sample_data.csv
│       └── mock_model.pkl
│
├── 📂 docs/                                # Documentation
│   ├── API.md                             # API documentation
│   ├── DEPLOYMENT.md                      # Deployment guide
│   ├── CONTRIBUTING.md                    # Contribution guidelines
│   └── CHANGELOG.md                       # Version history
│
├── 📂 scripts/                             # Utility scripts
│   ├── download_data.sh                   # Data download script
│   ├── setup_mongodb.sh                   # MongoDB setup
│   ├── run_training.sh                    # Training script
│   └── deploy.sh                          # Deployment script
│
├── 📂 .github/                             # GitHub specific files
│   ├── workflows/                         # CI/CD workflows
│   │   ├── test.yml
│   │   ├── deploy.yml
│   │   └── docker-publish.yml
│   └── ISSUE_TEMPLATE/
│
├── 📄 app.py                               # Flask application
├── 📄 main.py                              # Training entry point
├── 📄 push_data_etl_py                     # ETL script
├── 📄 fix_ssl.py                           # SSL certificate fix
│
├── 📄 requirements.txt                     # Python dependencies
├── 📄 requirements-dev.txt                 # Development dependencies
├── 📄 requirements-test.txt                # Testing dependencies
├── 📄 setup.py                             # Package setup
├── 📄 setup.cfg                            # Setup configuration
├── 📄 pyproject.toml                       # Project metadata
│
├── 📄 Dockerfile                           # Docker image definition
├── 📄 .dockerignore                        # Docker ignore patterns
├── 📄 docker-compose.yml                   # Multi-container setup
├── 📄 docker-compose.prod.yml              # Production compose
│
├── 📄 .env.example                         # Environment variables template
├── 📄 .gitignore                           # Git ignore patterns
├── 📄 .gitattributes                       # Git attributes
│
├── 📄 README.md                            # This file
├── 📄 LICENSE                              # License information
├── 📄 SECURITY.md                          # Security policy
├── 📄 CODE_OF_CONDUCT.md                   # Code of conduct
│
├── 📄 Makefile                             # Build automation
├── 📄 tox.ini                              # Tox configuration
├── 📄 pytest.ini                           # Pytest configuration
└── 📄 .coveragerc                          # Coverage configuration
```

### Key Files Description

#### Core Application Files

| File | Description |
|------|-------------|
| `app.py` | Flask web application - main entry point for web interface |
| `main.py` | Training pipeline execution script |
| `push_data_etl_py` | ETL pipeline for data extraction and loading |
| `setup.py` | Package installation and distribution setup |
| `requirements.txt` | Production Python dependencies |
| `Dockerfile` | Docker container configuration |
| `.env` | Environment variables (create from .env.example) |

---

## 🚀 Installation

### System Requirements

**Minimum Requirements:**
- CPU: 2 cores
- RAM: 4 GB
- Storage: 10 GB free space
- OS: Windows 10+, macOS 10.14+, Ubuntu 18.04+

**Recommended Requirements:**
- CPU: 4+ cores
- RAM: 8+ GB
- Storage: 20+ GB SSD
- OS: Latest stable version

### Step 1: Prerequisites Installation

#### Install Python 3.8+

**Windows:**
```bash
# Download from python.org and install
# Or use Chocolatey
choco install python --version=3.9.0
```

**macOS:**
```bash
# Using Homebrew
brew install python@3.9
```

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install python3.9 python3.9-venv python3-pip
```

#### Install Git

**Windows:**
```bash
choco install git
```

**macOS:**
```bash
brew install git
```

**Ubuntu/Debian:**
```bash
sudo apt install git
```

#### Install MongoDB

**Windows:**
```bash
# Download MongoDB Community Server from mongodb.com
# Or use Chocolatey
choco install mongodb
```

**macOS:**
```bash
brew tap mongodb/brew
brew install mongodb-community@4.4
brew services start mongodb-community@4.4
```

**Ubuntu/Debian:**
```bash
wget -qO - https://www.mongodb.org/static/pgp/server-4.4.asc | sudo apt-key add -
echo "deb [ arch=amd64,arm64 ] https://repo.mongodb.org/apt/ubuntu focal/mongodb-org/4.4 multiverse" | sudo tee /etc/apt/sources.list.d/mongodb-org-4.4.list
sudo apt update
sudo apt install -y mongodb-org
sudo systemctl start mongod
sudo systemctl enable mongod
```

### Step 2: Clone Repository

```bash
# Clone the repository
git clone https://github.com/MadarwalaHussain/NetworkSecurity2.git

# Navigate to project directory
cd NetworkSecurity2

# Check the repository
ls -la
```

### Step 3: Create Virtual Environment

**Windows:**
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
venv\Scripts\activate

# Verify activation (should show venv path)
where python
```

**macOS/Linux:**
```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Verify activation
which python
```

**Alternative: Using conda**
```bash
# Create conda environment
conda create -n network-security python=3.9

# Activate environment
conda activate network-security
```

### Step 4: Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install required packages
pip install -r requirements.txt

# For development (includes testing tools)
pip install -r requirements-dev.txt

# Verify installation
pip list
```

### Step 5: Install Package in Development Mode

```bash
# Install the network_security package
pip install -e .

# Verify installation
python -c "import network_security; print('Package installed successfully!')"
```

### Step 6: Configure Environment

#### Create .env File

```bash
# Copy example file
cp .env.example .env

# Edit with your values
nano .env  # or use your preferred editor
```

#### Environment Variables Configuration

```env
# ============================================
# DATABASE CONFIGURATION
# ============================================
# MongoDB Connection
MONGO_DB_URL=mongodb://localhost:27017
# For MongoDB Atlas:
# MONGO_DB_URL=mongodb+srv://username:password@cluster.mongodb.net/?retryWrites=true&w=majority

# Database and Collection Names
MONGO_DB_NAME=network_security
MONGO_COLLECTION_NAME=network_data
MONGO_COLLECTION_TRAIN=train_data
MONGO_COLLECTION_TEST=test_data

# ============================================
# AWS CONFIGURATION (Optional)
# ============================================
AWS_ACCESS_KEY_ID=your_access_key_id
AWS_SECRET_ACCESS_KEY=your_secret_access_key
AWS_REGION=us-east-1
S3_BUCKET_NAME=network-security-models

# ============================================
# MODEL CONFIGURATION
# ============================================
MODEL_DIR=final_model
MODEL_FILE_NAME=model.pkl
PREPROCESSOR_FILE_NAME=preprocessor.pkl
MODEL_METADATA_FILE=model_metadata.json

# ============================================
# FLASK CONFIGURATION
# ============================================
FLASK_APP=app.py
FLASK_ENV=development
FLASK_DEBUG=1
SECRET_KEY=your-super-secret-key-change-this-in-production
PORT=5000

# ============================================
# TRAINING CONFIGURATION
# ============================================
TRAINING_PIPELINE_NAME=training
ARTIFACT_DIR=artifacts
TEST_SIZE=0.2
RANDOM_STATE=42
CV_FOLDS=5

# ============================================
# DATA PATHS
# ============================================
RAW_DATA_DIR=network_data/raw
PROCESSED_DATA_DIR=network_data/processed
SCHEMA_FILE_PATH=data_schema/schema.json
PREDICTION_OUTPUT_DIR=prediction_output

# ============================================
# LOGGING CONFIGURATION
# ============================================
LOG_LEVEL=INFO
LOG_DIR=logs
LOG_FILE=app.log
LOG_FORMAT=%(asctime)s - %(name)s - %(levelname)s - %(message)s

# ============================================
# API CONFIGURATION
# ============================================
API_VERSION=v1
API_RATE_LIMIT=100
API_RATE_LIMIT_PERIOD=3600

# ============================================
# SECURITY
# ============================================
ALLOWED_ORIGINS=http://localhost:3000,http://localhost:5000
MAX_FILE_SIZE=10485760  # 10MB in bytes
ALLOWED_EXTENSIONS=csv,json

# ============================================
# PERFORMANCE
# ============================================
BATCH_SIZE=1000
MAX_WORKERS=4
```

### Step 7: Setup MongoDB

#### Initialize Database

```bash
# Connect to MongoDB
mongo

# Create database and collection
use network_security
db.createCollection("network_data")
db.createCollection("train_data")
db.createCollection("test_data")
db.createCollection("predictions")

# Create indexes for performance
db.network_data.createIndex({"timestamp": 1})
db.predictions.createIndex({"created_at": -1})

# Exit mongo shell
exit
```

#### Load Sample Data (Optional)

```bash
# If you have sample data
python push_data_etl_py --source network_data/sample_data.csv --database network_security --collection network_data
```

### Step 8: Verify Installation

```bash
# Test MongoDB connection
python -c "from pymongo import MongoClient; client = MongoClient('mongodb://localhost:27017'); print('MongoDB connected:', client.server_info()['version'])"

# Test package import
python -c "from network_security.pipeline.training_pipeline import TrainingPipeline; print('Package import successful!')"

# Check Flask
python -c "from flask import Flask; print('Flask version:', Flask.__version__)"

# Check scikit-learn
python -c "import sklearn; print('scikit-learn version:', sklearn.__version__)"
```

### Step 9: Create Required Directories

```bash
# Create all necessary directories
mkdir -p logs artifacts prediction_output final_model network_data/raw network_data/processed valid_data

# On Windows (PowerShell)
# New-Item -ItemType Directory -Force -Path logs, artifacts, prediction_output, final_model, network_data/raw, network_data/processed, valid_data
```

### Step 10: Run Initial Test

```bash
# Test the Flask application
python app.py

# You should see output like:
# * Running on http://127.0.0.1:5000
# * Debug mode: on

# Open browser and navigate to http://localhost:5000
```

### Troubleshooting Installation

#### Common Issues

**1. SSL Certificate Error**
```bash
# Run the fix_ssl script
python fix_ssl.py

# Or manually
pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org -r requirements.txt
```

**2. MongoDB Connection Failed**
```bash
# Check MongoDB status
# Linux/macOS:
sudo systemctl status mongod

# Windows: Check Services

# Restart MongoDB
# Linux/macOS:
sudo systemctl restart mongod

# Check connection
mongo --eval "db.adminCommand('ping')"
```

**3. Package Import Errors**
```bash
# Reinstall package
pip uninstall network-security
pip install -e .

# Clear Python cache
find . -type d -name "__pycache__" -exec rm -r {} +
find . -type f -name "*.pyc" -delete
```

**4. Permission Denied Errors**
```bash
# Linux/macOS: Fix permissions
chmod +x scripts/*.sh
sudo chown -R $USER:$USER .

# Windows: Run as Administrator
```

---

## ⚙️ Configuration

### Model Configuration

Edit `config/model_config.yaml`:

```yaml
# Model Training Configuration
model_training:
  algorithms:
    - name: RandomForest
      enabled: true
      params:
        n_estimators: 100
        max_depth: 20
        min_samples_split: 5
        min_samples_leaf: 2
        random_state: 42
        n_jobs: -1
    
    - name: XGBoost
      enabled: true
      params:
        n_estimators: 100
        max_depth: 6
        learning_rate: 0.1
        subsample: 0.8
        colsample_bytree: 0.8
        random_state: 42
    
    - name: LogisticRegression
      enabled: false
      params:
        max_iter: 1000
        random_state: 42
  
  cross_validation:
    enabled: true
    cv_folds: 5
    stratified: true
  
  hyperparameter_tuning:
    enabled: true
    method: GridSearchCV  # or RandomizedSearchCV
    n_iter: 10  # for RandomizedSearchCV
    scoring: f1_weighted

# Feature Engineering
feature_engineering:
  polynomial_features:
    enabled: false
    degree: 2
  
  feature_selection:
    enabled: true
    method: SelectKBest  # or RFE, SelectFromModel
    k: 20
  
  dimensionality_reduction:
    enabled: false
    method: PCA
    n_components: 0.95

# Preprocessing
preprocessing:
  missing_values:
    strategy: mean  # mean, median, mode, constant
    fill_value: 0  # for constant strategy
  
  scaling:
    method: StandardScaler  # StandardScaler, MinMaxScaler, RobustScaler
  
  encoding:
    categorical:
      method: OneHotEncoder  # OneHotEncoder, LabelEncoder, TargetEncoder
      handle_unknown: ignore
  
  imbalance_handling:
    enabled: true
    method: SMOTE
    sampling_strategy: auto
    k_neighbors: 5
```

### Training Configuration

Edit `config/training_config.yaml`:

```yaml
# Training Pipeline Configuration
training_pipeline:
  name: network_security_training
  version: "1.0.0"
  description: "Network security threat detection model"
  
  data:
    source: mongodb  # mongodb, csv, json
    train_test_split:
      test_size: 0.2
      random_state: 42
      stratify: true
    
    validation_split:
      enabled: true
      val_size: 0.1
  
  artifacts:
    base_dir: artifacts
    keep_n_versions: 5
    compress: true
  
  model:
    save_dir: final_model
    save_format: pickle  # pickle, joblib
    include_metadata: true
    include_feature_names: true
  
  evaluation:
    metrics:
      - accuracy
      - precision
      - recall
      - f1_score
      - roc_auc
    
    threshold: 0.5
    generate_reports: true
    save_predictions: true
  
  logging:
    level: INFO
    save_to_file: true
    log_file: logs/training.log
```

### Deployment Configuration

Edit `config/deployment_config.yaml`:

```yaml
# Deployment Configuration
deployment:
  environment: production  # development, staging, production
  
  server:
    host: 0.0.0.0
    port: 5000
    workers: 4
    threads: 2
    timeout: 120
    keepalive: 5
  
  cors:
    enabled: true
    origins:
      - http://localhost:3000
      - https://yourdomain.com
    methods:
      - GET
      - POST
    allow_credentials: true
  
  security:
    rate_limiting:
      enabled: true
      requests_per_hour: 1000
    
    authentication:
      enabled: false
      method: jwt  # jwt, api_key, oauth
    
    file_upload:
      max_size_mb: 10
      allowed_extensions:
        - csv
        - json
  
  monitoring:
    enabled: true
    log_requests: true
    track_predictions: true
    alert_on_errors: true
  
  cloud:
    provider: aws  # aws, gcp, azure
    region: us-east-1
    
    s3:
      bucket: network-security-models
      model_prefix: models/
      data_prefix: data/
```

---

## 🎬 Quick Start

### 1. Complete Setup (First Time)

```bash
# Clone and setup
git clone https://github.com/MadarwalaHussain/NetworkSecurity2.git
cd NetworkSecurity2
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install -e .

# Configure environment
cp .env.example .env
# Edit .env with your settings

# Create directories
mkdir -p logs artifacts prediction_output final_model
```

### 2. Prepare Data

```bash
# If you have data in CSV format
# Place it in network_data/raw/

# Load to MongoDB (optional)
python push_data_etl_py --source network_data/raw/your_data.csv
```

### 3. Train Model

```bash
# Run complete training pipeline
python main.py

# This will:
# 1. Load data from MongoDB or files
# 2. Validate data
# 3. Transform and preprocess
# 4. Train multiple models
# 5. Evaluate and select best model
# 6. Save model to final_model/
```

### 4. Start Application

```bash
# Development mode
python app.py

# Production mode
gunicorn --bind 0.0.0.0:8080 --workers 4 app:app
```

### 5. Make Predictions

**Via Web Interface:**
1. Open `http://localhost:5000`
2. Upload CSV file
3. View results

**Via API:**
```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"protocol_type": "tcp", "service": "http", "flag": "SF", "src_bytes": 181, "dst_bytes": 5450, "count": 8}'
```

**Via Python:**
```python
from network_security.pipeline.prediction_pipeline import PredictionPipeline
import pandas as pd

predictor = PredictionPipeline()
data = pd.read_csv('test_data.csv')
predictions = predictor.predict(data)
```

---

## 📖 Detailed Usage

### Training the Model

#### Using Command Line

```bash
# Basic training
python main.py

# With specific configuration
python main.py --config config/training_config.yaml

# With data source
python main.py --data-source mongodb --database network_security

# With custom artifact directory
python main.py --artifact-dir custom_artifacts
```

#### Using Python API

```python
from network_security.pipeline.training_pipeline import TrainingPipeline
from network_security.logger import logging

# Initialize logging
logging.info("Starting training pipeline")

# Create pipeline instance
pipeline = TrainingPipeline()

# Run complete pipeline
try:
    pipeline.run_pipeline()
    logging.info("Training completed successfully")
except Exception as e:
    logging.error(f"Training failed: {str(e)}")
    raise
```

#### Custom Training Script

```python
from network_security.components.data_ingestion import DataIngestion
from network_security.components.data_validation import DataValidation
from network_security.components.data_transformation import DataTransformation
from network_security.components.model_trainer import ModelTrainer
from network_security.entity.config_entity import (
    DataIngestionConfig,
    DataValidationConfig,
    DataTransformationConfig,
    ModelTrainerConfig
)

# Step 1: Data Ingestion
data_ingestion_config = DataIngestionConfig()
data_ingestion = DataIngestion(data_ingestion_config)
data_ingestion_artifact = data_ingestion.initiate_data_ingestion()

# Step 2: Data Validation
data_validation_config = DataValidationConfig()
data_validation = DataValidation(
    data_validation_config,
    data_ingestion_artifact
)
data_validation_artifact = data_validation.initiate_data_validation()

# Step 3: Data Transformation
data_transformation_config = DataTransformationConfig()
data_transformation = DataTransformation(
    data_transformation_config,
    data_validation_artifact
)
data_transformation_artifact = data_transformation.initiate_data_transformation()

# Step 4: Model Training
model_trainer_config = ModelTrainerConfig()
model_trainer = ModelTrainer(
    model_trainer_config,
    data_transformation_artifact
)
model_trainer_artifact = model_trainer.initiate_model_trainer()

print(f"Model saved at: {model_trainer_artifact.trained_model_file_path}")
print(f"Model metrics: {model_trainer_artifact.metric_artifact}")
```

###