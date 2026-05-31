# Credit Risk Model API

## Overview

This project develops a machine learning-based Credit Risk Prediction API that evaluates customer risk profiles based on transaction behavior and financial activity. The model predicts the probability of a customer being high risk and classifies the customer as either "High Risk" or "Low Risk".

The project includes:

- Data preprocessing and feature engineering
- Machine learning model training
- FastAPI deployment
- Docker containerization
- Automated testing and CI/CD using GitHub Actions

---

## Project Structure

credit-risk-model/
│
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── data/
├── notebooks/
├── reports/
│
├── src/
│   ├── api/
│   │   ├── main.py
│   │   └── pydantic_models.py
│   │
│   ├── data_preprocessing.py
│   └── train.py
│
├── tests/
│   └── test_data_processing.py
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── mlflow.db
└── README.md
---

## Features

- Transaction-based risk assessment
- Risk probability prediction
- Risk classification
- REST API interface
- Docker deployment
- GitHub Actions CI pipeline

---

## Technologies Used

- Python 3.10
- FastAPI
- Scikit-Learn
- Pandas
- NumPy
- Joblib
- Docker
- MLflow
- GitHub Actions
- Pytest

---

## Installation

### Clone Repository

git clone https://github.com/redu2127/credit-risk-model.git
cd credit-risk-model
### Create Virtual Environment

python -m venv venv
### Activate Environment

Windows:

venv\Scripts\activate
Linux/Mac:

source venv/bin/activate
### Install Dependencies

pip install -r requirements.txt
---

## Running the API

Start FastAPI:

uvicorn src.api.main:app --reload
API documentation:

http://localhost:8000/docs
---

## Docker Deployment

### Build Docker Image

docker build -t credit-risk-api .
### Run Docker Container

docker run -p 8000:8000 credit-risk-api
### Access API

http://localhost:8000/docs
---

## Example Request

POST

http://localhost:8000/predict
Request Body:

{
  "total_transaction_amount": 10,
  "average_transaction_amount": 1,
  "transaction_count": 1,
  "std_transaction_amount": 0,
  "total_value": 10,
  "average_value": 1,
  "average_transaction_hour": 12,
  "average_transaction_day": 15,
  "average_transaction_month": 6,
  "fraud_count": 0,
  "fraud_rate": 0,
  "Recency": 100,
  "Frequency": 1,
  "Monetary": 10
}
Example Response:

{
  "risk_probability": 0.84,
  "risk_class": "High Risk"
}
---

## Model Output

| Output | Description |
|----------|-------------|
| risk_probability | Probability that a customer belongs to the high-risk class |
| risk_class | Final classification (High Risk or Low Risk) |

---

## Testing

Run tests using:

pytest tests
---

## CI/CD

GitHub Actions automatically performs:

- Code linting using Flake8
- Automated unit testing using Pytest
- Continuous Integration validation on every push

---

## Results

The deployed API successfully:

- Accepts customer transaction features
- Performs real-time risk prediction
- Returns risk probability and classification
- Supports containerized deployment using Docker

