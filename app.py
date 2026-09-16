from typing import List
import os
import joblib
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

MODEL_PATH = os.path.join('model', 'churn_pipeline.pkl')
app = FastAPI(title='Telco Customer Churn Prediction API', version='1.0.0')

try:
    model_pipeline = joblib.load(MODEL_PATH)
except Exception as exc:
    model_pipeline = None
    MODEL_LOAD_ERROR = str(exc)
else:
    MODEL_LOAD_ERROR = None

class CustomerData(BaseModel):
    model_config = ConfigDict(extra='forbid')
    gender: str
    SeniorCitizen: int = Field(ge=0, le=1)
    Partner: str
    Dependents: str
    tenure: int = Field(ge=0)
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: float = Field(ge=0)

@app.get('/')
def root():
    return {
        'message': 'Telco Customer Churn Prediction API',
        'docs': '/docs',
        'health': '/health',
        'prediction_endpoint': '/predict',
    }

@app.get('/health')
def health():
    return {'status': 'healthy' if model_pipeline is not None else 'unhealthy', 'model_loaded': model_pipeline is not None}

def predict_rows(rows):
    if model_pipeline is None:
        raise RuntimeError(f'Model could not be loaded: {MODEL_LOAD_ERROR}')
    df = pd.DataFrame(rows)
    probs = model_pipeline.predict_proba(df)[:, 1]
    return [
        {'prediction': 'Yes' if float(p) >= 0.5 else 'No', 'churn_probability': round(float(p), 4)}
        for p in probs
    ]

@app.post('/predict')
def predict(customer: CustomerData):
    try:
        return predict_rows([customer.model_dump()])[0]
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))

@app.post('/predict_batch')
def predict_batch(customers: List[CustomerData]):
    if not customers:
        raise HTTPException(status_code=422, detail='customers must contain at least one record')
    try:
        return {'batch_predictions': predict_rows([c.model_dump() for c in customers])}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)
