import numpy as np
import pickle
import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

# Model definition 
class LSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=5,
                            hidden_size=128,
                            num_layers=2,
                            batch_first=True,
                            dropout=0.2)
        self.fc = nn.Linear(128, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

# Load model & scaler 
model = LSTMModel()
model.load_state_dict(torch.load('lstm_model.pth', map_location='cpu'))
model.eval()

from sklearn.preprocessing import MinMaxScaler

with open('scaler_params.pkl', 'rb') as f:
    params = pickle.load(f)

scaler = MinMaxScaler()
scaler.scale_        = np.array(params['scale_'])
scaler.min_          = np.array(params['min_'])
scaler.data_min_     = np.array(params['data_min_'])
scaler.data_max_     = np.array(params['data_max_'])
scaler.data_range_   = np.array(params['data_range_'])
scaler.feature_range = params['feature_range']
scaler.n_features_in_ = params['n_features_in_']

FEATURES   = ['Open', 'High', 'Low', 'Close', 'Volume']
TARGET_IDX = FEATURES.index('Close')
WINDOW     = 60

# FastAPI app 
app = FastAPI(
    title="Stock Price Prediction API",
    description="LSTM-based stock price prediction for AAPL",
    version="1.0.0"
)

# Request/Response schema 
class PredictRequest(BaseModel):
    # 60 days x 5 features: [[Open, High, Low, Close, Volume], ...]
    data: List[List[float]]

class PredictResponse(BaseModel):
    predicted_price: float
    currency: str
    model: str

# Endpoints 
@app.get("/")
def root():
    return {"message": "Stock Prediction API is running!",
            "endpoints": ["/predict", "/health"]}

@app.get("/health")
def health():
    return {"status": "healthy", "model": "LSTM", "window": WINDOW}

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    data = req.data

    # Validate input
    if len(data) != WINDOW:
        raise HTTPException(
            status_code=400,
            detail=f"Expected {WINDOW} timesteps, got {len(data)}"
        )
    if any(len(row) != len(FEATURES) for row in data):
        raise HTTPException(
            status_code=400,
            detail=f"Each timestep must have {len(FEATURES)} features: {FEATURES}"
        )

    # Scale input
    arr = np.array(data, dtype=np.float64)      # (60, 5)
    arr_sc = scaler.transform(arr)               # (60, 5)

    # Predict
    X = torch.tensor(arr_sc[np.newaxis], dtype=torch.float32)  # (1, 60, 5)
    with torch.no_grad():
        pred_sc = model(X).item()

    # Inverse transform
    dummy = np.zeros((1, len(FEATURES)))
    dummy[0, TARGET_IDX] = pred_sc
    pred_price = scaler.inverse_transform(dummy)[0, TARGET_IDX]

    return PredictResponse(
        predicted_price=round(float(pred_price), 2),
        currency="USD",
        model="LSTM-2layer-128hidden"
    )