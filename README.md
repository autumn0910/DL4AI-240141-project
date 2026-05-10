# DL4AI-240141-project
**CS313 Deep Learning for Artificial Intelligence — Spring 2026**

Time-series forecasting and trading analysis for stock markets using deep learning (LSTM), applied to both Nasdaq (AAPL) and Vietnam (FPT, HPG, VNM, ...) stock data.

---

## Project Structure

```
DL4AI-240141-project/
├── notebook/
│   └── Final_project_DL4AI.ipynb   # Main notebook: Tasks 1–4
├── task5/
│   ├── main.py                     # Task 5.1: FastAPI REST API
│   ├── streamlit_app.py            # Task 5.2: Streamlit web UI
│   ├── test_api.py                 # API test script
│   └── requirements.txt            # Dependencies
├── data/
│   └── README.md                   # Data sources and description
└── README.md
```

---

## Tasks Overview

| Task | Description | Key Results |
|------|-------------|-------------|
| 1.1 | Nasdaq multi-feature prediction (AAPL) | MAPE = 3.09% |
| 1.2 | N-th day forecast (K=1,3,7) | MAPE = 3.06–4.50% |
| 1.3 | K consecutive days forecast | MAPE = 3.41–4.24% |
| 2.1 | Vietnam multi-feature prediction (FPT) | MAPE = 2.24% |
| 2.2 | Vietnam N-th day forecast | MAPE = 2.07–3.57% |
| 2.3 | Vietnam K consecutive days forecast | MAPE = 3.0–4.5% |
| 3.1 | Buy signal identification (FPT) | F1 = 0.294 |
| 3.2 | Sell signal identification (FPT) | F1 = 0.297 |
| 4.1 | Profitable stock selection (8 VN stocks) | VIC +18.4% projected |
| 4.2 | Risk management & scoring | VNM safest (0.197) |
| 4.3 | Portfolio composition | Aggressive / Conservative |
| 5.1 | Model deployment as REST API | FastAPI on port 8000 |
| 5.2 | Model as SaaS | Streamlit on port 8501 |
| 5.3 | AI engineering workflow | Airflow + dbt + PostgreSQL |

---

## Setup & Running Instructions

### Requirements
- Python 3.10+
- Google Colab (for notebook) or local Python environment

### 1. Clone the repository
```bash
git clone https://github.com/autumn0910/DL4AI-240141-project.git
cd DL4AI-240141-project
```

### 2. Install dependencies
```bash
pip install -r task5/requirements.txt
```

### 3. Run the notebook (Tasks 1–4)
Open `notebook/Final_project_DL4AI.ipynb` in Google Colab or Jupyter.

Upload the required data files (see `data/README.md`) to `/content/` before running.

Run all cells sequentially from top to bottom.

### 4. Run the API (Task 5.1)
```bash
cd task5
uvicorn main:app --reload --port 8000
```

API will be available at `http://localhost:8000`

Endpoints:
- `GET /health` — check API status
- `POST /predict` — predict next-day stock price

Example request:
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"data": [[142.7, 143.0, 141.07, 142.32, 21904917], ...]}'
```

### 5. Run the Streamlit UI (Task 5.2)
Make sure the FastAPI server is running first, then:
```bash
cd task5
streamlit run streamlit_app.py
```

Open `http://localhost:8501` in your browser.

Upload AAPL.csv (2018 onwards) and click **Predict Next Day**.

---

## Model Architecture

All prediction models use a 2-layer LSTM:
- Input: 5 features (Open, High, Low, Close, Volume)
- Hidden size: 128
- Dropout: 0.2
- Optimizer: Adam (lr=0.0005, weight_decay=1e-4)
- Loss: MSELoss (regression) / BCEWithLogitsLoss (classification)
- Early stopping: patience=15

Trading signal models (Task 3) use LSTM feature extraction + GradientBoosting classifier.

---

## Data

See `data/README.md` for data sources and download instructions.

- **Nasdaq**: AAPL historical data from Yahoo Finance (2018–2022)
- **Vietnam**: FPT, HPG, MSN, VNM, MWG, TCB, VHM, VIC from VNINDEX (2018–2023)

---

## Key Design Decisions

- **Chronological split**: 70/15/15 train/val/test — no shuffling
- **Time-series cross-validation**: 5-fold TimeSeriesSplit on train+val only
- **Global scaler**: Used due to extreme price appreciation over long periods
- **Company filtering**: Stocks with at least 120 historical data points
- **Trading signals**: Defined as >1% price change within 3-day horizon

