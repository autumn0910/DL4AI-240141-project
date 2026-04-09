# DL4AI-240141-project
Overview
This project focuses on applying deep learning techniques to time-series data for stock market analysis and prediction. The main objectives include:
- Predicting stock prices for Nasdaq and Vietnam markets
- Identifying buy/sell trading signals
- Constructing an optimized investment portfolio
- (Optional) Deploying models as API / SaaS applications
Objectives
- Learn and apply time-series forecasting techniques
- Design and evaluate deep learning models
- Analyze financial data for decision-making
- Build an end-to-end pipeline from data → model → evaluation → (deployment)
Project Structure
.
├── data/                  # Raw and processed datasets
├── notebooks/             # Exploratory analysis & experiments
├── src/
│   ├── data/              # Data loading & preprocessing
│   ├── features/          # Feature engineering
│   ├── models/            # Model architectures (LSTM, etc.)
│   ├── training/          # Training pipeline
│   ├── evaluation/        # Metrics & validation
│   └── utils/             # Helper functions
│
├── configs/               # Experiment configurations
├── results/               # Saved outputs (metrics, plots)
├── report/                # Final report
├── requirements.txt
└── README.md
Methodology
- Data Handling
- Multi-source datasets (Nasdaq, Vietnam market)
- Chronological splitting:
  + Train → Validation → Test
- Rolling / expanding window validation
Feature Engineering
- Price-based features: Open, High, Low, Close, Volume
- Technical indicators (optional):
- Moving averages (SMA, EMA)
- Momentum indicators (RSI, MACD)
