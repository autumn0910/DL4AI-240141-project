# streamlit_app.py — Stock Price Prediction Web UI
import streamlit as st
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

API_URL = "http://localhost:8000"

st.markdown("""
<style>
    .stApp { background-color: #FFF8F5; }
    [data-testid="stSidebar"] { background-color: #FEE3CA; }
    .stButton button {
        background-color: #F06E95;
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 500;
    }
    .stButton button:hover { background-color: #AB82A4; }
    h1, h2, h3 { color: #3d3a5c; }
    [data-testid="stDataFrame"] { border: 1px solid #9295C0; border-radius: 8px; }
    [data-testid="stAlert"] { border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

st.title("Stock Price Prediction")
st.caption("LSTM-based prediction model — CS313 Deep Learning for AI")

st.sidebar.header("Input Method")
input_method = st.sidebar.radio("Choose input method:", ["Upload CSV", "Manual Input"])

def call_api(data_60days):
    try:
        res = requests.post(f"{API_URL}/predict",
                            json={"data": data_60days}, timeout=10)
        return res.json()
    except Exception as e:
        return {"error": str(e)}

def show_metrics(pred, last_close, change):
    col_a, col_b, col_c = st.columns(3)
    color = '#6999A1' if change >= 0 else '#F06E95'
    arrow = '▲' if change >= 0 else '▼'
    card = lambda label, val: f"""
    <div style='background:white;border:1px solid #FC9390;border-radius:10px;
    padding:16px;text-align:center;margin:4px'>
    <p style='color:#9295C0;margin:0 0 6px 0;font-size:13px'>{label}</p>
    <p style='color:#3d3a5c;margin:0;font-size:26px;font-weight:500'>{val}</p>
    </div>"""
    with col_a:
        st.markdown(card("Predicted price", f"${pred:,.2f}"), unsafe_allow_html=True)
    with col_b:
        st.markdown(card("Last close", f"${last_close:,.2f}"), unsafe_allow_html=True)
    with col_c:
        st.markdown(f"""
        <div style='background:white;border:1px solid #FC9390;border-radius:10px;
        padding:16px;text-align:center;margin:4px'>
        <p style='color:#9295C0;margin:0 0 6px 0;font-size:13px'>Change</p>
        <p style='color:{color};margin:0;font-size:26px;font-weight:500'>{arrow} {change:+.2f}%</p>
        </div>""", unsafe_allow_html=True)

def plot_prediction(close_vals, pred):
    close_vals = list(close_vals)
    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor('#FFF8F5')
    ax.set_facecolor('#FFF8F5')
    ax.plot(range(len(close_vals)), close_vals,
            color='#6999A1', linewidth=2, label='Last 60 days')
    ax.plot(len(close_vals), pred, 'o',
            color='#F06E95', markersize=12, label=f'Predicted: ${pred:,.2f}')
    ax.plot([len(close_vals)-1, len(close_vals)], [close_vals[-1], pred],
            '--', color='#AB82A4', linewidth=1.5, alpha=0.8)
    ax.set_title('Last 60 Days + Next Day Prediction', color='#3d3a5c', fontsize=13)
    ax.set_ylabel('Price (USD)', color='#3d3a5c')
    ax.tick_params(colors='#9295C0')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#FC9390')
    ax.spines['bottom'].set_color('#FC9390')
    ax.legend(framealpha=0.8)
    return fig

# ── Upload CSV ────────────────────────────────
if input_method == "Upload CSV":
    st.subheader("Upload stock CSV file")
    uploaded = st.file_uploader(
        "Upload CSV (must have: Open, High, Low, Close, Volume)", type=["csv"])

    if uploaded:
        df = pd.read_csv(uploaded)
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
            df = df[df['Date'] >= '2018-01-01'].reset_index(drop=True)
        elif 'TradingDate' in df.columns:
            df['TradingDate'] = pd.to_datetime(df['TradingDate'])
            df = df[df['TradingDate'] >= '2018-01-01'].reset_index(drop=True)

        st.write(f"Loaded {len(df)} rows")
        st.dataframe(df.tail(10), use_container_width=True)

        features = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing  = [f for f in features if f not in df.columns]

        if missing:
            st.error(f"Missing columns: {missing}")
        elif len(df) < 60:
            st.error(f"Need at least 60 rows, got {len(df)}")
        else:
            last_60    = df[features].tail(60).values.tolist()
            last_close = float(df['Close'].iloc[-1])
            close_vals = df['Close'].tail(60).values

            st.markdown(f"""
            <div style='background:white;border:1px solid #FC9390;border-radius:10px;
            padding:16px;display:inline-block;margin-bottom:12px'>
            <p style='color:#9295C0;margin:0;font-size:13px'>Last close price</p>
            <p style='color:#3d3a5c;margin:0;font-size:28px;font-weight:500'>${last_close:,.2f}</p>
            </div>""", unsafe_allow_html=True)

            if st.button("Predict Next Day", type="primary"):
                with st.spinner("Predicting..."):
                    result = call_api(last_60)
                if "error" in result:
                    st.error(f"API Error: {result['error']}")
                else:
                    pred   = result['predicted_price']
                    change = (pred - last_close) / last_close * 100
                    st.success("Prediction complete!")
                    show_metrics(pred, last_close, change)
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.pyplot(plot_prediction(close_vals, pred))

# ── Manual Input ──────────────────────────────
else:
    st.subheader("Manual Input")
    st.caption("Enter today's OHLCV values.")

    col1, col2, col3 = st.columns(3)
    with col1:
        open_ = st.number_input("Open",   value=150.0, step=0.1)
        high  = st.number_input("High",   value=155.0, step=0.1)
    with col2:
        low   = st.number_input("Low",    value=148.0, step=0.1)
        close = st.number_input("Close",  value=152.0, step=0.1)
    with col3:
        volume = st.number_input("Volume", value=80000000.0, step=1000000.0)

    try:
        df_base = pd.read_csv('AAPL.csv')
        df_base['Date'] = pd.to_datetime(df_base['Date'], dayfirst=True)
        df_base = df_base[df_base['Date'] >= '2018-01-01']
        features = ['Open', 'High', 'Low', 'Close', 'Volume']
        last_59  = df_base[features].tail(59).values.tolist()
        close_59 = df_base['Close'].tail(59).values.tolist()
        data_60  = last_59 + [[open_, high, low, close, volume]]

        if st.button("Predict", type="primary"):
            with st.spinner("Predicting..."):
                result = call_api(data_60)
            if "error" in result:
                st.error(f"API Error: {result['error']}")
            else:
                pred   = result['predicted_price']
                change = (pred - close) / close * 100
                st.success("Prediction complete!")
                show_metrics(pred, close, change)
                st.markdown("<br>", unsafe_allow_html=True)
                # close_59 + input close = 60 values
                st.pyplot(plot_prediction(close_59 + [close], pred))

    except FileNotFoundError:
        st.warning("AAPL.csv not found in task5 folder. Please use Upload CSV method.")

# ── Footer ────────────────────────────────────
st.divider()
st.caption("Model: 2-layer LSTM | Hidden: 128 | Window: 60 days | Features: Open, High, Low, Close, Volume")

try:
    res = requests.get(f"{API_URL}/health", timeout=2)
    st.success(f"API status: {res.json()['status']}")
except:
    st.error("API offline — start FastAPI on port 8000")