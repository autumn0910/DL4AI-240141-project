# test_api.py
import requests
import json
import pandas as pd

# Đổi path đúng về máy của bạn
df = pd.read_csv(r'C:\Users\LENOVO\OneDrive - Fulbright University Vietnam\Desktop\Major Minor\CS313 - Deep Learning\final\task5\AAPL.csv')
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
df = df[df['Date'] >= '2018-01-01'].tail(60)
data = df[['Open','High','Low','Close','Volume']].values.tolist()

# Health check
res = requests.get('http://localhost:8000/health')
print("Health:", res.json())

# Predict
res = requests.post('http://localhost:8000/predict', json={'data': data})
print("Predict:", json.dumps(res.json(), indent=2))