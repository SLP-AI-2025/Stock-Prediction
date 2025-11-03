import streamlit as st
import pandas as pd
import json
import matplotlib.pyplot as plt

# Title
st.title("Stock Predictions Dashboard")

# Load predictions
with open("predictions.json", "r") as predictions:
    data = json.load(predictions)

# Convert to DataFrame
df = pd.DataFrame(data)

# Convert 'Date' to datetime
df['Date'] = pd.to_datetime(df['Date'])

# Group by ticker
tickers = ['AAPL', 'MSFT', 'SPY']

for ticker in tickers:
    st.subheader(f"{ticker}")
    df_ticker = df[df['ticker'] == ticker].sort_values('Date')
    st.line_chart(df_ticker.set_index('Date')['close'])

    st.write("Latest Predictions:")
    st.dataframe(df_ticker[['Date', 'close', 'proba_up', 'signal']].tail(10))
