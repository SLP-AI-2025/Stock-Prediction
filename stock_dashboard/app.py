import streamlit as st
import pandas as pd
import json

# 1️ Page setup
st.title("Stock Predictions Dashboard")

# 2️ Load data
data = pd.read_json("predictions.json")
data["Date"] = pd.to_datetime(data["Date"])

# 3️ User selection
tickers = sorted(data["ticker"].unique())
choice = st.selectbox("Choose a stock:", tickers)

# 4️ Filter the data
df = data[data["ticker"] == choice]

# 5️ Plot the line chart
st.line_chart(df.set_index("Date")["close"])
