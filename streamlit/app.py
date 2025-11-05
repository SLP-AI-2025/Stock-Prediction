import streamlit as st
import pandas as pd

st.title("Stock Prediction")

# sort by date and ticker
data = pd.read_json('predictions.json')
df = pd.DataFrame(data)
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date")
tickers = ['AAPL', 'MSFT', 'SPY']

# display chart
st.line_chart(df, x="Date", y="close", y_label = "Close", color="ticker", height=520)

#display price tables
st.subheader("Price Predictions")
for ticker in tickers:
    st.write(f"Ticker: {ticker}")
    filtered_df = df[df["ticker"] == ticker]
    st.dataframe(filtered_df[["Date", "close", "proba_up", "signal"]])