import json, pathlib
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

DATA_DIR = pathlib.Path("data")

@st.cache_data
def load_preds_metrics():
    preds = pd.read_json(DATA_DIR / "predictions.json", convert_dates=False)
    preds["Date"] = pd.to_datetime(preds["Date"])
    with open(DATA_DIR / "metrics.json") as f:
        metrics = json.load(f)
    return preds, metrics

@st.cache_data
def load_features():
    f = DATA_DIR / "features.csv"
    return pd.read_csv(f, parse_dates=["Date"]) if f.exists() else None

def get_ticker_col(df: pd.DataFrame):
    for c in ("Ticker", "Symbol", "TickerSymbol", "Asset"):
        if c in df.columns:
            return c
    return None

def compute_signals(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Compute buy signals from probabilities."""
    df = df.copy()
    df["signal"] = (df["proba_up"] > threshold).map(lambda b: "buy" if b else "neutral")
    return df

def create_price_chart(ticker_data: pd.DataFrame, ticker: str, threshold: float):
    """Create price chart with buy signals."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ticker_data["Date"], y=ticker_data["close"], mode='lines',
        name='Close Price', line=dict(color='#1f77b4', width=2),
        hovertemplate='<b>%{x}</b><br>Price: $%{y:.2f}<extra></extra>'))
    
    buy_signals = ticker_data[ticker_data["signal"] == "buy"]
    if len(buy_signals) > 0:
        fig.add_trace(go.Scatter(x=buy_signals["Date"], y=buy_signals["close"], mode='markers',
            name='Buy Signal', marker=dict(symbol='triangle-up', size=12, color='#2ca02c', line=dict(width=2, color='white')),
            hovertemplate='<b>BUY SIGNAL</b><br>Date: %{x}<br>Price: $%{y:.2f}<br>Probability: %{customdata:.1%}<extra></extra>',
            customdata=buy_signals["proba_up"]))
    
    fig.update_layout(title=f'{ticker} - Price Chart with Buy Signals', xaxis_title='Date',
        yaxis_title='Price ($)', hovermode='x unified', height=400, template='plotly_white',
        showlegend=True, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    return fig, len(buy_signals)

def create_tech_chart(ticker_features: pd.DataFrame, ticker: str):
    """Create technical indicators chart (SMA and RSI)."""
    fig = make_subplots(rows=2, cols=1, subplot_titles=('Price with Moving Averages (SMA)', 'RSI (Relative Strength Index)'),
        vertical_spacing=0.15, row_heights=[0.6, 0.4])
    
    fig.add_trace(go.Scatter(x=ticker_features['Date'], y=ticker_features['close'], mode='lines',
        name='Close Price', line=dict(color='#1f77b4', width=2)), row=1, col=1)
    
    if 'sma_5' in ticker_features.columns:
        fig.add_trace(go.Scatter(x=ticker_features['Date'], y=ticker_features['sma_5'], mode='lines',
            name='SMA 5', line=dict(color='#ff7f0e', width=1.5, dash='dash')), row=1, col=1)
    if 'sma_20' in ticker_features.columns:
        fig.add_trace(go.Scatter(x=ticker_features['Date'], y=ticker_features['sma_20'], mode='lines',
            name='SMA 20', line=dict(color='#d62728', width=1.5, dash='dash')), row=1, col=1)
    
    if 'rsi_14' in ticker_features.columns:
        fig.add_trace(go.Scatter(x=ticker_features['Date'], y=ticker_features['rsi_14'], mode='lines',
            name='RSI 14', line=dict(color='#9467bd', width=2), fill='tozeroy',
            fillcolor='rgba(148, 103, 189, 0.1)'), row=2, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5, annotation_text="Overbought (70)", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5, annotation_text="Oversold (30)", row=2, col=1)
    
    fig.update_layout(height=600, template='plotly_white', showlegend=True, hovermode='x unified',
        title=f'{ticker} - Technical Indicators')
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="RSI", row=2, col=1)
    return fig

def format_signals_table(buy_signals: pd.DataFrame) -> pd.DataFrame:
    """Format buy signals for display."""
    display = buy_signals[["Date", "ticker", "close", "proba_up"]].copy()
    display["Date"] = display["Date"].dt.strftime("%Y-%m-%d")
    display["close"] = display["close"].apply(lambda x: f"${x:.2f}")
    display["proba_up"] = display["proba_up"].apply(lambda x: f"{x:.1%}")
    display.columns = ["Date", "Ticker", "Price", "Probability"]
    return display

# Load data
preds, metrics = load_preds_metrics()
features = load_features()

# Page configuration
st.set_page_config(page_title="Stock Prediction Dashboard", page_icon="📈", layout="wide")
st.title("Stock Prediction Dashboard")

# Global controls
options = []
for df in (preds, features):
    col = get_ticker_col(df)
    if col:
        options.extend(df[col].dropna().astype(str).unique())
options = sorted(set(options)) or ["AAPL", "MSFT", "SPY"]

tickers = st.multiselect("Select tickers:", options)
threshold = st.selectbox("Buy Threshold", options=[0.50, 0.55, 0.60, 0.65, 0.70], index=1,
    help="Probability threshold for generating buy signals. Lower threshold = more buy signals.")

if not tickers:
    st.warning("👆 Please select at least one ticker to continue.")
else:
    st.success(f"You selected: {', '.join(tickers)}")

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Overview", "Per-Stock", "Signals", "Metrics Insights", "Technical Indicators"])

with tab1:
    st.header("Overview")
    st.write("Quick summary of model performance and buy signals")
    
    # Model Performance Summary
    accuracy = metrics.get('accuracy', 0)
    avg_return = metrics.get('avg_next_day_return_when_buy', 0)
    
    st.subheader("📊 Model Performance")
    col1, col2, col3 = st.columns(3)
    col1.metric("Model Accuracy", f"{accuracy:.1%}")
    col2.metric("Avg Return (Buy)", f"{avg_return:.2%}")
    col3.metric("Buy Threshold", f"{threshold:.0%}")
    
    if not tickers:
        st.info("👆 Select tickers above to see detailed analysis.")
    else:
        st.divider()
        st.subheader(f"📈 Analysis for Selected Tickers: {', '.join(tickers)}")
        
        # Compute signals for selected tickers
        filtered_preds = compute_signals(preds[preds["ticker"].isin(tickers)], threshold)
        buy_signals = filtered_preds[filtered_preds["signal"] == "buy"].sort_values("Date", ascending=False)
        
        # Summary statistics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Buy Signals", len(buy_signals))
        col2.metric("Avg Signal Probability", f"{buy_signals['proba_up'].mean():.1%}" if len(buy_signals) > 0 else "N/A")
        date_min = filtered_preds['Date'].min().strftime('%m/%d/%y')
        date_max = filtered_preds['Date'].max().strftime('%m/%d/%y')
        col3.metric("Date Range", f"{date_min} - {date_max}")
        col4.metric("Total Predictions", len(filtered_preds))
        
        st.divider()
        
        # Per-ticker breakdown
        st.subheader("📋 Per-Ticker Summary")
        ticker_summary = []
        for ticker in tickers:
            ticker_data = filtered_preds[filtered_preds["ticker"] == ticker]
            ticker_buys = ticker_data[ticker_data["signal"] == "buy"]
            latest_price = ticker_data.iloc[-1]["close"] if len(ticker_data) > 0 else None
            avg_prob = ticker_data["proba_up"].mean()
            
            ticker_summary.append({
                "Ticker": ticker,
                "Buy Signals": len(ticker_buys),
                "Avg Probability": f"{avg_prob:.1%}",
                "Latest Price": f"${latest_price:.2f}" if latest_price else "N/A",
                "Data Points": len(ticker_data)
            })
        
        summary_df = pd.DataFrame(ticker_summary)
        st.dataframe(summary_df, use_container_width=True, hide_index=True)
        
        # Quick insights
        st.divider()
        st.subheader("💡 Quick Insights")
        
        if len(buy_signals) == 0:
            st.warning(f"⚠️ No buy signals found for selected tickers at threshold {threshold:.0%}. Try lowering the threshold or selecting different tickers.")
        else:
            latest_signal = buy_signals.iloc[0] if len(buy_signals) > 0 else None
            if latest_signal is not None:
                st.info(f"📅 **Latest Buy Signal**: {latest_signal['ticker']} on {latest_signal['Date'].strftime('%Y-%m-%d')} at ${latest_signal['close']:.2f} (Probability: {latest_signal['proba_up']:.1%})")
            
            signals_by_ticker = buy_signals.groupby("ticker").size().sort_values(ascending=False)
            top_ticker = signals_by_ticker.index[0] if len(signals_by_ticker) > 0 else None
            if top_ticker:
                st.success(f"🏆 **Most Signals**: {top_ticker} has {signals_by_ticker[top_ticker]} buy signal(s)")
            
            if len(buy_signals) > 0:
                high_confidence = buy_signals[buy_signals["proba_up"] >= 0.60]
                if len(high_confidence) > 0:
                    st.success(f"✨ **High Confidence Signals**: {len(high_confidence)} signal(s) with probability ≥60%")

with tab2:
    st.header("Per-Stock Analysis")
    if not tickers:
        st.info("Please select at least one ticker from the dropdown above to view charts.")
    else:
        filtered_preds = compute_signals(preds[preds["ticker"].isin(tickers)], threshold)
        for ticker in tickers:
            ticker_data = filtered_preds[filtered_preds["ticker"] == ticker].sort_values("Date")
            if len(ticker_data) == 0:
                st.warning(f"No data available for {ticker}")
                continue
            st.subheader(f"{ticker}")
            fig, buy_count = create_price_chart(ticker_data, ticker, threshold)
            st.plotly_chart(fig, use_container_width=True)
            st.caption(f"📊 {buy_count} buy signal(s) found for {ticker} at threshold {threshold:.0%}" if buy_count > 0 
                      else f"ℹ️ No buy signals for {ticker} at threshold {threshold:.0%}")
            st.divider()

with tab3:
    st.header("Buy Signals")
    st.write("Recent buy recommendations based on model predictions")
    st.caption(f"Showing signals for selected tickers with threshold {threshold:.0%}")
    
    if not tickers:
        st.info("👆 Please select at least one ticker from the dropdown above to view buy signals.")
    else:
        signals_df = compute_signals(preds, threshold)
        buy_signals = signals_df[(signals_df["signal"] == "buy") & (signals_df["ticker"].isin(tickers))].sort_values("Date", ascending=False)
        
        if len(buy_signals) == 0:
            st.info(f"No buy signals found with threshold {threshold:.0%} for selected tickers.")
        else:
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Signals", len(buy_signals))
            col2.metric("Unique Tickers", buy_signals["ticker"].nunique())
            col3.metric("Avg Probability", f"{buy_signals['proba_up'].mean():.1%}")
            col4.metric("Latest Signal", buy_signals["Date"].max().strftime("%Y-%m-%d"))
            st.divider()
            
            st.dataframe(format_signals_table(buy_signals).reset_index(drop=True), use_container_width=True, hide_index=True, height=400)
            
            csv_data = buy_signals[["Date", "ticker", "close", "proba_up"]].copy()
            csv_data["Date"] = csv_data["Date"].dt.strftime("%Y-%m-%d")
            csv_data.columns = ["Date", "Ticker", "Price", "Probability"]
            st.download_button("📥 Download Buy Signals (CSV)", csv_data.to_csv(index=False), "buy_signals.csv", "text/csv", key="download_signals")

with tab4:
    st.header("Metrics Insights")
    
    accuracy = metrics.get('accuracy', 0)
    avg_return = metrics.get('avg_next_day_return_when_buy', 0)
    
    # Key metrics
    col1, col2 = st.columns(2)
    col1.metric("Model Accuracy", f"{accuracy:.1%}")
    col2.metric("Avg Return (Buy)", f"{avg_return:.2%}")
    
    # Performance assessment with breakdowns
    st.divider()
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Model Accuracy Breakdown**")
        st.write(f"**Current**: {accuracy:.1%}")
        st.write("""
        - **< 50%**: Worse than random guessing
        - **50-52%**: Weak, barely better than random
        - **52-55%**: Moderate predictive capability
        - **55-60%**: Strong predictive capability
        - **> 60%**: Very strong predictive capability
        """)
        if accuracy > 0.55:
            st.success(f"✅ **Strong Model**: {accuracy:.1%} accuracy")
        elif accuracy > 0.50:
            st.info(f"ℹ️ **Moderate Model**: {accuracy:.1%} accuracy")
        else:
            st.warning(f"⚠️ **Weak Model**: {accuracy:.1%} accuracy")
    
    with col2:
        st.write("**Return per Signal Breakdown**")
        st.write(f"**Current**: {avg_return:.2%}")
        st.write("""
        - **< 0%**: Negative returns (losses)
        - **0-0.1%**: Minimal positive returns
        - **0.1-0.2%**: Modest positive returns
        - **0.2-0.5%**: Strong positive returns
        - **> 0.5%**: Very strong positive returns
        """)
        if avg_return > 0.002:
            st.success(f"✅ **Strong Returns**: {avg_return:.2%} per signal")
        elif avg_return > 0:
            st.info(f"ℹ️ **Positive Returns**: {avg_return:.2%} per signal")
        else:
            st.error(f"❌ **Negative Returns**: {avg_return:.2%} per signal")
    
    # Analysis charts
    st.divider()
    if len(preds) > 0:
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Probability Distribution**")
            fig_dist = go.Figure(data=[go.Histogram(x=preds["proba_up"], nbinsx=30, marker_color='#1f77b4', opacity=0.7)])
            fig_dist.add_vline(x=threshold, line_dash="dash", line_color="red", annotation_text=f"Threshold: {threshold:.0%}")
            fig_dist.update_layout(height=300, xaxis_title="Probability", yaxis_title="Frequency",
                template='plotly_white', title="Prediction Probabilities")
            st.plotly_chart(fig_dist, use_container_width=True)
        
        with col2:
            st.write("**Signal Statistics**")
            preds_copy = compute_signals(preds, threshold)
            buy_count = (preds_copy["signal"] == "buy").sum()
            total_count = len(preds_copy)
            buy_pct = buy_count / total_count if total_count > 0 else 0
            st.metric("Buy Signals", f"{buy_count:,}", f"{buy_pct:.1%} of total")
            st.metric("Neutral Signals", f"{total_count - buy_count:,}", f"{1 - buy_pct:.1%} of total")
            if buy_count > 0:
                st.metric("Avg Probability (Buy)", f"{preds_copy[preds_copy['signal']=='buy']['proba_up'].mean():.1%}")
        
        if "ticker" in preds.columns:
            st.write("**Performance by Ticker**")
            ticker_stats = preds.groupby("ticker").agg({"proba_up": ["mean", "std", "count"]}).round(4)
            ticker_stats.columns = ["Avg Probability", "Std Deviation", "Count"]
            st.dataframe(ticker_stats.sort_values("Avg Probability", ascending=False), use_container_width=True)

with tab5:
    st.header("Technical Indicators")
    st.write("SMA (Simple Moving Average) and RSI (Relative Strength Index) analysis")
    st.caption(f"Showing technical indicators for selected tickers")
    
    if features is None:
        st.warning("⚠️ Features data not available. Please ensure features.csv exists in the data folder.")
    elif not tickers:
        st.info("👆 Please select at least one ticker from the dropdown above to view technical indicators.")
    else:
        for ticker in tickers:
            ticker_features = features[features["ticker"] == ticker].sort_values("Date")
            if len(ticker_features) == 0:
                st.warning(f"No technical indicator data available for {ticker}")
                st.divider()
                continue
            
            st.subheader(f"{ticker}")
            st.plotly_chart(create_tech_chart(ticker_features, ticker), use_container_width=True)
            
            latest = ticker_features.iloc[-1]
            st.subheader("Current Indicator Values")
            col1, col2, col3, col4 = st.columns(4)
            if 'close' in latest:
                col1.metric("Current Price", f"${latest['close']:.2f}")
            if 'sma_5' in latest:
                col2.metric("SMA 5", f"${latest['sma_5']:.2f}")
            if 'sma_20' in latest:
                col3.metric("SMA 20", f"${latest['sma_20']:.2f}")
            if 'rsi_14' in latest:
                rsi_val = latest['rsi_14']
                rsi_status = "🟢 Oversold" if rsi_val < 30 else "🔴 Overbought" if rsi_val > 70 else "🟡 Neutral"
                col4.metric("RSI 14", f"{rsi_val:.1f}", rsi_status)
            
            st.divider()
            st.subheader("📊 Current Signals")
            col1, col2 = st.columns(2)
            with col1:
                if 'sma_5' in latest and 'sma_20' in latest:
                    if latest['sma_5'] > latest['sma_20']:
                        st.success("✅ **Bullish**: SMA 5 > SMA 20")
                    else:
                        st.warning("⚠️ **Bearish**: SMA 5 < SMA 20")
            with col2:
                if 'rsi_14' in latest:
                    if rsi_val > 70:
                        st.error("🔴 **Overbought**: RSI > 70")
                    elif rsi_val < 30:
                        st.success("🟢 **Oversold**: RSI < 30")
                    else:
                        st.info("🟡 **Neutral**: RSI 30-70")
            st.divider()