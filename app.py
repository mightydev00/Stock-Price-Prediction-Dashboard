import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from fpdf import FPDF
import os
import datetime
from curl_cffi import requests

# ==========================================
# Session Setup (Bypass Yahoo Finance 429 TLS Fingerprinting)
# ==========================================
# Impersonate a real Chrome browser to bypass advanced rate limits
session = requests.Session(impersonate="chrome")

st.set_page_config(page_title="Stock Predictor & Intelligence Report", layout="wide")

st.title("📈 Stock Price Predictor & Intelligence Dashboard")
st.write("Combines LSTM deep learning, technical indicators, news sentiment, interactive charts, and automated PDF reporting.")

# ==========================================
# Sidebar Configuration & Dynamic Dates
# ==========================================
st.sidebar.header("Configuration")
ticker = st.sidebar.text_input("Stock Ticker", value="AAPL").upper()

# Dynamically calculate today and 5 years ago
today = datetime.date.today()
five_years_ago = today - datetime.timedelta(days=5 * 365)

# Added format="YYYY-MM-DD" to fix the typing bug
start_date = st.sidebar.date_input("Start Date", value=five_years_ago, format="YYYY-MM-DD")
end_date = st.sidebar.date_input("End Date", value=today, format="YYYY-MM-DD")
epochs = st.sidebar.slider("Training Epochs", min_value=5, max_value=30, value=10)

if st.sidebar.button("Run Full Analysis & Generate Report"):
    
    # ==========================================
    # 1. Fetch Price Data, Fundamentals & News
    # ==========================================
    with st.spinner(f"Fetching market data, fundamentals, and news for {ticker}..."):
        ticker_obj = yf.Ticker(ticker, session=session)
        data = yf.download(ticker, start=start_date, end=end_date, multi_level_index=False, session=session)
        
        info = ticker_obj.info if hasattr(ticker_obj, 'info') else {}
        news_items = ticker_obj.news if hasattr(ticker_obj, 'news') else []

    # ==========================================
    # Safety Check: Minimum Data Required
    # ==========================================
    if data.empty:
        st.error("No historical data found. Please verify the ticker symbol and date range.")
    elif len(data) < 150:
        st.warning(f"⚠️ {ticker} only has {len(data)} days of trading history available in this range.")
        st.error("Our deep learning model requires a minimum of 150 trading days to calculate technical indicators (50-day SMA) and generate 60-day training sequences. Please select a company with a longer public history.")
    else:
        # ==========================================
        # 1.5 Interactive Historical Candlestick Chart
        # ==========================================
        st.subheader(f"📊 Historical Price Action ({start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')})")
        
        fig_candle = go.Figure(data=[go.Candlestick(x=data.index,
                        open=data['Open'],
                        high=data['High'],
                        low=data['Low'],
                        close=data['Close'],
                        name=ticker)])
        
        fig_candle.update_layout(
            yaxis_title="Price (USD)",
            xaxis_title="Date",
            template="plotly_dark", 
            margin=dict(l=0, r=0, t=30, b=0),
            height=450,
            dragmode="zoom"
        )
        fig_candle.update_xaxes(rangeslider_visible=False, fixedrange=False)
        fig_candle.update_yaxes(fixedrange=False)
        
        st.plotly_chart(fig_candle, use_container_width=True, config={'scrollZoom': True})
        st.divider()

        # ==========================================
        # 2. NLP Sentiment Analysis
        # ==========================================
        st.subheader("📰 Real-Time News Sentiment Analysis")
        analyzer = SentimentIntensityAnalyzer()
        
        sentiment_scores = []
        headlines = []

        if news_items:
            for item in news_items[:6]:
                content = item.get('content', {})
                title = content.get('title', '')
                
                click_through = content.get('clickThroughUrl', {})
                url = click_through.get('url', '') if isinstance(click_through, dict) else ''
                if not url:
                    canonical = content.get('canonicalUrl', {})
                    url = canonical.get('url', '') if isinstance(canonical, dict) else ''

                if title:
                    score = analyzer.polarity_scores(title)['compound']
                    sentiment_scores.append(score)
                    headlines.append((title, score, url))

            avg_sentiment = np.mean(sentiment_scores) if sentiment_scores else 0.0
        else:
            avg_sentiment = 0.0

        col_s1, col_s2 = st.columns([1, 2])
        with col_s1:
            if avg_sentiment >= 0.05:
                st.success(f"**Overall Sentiment: Bullish** (Score: {avg_sentiment:+.2f})")
            elif avg_sentiment <= -0.05:
                st.error(f"**Overall Sentiment: Bearish** (Score: {avg_sentiment:+.2f})")
            else:
                st.info(f"**Overall Sentiment: Neutral** (Score: {avg_sentiment:+.2f})")

        with col_s2:
            with st.expander("View Recent Headlines & NLP Scores"):
                for title, score, url in headlines:
                    if url:
                        st.markdown(f"- [{title}]({url}) `[Score: {score:+.2f}]`")
                    else:
                        st.markdown(f"- {title} `[Score: {score:+.2f}]`")

        st.divider()

        # ==========================================
        # 3. Technical Indicators & Data Prep
        # ==========================================
        df = data[['Close']].copy()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()

        delta = df['Close'].diff(1).dropna()
        gains = delta.where(delta > 0, 0)
        losses = -delta.where(delta < 0, 0)
        avg_gain = gains.rolling(window=14).mean()
        avg_loss = losses.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        df['RSI_14'] = 100 - (100 / (1 + rs))
        df = df.dropna()

        feature_scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_features = feature_scaler.fit_transform(df[['Close', 'SMA_50', 'RSI_14']].values)

        target_scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_target = target_scaler.fit_transform(df[['Close']].values)

        training_len = int(np.ceil(len(scaled_features) * 0.8))
        train_feat = scaled_features[:training_len]
        train_tgt = scaled_target[:training_len]

        seq_len = 60
        x_train, y_train = [], []
        for i in range(seq_len, len(train_feat)):
            x_train.append(train_feat[i-seq_len:i, :])
            y_train.append(train_tgt[i, 0])

        x_train, y_train = np.array(x_train), np.array(y_train)

        # ==========================================
        # 4. Model Training
        # ==========================================
        with st.spinner("Training LSTM network on price and technical momentum..."):
            model = Sequential([
                LSTM(units=50, return_sequences=True, input_shape=(seq_len, 3)),
                Dropout(0.2),
                LSTM(units=50, return_sequences=False),
                Dropout(0.2),
                Dense(units=25),
                Dense(units=1)
            ])
            model.compile(optimizer='adam', loss='mean_squared_error')
            model.fit(x_train, y_train, batch_size=32, epochs=epochs, verbose=0)

        # ==========================================
        # 5. Evaluation & Predictions
        # ==========================================
        test_feat = scaled_features[training_len - seq_len:, :]
        y_test_actual = df[['Close']].values[training_len:]
        x_test = []
        for i in range(seq_len, len(test_feat)):
            x_test.append(test_feat[i-seq_len:i, :])
        x_test = np.array(x_test)

        predictions = model.predict(x_test)
        predictions = target_scaler.inverse_transform(predictions)
        rmse = np.sqrt(np.mean(((predictions - y_test_actual) ** 2)))

        last_60 = scaled_features[-60:]
        x_future = np.array([last_60])
        pred_next = model.predict(x_future)
        pred_next_usd = float(target_scaler.inverse_transform(pred_next)[0][0])
        
        last_actual_price = float(df['Close'].iloc[-1])
        price_diff = pred_next_usd - last_actual_price
        pct_change = (price_diff / last_actual_price) * 100

        # ==========================================
        # 6. Dashboard Metrics & Visuals
        # ==========================================
        st.subheader("📊 Model Performance & Price Forecast")
        col1, col2, col3 = st.columns(3)
        col1.metric(label="Model RMSE", value=f"${rmse:.2f}")
        col2.metric(label="Predicted Next Day Close", value=f"${pred_next_usd:.2f}", delta=f"{pct_change:+.2f}%")
        col3.metric(label="NLP Market Mood", value=f"{'Bullish' if avg_sentiment > 0.05 else 'Bearish' if avg_sentiment < -0.05 else 'Neutral'}")

        valid = df[training_len:].copy()
        valid['Predicted'] = predictions

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(df.index[:training_len], df['Close'][:training_len], label="Training History")
        ax.plot(valid.index, valid['Close'], label="Actual Price")
        ax.plot(valid.index, valid['Predicted'], label="Predicted Price", linestyle="--")
        ax.set_title(f"{ticker} Price vs LSTM Forecast")
        ax.set_ylabel("USD ($)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

        # Save chart locally for PDF report embedding
        chart_filename = "chart.png"
        fig.savefig(chart_filename, bbox_inches='tight')

        # ==========================================
        # 7. Comprehensive Stock Intelligence Report
        # ==========================================
        st.divider()
        st.subheader(f"📑 Executive Intelligence Report: {ticker}")

        current_sma = float(df['SMA_50'].iloc[-1])
        current_rsi = float(df['RSI_14'].iloc[-1])

        sma_signal = "Bullish" if last_actual_price >= current_sma else "Bearish"
        rsi_signal = "Overbought" if current_rsi > 70 else "Oversold" if current_rsi < 30 else "Neutral"
        lstm_signal = "Bullish" if price_diff > 0 else "Bearish"

        score = 0
        if price_diff > 0: score += 1
        else: score -= 1
        if last_actual_price >= current_sma: score += 1
        else: score -= 1
        if current_rsi < 30: score += 1
        elif current_rsi > 70: score -= 1
        if avg_sentiment >= 0.05: score += 1
        elif avg_sentiment <= -0.05: score -= 1

        if score >= 2: consensus = "Strong Bullish"
        elif score == 1: consensus = "Moderately Bullish"
        elif score == 0: consensus = "Neutral / Consolidation"
        elif score == -1: consensus = "Moderately Bearish"
        else: consensus = "Strong Bearish"

        rep_col1, rep_col2 = st.columns(2)
        with rep_col1:
            st.markdown("### 🏢 Fundamentals")
            st.write(f"- **Name:** {info.get('longName', ticker)}")
            st.write(f"- **Sector:** {info.get('sector', 'N/A')}")
            st.write(f"- **Market Cap:** ${info.get('marketCap', 0):,}")
            st.write(f"- **Trailing P/E Ratio:** {info.get('trailingPE', 'N/A')}")

        with rep_col2:
            st.markdown("### 🎯 Signal Breakdown")
            st.write(f"- **LSTM Forecast:** ${pred_next_usd:.2f} $\\rightarrow$ **{lstm_signal}**")
            st.write(f"- **50-Day SMA:** ${current_sma:.2f} $\\rightarrow$ **{sma_signal}**")
            st.write(f"- **14-Day RSI:** {current_rsi:.2f} $\\rightarrow$ **{rsi_signal}**")
            st.write(f"- **News Sentiment:** {avg_sentiment:+.2f} $\\rightarrow$ **{'Bullish' if avg_sentiment >= 0.05 else 'Bearish' if avg_sentiment <= -0.05 else 'Neutral'}**")
            st.markdown(f"#### **Consensus Signal:** {consensus}")

        # ==========================================
        # 8. Generate & Download PDF
        # ==========================================
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", 'B', 16)
        pdf.cell(200, 10, txt=f"Executive Stock Intelligence Report: {ticker}", ln=True, align='C')
        pdf.set_font("Arial", size=10)
        pdf.cell(200, 10, txt=f"Generated: {pd.to_datetime('today').strftime('%Y-%m-%d %H:%M')}", ln=True, align='C')
        
        pdf.ln(5)
        
        pdf.image(chart_filename, x=10, w=190)
        pdf.ln(5)

        report_lines = [
            "1. COMPANY OVERVIEW",
            f"   - Name: {info.get('longName', ticker)}",
            f"   - Sector: {info.get('sector', 'N/A')}",
            f"   - Market Cap: ${info.get('marketCap', 0):,}",
            "",
            "2. QUANTITATIVE FORECAST (LSTM MODEL)",
            f"   - Last Recorded Close: ${last_actual_price:.2f}",
            f"   - Next-Day Forecast: ${pred_next_usd:.2f}",
            f"   - Expected Delta: {price_diff:+.2f} ({pct_change:+.2f}%)",
            f"   - Model Test RMSE: ${rmse:.2f}",
            "",
            "3. TECHNICAL INDICATORS",
            f"   - 50-Day SMA: ${current_sma:.2f} ({sma_signal})",
            f"   - 14-Day RSI: {current_rsi:.2f} ({rsi_signal})",
            "",
            "4. NEWS SENTIMENT ANALYSIS (VADER NLP)",
            f"   - Average Headline Polarity Score: {avg_sentiment:+.2f}",
            "",
            "5. OVERALL COMPOSITE SIGNAL",
            f"   - Verdict: {consensus}"
        ]

        pdf.set_font("Arial", size=12)
        for line in report_lines:
            pdf.cell(200, 7, txt=line, ln=True)

        pdf_filename = "report.pdf"
        pdf.output(pdf_filename)
        
        with open(pdf_filename, "rb") as pdf_file:
            pdf_bytes = pdf_file.read()

        if os.path.exists(chart_filename): os.remove(chart_filename)
        if os.path.exists(pdf_filename): os.remove(pdf_filename)

        st.download_button(
            label="📄 Download Executive Report as PDF",
            data=pdf_bytes,
            file_name=f"{ticker}_Intelligence_Report.pdf",
            mime="application/pdf"
        )
