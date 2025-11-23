import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from datetime import datetime, timedelta

# ==========================================
# 1. APP CONFIGURATION & BRANDING
# ==========================================
st.set_page_config(page_title="Just Chill & Trade", layout="wide", page_icon="🧊")

# Custom CSS for that "Chill" vibe
st.markdown("""
<style>
    .main-title {font-size: 3rem; color: #4DA8DA; text-align: center; font-weight: bold;}
    .sub-title {font-size: 1.2rem; color: #555; text-align: center;}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">Just Chill & Trade </div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">AI-Powered Deep Learning & Market Scanner</div>', unsafe_allow_html=True)

# ==========================================
# 2. STOCK UNIVERSE (High Volume Indian Stocks)
# ==========================================
STOCK_LIST = {
    # --- INDICES ---
    '^NSEI': 'NIFTY 50 Index',
    '^NSEBANK': 'BANK NIFTY Index',
    # --- BANKING ---
    'HDFCBANK.NS': 'HDFC Bank', 'ICICIBANK.NS': 'ICICI Bank', 'SBIN.NS': 'State Bank of India',
    'KOTAKBANK.NS': 'Kotak Mahindra Bank', 'AXISBANK.NS': 'Axis Bank', 'BAJFINANCE.NS': 'Bajaj Finance',
    # --- IT ---
    'TCS.NS': 'TCS', 'INFY.NS': 'Infosys', 'HCLTECH.NS': 'HCL Tech', 'WIPRO.NS': 'Wipro',
    # --- AUTO ---
    'MARUTI.NS': 'Maruti Suzuki', 'TATAMOTORS.NS': 'Tata Motors', 'M&M.NS': 'Mahindra & Mahindra',
    # --- ENERGY ---
    'RELIANCE.NS': 'Reliance Industries', 'ONGC.NS': 'ONGC', 'NTPC.NS': 'NTPC',
    # --- FMCG ---
    'HINDUNILVR.NS': 'Hindustan Unilever', 'ITC.NS': 'ITC Ltd', 'NESTLEIND.NS': 'Nestle India',
    # --- PHARMA ---
    'SUNPHARMA.NS': 'Sun Pharma', 'CIPLA.NS': 'Cipla', 'DRREDDY.NS': 'Dr Reddys',
    # --- METALS & INFRA ---
    'TATASTEEL.NS': 'Tata Steel', 'LT.NS': 'Larsen & Toubro', 'TITAN.NS': 'Titan Company',
    'ADANIENT.NS': 'Adani Enterprises'
}

# List used for the Scanner
SCANNER_LIST = list(STOCK_LIST.keys())

# ==========================================
# 3. DATA & TECHNICALS
# ==========================================
@st.cache_data(ttl=3600)
def fetch_data(ticker, period="2y"):
    try:
        # Download data
        df = yf.download(ticker, period=period, progress=False)
        if df.empty: return None
        
        # FIX: Handle MultiIndex columns (yfinance update)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        df.reset_index(inplace=True)
        return df
    except Exception as e:
        return None

def add_technicals(df):
    # MACD
    df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()

    # Bollinger Bands
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['Std_Dev'] = df['Close'].rolling(window=20).std()
    df['Upper_Band'] = df['SMA_20'] + (2 * df['Std_Dev'])
    df['Lower_Band'] = df['SMA_20'] - (2 * df['Std_Dev'])

    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    return df

def get_signal(row):
    """Determine Buy/Sell/Wait based on Technicals"""
    if row['MACD'] > row['Signal_Line'] and row['RSI'] < 70:
        return "BUY", "green"
    elif row['MACD'] < row['Signal_Line'] or row['RSI'] > 70:
        return "SELL", "red"
    return "WAIT", "gray"

# ==========================================
# 4. DEEP LEARNING MODEL (LSTM)
# ==========================================
def train_lstm_model(df, forecast_days=30):
    # Prepare Data
    data = df.filter(['Close']).values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    prediction_days = 60
    x_train, y_train = [], []

    for i in range(prediction_days, len(scaled_data)):
        x_train.append(scaled_data[i-prediction_days:i, 0])
        y_train.append(scaled_data[i, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    # Build LSTM Model
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=False, input_shape=(x_train.shape[1], 1)))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    # Train
    model.fit(x_train, y_train, epochs=3, batch_size=32, verbose=0)

    # Predict Future
    test_input = scaled_data[-prediction_days:].reshape(1, prediction_days, 1)
    future_predictions = []

    for _ in range(forecast_days):
        pred = model.predict(test_input, verbose=0)
        future_predictions.append(pred[0][0])
        test_input = np.append(test_input[:, 1:, :], [[pred[0]]], axis=1)

    future_prices = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
    return future_prices.flatten()

# ==========================================
# 5. BACKTEST ENGINE
# ==========================================
def run_backtest(df, initial_investment):
    balance = initial_investment
    shares = 0
    df['Signal_Text'], _ = zip(*df.apply(get_signal, axis=1)) # Unpack tuple
    
    log_data = []
    
    for i in range(1, len(df)):
        price = df['Close'].iloc[i]
        sig = df['Signal_Text'].iloc[i]
        
        if sig == "BUY" and balance > 0:
            shares = balance / price
            balance = 0
        elif sig == "SELL" and shares > 0:
            balance = shares * price
            shares = 0
            
    final_value = balance if balance > 0 else (shares * df['Close'].iloc[-1])
    return initial_investment, final_value

# ==========================================
# 6. MAIN APPLICATION UI
# ==========================================
def main():
    # Sidebar
    st.sidebar.header("⚙️ Configuration")
    selected_ticker = st.sidebar.selectbox("Select Stock", SCANNER_LIST, format_func=lambda x: f"{STOCK_LIST[x]} ({x})")
    investment_amount = st.sidebar.number_input("Amount to Invest (₹)", value=100000, step=5000)
    forecast_period = st.sidebar.slider("AI Prediction Days", 7, 60, 30)

    # Tabs
    tab1, tab2, tab3 = st.tabs(["📊 Analysis & Prediction", "🤖 Market Scanner", "🔙 Backtesting"])

    # --- TAB 1: ANALYSIS ---
    with tab1:
        st.subheader(f"Analysis for {STOCK_LIST[selected_ticker]}")
        
        df = fetch_data(selected_ticker)
        
        if df is not None:
            df = add_technicals(df)
            current_price = float(df['Close'].iloc[-1])
            signal, sig_color = get_signal(df.iloc[-1])
            
            # Top Metrics
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Current Price", f"₹{current_price:,.2f}")
            c2.metric("Market Signal", signal, delta_color="off")
            c3.metric("RSI (Strength)", f"{df['RSI'].iloc[-1]:.1f}")
            c4.metric("MACD Trend", "Bullish" if df['MACD'].iloc[-1] > df['Signal_Line'].iloc[-1] else "Bearish")

            # AI Section
            st.divider()
            st.markdown("### 🧠 Deep Learning Projection")
            
            with st.spinner("Just chill... The AI is thinking (Training LSTM Model)..."):
                predicted_prices = train_lstm_model(df, forecast_days=forecast_period)
            
            final_pred = predicted_prices[-1]
            roi = ((final_pred - current_price) / current_price) * 100
            future_value = investment_amount * (1 + roi/100)
            
            # Investment Result
            st.info(f"""
            **💰 Investment Outlook:**
            If you invest **₹{investment_amount:,.0f}** now, the AI predicts it could be worth **₹{future_value:,.0f}** in {forecast_period} days.
            Expected Return: **{roi:+.2f}%**
            """)

            # Charts
            fig = go.Figure()
            # Candlestick
            fig.add_trace(go.Candlestick(x=df['Date'], open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Actual Price'))
            # Bollinger
            fig.add_trace(go.Scatter(x=df['Date'], y=df['Upper_Band'], line=dict(color='gray', width=1, dash='dot'), name='Upper Band'))
            fig.add_trace(go.Scatter(x=df['Date'], y=df['Lower_Band'], line=dict(color='gray', width=1, dash='dot'), name='Lower Band'))
            # Prediction
            future_dates = [df['Date'].iloc[-1] + timedelta(days=i) for i in range(1, forecast_period+1)]
            fig.add_trace(go.Scatter(x=future_dates, y=predicted_prices, line=dict(color='#4DA8DA', width=3), name='AI Forecast'))
            
            fig.update_layout(height=600, title=f"{selected_ticker} - Price vs AI Prediction", xaxis_title="Date", yaxis_title="Price")
            st.plotly_chart(fig, use_container_width=True)

    # --- TAB 2: SCANNER ---
    with tab2:
        st.subheader("🔍 Find Buy Opportunities")
        st.write(f"Scanning {len(SCANNER_LIST)} major stocks for BUY signals...")
        
        if st.button("Start Chill Scan 🧊"):
            results = []
            progress = st.progress(0)
            
            for i, ticker in enumerate(SCANNER_LIST):
                d = fetch_data(ticker, period="6mo") # Shorter period for speed
                if d is not None:
                    d = add_technicals(d)
                    sig, _ = get_signal(d.iloc[-1])
                    if sig == "BUY":
                        results.append({
                            "Stock": STOCK_LIST[ticker],
                            "Ticker": ticker,
                            "Price": f"₹{d['Close'].iloc[-1]:.2f}",
                            "RSI": f"{d['RSI'].iloc[-1]:.1f}"
                        })
                progress.progress((i + 1) / len(SCANNER_LIST))
            
            if results:
                st.success(f"Found {len(results)} stocks to BUY!")
                st.dataframe(pd.DataFrame(results))
            else:
                st.warning("Market is cold. No clear BUY signals found right now.")

    # --- TAB 3: BACKTEST ---
    with tab3:
        st.subheader("🔙 Does this strategy work?")
        st.write(f"Testing Strategy on {STOCK_LIST[selected_ticker]} (Past 2 Years)")
        
        if df is not None:
            start_val, end_val = run_backtest(df.copy(), investment_amount)
            pnl = end_val - start_val
            pnl_pct = (pnl / start_val) * 100
            
            col1, col2 = st.columns(2)
            col1.metric("Starting Balance", f"₹{start_val:,.0f}")
            col2.metric("Final Balance", f"₹{end_val:,.0f}", f"{pnl_pct:.2f}%")
            
            if pnl > 0:
                st.success("✅ This strategy would have made money!")
            else:
                st.error("❌ This strategy would have lost money on this specific stock.")

if __name__ == "__main__":
    main()