"""
📈 Indian Stock LSTM Predictor
- Live NSE data in ₹ (INR)
- 3 LSTM variants trained on startup
- Next N-day price forecast
- High-accuracy feature engineering
"""
import os, warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ta

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, LSTM, Bidirectional, Dense, Dropout,
    LayerNormalization, MultiHeadAttention, GlobalAveragePooling1D,
    Conv1D, BatchNormalization
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

tf.get_logger().setLevel("ERROR")

# ─────────────────── PAGE CONFIG ───────────────────
st.set_page_config(
    page_title="Indian Stock LSTM Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
  [data-testid="stMetricValue"]  { font-size:1.7rem; font-weight:800 }
  [data-testid="stMetricLabel"]  { font-size:0.8rem; color:#888 }
  .block-container               { padding-top:1.2rem }
  h2                             { margin-top:1.4rem }
  [data-testid="stSidebar"] h1  { font-size:1.2rem }
</style>
""", unsafe_allow_html=True)

# ─────────────────── CONSTANTS ─────────────────────
STOCKS = {
    "Reliance Industries": "RELIANCE.NS",
    "TCS":                 "TCS.NS",
    "Infosys":             "INFY.NS",
    "HDFC Bank":           "HDFCBANK.NS",
    "NIFTY 50":            "^NSEI",
}
SEQ_LEN  = 30
EPOCHS   = 10
BATCH    = 64
FEATURES = ["Close","Open","High","Low","Volume",
            "RSI","MACD","Signal","EMA20","EMA50",
            "ATR","OBV","BB_up","BB_lo","BB_mid",
            "Ret1","Ret3","Ret5","Vol10","Vol20","Momentum"]

# ─────────────────── SIDEBAR ───────────────────────
st.sidebar.title("🇮🇳 Controls")
stock_name   = st.sidebar.selectbox("Select Stock", list(STOCKS.keys()))
model_choice = st.sidebar.radio("Select Model", ["Standard LSTM","Bidirectional LSTM","Attention LSTM"])
predict_days = st.sidebar.slider("Forecast Next N Days", 1, 15, 5)

TICKER = STOCKS[stock_name]

# ─────────────────── DATA ──────────────────────────
@st.cache_data(ttl=3600, show_spinner="⬇️  Fetching live data from NSE…")
def get_data(ticker):
    df = yf.download(ticker, period="3y", interval="1d",
                     auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.dropna(inplace=True)

    c, h, l, v = df["Close"], df["High"], df["Low"], df["Volume"]

    df["RSI"]      = ta.momentum.RSIIndicator(close=c, window=14).rsi()
    macd_obj       = ta.trend.MACD(close=c, window_fast=12, window_slow=26, window_sign=9)
    df["MACD"]     = macd_obj.macd()
    df["Signal"]   = macd_obj.macd_signal()
    df["EMA20"]    = ta.trend.EMAIndicator(close=c, window=20).ema_indicator()
    df["EMA50"]    = ta.trend.EMAIndicator(close=c, window=50).ema_indicator()
    df["ATR"]      = ta.volatility.AverageTrueRange(high=h, low=l, close=c).average_true_range()
    df["OBV"]      = ta.volume.OnBalanceVolumeIndicator(close=c, volume=v).on_balance_volume()
    bb             = ta.volatility.BollingerBands(close=c, window=20, window_dev=2)
    df["BB_up"]    = bb.bollinger_hband()
    df["BB_lo"]    = bb.bollinger_lband()
    df["BB_mid"]   = bb.bollinger_mavg()
    df["Ret1"]     = c.pct_change(1)
    df["Ret3"]     = c.pct_change(3)
    df["Ret5"]     = c.pct_change(5)
    df["Vol10"]    = c.rolling(10).std()
    df["Vol20"]    = c.rolling(20).std()
    df["Momentum"] = c - c.shift(10)

    df.dropna(inplace=True)
    return df

# ─────────────────── SEQUENCES ─────────────────────
def build_sequences(df, seq_len):
    data   = df[FEATURES].values.astype(np.float32)
    scaler = MinMaxScaler()
    ds     = scaler.fit_transform(data)

    xs, ys = [], []
    for i in range(seq_len, len(ds)):
        xs.append(ds[i - seq_len:i])
        ys.append(ds[i, 0])

    xs, ys  = np.array(xs), np.array(ys)
    split   = int(len(xs) * 0.82)
    return xs[:split], xs[split:], ys[:split], ys[split:], scaler, df.index[seq_len:]

# ─────────────────── MODEL BUILDERS ────────────────
def build_standard(shape):
    inp = Input(shape=shape)
    x   = Conv1D(64, 3, padding="same", activation="relu")(inp)
    x   = LSTM(128, return_sequences=True)(x)
    x   = Dropout(0.25)(x)
    x   = LSTM(64, return_sequences=True)(x)
    x   = Dropout(0.25)(x)
    x   = LSTM(32)(x)
    x   = Dense(32, activation="relu")(x)
    out = Dense(1)(x)
    return Model(inp, out, name="Standard_LSTM")

def build_bidi(shape):
    inp = Input(shape=shape)
    x   = Conv1D(64, 3, padding="same", activation="relu")(inp)
    x   = Bidirectional(LSTM(128, return_sequences=True))(x)
    x   = Dropout(0.25)(x)
    x   = Bidirectional(LSTM(64, return_sequences=True))(x)
    x   = Dropout(0.25)(x)
    x   = Bidirectional(LSTM(32))(x)
    x   = Dense(32, activation="relu")(x)
    out = Dense(1)(x)
    return Model(inp, out, name="Bidi_LSTM")

def build_attention(shape):
    inp  = Input(shape=shape)
    x    = Conv1D(64, 3, padding="same", activation="relu")(inp)
    x    = LSTM(128, return_sequences=True)(x)
    x    = LayerNormalization()(x)
    attn = MultiHeadAttention(num_heads=4, key_dim=32)(x, x)
    x    = x + attn
    x    = LayerNormalization()(x)
    x    = LSTM(64, return_sequences=True)(x)
    attn2= MultiHeadAttention(num_heads=2, key_dim=16)(x, x)
    x    = x + attn2
    x    = GlobalAveragePooling1D()(x)
    x    = Dense(64, activation="relu")(x)
    x    = Dropout(0.25)(x)
    out  = Dense(1)(x)
    return Model(inp, out, name="Attention_LSTM")

# ─────────────────── TRAIN ALL ─────────────────────
@st.cache_resource(show_spinner="🧠  Training LSTM models on live NSE data (one-time, ~60s)…")
def train_all(ticker):
    df = get_data(ticker)
    X_tr, X_te, y_tr, y_te, scaler, dates = build_sequences(df, SEQ_LEN)
    shape = (X_tr.shape[1], X_tr.shape[2])
    cbs = [
        EarlyStopping(patience=6, restore_best_weights=True, verbose=0),
        ReduceLROnPlateau(patience=3, factor=0.5, min_lr=1e-5, verbose=0)
    ]

    out = {}
    n_feat = len(FEATURES)

    for name, builder in [
        ("Standard LSTM",      build_standard),
        ("Bidirectional LSTM", build_bidi),
        ("Attention LSTM",     build_attention),
    ]:
        m = builder(shape)
        m.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse", metrics=["mae"])
        m.fit(X_tr, y_tr, epochs=EPOCHS, batch_size=BATCH,
              validation_split=0.12, callbacks=cbs, verbose=0)

        y_pred_sc = m.predict(X_te, verbose=0).flatten()

        def inv(col_vals):
            tmp = np.zeros((len(col_vals), n_feat))
            tmp[:, 0] = col_vals
            return scaler.inverse_transform(tmp)[:, 0]

        y_actual = inv(y_te)
        y_pred   = inv(y_pred_sc)

        rmse = float(np.sqrt(mean_squared_error(y_actual, y_pred)))
        mae  = float(mean_absolute_error(y_actual, y_pred))
        mape = float(np.mean(np.abs((y_actual - y_pred) / (y_actual + 1e-8))) * 100)
        r2   = float(r2_score(y_actual, y_pred))
        da   = float(np.mean(np.sign(np.diff(y_actual)) == np.sign(np.diff(y_pred))) * 100)

        # ── future forecast ────────────────────────────
        last_seq  = X_te[-1].copy()         # shape (SEQ_LEN, n_feat)
        scaled_data = scaler.transform(df[FEATURES].values[-SEQ_LEN:].astype(np.float32))
        last_seq  = scaled_data.copy()

        future_sc = []
        cur = last_seq.copy()
        for _ in range(15):                   # predict up to 15 days
            pred = m.predict(cur[np.newaxis], verbose=0)[0, 0]
            future_sc.append(pred)
            nxt = np.zeros(n_feat)
            nxt[0] = pred
            nxt[1:] = cur[-1, 1:]            # carry other features forward
            cur = np.vstack([cur[1:], nxt])

        future_prices = inv(np.array(future_sc))
        last_date     = df.index[-1]
        future_dates  = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=15)

        out[name] = {
            "y_actual":     y_actual,
            "y_pred":       y_pred,
            "test_dates":   dates[-len(y_te):],
            "future_prices":future_prices,
            "future_dates": future_dates,
            "model":        m,
            "metrics": {
                "RMSE":  rmse,
                "MAE":   mae,
                "MAPE":  mape,
                "R²":    r2,
                "Dir %": da,
            }
        }

    return out, df, scaler

# ─────────────────── CURRENCY HELPER ───────────────
def fmt(v):
    return f"₹{v:,.2f}"

# ─────────────────── LOAD ──────────────────────────
st.title(f"📈 {stock_name} — Live LSTM Predictor  ( ₹ INR )")
st.caption("NSE/BSE live data · All 3 models trained on startup · Instant model switch · Next-day forecast")

with st.spinner("Loading…"):
    all_res, raw_df, scaler = train_all(TICKER)

res     = all_res[model_choice]
metrics = res["metrics"]

# ─────────────────── METRIC CARDS ──────────────────
st.subheader(f"📐 {model_choice} — Performance Metrics")
c1,c2,c3,c4,c5 = st.columns(5)
c1.metric("RMSE",           fmt(metrics["RMSE"]))
c2.metric("MAE",            fmt(metrics["MAE"]))
c3.metric("MAPE",           f"{metrics['MAPE']:.2f}%")
c4.metric("R² Score",       f"{metrics['R²']:.3f}")
c5.metric("Directional ✓",  f"{metrics['Dir %']:.1f}%")

# ─────────────────── FUTURE FORECAST ───────────────
st.subheader(f"🔮 Next {predict_days}-Day Price Forecast")
fp = res["future_prices"][:predict_days]
fd = res["future_dates"][:predict_days]
last_close = float(raw_df["Close"].iloc[-1])

fig_fut = go.Figure()
# show last 30 actual days for context
ctx_dates  = raw_df.index[-30:]
ctx_prices = raw_df["Close"].values[-30:]
fig_fut.add_trace(go.Scatter(
    x=ctx_dates, y=ctx_prices, mode="lines",
    name="Recent Actual", line=dict(color="#00d4ff", width=2)
))
fig_fut.add_trace(go.Scatter(
    x=[ctx_dates[-1], fd[0]], y=[ctx_prices[-1], fp[0]],
    mode="lines", line=dict(color="#ff6b35", dash="dot"), showlegend=False
))
fig_fut.add_trace(go.Scatter(
    x=fd, y=fp, mode="lines+markers+text",
    name="Forecast",
    line=dict(color="#ff6b35", width=2.5, dash="dot"),
    marker=dict(size=8, symbol="diamond"),
    text=[fmt(v) for v in fp],
    textposition="top center",
    textfont=dict(size=11, color="#ffcc44")
))
fig_fut.update_layout(
    template="plotly_dark", height=370,
    margin=dict(l=10,r=10,t=30,b=10),
    legend=dict(orientation="h",y=1.05),
    yaxis_title="Price (₹)"
)
st.plotly_chart(fig_fut, use_container_width=True)

# forecast table
forecast_df = pd.DataFrame({
    "Date":           [d.strftime("%a, %d %b %Y") for d in fd],
    "Predicted Price":[ fmt(v) for v in fp],
    "Change vs Today":[ f"{(v-last_close)/last_close*100:+.2f}%" for v in fp]
})
st.dataframe(forecast_df.set_index("Date"), use_container_width=True)

# ─────────────────── ACTUAL vs PREDICTED ───────────
st.subheader("📉 Actual vs Predicted Close Price (Test Set)")
fig_pred = go.Figure()
fig_pred.add_trace(go.Scatter(
    x=res["test_dates"], y=res["y_actual"],
    mode="lines", name="Actual", line=dict(color="#00d4ff", width=2)
))
fig_pred.add_trace(go.Scatter(
    x=res["test_dates"], y=res["y_pred"],
    mode="lines", name=f"Predicted ({model_choice})",
    line=dict(color="#ff6b35", width=2, dash="dot")
))
fig_pred.update_layout(
    template="plotly_dark", height=380,
    margin=dict(l=10,r=10,t=30,b=10),
    legend=dict(orientation="h",y=1.05),
    yaxis_title="Price (₹)"
)
st.plotly_chart(fig_pred, use_container_width=True)

# ─────────────────── CANDLESTICK ───────────────────
st.subheader(f"🕯️ {stock_name} Candlestick — Live 3Y")
fig_c = make_subplots(rows=2, cols=1, shared_xaxes=True,
                      row_heights=[0.75, 0.25], vertical_spacing=0.02)

fig_c.add_trace(go.Candlestick(
    x=raw_df.index,
    open=raw_df["Open"], high=raw_df["High"],
    low=raw_df["Low"],   close=raw_df["Close"],
    name="OHLC",
    increasing_line_color="#26a69a",
    decreasing_line_color="#ef5350"
), row=1, col=1)
fig_c.add_trace(go.Scatter(x=raw_df.index, y=raw_df["EMA20"],
    line=dict(color="orange",width=1.2),name="EMA 20"), row=1, col=1)
fig_c.add_trace(go.Scatter(x=raw_df.index, y=raw_df["EMA50"],
    line=dict(color="#5b7fff",width=1.2),name="EMA 50"), row=1, col=1)
fig_c.add_trace(go.Scatter(x=raw_df.index, y=raw_df["BB_up"],
    line=dict(color="rgba(180,180,180,0.6)",width=0.8,dash="dash"),name="BB Upper"), row=1, col=1)
fig_c.add_trace(go.Scatter(x=raw_df.index, y=raw_df["BB_lo"],
    line=dict(color="rgba(180,180,180,0.6)",width=0.8,dash="dash"),
    fill="tonexty", fillcolor="rgba(128,128,128,0.07)", name="BB Lower"), row=1, col=1)

colors = ["#26a69a" if c >= o else "#ef5350"
          for c,o in zip(raw_df["Close"], raw_df["Open"])]
fig_c.add_trace(go.Bar(x=raw_df.index,y=raw_df["Volume"],name="Volume",
    marker_color=colors, showlegend=False), row=2, col=1)

fig_c.update_layout(template="plotly_dark", height=550,
    margin=dict(l=10,r=10,t=30,b=10),
    xaxis_rangeslider_visible=False,
    legend=dict(orientation="h",y=1.02))
fig_c.update_yaxes(title_text="Price (₹)", row=1, col=1)
fig_c.update_yaxes(title_text="Volume",     row=2, col=1)
st.plotly_chart(fig_c, use_container_width=True)

# ─────────────────── INDICATORS PANEL ──────────────
st.subheader("📊 Technical Indicators")
t1,t2,t3 = st.columns(3)

with t1:
    fg = go.Figure()
    fg.add_trace(go.Scatter(x=raw_df.index, y=raw_df["RSI"],
        line=dict(color="#ff6b35"), name="RSI", fill="tozeroy",
        fillcolor="rgba(255,107,53,0.08)"))
    fg.add_hline(y=70, line_dash="dot", line_color="red",   annotation_text="70", annotation_position="right")
    fg.add_hline(y=30, line_dash="dot", line_color="lime",  annotation_text="30", annotation_position="right")
    fg.update_layout(template="plotly_dark", height=260, title="RSI (14)",
        margin=dict(l=5,r=5,t=40,b=5), showlegend=False)
    st.plotly_chart(fg, use_container_width=True)

with t2:
    fg2 = go.Figure()
    fg2.add_trace(go.Scatter(x=raw_df.index, y=raw_df["MACD"],
        line=dict(color="#00d4ff"), name="MACD"))
    fg2.add_trace(go.Scatter(x=raw_df.index, y=raw_df["Signal"],
        line=dict(color="#ffcc44",dash="dot"), name="Signal"))
    fg2.add_trace(go.Bar(x=raw_df.index,
        y=raw_df["MACD"]-raw_df["Signal"],
        name="Histogram", marker_color="rgba(162,155,254,0.4)"))
    fg2.update_layout(template="plotly_dark", height=260, title="MACD",
        margin=dict(l=5,r=5,t=40,b=5),
        legend=dict(orientation="h",y=1.15,font=dict(size=10)))
    st.plotly_chart(fg2, use_container_width=True)

with t3:
    fg3 = go.Figure()
    fg3.add_trace(go.Scatter(x=raw_df.index, y=raw_df["OBV"],
        line=dict(color="#a29bfe"), name="OBV",
        fill="tozeroy", fillcolor="rgba(162,155,254,0.1)"))
    fg3.update_layout(template="plotly_dark", height=260, title="OBV",
        margin=dict(l=5,r=5,t=40,b=5), showlegend=False)
    st.plotly_chart(fg3, use_container_width=True)

# ─────────────────── MODEL COMPARISON ──────────────
st.subheader("🔬 All 3 Models — Side-by-Side")
rows=[]
for mn, mr in all_res.items():
    m = mr["metrics"]
    rows.append({"Model":mn,"RMSE":fmt(m["RMSE"]),"MAE":fmt(m["MAE"]),
                 "MAPE":f"{m['MAPE']:.2f}%","R²":f"{m['R²']:.3f}",
                 "Dir %":f"{m['Dir %']:.1f}%"})
st.dataframe(pd.DataFrame(rows).set_index("Model"), use_container_width=True)

# ─────────────────── LIVE PRICE SUMMARY ────────────
st.subheader(f"📋 Live {stock_name} Snapshot")
last = raw_df.iloc[-1]
prev = raw_df.iloc[-2]
change     = float(last["Close"] - prev["Close"])
change_pct = change / float(prev["Close"]) * 100
arrow = "🟢" if change >= 0 else "🔴"

s1,s2,s3,s4,s5,s6 = st.columns(6)
s1.metric("Last Close",   fmt(float(last["Close"])),  f"{arrow} {change:+.2f} ({change_pct:+.2f}%)")
s2.metric("Day High",     fmt(float(last["High"])))
s3.metric("Day Low",      fmt(float(last["Low"])))
s4.metric("RSI",          f"{float(last['RSI']):.1f}")
s5.metric("ATR",          fmt(float(last["ATR"])))
s6.metric("EMA 20",       fmt(float(last["EMA20"])))

st.caption(f"Data: Yahoo Finance (NSE). Epochs: {EPOCHS}. Models retrain once per session. Forecast by iterative 1-step rollout.")
