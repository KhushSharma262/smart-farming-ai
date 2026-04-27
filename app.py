import streamlit as st
import pandas as pd
import numpy as np
import requests
import time
import random
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import plotly.graph_objects as go

st.set_page_config(
    page_title="AgriMind — Smart Farm AI",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');
:root {
    --green:  #00ff88;
    --amber:  #ffb340;
    --red:    #ff4d6d;
    --blue:   #38bdf8;
    --bg:     #0a0f0d;
    --panel:  #111a14;
    --border: #1f3028;
    --text:   #d4e8da;
    --muted:  #5a7a63;
}
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; background-color: var(--bg); color: var(--text); }
.stApp { background-color: var(--bg); }
h1, h2, h3, .mono { font-family: 'Space Mono', monospace; }
.metric-card { background: var(--panel); border: 1px solid var(--border); border-radius: 12px; padding: 20px 24px; text-align: center; position: relative; overflow: hidden; }
.metric-card::before { content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px; background: linear-gradient(90deg, var(--green), transparent); }
.metric-label { font-size: 11px; letter-spacing: 2px; text-transform: uppercase; color: var(--muted); margin-bottom: 8px; font-family: 'Space Mono', monospace; }
.metric-value { font-size: 36px; font-weight: 700; font-family: 'Space Mono', monospace; color: var(--green); line-height: 1; }
.metric-unit  { font-size: 13px; color: var(--muted); margin-top: 4px; }
.metric-card.warn   .metric-value { color: var(--amber); }
.metric-card.warn::before         { background: linear-gradient(90deg, var(--amber), transparent); }
.metric-card.danger .metric-value { color: var(--red); }
.metric-card.danger::before       { background: linear-gradient(90deg, var(--red), transparent); }
.metric-card.info   .metric-value { color: var(--blue); }
.metric-card.info::before         { background: linear-gradient(90deg, var(--blue), transparent); }
.alert-box    { border-radius: 10px; padding: 14px 20px; margin: 8px 0; font-size: 14px; font-weight: 500; display: flex; align-items: center; gap: 10px; }
.alert-danger { background: rgba(255,77,109,0.12); border: 1px solid rgba(255,77,109,0.4); color: #ff8099; }
.alert-warn   { background: rgba(255,179,64,0.12); border: 1px solid rgba(255,179,64,0.4); color: #ffc966; }
.alert-ok     { background: rgba(0,255,136,0.08);  border: 1px solid rgba(0,255,136,0.3);  color: #66ffb3; }
.section-title { font-family: 'Space Mono', monospace; font-size: 11px; letter-spacing: 3px; text-transform: uppercase; color: var(--muted); padding-bottom: 8px; border-bottom: 1px solid var(--border); margin-bottom: 16px; }
.crop-result { background: linear-gradient(135deg, rgba(0,255,136,0.08), rgba(0,255,136,0.02)); border: 1px solid rgba(0,255,136,0.3); border-radius: 16px; padding: 28px; text-align: center; }
.crop-name   { font-family: 'Space Mono', monospace; font-size: 42px; font-weight: 700; color: var(--green); text-transform: uppercase; letter-spacing: 4px; }
.rec-card         { background: var(--panel); border: 1px solid var(--border); border-left: 3px solid var(--green); border-radius: 8px; padding: 12px 16px; margin: 6px 0; font-size: 14px; }
.rec-card.warn    { border-left-color: var(--amber); }
.rec-card.danger  { border-left-color: var(--red); }
.cost-panel { background: var(--panel); border: 1px solid var(--border); border-radius: 12px; padding: 20px; }
.cost-row   { display: flex; justify-content: space-between; align-items: center; padding: 8px 0; border-bottom: 1px solid var(--border); font-size: 14px; }
.cost-row:last-child { border-bottom: none; font-weight: 600; color: var(--green); }
.live-dot { display: inline-block; width: 8px; height: 8px; background: var(--green); border-radius: 50%; animation: pulse 1.5s infinite; margin-right: 6px; }
@keyframes pulse { 0%,100% { opacity:1; transform:scale(1); } 50% { opacity:0.4; transform:scale(1.4); } }
.stButton>button { background: var(--green) !important; color: #0a0f0d !important; font-family: 'Space Mono', monospace !important; font-weight: 700 !important; border: none !important; border-radius: 8px !important; padding: 10px 28px !important; font-size: 13px !important; letter-spacing: 1px !important; width: 100% !important; }
.stButton>button:hover { opacity: 0.85 !important; }
footer { visibility: hidden; } #MainMenu { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── SESSION STATE ──────────────────────────────────────────
for key in ["temp_history","humid_history","moisture_history","time_history","refresh_count"]:
    if key not in st.session_state:
        st.session_state[key] = [] if key != "refresh_count" else 0

# ── ML MODEL ───────────────────────────────────────────────
@st.cache_resource
def load_model():
    df = pd.read_csv("crops.csv")
    X  = df[["N","P","K","temperature","humidity","ph","rainfall"]]
    y  = df["crop"]
    le = LabelEncoder()
    ye = le.fit_transform(y)
    Xtr, Xte, ytr, yte = train_test_split(X, ye, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    clf.fit(Xtr, ytr)
    acc = accuracy_score(yte, clf.predict(Xte))
    return clf, le, acc

model, le, accuracy = load_model()

# ── WEATHERAPI.COM INTEGRATION ─────────────────────────────
def fetch_weather(api_key: str, city: str):
    if not api_key or api_key.strip() == "":
        raise ValueError("no key")
    url = f"http://api.weatherapi.com/v1/current.json?key={api_key}&q={city}&aqi=no"
    r   = requests.get(url, timeout=6)
    r.raise_for_status()
    d   = r.json()
    return {
        "temp_c"     : float(d["current"]["temp_c"]),
        "humidity"   : float(d["current"]["humidity"]),
        "wind_kph"   : float(d["current"]["wind_kph"]),
        "cloud"      : float(d["current"]["cloud"]),
        "condition"  : d["current"]["condition"]["text"],
        "feelslike_c": float(d["current"]["feelslike_c"]),
        "uv"         : float(d["current"]["uv"]),
        "city"       : d["location"]["name"],
        "country"    : d["location"]["country"],
    }

def get_sensor_data(api_key: str, city: str):
    source = "📡 Simulated"
    live   = {}
    try:
        live   = fetch_weather(api_key, city)
        source = f"🌐 WeatherAPI · {live['condition']}"
        temp   = live["temp_c"]
        humid  = live["humidity"]
    except Exception:
        base_temp  = 28.0 + np.sin(time.time() / 300) * 4
        base_humid = 65.0 + np.cos(time.time() / 240) * 10
        temp   = round(base_temp  + random.uniform(-0.5, 0.5), 1)
        humid  = round(base_humid + random.uniform(-1.0, 1.0), 1)

    return {
        "temperature": temp,
        "humidity"   : humid,
        "moisture"   : round(random.uniform(28, 75), 1),
        "health"     : round(random.uniform(60, 98), 1),
        "soil_temp"  : round(temp - random.uniform(2, 5), 1),
        "light_lux"  : round(random.uniform(30000, 95000)),
        "wind_kph"   : live.get("wind_kph", round(random.uniform(5,30),1)),
        "uv_index"   : live.get("uv", round(random.uniform(1,10),1)),
        "source"     : source,
        "ts"         : datetime.now().strftime("%H:%M:%S"),
    }

# ── ALERTS ─────────────────────────────────────────────────
def detect_anomalies(d):
    alerts = []
    if d["temperature"] > 38:
        alerts.append(("danger", f"🔥 CRITICAL: Temperature {d['temperature']}°C — Heat stress risk!"))
    elif d["temperature"] > 33:
        alerts.append(("warn",   f"⚠️ High temperature {d['temperature']}°C — Monitor closely"))
    if d["moisture"] < 35:
        alerts.append(("danger", f"💧 CRITICAL: Soil moisture {d['moisture']}% — Irrigation required!"))
    elif d["moisture"] < 45:
        alerts.append(("warn",   f"⚠️ Low moisture {d['moisture']}% — Consider irrigating"))
    if d["humidity"] < 40:
        alerts.append(("warn",   f"⚠️ Low humidity {d['humidity']}% — Crop stress possible"))
    if d["health"] < 70:
        alerts.append(("danger", f"🌿 CRITICAL: Crop health {d['health']}% — Intervention needed!"))
    if d["uv_index"] > 8:
        alerts.append(("warn",   f"☀️ High UV Index {d['uv_index']} — Shade protection advised"))
    if d["wind_kph"] > 40:
        alerts.append(("warn",   f"💨 Strong winds {d['wind_kph']} kph — Check crop supports"))
    if not alerts:
        alerts.append(("ok", "✅ All parameters within normal range"))
    return alerts

def generate_recommendations(d):
    recs = []
    if d["moisture"] < 40:
        recs.append(("danger", "💧 Increase irrigation immediately — soil moisture critical"))
    elif d["moisture"] < 50:
        recs.append(("warn",   "💧 Schedule irrigation in next 6 hours"))
    else:
        recs.append(("ok",     "💧 Irrigation schedule on track"))
    if d["temperature"] > 35:
        recs.append(("warn",   "🌡️ Apply shade netting or misting to reduce heat load"))
    elif d["temperature"] < 15:
        recs.append(("warn",   "🌡️ Deploy frost protection — temperature dropping"))
    else:
        recs.append(("ok",     "🌡️ Temperature optimal for crop growth"))
    if d["health"] < 75:
        recs.append(("danger", "🌿 Inspect for pests/disease — crop health declining"))
    else:
        recs.append(("ok",     "🌿 Crop health nominal — continue monitoring"))
    if d["humidity"] > 85:
        recs.append(("warn",   "💨 High humidity — fungal risk elevated"))
    if d["uv_index"] > 7:
        recs.append(("warn",   "☀️ High UV — consider UV-protective covers for seedlings"))
    return recs

PLOT_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(17,26,20,0.6)",
    font=dict(family="Space Mono", color="#5a7a63", size=11),
    margin=dict(l=40, r=20, t=30, b=40),
    xaxis=dict(gridcolor="#1f3028", showgrid=True, zeroline=False),
    yaxis=dict(gridcolor="#1f3028", showgrid=True, zeroline=False),
)

# ── SIDEBAR ────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<p class="section-title">⚙ System Config</p>', unsafe_allow_html=True)
    api_key  = st.text_input("WeatherAPI.com Key", value="", type="password", placeholder="Paste your key here")
    city     = st.text_input("Farm Location / City", value="Pune")
    auto_ref = st.toggle("Auto-Refresh (5s)", value=True)
    st.divider()

    st.markdown('<p class="section-title">🌱 Crop Prediction Inputs</p>', unsafe_allow_html=True)
    N          = st.slider("Nitrogen (N)",      10, 120, 70)
    P          = st.slider("Phosphorus (P)",    20,  90, 45)
    K          = st.slider("Potassium (K)",      5,  60, 30)
    ph         = st.slider("Soil pH",           4.0, 9.0, 6.5)
    rainfall   = st.slider("Rainfall (mm)",     30.0, 300.0, 180.0)
    predict_btn = st.button("🌾 PREDICT CROP")
    st.divider()

    st.markdown('<p class="section-title">💸 Cost Calculator</p>', unsafe_allow_html=True)
    water_liters = st.number_input("Water Used (L/day)",       min_value=0, value=5000, step=100)
    fert_kg      = st.number_input("Fertilizer Used (kg/day)", min_value=0, value=20,   step=1)
    water_rate   = st.number_input("Water Cost (₹/L)",         min_value=0.0, value=0.05, step=0.01, format="%.2f")
    fert_rate    = st.number_input("Fertilizer Cost (₹/kg)",   min_value=0.0, value=25.0, step=1.0)
    calc_btn     = st.button("💰 CALCULATE COSTS")

# ── FETCH DATA ─────────────────────────────────────────────
data = get_sensor_data(api_key, city)
now  = datetime.now()

MAX_HISTORY = 30
st.session_state.temp_history.append(data["temperature"])
st.session_state.humid_history.append(data["humidity"])
st.session_state.moisture_history.append(data["moisture"])
st.session_state.time_history.append(now)
for key in ["temp_history","humid_history","moisture_history","time_history"]:
    if len(st.session_state[key]) > MAX_HISTORY:
        st.session_state[key] = st.session_state[key][-MAX_HISTORY:]
st.session_state.refresh_count += 1

# ── HEADER ─────────────────────────────────────────────────
col_h1, col_h2 = st.columns([3,1])
with col_h1:
    st.markdown("""
    <h1 style="font-family:'Space Mono',monospace;font-size:28px;color:#00ff88;margin:0;letter-spacing:2px;">🌾 AGRIMIND</h1>
    <p style="color:#5a7a63;font-size:13px;margin:2px 0 0 0;font-family:'Space Mono',monospace;letter-spacing:1px;">AI-POWERED SMART FARM MONITORING SYSTEM</p>
    """, unsafe_allow_html=True)
with col_h2:
    st.markdown(f"""
    <div style="text-align:right;padding-top:8px;">
        <span class="live-dot"></span>
        <span style="font-family:'Space Mono',monospace;font-size:11px;color:#5a7a63;">
            {data['source']}<br>{data['ts']} &nbsp;|&nbsp; Refresh #{st.session_state.refresh_count}
        </span>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<hr style='border-color:#1f3028;margin:12px 0;'>", unsafe_allow_html=True)

# ── LIVE CARDS ─────────────────────────────────────────────
st.markdown('<p class="section-title">📡 Live Sensor Readings</p>', unsafe_allow_html=True)

def card_class(val, warn_thresh, danger_thresh, invert=False):
    if invert:
        if val < danger_thresh: return "danger"
        if val < warn_thresh:   return "warn"
        return ""
    if val > danger_thresh: return "danger"
    if val > warn_thresh:   return "warn"
    return ""

c1,c2,c3,c4,c5,c6,c7,c8 = st.columns(8)
cards = [
    (c1, "🌡️ Air Temp",      data["temperature"],            "°C",  card_class(data["temperature"], 33, 38)),
    (c2, "💧 Humidity",      data["humidity"],               "%",   card_class(data["humidity"], 85, 92)),
    (c3, "🪱 Soil Moisture", data["moisture"],               "%",   card_class(data["moisture"], 45, 35, invert=True)),
    (c4, "🌿 Crop Health",   data["health"],                 "%",   card_class(data["health"], 75, 60, invert=True)),
    (c5, "🌍 Soil Temp",     data["soil_temp"],              "°C",  card_class(data["soil_temp"], 30, 36)),
    (c6, "☀️ Light",         f'{data["light_lux"]:,}',       "lux", "info"),
    (c7, "💨 Wind",          data["wind_kph"],               "kph", card_class(data["wind_kph"], 30, 45)),
    (c8, "🔆 UV Index",      data["uv_index"],               "",    card_class(data["uv_index"], 6, 8)),
]
for col, label, val, unit, cls in cards:
    with col:
        st.markdown(f"""
        <div class="metric-card {cls}">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{val}</div>
            <div class="metric-unit">{unit}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── ALERTS + RECS ──────────────────────────────────────────
col_al, col_rec = st.columns(2)
with col_al:
    st.markdown('<p class="section-title">🚨 Anomaly Alerts</p>', unsafe_allow_html=True)
    for kind, msg in detect_anomalies(data):
        css = {"danger":"alert-danger","warn":"alert-warn","ok":"alert-ok"}[kind]
        st.markdown(f'<div class="alert-box {css}">{msg}</div>', unsafe_allow_html=True)
with col_rec:
    st.markdown('<p class="section-title">🧠 Smart Recommendations</p>', unsafe_allow_html=True)
    for kind, msg in generate_recommendations(data):
        st.markdown(f'<div class="rec-card {kind}">{msg}</div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── CHARTS ─────────────────────────────────────────────────
st.markdown('<p class="section-title">📊 Live Trend Analysis</p>', unsafe_allow_html=True)
times = [t.strftime("%H:%M:%S") for t in st.session_state.time_history]
ch1, ch2, ch3 = st.columns(3)

with ch1:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=times, y=st.session_state.temp_history,
        mode="lines+markers", line=dict(color="#ff4d6d",width=2), marker=dict(size=4),
        fill="tozeroy", fillcolor="rgba(255,77,109,0.07)"))
    fig.add_hline(y=33, line=dict(color="#ffb340",dash="dot",width=1))
    fig.add_hline(y=38, line=dict(color="#ff4d6d",dash="dot",width=1))
    fig.update_layout(title="Temperature (°C)", height=220, showlegend=False, **PLOT_LAYOUT)
    fig.update_xaxes(tickangle=45, nticks=6)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})

with ch2:
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=times, y=st.session_state.humid_history,
        mode="lines+markers", line=dict(color="#38bdf8",width=2), marker=dict(size=4),
        fill="tozeroy", fillcolor="rgba(56,189,248,0.07)"))
    fig2.update_layout(title="Humidity (%)", height=220, showlegend=False, **PLOT_LAYOUT)
    fig2.update_xaxes(tickangle=45, nticks=6)
    st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar":False})

with ch3:
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=times, y=st.session_state.moisture_history,
        mode="lines+markers", line=dict(color="#00ff88",width=2), marker=dict(size=4),
        fill="tozeroy", fillcolor="rgba(0,255,136,0.07)"))
    fig3.add_hline(y=45, line=dict(color="#ffb340",dash="dot",width=1))
    fig3.add_hline(y=35, line=dict(color="#ff4d6d",dash="dot",width=1))
    fig3.update_layout(title="Soil Moisture (%)", height=220, showlegend=False, **PLOT_LAYOUT)
    fig3.update_xaxes(tickangle=45, nticks=6)
    st.plotly_chart(fig3, use_container_width=True, config={"displayModeBar":False})

st.markdown("<br>", unsafe_allow_html=True)

# ── CROP PREDICTION ────────────────────────────────────────
if predict_btn:
    st.markdown('<p class="section-title">🌱 Crop Prediction Result</p>', unsafe_allow_html=True)
    X_in  = np.array([[N, P, K, data["temperature"], data["humidity"], ph, rainfall]])
    pred  = model.predict(X_in)
    proba = model.predict_proba(X_in)[0]
    crop  = le.inverse_transform(pred)[0]
    conf  = round(max(proba) * 100, 1)
    col_r1, col_r2 = st.columns(2)
    with col_r1:
        st.markdown(f"""
        <div class="crop-result">
            <div class="metric-label" style="margin-bottom:12px;">RECOMMENDED CROP</div>
            <div class="crop-name">{crop}</div>
            <div style="margin-top:14px;font-family:'Space Mono',monospace;color:#5a7a63;font-size:12px;">
                CONFIDENCE &nbsp;<span style="color:#00ff88">{conf}%</span>
                &nbsp;|&nbsp; MODEL ACC &nbsp;<span style="color:#00ff88">{round(accuracy*100,1)}%</span>
            </div>
        </div>""", unsafe_allow_html=True)
    with col_r2:
        proba_df = pd.DataFrame({"Crop":le.classes_,"Probability":proba}).sort_values("Probability",ascending=True).tail(6)
        fig4 = go.Figure(go.Bar(
            x=proba_df["Probability"]*100, y=proba_df["Crop"], orientation="h",
            marker=dict(color=proba_df["Probability"]*100,
                colorscale=[[0,"#1f3028"],[0.5,"#005533"],[1,"#00ff88"]],showscale=False)
        ))
        fig4.update_layout(title="Crop Probability (%)", height=250, xaxis_title=None, yaxis_title=None, **PLOT_LAYOUT)
        st.plotly_chart(fig4, use_container_width=True, config={"displayModeBar":False})
    st.markdown("<br>", unsafe_allow_html=True)

# ── COST INTELLIGENCE ──────────────────────────────────────
if calc_btn:
    st.markdown('<p class="section-title">💸 Cost Intelligence Report</p>', unsafe_allow_html=True)
    water_cost     = water_liters * water_rate
    fert_cost      = fert_kg * fert_rate
    total_daily    = water_cost + fert_cost
    total_monthly  = total_daily * 30
    total_yearly   = total_daily * 365
    opt_daily      = water_cost*0.78 + fert_cost*0.85
    savings_monthly= (total_daily - opt_daily) * 30
    col_c1, col_c2 = st.columns(2)
    with col_c1:
        st.markdown(f"""
        <div class="cost-panel">
            <div class="section-title" style="margin-bottom:12px;">Current Costs</div>
            <div class="cost-row"><span>💧 Water ({water_liters:,} L)</span><span>₹ {water_cost:,.2f}</span></div>
            <div class="cost-row"><span>🧪 Fertilizer ({fert_kg} kg)</span><span>₹ {fert_cost:,.2f}</span></div>
            <div class="cost-row"><span>📅 Daily Total</span><span>₹ {total_daily:,.2f}</span></div>
            <div class="cost-row"><span>📆 Monthly Est.</span><span style="color:#ffb340;">₹ {total_monthly:,.2f}</span></div>
            <div class="cost-row"><span>📈 Annual Est.</span><span style="color:#ff4d6d;">₹ {total_yearly:,.2f}</span></div>
        </div>""", unsafe_allow_html=True)
    with col_c2:
        st.markdown(f"""
        <div class="cost-panel">
            <div class="section-title" style="margin-bottom:12px;">Optimized Scenario (AI)</div>
            <div class="cost-row"><span>💧 Optimized Water (-22%)</span><span>₹ {water_cost*0.78:,.2f}</span></div>
            <div class="cost-row"><span>🧪 Optimized Fertilizer (-15%)</span><span>₹ {fert_cost*0.85:,.2f}</span></div>
            <div class="cost-row"><span>📅 Optimized Daily</span><span>₹ {opt_daily:,.2f}</span></div>
            <div class="cost-row"><span>💰 Monthly Savings</span><span style="color:#00ff88;">₹ {savings_monthly:,.2f}</span></div>
            <div class="cost-row"><span>📊 Annual Savings Est.</span><span style="color:#00ff88;">₹ {savings_monthly*12:,.2f}</span></div>
        </div>""", unsafe_allow_html=True)
    st.markdown(f"""
    <div class="alert-box alert-ok" style="margin-top:12px;">
        🧠 AI Suggestion: Drip irrigation saves ~22% water · Slow-release fertilizers save ~15% · Annual saving: ₹ {savings_monthly*12:,.2f}
    </div>""", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

# ── FOOTER + AUTO-REFRESH ──────────────────────────────────
st.markdown("""
<div style="text-align:center;padding:20px 0 8px;font-family:'Space Mono',monospace;font-size:10px;color:#1f3028;letter-spacing:2px;">
    AGRIMIND v2.1 &nbsp;|&nbsp; WEATHERAPI.COM LIVE DATA &nbsp;|&nbsp; SCIKIT-LEARN + STREAMLIT
</div>""", unsafe_allow_html=True)

if auto_ref:
    time.sleep(5)
    st.rerun()
