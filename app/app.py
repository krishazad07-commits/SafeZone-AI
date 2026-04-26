import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import random
from datetime import datetime

st.set_page_config(
    page_title="SafeZone AI | Smart Safety Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

with open("app/styles.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

AREA_COORDS = {
    "Chandkheda": (23.110, 72.580), "Gota": (23.090, 72.540),
    "Bopal": (23.030, 72.470),      "Thaltej": (23.050, 72.510),
    "Naroda": (23.070, 72.670),     "Shilaj": (23.080, 72.460),
    "Science City": (23.080, 72.500), "Ognaj": (23.120, 72.480),
    "Chiloda": (23.150, 72.650),    "Gurukul": (23.040, 72.530),
}
CRIME_COLORS = {"Theft": "#00d4ff", "Robbery": "#ff4757", "Assault": "#ffa502", "Burglary": "#a29bfe"}
PLOTLY_LAYOUT = dict(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                     font=dict(color="white", family="Inter, sans-serif"), margin=dict(t=10, b=10, l=10, r=10))

@st.cache_data
def load_data():
    df = pd.read_csv("data/crime_data.csv")
    df["date"] = pd.to_datetime(df["date"])
    df["time_of_day"] = df["hour"].apply(lambda h: "Morning" if 6<=h<12 else ("Afternoon" if 12<=h<18 else "Night"))
    df["day_of_week"] = df["date"].dt.day_name()
    return df

df = load_data()

@st.cache_resource
def train_model(data):
    area_enc, time_enc = LabelEncoder(), LabelEncoder()
    d = data.copy()
    d["area_enc"] = area_enc.fit_transform(d["area"])
    d["time_enc"] = time_enc.fit_transform(d["time_of_day"])
    counts = d.groupby(["area","time_of_day"]).size().reset_index(name="cnt")
    counts["risk"] = (counts["cnt"] > counts["cnt"].median()).astype(int)
    merged = d.merge(counts[["area","time_of_day","risk"]], on=["area","time_of_day"])
    clf = LogisticRegression(max_iter=1000)
    clf.fit(merged[["area_enc","time_enc"]], merged["risk"])
    return clf, area_enc, time_enc

model, area_enc, time_enc = train_model(df)

# ── SIDEBAR ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-header">
        <div class="shield-icon">🛡️</div>
        <div class="sidebar-title">SafeZone AI</div>
        <div class="sidebar-subtitle">SMART PUBLIC SAFETY</div>
        <div class="live-badge"><span class="dot"></span>LIVE MONITORING</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<p class="filter-label">FILTERS</p>', unsafe_allow_html=True)
    selected_areas  = st.multiselect("Areas",       sorted(df["area"].unique()),       default=sorted(df["area"].unique()))
    selected_crimes = st.multiselect("Crime Types", sorted(df["crime_type"].unique()), default=sorted(df["crime_type"].unique()))
    selected_times  = st.multiselect("Time of Day", ["Morning","Afternoon","Night"],   default=["Morning","Afternoon","Night"])

    st.markdown("---")
    st.markdown('<p class="filter-label">RISK PREDICTOR</p>', unsafe_allow_html=True)
    pred_area = st.selectbox("Area", sorted(df["area"].unique()), key="pred_area")
    pred_time = st.selectbox("Time", ["Morning","Afternoon","Night"], key="pred_time")

    if st.button("🔍 Predict Risk", key="predict_btn"):
        a_enc = area_enc.transform([pred_area])[0]
        t_enc = time_enc.transform([pred_time])[0]
        pred  = model.predict([[a_enc, t_enc]])[0]
        proba = model.predict_proba([[a_enc, t_enc]])[0]
        risk_score = int(proba[1] * 100)
        gauge_color = "#ff4757" if risk_score>60 else ("#ffa502" if risk_score>35 else "#2ed573")

        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number", value=risk_score,
            domain={"x":[0,1],"y":[0,1]},
            title={"text":"Risk Score %","font":{"color":"white","size":13}},
            number={"font":{"color":"white","size":28},"suffix":"%"},
            gauge={"axis":{"range":[0,100],"tickcolor":"rgba(255,255,255,0.4)","tickfont":{"color":"rgba(255,255,255,0.5)"}},
                   "bar":{"color":gauge_color,"thickness":0.25},"bgcolor":"rgba(0,0,0,0)","borderwidth":0,
                   "steps":[{"range":[0,35],"color":"rgba(46,213,115,0.1)"},
                             {"range":[35,60],"color":"rgba(255,165,2,0.1)"},
                             {"range":[60,100],"color":"rgba(255,71,87,0.1)"}]}))
        fig_gauge.update_layout(**PLOTLY_LAYOUT, height=190, margin=dict(t=30,b=0,l=20,r=20))
        st.plotly_chart(fig_gauge, use_container_width=True)

        label = "HIGH RISK" if pred==1 else "SAFE"
        color = "#ff4757" if pred==1 else "#2ed573"
        st.markdown(f'<div class="risk-result-box" style="border-color:{color};background:{color}18;">'
                    f'<span style="color:{color};font-weight:700;font-size:1rem;">{label}</span>'
                    f'<br><small style="color:rgba(255,255,255,0.5);">{pred_area} · {pred_time} · {risk_score}% probability</small>'
                    f'</div>', unsafe_allow_html=True)

# ── FILTERED DATA ─────────────────────────────────────────────
fdf = df[df["area"].isin(selected_areas) & df["crime_type"].isin(selected_crimes) & df["time_of_day"].isin(selected_times)]

# ── HEADER ───────────────────────────────────────────────────
now_str = datetime.now().strftime("%B %d, %Y  ·  %H:%M")
st.markdown(f"""
<div class="main-header">
    <div>
        <div class="header-title">🛡️ SafeZone AI</div>
        <div class="header-sub">Intelligent Crime Analytics &amp; Public Safety Dashboard — Ahmedabad</div>
    </div>
    <div style="text-align:right;">
        <div class="header-time">{now_str}</div>
        <div class="status-pill"><span class="dot"></span>SYSTEM ACTIVE</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── KPI CARDS ────────────────────────────────────────────────
total   = len(fdf)
night   = len(fdf[fdf["time_of_day"]=="Night"])
n_pct   = round(night/total*100) if total else 0
zones   = fdf["area"].nunique()
area_sz = fdf.groupby("area").size()
hotspot = int((area_sz > area_sz.median()).sum()) if len(area_sz) else 0

k1,k2,k3,k4 = st.columns(4)
cards = [
    (k1,"kpi-red",   "🚨",f"{total:,}","Total Incidents", "Filtered selection"),
    (k2,"kpi-orange","⚠️",f"{hotspot}", "High-Risk Zones", "Above median crime rate"),
    (k3,"kpi-blue",  "📍",f"{zones}",   "Areas Monitored", "Active surveillance"),
    (k4,"kpi-purple","🌙",f"{n_pct}%",  "Night Crime Rate", f"{night} incidents after dark"),
]
for col,cls,icon,val,label,sub in cards:
    with col:
        st.markdown(f'<div class="kpi-card {cls}"><div class="kpi-icon">{icon}</div>'
                    f'<div class="kpi-val">{val}</div><div class="kpi-label">{label}</div>'
                    f'<div class="kpi-sub">{sub}</div></div>', unsafe_allow_html=True)

st.markdown("<div style='height:1.25rem'></div>", unsafe_allow_html=True)

# ── TABS ──────────────────────────────────────────────────────
tab1,tab2,tab3,tab4 = st.tabs(["🗺️  Overview","📊  Crime Analysis","🎯  Risk Assessment","🚨  Live Alerts"])

# ── TAB 1: OVERVIEW ──────────────────────────────────────────
with tab1:
    left, right = st.columns([3,2])
    with left:
        st.markdown('<div class="section-title">📍 Crime Incident Map</div>', unsafe_allow_html=True)
        map_df = fdf.copy()
        rng = np.random.default_rng(42)
        map_df["lat"] = map_df["area"].map(lambda a: AREA_COORDS[a][0]) + rng.uniform(-0.006,0.006,len(map_df))
        map_df["lon"] = map_df["area"].map(lambda a: AREA_COORDS[a][1]) + rng.uniform(-0.006,0.006,len(map_df))
        map_df["date_str"] = map_df["date"].dt.strftime("%b %d, %Y")
        fig_map = px.scatter_mapbox(map_df, lat="lat", lon="lon", color="crime_type",
            color_discrete_map=CRIME_COLORS, hover_name="area",
            hover_data={"crime_type":True,"date_str":True,"time_of_day":True,"lat":False,"lon":False},
            zoom=11, center={"lat":23.08,"lon":72.56}, mapbox_style="carto-darkmatter")
        fig_map.update_traces(marker=dict(size=8,opacity=0.85))
        fig_map.update_layout(**PLOTLY_LAYOUT, height=420,
            legend=dict(bgcolor="rgba(10,14,26,0.85)",bordercolor="rgba(255,255,255,0.1)",font=dict(color="white"),title_text=""))
        st.plotly_chart(fig_map, use_container_width=True)

    with right:
        st.markdown('<div class="section-title">📊 Crimes by Area</div>', unsafe_allow_html=True)
        area_cnt = fdf.groupby("area").size().sort_values().reset_index(name="count")
        fig_bar = go.Figure(go.Bar(x=area_cnt["count"], y=area_cnt["area"], orientation="h",
            marker=dict(color=area_cnt["count"], colorscale=[[0,"#2ed573"],[0.5,"#ffa502"],[1,"#ff4757"]]),
            text=area_cnt["count"], textposition="outside", textfont=dict(color="white",size=11)))
        fig_bar.update_layout(**PLOTLY_LAYOUT, height=420,
            xaxis=dict(showgrid=True,gridcolor="rgba(255,255,255,0.08)",tickfont=dict(color="rgba(255,255,255,0.6)")),
            yaxis=dict(showgrid=False,tickfont=dict(color="white")))
        st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown('<div class="section-title">📅 Crime Trend Over Time</div>', unsafe_allow_html=True)
    trend = fdf.groupby(["date","crime_type"]).size().reset_index(name="count")
    fig_trend = px.line(trend, x="date", y="count", color="crime_type",
        color_discrete_map=CRIME_COLORS, line_shape="spline", markers=True)
    fig_trend.update_traces(marker=dict(size=5))
    fig_trend.update_layout(**PLOTLY_LAYOUT, height=280,
        xaxis=dict(showgrid=True,gridcolor="rgba(255,255,255,0.06)",tickfont=dict(color="rgba(255,255,255,0.6)"),title=""),
        yaxis=dict(showgrid=True,gridcolor="rgba(255,255,255,0.06)",tickfont=dict(color="rgba(255,255,255,0.6)"),title=""),
        legend=dict(bgcolor="rgba(10,14,26,0.85)",font=dict(color="white"),title_text=""), hovermode="x unified")
    st.plotly_chart(fig_trend, use_container_width=True)

# ── TAB 2: CRIME ANALYSIS ─────────────────────────────────────
with tab2:
    c1,c2 = st.columns(2)
    with c1:
        st.markdown('<div class="section-title">🍩 Crime Type Distribution</div>', unsafe_allow_html=True)
        ct = fdf["crime_type"].value_counts().reset_index()
        ct.columns = ["crime_type","count"]
        fig_donut = go.Figure(go.Pie(labels=ct["crime_type"], values=ct["count"], hole=0.62,
            marker=dict(colors=[CRIME_COLORS.get(c,"#fff") for c in ct["crime_type"]],
                        line=dict(color="rgba(0,0,0,0.3)",width=2)),
            textfont=dict(color="white"),
            hovertemplate="<b>%{label}</b><br>Count: %{value}<br>Share: %{percent}<extra></extra>"))
        fig_donut.add_annotation(text=f"<b>{total}</b><br><span style='font-size:11px'>Total</span>",
            x=0.5, y=0.5, showarrow=False, font=dict(size=20,color="white"))
        fig_donut.update_layout(**PLOTLY_LAYOUT
