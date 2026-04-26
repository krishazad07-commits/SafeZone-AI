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
    selected_areas  = st.multiselect("Areas", sorted(df["area"].unique()), default=sorted(df["area"].unique()))
    selected_crimes = st.multiselect("Crime Types", sorted(df["crime_type"].unique()), default=sorted(df["crime_type"].unique()))
    selected_times  = st.multiselect("Time of Day", ["Morning","Afternoon","Night"], default=["Morning","Afternoon","Night"])

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

fdf = df[df["area"].isin(selected_areas) & df["crime_type"].isin(selected_crimes) & df["time_of_day"].isin(selected_times)]

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

tab1,tab2,tab3,tab4 = st.tabs(["🗺️  Overview","📊  Crime Analysis","🎯  Risk Assessment","🚨  Live Alerts"])

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
        fig_donut.update_layout(**PLOTLY_LAYOUT, height=360,
            legend=dict(bgcolor="rgba(10,14,26,0.85)",font=dict(color="white"),title_text=""))
        st.plotly_chart(fig_donut, use_container_width=True)

    with c2:
        st.markdown('<div class="section-title">⏰ Crimes by Time of Day</div>', unsafe_allow_html=True)
        td = fdf.groupby(["time_of_day","crime_type"]).size().reset_index(name="count")
        fig_tod = px.bar(td, x="time_of_day", y="count", color="crime_type",
            color_discrete_map=CRIME_COLORS,
            category_orders={"time_of_day":["Morning","Afternoon","Night"]}, barmode="group")
        fig_tod.update_layout(**PLOTLY_LAYOUT, height=360,
            xaxis=dict(showgrid=False,tickfont=dict(color="white"),title=""),
            yaxis=dict(showgrid=True,gridcolor="rgba(255,255,255,0.06)",tickfont=dict(color="rgba(255,255,255,0.6)"),title=""),
            legend=dict(bgcolor="rgba(10,14,26,0.85)",font=dict(color="white"),title_text=""), bargap=0.2)
        st.plotly_chart(fig_tod, use_container_width=True)

    st.markdown('<div class="section-title">🔥 Crime Heatmap: Area × Time of Day</div>', unsafe_allow_html=True)
    if len(fdf):
        pivot = fdf.groupby(["area","time_of_day"]).size().reset_index(name="count")
        pivot_tbl = pivot.pivot(index="area", columns="time_of_day", values="count").fillna(0)
        col_order = [c for c in ["Morning","Afternoon","Night"] if c in pivot_tbl.columns]
        pivot_tbl = pivot_tbl[col_order]
        fig_heat = go.Figure(go.Heatmap(z=pivot_tbl.values, x=pivot_tbl.columns.tolist(),
            y=pivot_tbl.index.tolist(),
            colorscale=[[0,"#0d1f37"],[0.35,"#1a4f7a"],[0.65,"#f39c12"],[1,"#e74c3c"]],
            text=pivot_tbl.values.astype(int), texttemplate="%{text}",
            textfont=dict(color="white",size=14), showscale=True,
            colorbar=dict(tickfont=dict(color="white"),title=dict(text="Crimes",font=dict(color="white"))),
            hoverongaps=False))
        fig_heat.update_layout(**PLOTLY_LAYOUT, height=340,
            xaxis=dict(tickfont=dict(color="white"),title=""),
            yaxis=dict(tickfont=dict(color="white"),title=""))
        st.plotly_chart(fig_heat, use_container_width=True)

    st.markdown('<div class="section-title">📅 Crime by Day of Week</div>', unsafe_allow_html=True)
    dow_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    dow = fdf.groupby("day_of_week").size().reindex(dow_order, fill_value=0).reset_index()
    dow.columns = ["day","count"]
    fig_dow = go.Figure(go.Bar(x=dow["day"], y=dow["count"],
        marker=dict(color=dow["count"], colorscale=[[0,"#2ed573"],[1,"#ff4757"]], line=dict(width=0)),
        text=dow["count"], textposition="outside", textfont=dict(color="white",size=11)))
    fig_dow.update_layout(**PLOTLY_LAYOUT, height=280,
        xaxis=dict(showgrid=False,tickfont=dict(color="white"),title=""),
        yaxis=dict(showgrid=True,gridcolor="rgba(255,255,255,0.06)",tickfont=dict(color="rgba(255,255,255,0.6)"),title=""))
    st.plotly_chart(fig_dow, use_container_width=True)

with tab3:
    area_risk = fdf.groupby("area").agg(total=("crime_id","count"), types=("crime_type","nunique")).reset_index()
    area_risk["score"] = (area_risk["total"] / area_risk["total"].max() * 100).round(1)
    area_risk = area_risk.sort_values("score", ascending=False).reset_index(drop=True)
    area_risk["rank"] = range(1, len(area_risk) + 1)

    rank_col, radar_col = st.columns([1,1])
    with rank_col:
        st.markdown('<div class="section-title">🏆 Area Risk Rankings</div>', unsafe_allow_html=True)
        for _, row in area_risk.iterrows():
            color = "#ff4757" if row["score"]>70 else ("#ffa502" if row["score"]>40 else "#2ed573")
            level = "🔴 HIGH" if row["score"]>70 else ("🟡 MED" if row["score"]>40 else "🟢 LOW")
            st.markdown(f'<div class="rank-row"><span class="rank-num">#{int(row["rank"])}</span>'
                        f'<span class="rank-name">{row["area"]}</span>'
                        f'<div class="rank-bar-wrap"><div class="rank-bar" '
                        f'style="width:{row["score"]:.0f}%;background:{color};box-shadow:0 0 8px {color}66;"></div></div>'
                        f'<span class="rank-score" style="color:{color};">{row["score"]:.0f}</span>'
                        f'<span class="rank-level">{level}</span></div>', unsafe_allow_html=True)

    with radar_col:
        st.markdown('<div class="section-title">🕸️ Crime Profile Radar (Top 5 Areas)</div>', unsafe_allow_html=True)
        top5 = area_risk.head(5)["area"].tolist()
        categories = sorted(fdf["crime_type"].unique())
        radar_colors = ["#00d4ff","#ff4757","#2ed573","#ffa502","#a29bfe"]
        fig_radar = go.Figure()
        for i, area in enumerate(top5):
            vals = [len(fdf[(fdf["area"]==area) & (fdf["crime_type"]==ct)]) for ct in categories]
            vals_closed = vals + [vals[0]]
            cats_closed = categories + [categories[0]]
            fig_radar.add_trace(go.Scatterpolar(r=vals_closed, theta=cats_closed, fill="toself", name=area,
                line=dict(color=radar_colors[i % len(radar_colors)]),
                fillcolor=radar_colors[i % len(radar_colors)], opacity=0.25))
        fig_radar.update_layout(**PLOTLY_LAYOUT, height=420,
            polar=dict(bgcolor="rgba(0,0,0,0)",
                       radialaxis=dict(tickfont=dict(color="rgba(255,255,255,0.4)"),gridcolor="rgba(255,255,255,0.1)"),
                       angularaxis=dict(tickfont=dict(color="white"),gridcolor="rgba(255,255,255,0.1)")),
            legend=dict(bgcolor="rgba(10,14,26,0.85)",font=dict(color="white"),title_text=""))
        st.plotly_chart(fig_radar, use_container_width=True)

    st.markdown('<div class="section-title">🕐 24-Hour Crime Clock</div>', unsafe_allow_html=True)
    hourly = fdf.groupby("hour").size().reset_index(name="count")
    all_hrs = pd.DataFrame({"hour": range(24)})
    hourly = all_hrs.merge(hourly, on="hour", how="left").fillna(0)
    hourly["count"] = hourly["count"].astype(int)
    fig_clock = go.Figure(go.Barpolar(r=hourly["count"], theta=hourly["hour"]*15, width=13,
        marker=dict(color=hourly["count"], colorscale=[[0,"#2ed573"],[0.5,"#ffa502"],[1,"#ff4757"]],
                    showscale=True, colorbar=dict(tickfont=dict(color="white"),title=dict(text="Crimes",font=dict(color="white")))),
        hovertemplate="<b>%{customdata}:00</b><br>Crimes: %{r}<extra></extra>", customdata=hourly["hour"]))
    fig_clock.update_layout(**PLOTLY_LAYOUT, height=440,
        polar=dict(bgcolor="rgba(0,0,0,0)",
                   radialaxis=dict(tickfont=dict(color="rgba(255,255,255,0.4)"),gridcolor="rgba(255,255,255,0.08)"),
                   angularaxis=dict(tickvals=[h*15 for h in range(24)], ticktext=[f"{h:02d}:00" for h in range(24)],
                                    tickfont=dict(color="white",size=9), gridcolor="rgba(255,255,255,0.08)",
                                    direction="clockwise", rotation=90)))
    st.plotly_chart(fig_clock, use_container_width=True)

with tab4:
    a_left, a_right = st.columns([1,1])
    with a_left:
        st.markdown('<div class="alert-header-bar"><span class="dot"></span>ALERT MONITORING ACTIVE</div>', unsafe_allow_html=True)
        if st.button("▶ Generate Live Alerts", key="gen_alerts"):
            html_alerts = ""
            for i in range(6):
                a  = random.choice(df["area"].tolist())
                c  = random.choice(df["crime_type"].tolist())
                t  = random.choice(["Morning","Afternoon","Night"])
                sv = random.choice(["HIGH","MEDIUM","LOW"])
                sc = {"HIGH":"#ff4757","MEDIUM":"#ffa502","LOW":"#2ed573"}[sv]
                ts = datetime.now().strftime("%H:%M:%S")
                html_alerts += f'<div class="alert-card" style="border-left:4px solid {sc};animation:slideIn 0.4s ease {i*0.12}s both;">' \
                               f'<div class="alert-top"><span style="color:{sc};font-weight:700;font-size:0.72rem;letter-spacing:1.5px;">[{sv}]</span>' \
                               f'<span style="color:rgba(255,255,255,0.35);font-size:0.72rem;">{ts}</span></div>' \
                               f'<div style="color:#fff;font-weight:600;font-size:0.9rem;margin:0.2rem 0;">🚨 {c} Reported</div>' \
                               f'<div style="color:rgba(255,255,255,0.5);font-size:0.8rem;">📍 {a} &nbsp;·&nbsp; ⏰ {t}</div></div>'
            st.markdown(html_alerts, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🚒 Send Emergency Alert", key="emergency"):
            st.markdown('<div class="emergency-box"><div style="font-size:2rem;">🚨</div><div>'
                        '<div style="color:#ff4757;font-weight:700;font-size:1rem;letter-spacing:1px;">EMERGENCY BROADCAST SENT</div>'
                        '<div style="color:rgba(255,255,255,0.55);font-size:0.82rem;margin-top:0.2rem;">All units notified · Response teams dispatched</div>'
                        '</div></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-title">🧠 Smart Insights</div>', unsafe_allow_html=True)
        if len(fdf):
            az = fdf.groupby("area").size()
            most_d = az.idxmax()
            least_d = az.idxmin()
            peak_h = int(fdf["hour"].mode()[0])
            top_crime = fdf["crime_type"].value_counts().idxmax()
            st.markdown(f'<div class="insight-card danger-card">🔴 Most Dangerous: <strong>{most_d}</strong> ({az[most_d]} incidents)</div>'
                        f'<div class="insight-card safe-card">🟢 Safest Area: <strong>{least_d}</strong> ({az[least_d]} incidents)</div>'
                        f'<div class="insight-card info-card">⏰ Peak Hour: <strong>{peak_h:02d}:00 – {(peak_h+1):02d}:00</strong></div>'
                        f'<div class="insight-card warn-card">⚠️ Top Crime: <strong>{top_crime}</strong></div>', unsafe_allow_html=True)

    with a_right:
        st.markdown('<div class="section-title">📋 Recent Incidents</div>', unsafe_allow_html=True)
        recent = fdf.tail(12)[["area","crime_type","time_of_day","date"]].copy()
        recent["date"] = recent["date"].dt.strftime("%b %d")
        recent.columns = ["Area","Crime Type","Time","Date"]
        st.dataframe(recent, use_container_width=True, hide_index=True)

        st.markdown('<div class="section-title" style="margin-top:1rem;">📊 Alert Breakdown</div>', unsafe_allow_html=True)
        ac = fdf["crime_type"].value_counts().reset_index()
        ac.columns = ["type","count"]
        fig_ac = go.Figure(go.Bar(x=ac["type"], y=ac["count"],
            marker=dict(color=[CRIME_COLORS.get(t,"#fff") for t in ac["type"]], line=dict(width=0)),
            text=ac["count"], textposition="outside", textfont=dict(color="white",size=11)))
        fig_ac.update_layout(**PLOTLY_LAYOUT, height=260,
            xaxis=dict(showgrid=False,tickfont=dict(color="white"),title=""),
            yaxis=dict(showgrid=True,gridcolor="rgba(255,255,255,0.06)",tickfont=dict(color="rgba(255,255,255,0.5)"),title=""))
        st.plotly_chart(fig_ac, use_container_width=True)

st.markdown('<div class="footer"><span>SafeZone AI &nbsp;·&nbsp; Powered by Machine Learning</span>'
            '<span>🛡️ Protecting Communities with Data Intelligence</span>'
            '<span>Ahmedabad Smart City Initiative · 2024</span></div>', unsafe_allow_html=True)
