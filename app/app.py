import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import time
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression

# ------------------------
# LOAD CSS
# ------------------------
def load_css():
    with open("app/styles.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# ------------------------
# PAGE CONFIG
# ------------------------
st.set_page_config(page_title="SafeZone AI", layout="wide")

# ------------------------
# LOAD DATA
# ------------------------
df = pd.read_csv('data/crime_data.csv')

# ------------------------
# TITLE
# ------------------------
st.markdown('<p class="title">🚨 SafeZone AI Dashboard</p>', unsafe_allow_html=True)

# ------------------------
# PROCESSING
# ------------------------
def time_category(hour):
    if 6 <= hour < 12:
        return "Morning"
    elif 12 <= hour < 18:
        return "Afternoon"
    else:
        return "Night"

df['time_category'] = df['hour'].apply(time_category)

area_crime = df.groupby('area').size().reset_index(name='crime_count')
df = df.merge(area_crime, on='area')

# Balanced risk
df = df.sort_values(by='crime_count')
df['risk'] = 0
df.iloc[len(df)//2:, df.columns.get_loc('risk')] = 1

# ------------------------
# KPI VALUES (IMPORTANT FIX)
# ------------------------
total_crimes = len(df)
high_risk = df['risk'].sum()
areas = df['area'].nunique()

# ------------------------
# KPI CARDS
# ------------------------
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(
        f'<div class="metric-card"><h2>{total_crimes}</h2><p>Total Crimes</p></div>',
        unsafe_allow_html=True
    )

with col2:
    st.markdown(
        f'<div class="metric-card"><h2>{int(high_risk)}</h2><p>High Risk</p></div>',
        unsafe_allow_html=True
    )

with col3:
    st.markdown(
        f'<div class="metric-card"><h2>{areas}</h2><p>Areas Covered</p></div>',
        unsafe_allow_html=True
    )

st.markdown("---")

# ------------------------
# MAP DATA
# ------------------------
area_coords = {
    "Chandkheda": [23.110, 72.580],
    "Gota": [23.090, 72.540],
    "Bopal": [23.030, 72.470],
    "Thaltej": [23.050, 72.510],
    "Naroda": [23.070, 72.670],
    "Shilaj": [23.080, 72.460],
    "Science City": [23.080, 72.500],
    "Ognaj": [23.120, 72.480],
    "Chiloda": [23.150, 72.650],
    "Gurukul": [23.040, 72.530]
}

df['lat'] = df['area'].map(lambda x: area_coords[x][0])
df['lon'] = df['area'].map(lambda x: area_coords[x][1])

# ------------------------
# MODEL
# ------------------------
le_area = LabelEncoder()
le_time = LabelEncoder()

df['area_encoded'] = le_area.fit_transform(df['area'])
df['time_encoded'] = le_time.fit_transform(df['time_category'])

X = df[['area_encoded', 'time_encoded']]
y = df['risk']

model = LogisticRegression()
model.fit(X, y)

# ------------------------
# SIDEBAR
# ------------------------
st.sidebar.title("🔍 Analyze Area")

area = st.sidebar.selectbox("Area", df['area'].unique())
time_input = st.sidebar.selectbox("Time", ["Morning", "Afternoon", "Night"])

area_val = le_area.transform([area])[0]
time_val = le_time.transform([time_input])[0]

# FIXED prediction warning
input_df = pd.DataFrame([[area_val, time_val]],
                        columns=['area_encoded', 'time_encoded'])

prediction = model.predict(input_df)

crime_count = len(df[df['area'] == area])
safety_score = max(0, 100 - crime_count)

# ------------------------
# RESULT
# ------------------------
st.subheader("📊 Area Safety Analysis")

col1, col2 = st.columns(2)

col1.metric("Safety Score", safety_score)

if prediction[0] == 1:
    col2.error("🔴 High Risk Area")
else:
    col2.success("🟢 Safe Area")

st.markdown("---")

# ------------------------
# SMART INSIGHTS
# ------------------------
st.subheader("🧠 Smart Insights")

most_dangerous = df.groupby('area').size().idxmax()
least_dangerous = df.groupby('area').size().idxmin()

st.warning(f"🚨 Most Dangerous Area: {most_dangerous}")
st.success(f"✅ Safest Area: {least_dangerous}")

# ------------------------
# MAP
# ------------------------
st.subheader("📍 Crime Map (Ahmedabad)")
st.map(df[['lat', 'lon']])

# ------------------------
# LIVE ALERTS
# ------------------------
st.subheader("🔔 Live Crime Alerts")

if st.button("Start Live Alerts"):
    placeholder = st.empty()
    
    for i in range(5):
        random_area = np.random.choice(df['area'])
        random_crime = np.random.choice(df['crime_type'])
        
        placeholder.warning(f"🚨 {random_crime} in {random_area}")
        time.sleep(1)

st.markdown("---")

# ------------------------
# VISUALS
# ------------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Crime by Area")
    fig, ax = plt.subplots()
    df['area'].value_counts().plot(kind='bar', ax=ax)
    st.pyplot(fig)

with col2:
    st.subheader("📊 Crime Types")
    fig, ax = plt.subplots()
    df['crime_type'].value_counts().plot(kind='pie', autopct='%1.1f%%', ax=ax)
    st.pyplot(fig)

# ------------------------
# HEATMAP
# ------------------------
st.subheader("🔥 Crime Heatmap")

pivot = df.pivot_table(values='crime_id', index='area', columns='time_category', aggfunc='count')

fig, ax = plt.subplots()
sns.heatmap(pivot, annot=True, cmap="Reds", ax=ax)
st.pyplot(fig)

# ------------------------
# TREND GRAPH
# ------------------------
st.subheader("📊 Crime Trend Over Time")

df['date'] = pd.to_datetime(df['date'])
trend = df.groupby('date').size()

fig, ax = plt.subplots()
trend.plot(label="Actual")
trend.rolling(3).mean().plot(label="Trend", linestyle="--")

ax.legend()
st.pyplot(fig)

# ------------------------
# EMERGENCY BUTTON
# ------------------------
st.subheader("🚨 Emergency")

if st.button("Send Emergency Alert"):
    st.error("🚨 Alert Sent to Authorities!")