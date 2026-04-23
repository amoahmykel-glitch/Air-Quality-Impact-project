# ==========================================
# Air Quality & Health Impact Prediction App
# ==========================================

import streamlit as st
import numpy as np
import pandas as pd
import joblib


# ==========================================
# Page Configuration
# ==========================================

st.set_page_config(
    page_title="Air Quality Health Impact Predictor",
    page_icon="🌍",
    layout="wide"
)

st.title("🌍 Air Quality & Health Impact Classification Dashboard")
st.markdown(
"""
Predict **Health Impact Severity** using environmental and healthcare indicators.

Model Used: **Random Forest Classifier**
"""
)

st.divider()


# ==========================================
# Load Model Safely
# ==========================================

@st.cache_resource
def load_model():
    try:
        model = joblib.load("model/random_forest.pkl")
        scaler = joblib.load("model/scaler.pkl")
        return model, scaler
    except:
        st.error("Model files not found. Please ensure `.pkl` files exist in /model folder.")
        st.stop()

model, scaler = load_model()


# ==========================================
# Sidebar Inputs
# ==========================================

st.sidebar.header("⚙️ Enter Environmental Data")

AQI = st.sidebar.number_input("AQI", 0.0, 500.0, 150.0)

st.sidebar.subheader("Pollution Indicators")

PM10 = st.sidebar.number_input("PM10", 0.0, 300.0, 80.0)
PM2_5 = st.sidebar.number_input("PM2.5", 0.0, 200.0, 50.0)
NO2 = st.sidebar.number_input("NO2", 0.0, 200.0, 60.0)
SO2 = st.sidebar.number_input("SO2", 0.0, 100.0, 20.0)
O3 = st.sidebar.number_input("O3", 0.0, 300.0, 100.0)

st.sidebar.subheader("Environmental Conditions")

Temperature = st.sidebar.number_input("Temperature (°C)", -10.0, 50.0, 25.0)
Humidity = st.sidebar.number_input("Humidity (%)", 0.0, 100.0, 50.0)
WindSpeed = st.sidebar.number_input("Wind Speed", 0.0, 30.0, 5.0)

st.sidebar.subheader("Healthcare Data")

RespiratoryCases = st.sidebar.number_input("Respiratory Cases", 0, 50, 5)
CardiovascularCases = st.sidebar.number_input("Cardiovascular Cases", 0, 30, 3)
HospitalAdmissions = st.sidebar.number_input("Hospital Admissions", 0, 20, 1)

HealthImpactScore = st.sidebar.number_input("Health Impact Score", 0.0, 100.0, 90.0)


# ==========================================
# AQI Category
# ==========================================

st.header("🌫 AQI Category")

if AQI <= 50:
    category = "Good"
elif AQI <= 100:
    category = "Moderate"
elif AQI <= 150:
    category = "Unhealthy for Sensitive Groups"
elif AQI <= 200:
    category = "Unhealthy"
elif AQI <= 300:
    category = "Very Unhealthy"
else:
    category = "Hazardous"

col1, col2 = st.columns(2)

with col1:
    st.metric("AQI Value", AQI)

with col2:
    st.write(f"Air Quality Category: **{category}**")


st.divider()


# ==========================================
# Charts Section
# ==========================================

st.header("📊 Air Pollution Dashboard")

col1, col2 = st.columns(2)

with col1:

    st.subheader("Pollution Indicators")

    pollution_df = pd.DataFrame({
        "Pollutant": ["PM10", "PM2.5", "NO2", "SO2", "O3"],
        "Value": [PM10, PM2_5, NO2, SO2, O3]
    })

    st.bar_chart(pollution_df.set_index("Pollutant"))

with col2:

    st.subheader("Environmental Conditions")

    env_df = pd.DataFrame({
        "Metric": ["Temperature", "Humidity", "Wind Speed"],
        "Value": [Temperature, Humidity, WindSpeed]
    })

    st.line_chart(env_df.set_index("Metric"))


st.divider()


# ==========================================
# Health Indicators
# ==========================================

st.header("🏥 Health Indicators")

health_df = pd.DataFrame({
    "Indicator": [
        "Respiratory Cases",
        "Cardiovascular Cases",
        "Hospital Admissions"
    ],
    "Value": [
        RespiratoryCases,
        CardiovascularCases,
        HospitalAdmissions
    ]
})

st.bar_chart(health_df.set_index("Indicator"))


# ==========================================
# Feature Importance
# ==========================================

st.header("🔎 Feature Importance")

importance = model.feature_importances_

features = [
    "AQI", "PM10", "PM2_5", "NO2", "SO2", "O3",
    "Temperature", "Humidity", "WindSpeed",
    "RespiratoryCases", "CardiovascularCases",
    "HospitalAdmissions", "HealthImpactScore"
]

importance_df = pd.DataFrame({
    "Feature": features,
    "Importance": importance
}).sort_values("Importance", ascending=False)

st.bar_chart(importance_df.set_index("Feature"))


# ==========================================
# Footer
# ==========================================

st.divider()

st.caption(
"""
Air Quality Health Impact Classification  
Machine Learning Deployment Project
"""
)
