
import streamlit as st
import pandas as pd
import joblib

# ------------------------------
# PAGE CONFIG
# ------------------------------
st.set_page_config(
    page_title="Air Quality Health Impact Classification",
    page_icon="🌍",
    layout="centered"
)

# ------------------------------
# TITLE
# ------------------------------
st.title("🌍 Air Quality Health Impact Classification")
st.write("Predict the health impact level based on air quality parameters.")

# ------------------------------
# LOAD MODEL
# ------------------------------
@st.cache_resource

def load_model():
    return joblib.load("air_quality_model.pkl")

model = load_model()

# ------------------------------
# INPUT FIELDS
# ------------------------------
st.subheader("Enter Air Quality Details")

pm25 = st.number_input("PM2.5", min_value=0.0, value=25.0)
pm10 = st.number_input("PM10", min_value=0.0, value=40.0)
no2 = st.number_input("NO2", min_value=0.0, value=20.0)
so2 = st.number_input("SO2", min_value=0.0, value=10.0)
co = st.number_input("CO", min_value=0.0, value=1.0)
o3 = st.number_input("O3", min_value=0.0, value=30.0)

# ------------------------------
# PREDICTION
# ------------------------------
if st.button("Predict Health Impact"):

    input_data = pd.DataFrame([
        {
            "PM2.5": pm25,
            "PM10": pm10,
            "NO2": no2,
            "SO2": so2,
            "CO": co,
            "O3": o3
        }
    ])

    prediction = model.predict(input_data)[0]

    st.success(f"Predicted Health Impact: {prediction}")

    # Optional health messages
    if str(prediction).lower() == "good":
        st.info("Air quality is acceptable.")

    elif str(prediction).lower() == "moderate":
        st.warning("Sensitive people should limit outdoor activity.")

    elif str(prediction).lower() == "poor":
        st.error("Air quality may affect everyone.")

# ------------------------------
# FOOTER
# ------------------------------
st.markdown("---")
st.caption("Built with Streamlit")
