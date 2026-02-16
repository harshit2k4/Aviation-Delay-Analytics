import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os

# Set page configuration
st.set_page_config(page_title="Aviation Delay Analytics", page_icon="✈️", layout="wide")

# Load Assets
@st.cache_resource
def load_assets():
    model = joblib.load(os.path.join('models', 'best_model.pkl'))
    scaler = joblib.load(os.path.join('models', 'scaler.pkl'))
    airlines = pd.read_csv(os.path.join('data', 'airlines.csv'))
    airports = pd.read_csv(os.path.join('data', 'airports.csv'))
    return model, scaler, airlines, airports

try:
    model, scaler, airlines_df, airports_df = load_assets()
except Exception as e:
    st.error(f"Error loading assets: {e}")
    st.stop()

# UI Design
st.title("Flight Delay Prediction System")
st.markdown("Use this dashboard to predict flight delays based on real-time operational metrics.")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Schedule Details")
    month = st.selectbox("Month", options=range(1, 13), index=4)
    sched_dep = st.number_input("Scheduled Departure (HHMM)", value=1000)
    sched_arr = st.number_input("Scheduled Arrival (HHMM)", value=1200)
    elapsed = st.number_input("Elapsed Time (min)", value=120)

    airline_name = st.selectbox("Airline", options=airlines_df['AIRLINE'].unique())

with col2:
    st.subheader("Operational Metrics")
    taxi_out = st.slider("Taxi Out Time (min)", 0, 100, 15)
    taxi_in = st.slider("Taxi In Time (min)", 0, 100, 7)
    dep_time = st.number_input("Actual Departure (HHMM)", value=1005)
    wheels_off = st.number_input("Wheels Off (HHMM)", value=1020)
    wheels_on = st.number_input("Wheels On (HHMM)", value=1150)
    arr_time = st.number_input("Actual Arrival (HHMM)", value=1158)

# Prediction Logic
# TAXI_OUT, WHEELS_OFF, DEPARTURE_TIME, TAXI_IN,
# SCHEDULED_DEPARTURE, SCHEDULED_ARRIVAL, WHEELS_ON, ARRIVAL_TIME, ELAPSED_TIME, MONTH
features = np.array([[taxi_out, wheels_off, dep_time, taxi_in, sched_dep,
                     sched_arr, wheels_on, arr_time, elapsed, month]])

if st.button("Run Delay Analysis", use_container_width=True):
    # Apply the scaler first
    scaled_features = scaler.transform(features)
    prediction = model.predict(scaled_features)

    st.divider()
    if prediction[0] == 1:
        st.error("### Predicted Result: DELAYED")
    else:
        st.success("### Predicted Result: ON-TIME")
