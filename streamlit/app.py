import streamlit as st
import joblib
import numpy as np

model = joblib.load("stress_model.pkl")

st.title("🧠 Stress Detection System")
st.write("Predict stress level using wearable sensor data")

x = st.number_input("X (Motion)", 0.0)
y = st.number_input("Y (Motion)", 0.0)
z = st.number_input("Z (Motion)", 0.0)
eda = st.number_input("EDA", 0.0)
hr = st.number_input("Heart Rate", 70.0)
temp = st.number_input("Temperature", 30.0)

if st.button("Predict Stress"):
    features = np.array([[x, y, z, eda, hr, temp]])
    pred = model.predict(features)[0]

    if pred == 0:
        st.success("🟢 Low Stress")
    elif pred == 1:
        st.warning("🟡 Medium Stress")
    else:
        st.error("🔴 High Stress")