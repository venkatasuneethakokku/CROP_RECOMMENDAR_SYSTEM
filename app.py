import streamlit as st
import numpy as np
import pickle

model = pickle.load(open('crop_model.pkl', 'rb'))
label_encoder = pickle.load(open('label_encoder.pkl', 'rb'))

st.title("🌾 Crop Recommender System")

st.markdown("Provide the following soil and weather parameters to get a recommended crop:")

N = st.number_input("Nitrogen (N)", 0, 200)
P = st.number_input("Phosphorus (P)", 0, 200)
K = st.number_input("Potassium (K)", 0, 200)
temperature = st.number_input("Temperature (°C)", 0.0, 50.0)
humidity = st.number_input("Humidity (%)", 0.0, 100.0)
ph = st.number_input("pH", 0.0, 14.0)
rainfall = st.number_input("Rainfall (mm)", 0.0, 300.0)

if st.button("Predict Crop"):
    input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    prediction_encoded = model.predict(input_data)
    prediction = label_encoder.inverse_transform(prediction_encoded)
    st.success(f"✅ Recommended Crop: **{prediction[0]}**")
