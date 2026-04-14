import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="Timelytics", page_icon="🚚", layout="centered")

st.title("Timelytics - Delivery Time Predictor")
st.caption("Predict delivery time using machine learning")

# Load model
model_path = os.path.join("model", "timelytics_model.pkl")

try:
    model = joblib.load(model_path)
except:
    st.error("Model not found. Run train_model.py first.")
    st.stop()

# Input form
with st.form("input_form"):
    st.subheader("Order Details")

    product_category = st.selectbox(
        "Product Category",
        ["Electronics", "Clothing", "Groceries", "Furniture"]
    )

    origin_city = st.selectbox(
        "Origin City",
        ["Mumbai", "Delhi", "Bangalore", "Chennai", "Hyderabad", "Kolkata", "Pune", "Jaipur"]
    )

    destination_city = st.selectbox(
        "Destination City",
        ["Mumbai", "Delhi", "Bangalore", "Chennai", "Hyderabad", "Kolkata", "Pune", "Jaipur"]
    )

    shipping_method = st.selectbox(
        "Shipping Method",
        ["Air", "Ground"]
    )

    distance_km = st.slider("Distance (km)", 50, 2000, 500)

    traffic_level = st.selectbox(
        "Traffic Level",
        ["Low", "Medium", "High"]
    )

    weather = st.selectbox(
        "Weather",
        ["Clear", "Rain", "Fog", "Storm"]
    )

    submitted = st.form_submit_button("Predict Delivery Time")

# Prediction
if submitted:
    try:
        input_df = pd.DataFrame({
            "product_category": [product_category],
            "origin_city": [origin_city],
            "destination_city": [destination_city],
            "shipping_method": [shipping_method],
            "traffic_level": [traffic_level],
            "weather": [weather],
            "distance_km": [distance_km]
        })

        prediction = model.predict(input_df)[0]

        st.success(f"Estimated Delivery Time: **{round(prediction, 2)} days**")

    except Exception as e:
        st.error(f"Prediction failed: {e}")