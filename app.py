import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="Timelytics", page_icon="🚚", layout="wide")

# ---------- CUSTOM CSS ----------
st.markdown("""
<style>
.main {
    background-color: #0f172a;
    color: white;
}
.stButton>button {
    background: linear-gradient(90deg, #6366f1, #22c55e);
    color: white;
    border-radius: 10px;
    padding: 10px 20px;
}
</style>
""", unsafe_allow_html=True)

# ---------- TITLE ----------
st.title("Timelytics")
st.caption("Smart Delivery Time Prediction using Machine Learning")

# ---------- LOAD ----------
model = joblib.load("model/timelytics_model.pkl")
metrics = joblib.load("model/metrics.pkl")

# ---------- METRICS DISPLAY ----------
col1, col2, col3 = st.columns(3)
col1.metric("MAE", f"{metrics['MAE']:.2f}")
col2.metric("RMSE", f"{metrics['RMSE']:.2f}")
col3.metric("R² Score", f"{metrics['R2']:.2f}")

st.divider()

# ---------- INPUT ----------
col1, col2 = st.columns(2)

with col1:
    product_category = st.selectbox("Product", ["Electronics", "Clothing", "Groceries", "Furniture"])
    origin_city = st.selectbox("Origin", ["Mumbai","Delhi","Bangalore","Chennai","Hyderabad","Kolkata","Pune","Jaipur"])
    shipping_method = st.selectbox("Shipping", ["Air", "Ground"])

with col2:
    destination_city = st.selectbox("Destination", ["Mumbai","Delhi","Bangalore","Chennai","Hyderabad","Kolkata","Pune","Jaipur"])
    traffic_level = st.selectbox("Traffic", ["Low", "Medium", "High"])
    weather = st.selectbox("Weather", ["Clear", "Rain", "Fog", "Storm"])

distance_km = st.slider("Distance (km)", 50, 2000, 500)

# ---------- PREDICT ----------
if st.button("Predict Delivery Time"):
    with st.spinner("Analyzing logistics data..."):
        input_df = pd.DataFrame({
            "product_category": [product_category],
            "origin_city": [origin_city],
            "destination_city": [destination_city],
            "shipping_method": [shipping_method],
            "traffic_level": [traffic_level],
            "weather": [weather],
            "distance_km": [distance_km]
        })

        result = model.predict(input_df)[0]

        st.success(f"Estimated Delivery Time: {round(result,2)} days")

# ---------- FEATURE IMPORTANCE ----------
st.subheader("Feature Importance")

def get_feature_importance(model):
    regressor = model.named_steps["regressor"]
    preprocessor = model.named_steps["preprocessor"]

    cat_features = preprocessor.transformers_[0][2]
    ohe = preprocessor.transformers_[0][1]

    encoded_cat = ohe.get_feature_names_out(cat_features)
    all_features = list(encoded_cat) + ["distance_km"]

    importances = regressor.feature_importances_

    return pd.DataFrame({
        "feature": all_features,
        "importance": importances
    }).sort_values(by="importance", ascending=False)

fi_df = get_feature_importance(model)
st.bar_chart(fi_df.set_index("feature"))