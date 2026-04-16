import streamlit as st
import pandas as pd
import joblib
import time

st.set_page_config(page_title="Timelytics", page_icon="🚚", layout="wide")

# ---------- CSS (UBER STYLE) ----------
st.markdown("""
<style>

/* Background */
.stApp {
    background: radial-gradient(circle at top, #0f172a, #020617);
    color: white;
}

/* Glass Card */
.card {
    background: rgba(30, 41, 59, 0.6);
    backdrop-filter: blur(12px);
    border-radius: 16px;
    padding: 20px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.4);
    animation: fadeIn 0.8s ease-in-out;
}

/* Title */
.title {
    font-size: 42px;
    font-weight: 700;
    letter-spacing: 1px;
}

/* Gradient text */
.gradient {
    background: linear-gradient(90deg, #6366f1, #22c55e);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* Button */
.stButton>button {
    background: linear-gradient(90deg, #6366f1, #22c55e);
    border-radius: 12px;
    padding: 12px 25px;
    font-weight: bold;
    border: none;
    transition: all 0.3s ease;
}
.stButton>button:hover {
    transform: scale(1.07);
    box-shadow: 0 0 20px rgba(99,102,241,0.6);
}

/* Result animation */
.result {
    font-size: 32px;
    font-weight: bold;
    text-align: center;
    padding: 25px;
    border-radius: 15px;
    background: linear-gradient(90deg, #6366f1, #22c55e);
    animation: pop 0.6s ease;
}

/* Animations */
@keyframes fadeIn {
    from {opacity: 0; transform: translateY(15px);}
    to {opacity: 1; transform: translateY(0);}
}

@keyframes pop {
    0% {transform: scale(0.8); opacity: 0;}
    100% {transform: scale(1); opacity: 1;}
}

</style>
""", unsafe_allow_html=True)

# ---------- HEADER ----------
st.markdown('<div class="title gradient">🚚 Timelytics Dashboard</div>', unsafe_allow_html=True)
st.caption("AI-powered logistics intelligence")

# ---------- LOAD ----------
model = joblib.load("model/timelytics_model.pkl")
metrics = joblib.load("model/metrics.pkl")

# ---------- METRICS ----------
c1, c2, c3 = st.columns(3)

c1.markdown(f'<div class="card"><br>MAE<br><h2>{metrics["MAE"]:.2f}</h2></div>', unsafe_allow_html=True)
c2.markdown(f'<div class="card"><br>RMSE<br><h2>{metrics["RMSE"]:.2f}</h2></div>', unsafe_allow_html=True)
c3.markdown(f'<div class="card"><br>R² Score<br><h2>{metrics["R2"]:.2f}</h2></div>', unsafe_allow_html=True)

st.divider()

# ---------- MAIN ----------
left, right = st.columns([1,1])

# ---------- INPUT ----------
with left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Shipment Details")

    product_category = st.selectbox("Product", ["Electronics", "Clothing", "Groceries", "Furniture"])
    origin_city = st.selectbox("Origin", ["Mumbai","Delhi","Bangalore","Chennai","Hyderabad","Kolkata","Pune","Jaipur"])
    destination_city = st.selectbox("Destination", ["Mumbai","Delhi","Bangalore","Chennai","Hyderabad","Kolkata","Pune","Jaipur"])

    shipping_method = st.selectbox("Shipping", ["Air", "Ground"])
    traffic_level = st.selectbox("Traffic", ["Low", "Medium", "High"])
    weather = st.selectbox("Weather", ["Clear", "Rain", "Fog", "Storm"])

    distance_km = st.slider("Distance (km)", 50, 2000, 500)

    predict = st.button("Predict Delivery Time")
    st.markdown('</div>', unsafe_allow_html=True)

# ---------- RESULT ----------
with right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Prediction")

    if predict:
        progress = st.progress(0)

        # Fake loading animation (Uber-style feel)
        for i in range(100):
            time.sleep(0.01)
            progress.progress(i + 1)

        with st.spinner("Optimizing delivery route..."):
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

        st.markdown(
            f'<div class="result"> {round(result,2)} Days</div>',
            unsafe_allow_html=True
        )
    else:
        st.info("Enter details to generate prediction")

    st.markdown('</div>', unsafe_allow_html=True)

# ---------- FEATURE IMPORTANCE ----------
st.markdown('<div class="card">', unsafe_allow_html=True)
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

st.markdown('</div>', unsafe_allow_html=True)