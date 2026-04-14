import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import os

# Load dataset
df = pd.read_csv("delivery_data.csv")

# Features
categorical_features = [
    "product_category",
    "origin_city",
    "destination_city",
    "shipping_method",
    "traffic_level",
    "weather"
]

numeric_features = ["distance_km"]

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ("num", "passthrough", numeric_features)
    ]
)

# Pipeline
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(n_estimators=150, random_state=42))
])

# Split data
X = df.drop("delivery_time_days", axis=1)
y = df["delivery_time_days"]

# Train
model.fit(X, y)

# Save model
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/timelytics_model.pkl")

print("Model trained and saved at model/timelytics_model.pkl")