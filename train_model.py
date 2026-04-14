import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os
import numpy as np

# Load data
df = pd.read_csv("delivery_data.csv")

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

X = df.drop("delivery_time_days", axis=1)
y = df["delivery_time_days"]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Metrics
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R² Score: {r2:.2f}")

# Save model
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/timelytics_model.pkl")

# Save metrics
metrics = {"MAE": mae, "RMSE": rmse, "R2": r2}
joblib.dump(metrics, "model/metrics.pkl")

print("Model + Metrics saved")