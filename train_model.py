import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

DATA_FILE = "jorhat_heat_dataset_clean.csv"
MODEL_FILE = "heat_model.pkl"
FEATURES = ["NDVI", "NDBI", "MNDWI", "elevation"]
TARGET = "LST_C"

df = pd.read_csv(DATA_FILE)
required = FEATURES + [TARGET]
missing = [col for col in required if col not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

df = df.dropna(subset=required).copy()
X = df[FEATURES]
y = df[TARGET]

model = RandomForestRegressor(
    n_estimators=300,
    max_depth=20,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)
model.fit(X, y)
joblib.dump(model, MODEL_FILE)

print("Final model trained on all", len(df), "valid samples.")
print("Features:", FEATURES)
print("Saved:", MODEL_FILE)
print("Note: use evaluate_model.py for spatial holdout metrics.")
