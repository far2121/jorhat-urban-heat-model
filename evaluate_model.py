import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit

DATA_FILE = "jorhat_heat_dataset_clean.csv"
FEATURES = ["NDVI", "NDBI", "MNDWI", "elevation"]
TARGET = "LST_C"

df = pd.read_csv(DATA_FILE)
required = FEATURES + [TARGET, "longitude", "latitude"]
missing = [col for col in required if col not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

df = df.dropna(subset=required).copy()
X = df[FEATURES]
y = df[TARGET]

# Approx. 1 km spatial blocks; coordinates are used only for grouping.
lat_block = np.floor(df["latitude"] / 0.01).astype(int)
lon_block = np.floor(df["longitude"] / 0.01).astype(int)
groups = lat_block.astype(str) + "_" + lon_block.astype(str)

splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(splitter.split(X, y, groups))

X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

model = RandomForestRegressor(
    n_estimators=300,
    max_depth=20,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)
predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)

print("\nSPATIAL HOLDOUT EVALUATION")
print("--------------------------")
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))
print("Training spatial blocks:", groups.iloc[train_idx].nunique())
print("Testing spatial blocks:", groups.iloc[test_idx].nunique())
print(f"MAE:  {mae:.3f} °C")
print(f"RMSE: {rmse:.3f} °C")
print(f"R²:   {r2:.3f}")

metrics = {
    "MAE_C": float(mae),
    "RMSE_C": float(rmse),
    "R2": float(r2),
    "training_samples": int(len(X_train)),
    "testing_samples": int(len(X_test)),
    "training_spatial_blocks": int(groups.iloc[train_idx].nunique()),
    "testing_spatial_blocks": int(groups.iloc[test_idx].nunique()),
    "validation": "single GroupShuffleSplit spatial holdout",
    "target": "multi-year March-April median Landsat LST (surface temperature)"
}
with open("model_metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

results = pd.DataFrame({
    "longitude": df.iloc[test_idx]["longitude"].values,
    "latitude": df.iloc[test_idx]["latitude"].values,
    "Actual_LST_C": y_test.values,
    "Predicted_LST_C": predictions,
    "Residual_C": y_test.values - predictions
})
results.to_csv("spatial_test_predictions.csv", index=False)

plt.figure(figsize=(7, 6))
plt.scatter(y_test, predictions, alpha=0.4)
limits = [min(y_test.min(), predictions.min()), max(y_test.max(), predictions.max())]
plt.plot(limits, limits, "r--")
plt.xlabel("Actual LST (°C)")
plt.ylabel("Predicted LST (°C)")
plt.title("Spatial Holdout: Actual vs Predicted LST")
plt.tight_layout()
plt.savefig("actual_vs_predicted.png", dpi=300)
plt.close()

residuals = y_test.values - predictions
plt.figure(figsize=(7, 5))
plt.hist(residuals, bins=30)
plt.xlabel("Residual (actual - predicted) °C")
plt.ylabel("Frequency")
plt.title("Spatial Holdout Residual Distribution")
plt.tight_layout()
plt.savefig("residual_distribution.png", dpi=300)
plt.close()

perm = permutation_importance(
    model, X_test, y_test, n_repeats=10, random_state=42,
    scoring="neg_mean_absolute_error", n_jobs=-1
)
importance = pd.DataFrame({
    "Feature": FEATURES,
    "Permutation_Importance": perm.importances_mean
}).sort_values("Permutation_Importance", ascending=False)
importance.to_csv("feature_importance.csv", index=False)
print("\nPermutation importance (held-out spatial data):")
print(importance.to_string(index=False))
print("\nSaved metrics, test predictions, plots and feature_importance.csv.")
