import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import json

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# 1. Load dataset
df = pd.read_csv("jorhat_heat_dataset.csv")

features = ["NDVI", "NDBI", "MNDWI", "elevation"]

X = df[features]
y = df["LST_C"]

# 2. Create spatial groups
groups = (
    np.floor(df["latitude"] / 0.01).astype(str)
    + "_"
    + np.floor(df["longitude"] / 0.01).astype(str)
)

# 3. Spatial train-test split
splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.2,
    random_state=42
)

train_idx, test_idx = next(
    splitter.split(X, y, groups)
)

X_train = X.iloc[train_idx]
X_test = X.iloc[test_idx]

y_train = y.iloc[train_idx]
y_test = y.iloc[test_idx]

# 4. Train evaluation model
model = RandomForestRegressor(
    n_estimators=150,
    max_depth=20,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# 5. Predict unseen spatial blocks
predictions = model.predict(X_test)

# 6. Calculate metrics
mae = mean_absolute_error(y_test, predictions)

rmse = np.sqrt(
    mean_squared_error(y_test, predictions)
)

r2 = r2_score(y_test, predictions)

print("\nSPATIAL MODEL EVALUATION")
print("------------------------")
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))
print(f"MAE: {mae:.3f} °C")
print(f"RMSE: {rmse:.3f} °C")
print(f"R2: {r2:.3f}")

# 7. Save evaluation metrics
metrics = {
    "MAE": float(mae),
    "RMSE": float(rmse),
    "R2": float(r2),
    "training_samples": len(X_train),
    "testing_samples": len(X_test)
}

with open("model_metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

# 8. Save test predictions
results = pd.DataFrame({
    "Actual_LST": y_test.values,
    "Predicted_LST": predictions,
    "Error": y_test.values - predictions
})

results.to_csv(
    "spatial_test_predictions.csv",
    index=False
)

# 9. Actual vs predicted graph
plt.figure(figsize=(7, 6))

plt.scatter(
    y_test,
    predictions,
    alpha=0.4
)

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    "r--"
)

plt.xlabel("Actual LST (°C)")
plt.ylabel("Predicted LST (°C)")
plt.title("Actual vs Predicted LST")

plt.tight_layout()
plt.savefig("actual_vs_predicted.png", dpi=300)
plt.close()

# 10. Residual distribution
residuals = y_test.values - predictions

plt.figure(figsize=(7, 5))

plt.hist(residuals, bins=30)

plt.xlabel("Prediction Error (°C)")
plt.ylabel("Frequency")
plt.title("Prediction Error Distribution")

plt.tight_layout()
plt.savefig("residual_distribution.png", dpi=300)
plt.close()

# 11. Train final model on all samples
final_model = RandomForestRegressor(
    n_estimators=150,
    max_depth=20,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

final_model.fit(X, y)

joblib.dump(final_model, "heat_model.pkl")

print("\nFinal model trained on all samples.")
print("Model saved: heat_model.pkl")
print("Evaluation plots and metrics saved.")