import pandas as pd
import numpy as np
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset
df = pd.read_csv("jorhat_heat_dataset.csv")

features = ["NDVI", "NDBI", "MNDWI", "elevation"]
X = df[features]
y = df["LST_C"]

# Create approximately 1 km spatial blocks
lat_block = np.floor(df["latitude"] / 0.01)
lon_block = np.floor(df["longitude"] / 0.01)

groups = lat_block.astype(str) + "_" + lon_block.astype(str)

# Split by geographic blocks
splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.2,
    random_state=42
)

train_idx, test_idx = next(
    splitter.split(X, y, groups)
)

X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

# Train model
model = RandomForestRegressor(
    n_estimators=150,
    max_depth=20,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# Evaluate
predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)

print("\nSPATIAL VALIDATION RESULTS")
print("--------------------------")
print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")
print(f"MAE:  {mae:.3f} °C")
print(f"RMSE: {rmse:.3f} °C")
print(f"R²:   {r2:.3f}")

# Feature importance
importance = pd.DataFrame({
    "Feature": features,
    "Importance": model.feature_importances_
}).sort_values("Importance", ascending=False)

print("\nFEATURE IMPORTANCE")
print(importance)

# Save model
joblib.dump(model, "heat_model.pkl")

print("\nSpatially validated model saved.")