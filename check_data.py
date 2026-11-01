import pandas as pd

df = pd.read_csv("jorhat_heat_dataset.csv")

print("Shape:", df.shape)
print("\nColumns:", df.columns.tolist())
print("\nMissing values:\n", df.isnull().sum())
print("\nLST statistics:\n", df["LST_C"].describe())
print("\nFirst five rows:\n", df.head())