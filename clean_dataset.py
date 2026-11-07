import pandas as pd

INPUT_FILE = "jorhat_heat_dataset.csv"
OUTPUT_FILE = "jorhat_heat_dataset_clean.csv"

df = pd.read_csv(INPUT_FILE)

required = ["LST_C", "NDVI", "NDBI", "MNDWI", "elevation", "longitude", "latitude"]
missing = [col for col in required if col not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

print("Original rows:", len(df))
print("Missing values by column:")
print(df[required].isna().sum())

df_clean = df.drop_duplicates().copy()
print("Rows after exact-duplicate removal:", len(df_clean))
print("Rows removed:", len(df) - len(df_clean))

duplicate_coords = df_clean.duplicated(["longitude", "latitude"]).sum()
print("Repeated coordinate rows:", duplicate_coords)

df_clean = df_clean.dropna(subset=required)
print("Rows after removing incomplete records:", len(df_clean))

df_clean.to_csv(OUTPUT_FILE, index=False)
print(f"Clean dataset saved: {OUTPUT_FILE}")
