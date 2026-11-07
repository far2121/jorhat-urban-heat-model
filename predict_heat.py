import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio

MODEL_FILE = "heat_model.pkl"
INPUT_FILE = "jorhat_heat_predictors.tif"
FEATURES = ["NDVI", "NDBI", "MNDWI", "elevation"]

model = joblib.load(MODEL_FILE)

with rasterio.open(INPUT_FILE) as src:
    if src.count < 5:
        raise ValueError(f"Expected 5 raster bands, found {src.count}")
    if src.crs is None:
        raise ValueError("Input raster has no CRS.")

    profile = src.profile.copy()
    ndvi = src.read(2).astype("float32")
    ndbi = src.read(3).astype("float32")
    mndwi = src.read(4).astype("float32")
    elevation = src.read(5).astype("float32")

    valid = (
        (src.dataset_mask() > 0)
        & np.isfinite(ndvi) & np.isfinite(ndbi)
        & np.isfinite(mndwi) & np.isfinite(elevation)
    )

    X = pd.DataFrame({
        "NDVI": ndvi[valid],
        "NDBI": ndbi[valid],
        "MNDWI": mndwi[valid],
        "elevation": elevation[valid]
    }, columns=FEATURES)

    if X.empty:
        raise ValueError("No valid raster pixels found.")

    predictions = model.predict(X)
    print("\nPREDICTION STATISTICS (estimated Landsat-morning LST)")
    print("Valid pixels:", len(X))
    print(f"Minimum: {predictions.min():.2f} °C")
    print(f"Maximum: {predictions.max():.2f} °C")
    print(f"Mean: {predictions.mean():.2f} °C")
    print(f"Standard deviation: {predictions.std():.2f} °C")

    predicted_map = np.full((src.height, src.width), -9999, dtype="float32")
    predicted_map[valid] = predictions

    profile.update(count=1, dtype="float32", nodata=-9999, compress="deflate")
    with rasterio.open("heat_predictions.tif", "w", **profile) as dst:
        dst.write(predicted_map, 1)
        dst.set_band_description(1, "Predicted_LST_C")

    low_threshold, high_threshold = np.percentile(predictions, [33.33, 66.67])
    zones = np.zeros((src.height, src.width), dtype="uint8")
    zones[valid & (predicted_map <= low_threshold)] = 1
    zones[valid & (predicted_map > low_threshold) & (predicted_map <= high_threshold)] = 2
    zones[valid & (predicted_map > high_threshold)] = 3

    print(f"Relative heat-zone thresholds: {low_threshold:.2f}, {high_threshold:.2f} °C")
    print("Zone pixel counts:", {i: int(np.sum(zones == i)) for i in (1, 2, 3)})

    zone_profile = profile.copy()
    zone_profile.update(dtype="uint8", nodata=0)
    with rasterio.open("heat_zones.tif", "w", **zone_profile) as dst:
        dst.write(zones, 1)
        dst.set_band_description(1, "Relative_Heat_Zone")
        dst.write_colormap(1, {
            0: (0, 0, 0, 0),
            1: (50, 150, 255, 255),
            2: (255, 210, 50, 255),
            3: (230, 60, 40, 255)
        })

    plt.figure(figsize=(10, 8))
    plt.imshow(np.ma.masked_where(zones == 0, zones), cmap="RdYlBu_r", vmin=1, vmax=3)
    plt.colorbar(ticks=[1, 2, 3], label="Relative heat zone (quantiles)")
    plt.title("Relative Predicted LST Heat Zones - Jorhat")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig("jorhat_heat_zones.png", dpi=300, bbox_inches="tight")
    plt.close()

print("\nPrediction complete: heat_predictions.tif, heat_zones.tif, jorhat_heat_zones.png")
