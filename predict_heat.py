import pandas as pd
import rasterio
import numpy as np
import joblib
import matplotlib.pyplot as plt

# 1. Load trained model
model = joblib.load("heat_model.pkl")

input_file = "jorhat_heat_predictors.tif"

with rasterio.open(input_file) as src:

    profile = src.profile.copy()

    # Read predictor bands (NOT the observed LST band)
    ndvi = src.read(2).astype("float32")
    ndbi = src.read(3).astype("float32")
    mndwi = src.read(4).astype("float32")
    elevation = src.read(5).astype("float32")

    # Identify invalid pixels
    valid = (
        (src.dataset_mask() > 0)
        & np.isfinite(ndvi)
        & np.isfinite(ndbi)
        & np.isfinite(mndwi)
        & np.isfinite(elevation)
    )

    # Prepare prediction matrix
    X = pd.DataFrame({
    "NDVI": ndvi[valid],
    "NDBI": ndbi[valid],
    "MNDWI": mndwi[valid],
    "elevation": elevation[valid]
    })

    print("Valid pixels:", len(X))

    # 2. Predict LST
    predictions = model.predict(X)

    print("\nPREDICTION STATISTICS")
    print("Minimum:", predictions.min())
    print("Maximum:", predictions.max())
    print("Mean:", predictions.mean())
    print("Standard deviation:", predictions.std())

    predicted_map = np.full(
        (src.height, src.width),
        -9999,
        dtype="float32"
    )

    predicted_map[valid] = predictions

    # 3. Save predicted temperature GeoTIFF
    profile.update(
        count=1,
        dtype="float32",
        nodata=-9999,
        compress="deflate"
    )

    with rasterio.open(
        "heat_predictions.tif", "w", **profile
    ) as dst:
        dst.write(predicted_map, 1)

    # 4. Calculate heat-zone thresholds
    low_threshold, high_threshold = np.percentile(
        predictions, [33.33, 66.67]
    )

    print("Low/medium threshold:", low_threshold)
    print("Medium/high threshold:", high_threshold)

    # 5. Classify heat zones
    zones = np.zeros(
        (src.height, src.width),
        dtype="uint8"
    )

    zones[valid & (predicted_map <= low_threshold)] = 1

    zones[
        valid
        & (predicted_map > low_threshold)
        & (predicted_map <= high_threshold)
    ] = 2

    zones[valid & (predicted_map > high_threshold)] = 3

    print("\nHEAT ZONE PIXEL COUNTS")
    print("Low:", np.sum(zones == 1))
    print("Medium:", np.sum(zones == 2))
    print("High:", np.sum(zones == 3))

    # 6. Save heat-zone GeoTIFF
    zone_profile = profile.copy()
    zone_profile.update(
        dtype="uint8",
        nodata=0
    )

    with rasterio.open(
        "heat_zones.tif", "w", **zone_profile
    ) as dst:
        dst.write(zones, 1)

        dst.write_colormap(1, {
            0: (0, 0, 0, 0),
            1: (50, 150, 255, 255),
            2: (255, 210, 50, 255),
            3: (230, 60, 40, 255)
        })

    # 7. Visualise heat zones
    plt.figure(figsize=(10, 8))

    plt.imshow(
        np.ma.masked_where(zones == 0, zones),
        cmap="RdYlBu_r",
        vmin=1,
        vmax=3
    )

    plt.colorbar(
        ticks=[1, 2, 3],
        label="Heat Zone"
    )

    plt.title("Predicted LST Heat Zones - Jorhat")
    plt.axis("off")
    plt.tight_layout()

    plt.savefig(
        "jorhat_heat_zones.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("\nPREDICTION COMPLETE")
    print("Saved: heat_predictions.tif")
    print("Saved: heat_zones.tif")
    print("Saved: jorhat_heat_zones.png")
