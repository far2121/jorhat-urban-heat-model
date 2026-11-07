
from pathlib import Path

import rasterio
from rasterio.warp import transform
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Jorhat Urban Heat API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

RASTER_PATH = Path(__file__).resolve().parent.parent / "heat_predictions.tif"


@app.get("/")
def home():
    return {"message": "Jorhat Urban Heat API is running"}


@app.get("/api/health")
def health():
    return {
        "status": "healthy",
        "raster_available": RASTER_PATH.exists()
    }


@app.get("/api/heat/metadata")
def heat_metadata():
    if not RASTER_PATH.exists():
        raise HTTPException(status_code=404, detail="Prediction raster not found")

    with rasterio.open(RASTER_PATH) as src:
        return {
            "width": src.width,
            "height": src.height,
            "crs": str(src.crs),
            "bounds": list(src.bounds),
            "resolution": list(src.res),
            "nodata": src.nodata
        }


@app.get("/api/heat/value")
def heat_value(lon: float, lat: float):
    if not RASTER_PATH.exists():
        raise HTTPException(status_code=404, detail="Prediction raster not found")

    with rasterio.open(RASTER_PATH) as src:
        x, y = transform("EPSG:4326", src.crs, [lon], [lat])
        row, col = src.index(x[0], y[0])

        if not (0 <= row < src.height and 0 <= col < src.width):
            raise HTTPException(status_code=404, detail="Location outside study area")

        value = src.read(1)[row, col]

        if src.nodata is not None and value == src.nodata:
            raise HTTPException(status_code=404, detail="No data at this location")

        return {
            "longitude": lon,
            "latitude": lat,
            "predicted_lst_c": round(float(value), 2)
        }
