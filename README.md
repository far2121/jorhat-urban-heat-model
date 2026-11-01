# Jorhat Urban Heat Prediction Model

An AI-powered geospatial prototype for predicting Land Surface Temperature (LST) and identifying relative surface heat zones in Jorhat, Assam, India.

## Project Overview

This project uses satellite-derived environmental variables and machine learning to estimate spatial variations in Land Surface Temperature.

A Random Forest regression model is trained using Landsat-derived data and environmental predictors. The trained model is then applied to a geospatial raster to produce predicted LST and relative heat-zone classifications.

## Objectives

- Process satellite imagery for the Jorhat study area.
- Derive vegetation, built-up and water-related indices.
- Train a machine learning model to estimate LST.
- Generate spatial predictions using a trained Random Forest model.
- Classify locations into low, medium and high relative surface-temperature zones.

## Technology Stack

- Python
- Google Earth Engine
- Landsat 8/9 Collection 2 Level 2
- Dynamic World
- SRTM DEM
- Pandas and NumPy
- Scikit-learn
- Rasterio
- Matplotlib

## Model Features

| Feature | Description |
|---|---|
| NDVI | Normalized Difference Vegetation Index |
| NDBI | Normalized Difference Built-up Index |
| MNDWI | Modified Normalized Difference Water Index |
| Elevation | Terrain elevation |

**Target variable:** Land Surface Temperature (°C).

## Model

Algorithm: Random Forest Regressor

Spatial validation results:

| Metric | Result |
|---|---:|
| MAE | 0.762 °C |
| RMSE | 0.981 °C |
| R² | 0.349 |

These results represent the current baseline model and indicate that further improvement is needed.

## Heat-Zone Classification

Predicted LST values are divided into three relative classes using the 33rd and 67th percentiles:

1. Low relative surface temperature
2. Medium relative surface temperature
3. High relative surface temperature

These are relative classes, not absolute heat-risk thresholds.

## Project Outputs

- `heat_predictions.tif` — predicted LST raster
- `heat_zones.tif` — classified heat-zone raster
- `jorhat_heat_zones.png` — heat-zone visualisation
- `model_metrics.json` — model evaluation metrics

## How to Run

Clone the repository:

```bash
git clone https://github.com/far2121/jorhat-urban-heat-model.git
cd jorhat-urban-heat-model
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run prediction:

```bash
python predict_heat.py
```

The input raster `jorhat_heat_predictors.tif` and trained model `heat_model.pkl` must be available in the project directory.

## Limitations

- This is a prototype based on a limited set of satellite observations.
- Spatial validation performance is currently modest.
- The output represents estimated satellite-derived land surface temperature, not near-surface air temperature.
- Heat-zone classes are relative to the study-area predictions.
- The study boundary is a provisional project-defined polygon and should not be interpreted as an official municipal boundary.
- Further validation, seasonal analysis and model improvement are required.

## Future Work

- Improve model accuracy through expanded seasonal datasets and additional predictors.
- Validate predictions against independent observations.
- Integrate flood-risk prediction.
- Develop a unified geospatial decision-support platform for urban planning.

## Status

**Heat prediction prototype — Version 1.0 (baseline).**