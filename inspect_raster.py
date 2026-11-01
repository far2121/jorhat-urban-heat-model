import rasterio

path = "jorhat_heat_predictors.tif"

with rasterio.open(path) as src:
    print("Width:", src.width)
    print("Height:", src.height)
    print("Bands:", src.count)
    print("CRS:", src.crs)
    print("Resolution:", src.res)
    print("Bounds:", src.bounds)
    print("Band descriptions:", src.descriptions)