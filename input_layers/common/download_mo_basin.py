import os
import requests
import zipfile
import geopandas as gpd
from shapely.geometry import Polygon
from pathlib import Path

# -------------------------
# Minimal-adapted version
# -------------------------
# This uses your original logic and path style but points to the repo-relative
# common/mo_basin folder (no extra folders created).
BASE_DIR = Path(__file__).resolve().parents[2]   # SOS_DROT
path_shp = os.path.join(str(BASE_DIR), "input_layers", "common", "mo_basin", "")

print("Start")
# Automatically download/extract watershed boundary shapefiles
for wbd in [10]:
    shp_expected = path_shp + f"WBD_{wbd}_HU2_Shape/Shape/WBDHU2.shp"
    if not os.path.isfile(shp_expected):
        print(f"Downloading/extracting WBD {wbd}")
        r = requests.get(
            f"https://prd-tnm.s3.amazonaws.com/StagedProducts/Hydrography/WBD/HU2/Shape/WBD_{wbd}_HU2_Shape.zip",
            stream=True,
            timeout=60
        )
        r.raise_for_status()
        zip_path = path_shp + f"WBD_{wbd}_HU2_Shape.zip"
        # stream write to file (safer for large files)
        with open(zip_path, "wb") as zip_file:
            for chunk in r.iter_content(chunk_size=1024*1024):
                if chunk:
                    zip_file.write(chunk)
        # extract into the existing mo_basin folder
        with zipfile.ZipFile(zip_path, 'r') as zip_file:
            zip_file.extractall(path_shp + f"WBD_{wbd}_HU2_Shape")
    else:
        print(f"Shapefile already present: {shp_expected}")

# now read the shapefile from the exact expected path (same as your original)
shp_path = path_shp + "WBD_10_HU2_Shape/Shape/WBDHU2.shp"
if not os.path.isfile(shp_path):
    raise FileNotFoundError(f"Expected shapefile not found at: {shp_path}\n"
                            "Check the downloaded zip/extraction or run the script again.")

mo_basin = gpd.read_file(shp_path)
mo_basin = gpd.GeoSeries(Polygon(mo_basin.iloc[0].geometry.exterior), crs="EPSG:4326")
print(mo_basin)
print("Completed")
