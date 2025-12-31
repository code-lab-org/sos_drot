"""
Snowcover analysis using MODIS data for SOS_DROT.

Author: Divya Ramachandran
Affiliation: Arizona State University
Created: March 2025

License: MIT
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import earthaccess
import xarray as xr
import rioxarray as rxr
import os
import regex as rgx
from datetime import datetime,date,timedelta
import dask
import glob
import geopandas as gpd
from shapely.geometry import Polygon
import os
import requests
import zipfile

# ==================================================
# Paths (repo-relative, GitHub-safe)
# ==================================================
BASE_DIR = Path(__file__).resolve().parents[2]   # SOS_DROT

# Common basin shapefile
path_shp = (
    BASE_DIR /
    "input_layers" /
    "common" /
    "mo_basin" /
    "WBD_10_HU2_Shape" /
    "Shape"
)

# Resolution layer directories
SNOW_DIR = BASE_DIR / "input_layers" / "snowcover"
raw_path = SNOW_DIR / "raw"
hdf_path = raw_path / "hdf"
nc_path = raw_path / "nc"

# snodas_dir = Path(raw_path) / "SNODAS"
path_preprocessed = SNOW_DIR / "processed"
file_name_preprocessed = "preprocessed_snowcover.nc"

# Ensure directories exist
path_preprocessed.mkdir(parents=True, exist_ok=True)
raw_path.mkdir(parents=True, exist_ok=True)
hdf_path.mkdir(parents=True, exist_ok=True)     
nc_path.mkdir(parents=True, exist_ok=True)

# Each data has a code number that can be conveniently used to download data, first login to the earthacess account
# Downloading Data from earthaccess
# 1. Logging in
earthaccess.login(persist=True)
# 2. Search
results = earthaccess.search_data(
    short_name = 'MOD10C1',
    temporal = ("2025.01.01","2025.01.3")
)
# 3. Downloading Data
files = earthaccess.download(results,hdf_path)
# Downloaded files are in hdf format
# Code to loop through file in the folder, add date component, convert to netcdf, and collate to one data set
# Code for Snow Cover

ctr = 0
lon = np.linspace(-180,180,7200)
lat = np.flip(np.linspace(-90,90,3600))
time_sc = []

for filename in os.listdir(hdf_path):    
    year = filename[9:13]
    day = filename[13:16]
    name = filename[0:34]

    # converting day of year to time

    dates = pd.to_datetime(int(day)-1,unit = 'D', origin=year)     
    time_sc.append(dates)
    f_nc = xr.open_dataset(hdf_path / filename,engine = 'netcdf4')  
    snow = f_nc['Day_CMG_Snow_Cover']
    temp_arr = xr.DataArray(
    data=snow,
    dims=['lat','lon'],
    coords=dict(
        lon = lon,
        lat = lat,
    )
    )
    temp_arr.to_netcdf(nc_path / (name + ".nc"))
    files = glob.glob(os.path.join(nc_path,"*.nc"))
    print(glob.glob(os.path.join(nc_path,"*.nc")))

# Merged code
print("Writing snowcover-merged.nc")
ds = xr.combine_by_coords(
    [        
        rxr.open_rasterio(files[i]).drop_vars("band").assign_coords(time=time_sc[i]).expand_dims(dim="time")         
         for i in range(len(time_sc))            
        
    ], 
    combine_attrs="drop_conflicts"
)
ds = ds.rio.write_crs("EPSG:4326")
ds.to_netcdf(os.path.join(path_preprocessed, "snowcover-merged.nc"))
print("Snow Cover merged nc file completed processing.")

# Missouri Basin file read
print("Read Mo basin file")
mo_basin = gpd.read_file(path_shp / "WBDHU2.shp")
mo_basin = gpd.GeoSeries(
    Polygon(mo_basin.iloc[0].geometry.exterior),
    crs="EPSG:4326"
)

# Opening the merged netcdf files
snow_layer = rxr.open_rasterio(os.path.join(path_preprocessed,"snowcover-merged.nc"),crs = "EPSG:4326")
snow_layer_mo = snow_layer.rio.clip(mo_basin.envelope)
snow_layer_mo = snow_layer_mo.convert_calendar(calendar='standard')
temp = snow_layer_mo.groupby(snow_layer_mo.time.dt.isocalendar().week).max()
temp = temp.to_dataset()
temp = temp.rename({'Day_CMG_Snow_Cover': 'Weekly_Snow_Cover'})
temp_resampled = temp.sel(week=snow_layer_mo.time.dt.isocalendar().week)
temp_resampled.to_netcdf(path_preprocessed / file_name_preprocessed)

print("Preprocessing of snowcover completed.")