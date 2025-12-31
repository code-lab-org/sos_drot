"""
Resolution efficiency layer using prprocessed SNODAS for SOS_DROT.

Author: Divya Ramachandran
Affiliation: Arizona State University
Created: March 2025

License: MIT
"""
from pathlib import Path
import sys
import traceback
import xarray as xr
import numpy as np
from scipy.special import expit
from rasterio.enums import Resampling
import rioxarray as rxr

# ---------------------------
# Repo-aware paths
# ---------------------------
BASE_DIR = Path(__file__).resolve().parents[2]  # SOS_DROT

preprocessed_dir = BASE_DIR / "input_layers" / "resolution" / "processed"
output_dir = BASE_DIR / "output_efficiency_layers"
output_dir.mkdir(parents=True, exist_ok=True)

print(f"[INFO] BASE_DIR: {BASE_DIR}")
print(f"[INFO] Looking for preprocessed files in: {preprocessed_dir}")
print(f"[INFO] Efficiency outputs will be written to: {output_dir}")

# ---------------------------
# Find preprocessed files
# ---------------------------
if not preprocessed_dir.exists():
    raise FileNotFoundError(f"Preprocessed folder not found: {preprocessed_dir}")

pre_files = list(preprocessed_dir.glob("preprocessed_resolution*.nc"))
if not pre_files:
    raise FileNotFoundError(
        f"No preprocessed resolution files found in {preprocessed_dir}.\n"
        "Expected 'preprocessed_resolution.nc' or 'preprocessed_resolution_SAR.nc'."
    )

pre_file = next((p for p in pre_files if p.name == "preprocessed_resolution.nc"), pre_files[0])
pre_file_SAR = next((p for p in pre_files if "SAR" in p.name), None)

print(f"[INFO] Using resolution file: {pre_file}")
if pre_file_SAR:
    print(f"[INFO] Using SAR resolution file: {pre_file_SAR}")
else:
    print("[INFO] No SAR preprocessed file found (SAR step will be skipped).")

# ---------------------------
# Optional target dataset (reprojection target)
# ---------------------------
# snodas_candidates = (
#     list(BASE_DIR.rglob("snodas-merged_20km*.nc")) +
#     list(BASE_DIR.rglob("snodas-merged*.nc"))
# )
# target_path = snodas_candidates[0] if snodas_candidates else None

# Place target file (to reproject to grid) in the SOS_DROT root folder

target_path = BASE_DIR / "snodas-merged_20km.nc"

if target_path:
    print(f"[INFO] Found target file for reprojection: {target_path}")
else:
    print("[INFO] No target file found — reprojection will be skipped and native grid used.")
    target_path = None

# ---------------------------
# Utility functions
# ---------------------------
def open_with_rio(path: Path):
    """
    Try to open a dataset so it supports the .rio accessor.
    Returns an xarray object with rio methods available (DataArray or Dataset).
    """
    # Try normal xarray open first
    try:
        ds = xr.open_dataset(path)
        # check rio accessor
        try:
            _ = ds.rio
            return ds
        except Exception:
            # try rioxarray open_rasterio (returns DataArray)
            pass
    except Exception:
        # fall back to rioxarray below
        pass

    # Try rioxarray open (safe for many raster netcdf cases)
    try:
        da = rxr.open_rasterio(path)
        return da
    except Exception as e:
        raise RuntimeError(f"Failed to open {path} with xarray or rioxarray: {e}")


def reproject_coarser(source, target):
    """
    Reproject 'source' to the grid of 'target' using rioxarray.reproject_match.
    Both source and target must support the `.rio` accessor.
    Returns a reprojected DataArray or Dataset.
    """
    # Ensure .rio accessibility (assume source/target were opened with helpers above)
    try:
        source = source.rio.write_crs("EPSG:4326")
    except Exception:
        raise RuntimeError("Source dataset does not have rio accessor or CRS cannot be set.")

    try:
        target = target.rio.write_crs("EPSG:4326")
    except Exception:
        raise RuntimeError("Target dataset does not have rio accessor or CRS cannot be set.")

    reprojected = source.rio.reproject_match(
        target,
        resampling=Resampling.bilinear,
        nodata=np.nan
    )
    return reprojected


def efficiency_resolution(T: float, k: float, datarray):
    """
    Compute efficiency eta using a logistic (expit) function.
    resolution_eta = expit(-k * (datarray - T))
    Works element-wise on xarray DataArray / Dataset.
    """
    return expit(-k * (datarray - T))


# ---------------------------
# Load datasets (ensure rio-capable objects)
# ---------------------------
print("[INFO] Opening preprocessed resolution file(s)...")
try:
    file_ds = open_with_rio(pre_file)
    file_SAR_ds = open_with_rio(pre_file_SAR) if pre_file_SAR else None
    print("[INFO] Preprocessed file(s) opened successfully.")
except Exception as e:
    print(f"[ERROR] Failed to open preprocessed file(s): {e}")
    traceback.print_exc()
    sys.exit(1)

target_ds = None
if target_path:
    try:
        target_ds = open_with_rio(target_path)
        print("[INFO] Target file opened successfully.")
    except Exception as e:
        print(f"[WARN] Found target but failed to open it for reprojection: {e}")
        print("[WARN] Proceeding without reprojection (native grid will be used).")
        target_ds = None

# ---------------------------
# Compute efficiencies and save
# ---------------------------
T = 40.0
k = 0.3

# Helper: perform compute+save
def compute_and_save(input_ds, kind_label, out_name, target_ds_optional):
    """
    input_ds: DataArray or Dataset to compute efficiency on
    kind_label: string for logging ("regular" or "SAR")
    out_name: output filename (basename)
    target_ds_optional: if provided, reproject to this target first
    """
    print(f"[INFO] Processing {kind_label} data...")
    if target_ds_optional is not None:
        print(f"[INFO] Reprojecting {kind_label} to target grid...")
        try:
            data_for_eff = reproject_coarser(input_ds, target_ds_optional)
        except Exception as e:
            print(f"[ERROR] Reprojection failed for {kind_label}: {e}")
            print("[ERROR] Falling back to native grid for this dataset.")
            data_for_eff = input_ds
    else:
        print(f"[INFO] No target provided — using native grid for {kind_label}.")
        data_for_eff = input_ds

    print(f"[INFO] Computing efficiency for {kind_label} (T={T}, k={k})...")
    try:
        eff = efficiency_resolution(T, k, data_for_eff)
    except Exception as e:
        print(f"[ERROR] Efficiency computation failed for {kind_label}: {e}")
        traceback.print_exc()
        return

    out_path = output_dir / out_name
    print(f"[INFO] Writing output to: {out_path}")
    try:
        eff.to_netcdf(out_path)
        print(f"[INFO] Wrote {out_path}")
    except Exception as e:
        print(f"[ERROR] Failed to write {out_path}: {e}")
        traceback.print_exc()


# Regular resolution
# compute_and_save(file_ds, "regular resolution", "efficiency_resolution.nc", target_ds)
if file_ds is None:
    raise RuntimeError(
        "Regular preprocessed resolution file is missing. "
        "Cannot compute efficiency."
    )
else:
    compute_and_save(
        file_ds,
        "regular resolution",
        "efficiency_resolution.nc",
        target_ds
    )

# SAR resolution (if present)
if file_SAR_ds is not None:
    compute_and_save(file_SAR_ds, "SAR resolution", "efficiency_resolution_SAR.nc", target_ds)
else:
    print("[INFO] No SAR preprocessed file detected; skipped SAR output.")

print("[INFO] All done.")









