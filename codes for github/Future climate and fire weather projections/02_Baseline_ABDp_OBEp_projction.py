# -*- coding: utf-8 -*-
"""
Created on Sat Jan 24 11:28:13 2026

@author: Kaiwei Luo
"""

# -*- coding: utf-8 -*-
"""
Baseline projection script (1991-2020).
- Regrids historical observation data to match future projection templates.
- Calculates baseline ABDp and OBEp distributions.
"""

import pandas as pd
import numpy as np
import xarray as xr
import joblib
import os
from tqdm import tqdm
import time
from datetime import datetime
import glob
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# ========================= Modifications (only these three) =========================
# 1) New model directory (same as used in the future script)
MODEL_DIR = r"<MODEL_ROOT>\models_skl161"

# 2) Output directories: baseline results and plots
RESULTS_DIR = r"<OUTPUT_ROOT>\outputs_baseline"
PLOTS_DIR   = r"<OUTPUT_ROOT>\baseline_plots"

# 3) Template grid used to standardize to 512-longitude columns (use any future file as a template)
#    If outputs_future has not been generated yet, you can use any future input file (e.g., BUI) as the template.
FUTURE_TEMPLATE_NC = r"<OUTPUT_ROOT>\outputs_future\2040_245_CANESM_pred.nc"
# FUTURE_TEMPLATE_NC = r"<DATA_ROOT>\CMIP6_final\2040 245\2040_CANESM245_BUI.nc"

# Baseline data and biome paths remain as in the original setup
BASE_PATH  = r"<DATA_ROOT>\FireWeather79-16"
BIOME_NC   = r"<DATA_ROOT>\biome_hourmodeling.nc"
# ====================================================================

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR,   exist_ok=True)

# ------------------ Original helper functions (core logic unchanged) ------------------
def get_variable_values(dataset, var_name):
    """Get variable values from dataset."""
    if var_name in dataset.data_vars:
        return dataset[var_name].values
    else:
        raise KeyError(f"Variable {var_name} not found in dataset")

def merge_datasets(datasets):
    """Merge multiple datasets on common coordinates."""
    try:
        merged = xr.merge(datasets, join='inner')
        return merged
    except Exception as e:
        print(f"Error merging datasets: {str(e)}"); return None
# ------------------------------------------------------------------

def nearest_regrid_biome(biome_data, target_lats, target_lons):
    """Nearest-neighbor regrid biome to the target (lat, lon) grid (consistent with the future script)"""
    b = biome_data
    if 'latitude' not in b.coords or 'longitude' not in b.coords:
        rename_map = {}
        if 'lat' in b.coords: rename_map['lat'] = 'latitude'
        if 'lon' in b.coords: rename_map['lon'] = 'longitude'
        b = b.rename(rename_map)
    # 2D nearest-neighbor interpolation
    b2 = b.interp(latitude=target_lats, longitude=target_lons, method='nearest')
    return b2

def plot_spatial_predictions(data_array, title, filename, cmap='viridis', vmin=None, vmax=None):
    plt.figure(figsize=(12, 8))
    if cmap == 'fire':
        colors = [(1, 1, 1), (1, 0.9, 0), (0.8, 0, 0)]
        cmap = LinearSegmentedColormap.from_list('fire', colors, N=100)
    plt.pcolormesh(data_array.longitude, data_array.latitude, data_array, cmap=cmap, vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(extend='both'); cbar.set_label(title)
    plt.title(title); plt.xlabel('Longitude'); plt.ylabel('Latitude')
    plt.tight_layout(); plt.savefig(filename, dpi=300); plt.close()

def load_annual_data(base_path, year, var_folders):
    datasets = []
    for var, folder in var_folders.items():
        folder_path = os.path.join(base_path, folder)
        file_pattern = os.path.join(folder_path, f"*{year}*.nc")
        files = glob.glob(file_pattern)
        if not files:
            print(f"[WARNING] No file found for {var} in {year}, pattern: {file_pattern}")
            return None
        ds = xr.open_dataset(files[0])
        if 'latitude' not in ds.coords and 'lat' in ds.coords:
            ds = ds.rename({'lat':'latitude'})
        if 'longitude' not in ds.coords and 'lon' in ds.coords:
            ds = ds.rename({'lon':'longitude'})
        datasets.append(ds[[list(ds.data_vars.keys())[0]]].rename({list(ds.data_vars.keys())[0]: var}))
    merged = merge_datasets(datasets)
    for ds in datasets:
        ds.close()
    return merged

def main():
    start_time = time.time()

    # Models
    print("Loading models and preprocessors from NEW model dir...")
    encoder         = joblib.load(os.path.join(MODEL_DIR, 'encoder_biome_month.pkl'))
    scaler          = joblib.load(os.path.join(MODEL_DIR, 'scaler_features.pkl'))
    fire_model      = joblib.load(os.path.join(MODEL_DIR, 'fire_model.pkl'))
    duration_model  = joblib.load(os.path.join(MODEL_DIR, 'duration_model.pkl'))
    obe_model       = joblib.load(os.path.join(MODEL_DIR, 'obe_model.pkl'))
    h24event_model  = joblib.load(os.path.join(MODEL_DIR, 'h24event_model.pkl'))
    thresholds      = joblib.load(os.path.join(MODEL_DIR, 'thresholds.pkl'))

    print(f"Loaded thresholds: fire={thresholds['fire_threshold']:.4f}, "
          f"obe={thresholds['obe_threshold']:.4f}, h24={thresholds['h24_threshold']:.4f}")

    base_path = BASE_PATH
    var_folders = {
        'BUI': 'Daily_BUI_NA',
        'DMC': 'Daily_DMC_NA',
        'DC':  'Daily_DC_NA',
        'FWI': 'Daily_FWI_NA',
        'FFMC':'Daily_FFMC_forProjection',
        'ISI': 'Daily_ISI_forProjection'
    }
    years = range(1991, 2021)

    print("Loading biome data...")
    biome_data = xr.open_dataset(BIOME_NC)["gez_code"]

    feature_vars = ['BUI', 'DMC', 'DC', 'FWI', 'FFMC', 'ISI']
    results_dir = RESULTS_DIR; plots_dir = PLOTS_DIR
    os.makedirs(results_dir, exist_ok=True); os.makedirs(plots_dir, exist_ok=True)

    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    yearly_results = {}
    monthly_statistics = {m: {'fire': [], 'obe': [], 'h24': []} for m in range(1, 13)}

    # Read template grid
    tmpl = xr.open_dataset(FUTURE_TEMPLATE_NC, decode_times=False)
    if 'lat' in tmpl.coords and 'latitude' not in tmpl.coords: tmpl = tmpl.rename({'lat':'latitude'})
    if 'lon' in tmpl.coords and 'longitude' not in tmpl.coords: tmpl = tmpl.rename({'lon':'longitude'})
    tmpl_lat = tmpl['latitude'].values
    tmpl_lon = tmpl['longitude'].values
    tmpl.close()

    for year in tqdm(years, desc="Processing years"):
        print(f"\n{'='*30} Processing Year {year} {'='*30}")
        merged_ds = load_annual_data(base_path, year, var_folders)
        if merged_ds is None:
            print(f"Skipping year {year} due to data loading failure"); continue

        # === NEW: Regrid baseline data to the 237×512 template grid (nearest-neighbor) ===
        try:
            merged_ds = merged_ds.interp(latitude=tmpl_lat, longitude=tmpl_lon, method='nearest')
            print(f"[Regrid] Baseline {year} -> template grid: "
                  f"lat={merged_ds.dims['latitude']}, lon={merged_ds.dims['longitude']} (expect 237×512)")
        except Exception as e:
            print(f"[Regrid WARNING] Failed to regrid baseline {year} to template: {e}")

        fw_lats = merged_ds.latitude.values
        fw_lons = merged_ds.longitude.values
        times = merged_ds.time.values
        num_days = len(times)

        # Regrid biome to the same grid as well
        biome_on_grid = nearest_regrid_biome(biome_data, fw_lats, fw_lons).values.astype(int)

        fire_probability = np.full((num_days, len(fw_lats), len(fw_lons)), np.nan, dtype=np.float32)
        obe_probability  = np.full((num_days, len(fw_lats), len(fw_lons)), np.nan, dtype=np.float32)
        h24_probability  = np.full((num_days, len(fw_lats), len(fw_lons)), np.nan, dtype=np.float32)
        fire_prediction  = np.full((num_days, len(fw_lats), len(fw_lons)), np.nan, dtype=np.float32)
        obe_prediction   = np.full((num_days, len(fw_lats), len(fw_lons)), np.nan, dtype=np.float32)
        h24_prediction   = np.full((num_days, len(fw_lats), len(fw_lons)), np.nan, dtype=np.float32)
        duration_class   = np.full((num_days, len(fw_lats), len(fw_lons)), np.nan, dtype=np.float32)

        for d in tqdm(range(num_days), desc=f"Days {year}", leave=False):
            month = pd.Timestamp(times[d]).month

            day_features = []
            valid_indices = []

            for i in range(len(fw_lats)):
                for j in range(len(fw_lons)):
                    vals = []
                    # Extract the six variables for the current day
                    for var in feature_vars:
                        v = merged_ds[var].isel(time=d).values[i, j]
                        vals.append(v)
                    if np.any(np.isnan(vals)):
                        continue
                    biome_val = biome_on_grid[i, j]
                    if np.isnan(biome_val):
                        continue
                    day_features.append(vals + [biome_val, month])
                    valid_indices.append((i, j))

            if len(day_features) == 0:
                continue

            df_feat = pd.DataFrame(day_features, columns=feature_vars + ['biome', 'month'])
            cat = df_feat[['biome', 'month']].astype(int)
            num = df_feat[feature_vars].astype(float)

            cat_enc = encoder.transform(cat)
            num_std = scaler.transform(num)
            X = np.concatenate([num_std, cat_enc], axis=1)

            p_fire = fire_model.predict_proba(X)[:, 1]
            p_obe  = obe_model.predict_proba(X)[:, 1]
            p_h24  = h24event_model.predict_proba(X)[:, 1]

            y_fire = (p_fire >= thresholds['fire_threshold']).astype(int)
            y_obe  = (p_obe  >= thresholds['obe_threshold']).astype(int)
            y_h24  = (p_h24  >= thresholds['h24_threshold']).astype(int)

            y_dur = np.zeros_like(y_fire)
            if y_fire.any():
                idx_fire = np.where(y_fire == 1)[0]
                if len(idx_fire) > 0:
                    y_dur[idx_fire] = duration_model.predict(X[idx_fire]).astype(int)

            for k, (i, j) in enumerate(valid_indices):
                fire_probability[d, i, j] = p_fire[k]
                obe_probability[d,  i, j] = p_obe[k]
                h24_probability[d,  i, j] = p_h24[k]
                fire_prediction[d,  i, j] = y_fire[k]
                obe_prediction[d,   i, j] = y_obe[k]
                h24_prediction[d,   i, j] = y_h24[k]
                duration_class[d,   i, j] = y_dur[k]

        ds_out = xr.Dataset(
            data_vars=dict(
                fire_probability=(['time','latitude','longitude'], fire_probability),
                obe_probability =(['time','latitude','longitude'], obe_probability),
                h24_probability =(['time','latitude','longitude'], h24_probability),
                fire_prediction =(['time','latitude','longitude'], fire_prediction),
                obe_prediction  =(['time','latitude','longitude'], obe_prediction),
                h24_prediction  =(['time','latitude','longitude'], h24_prediction),
                duration_class  =(['time','latitude','longitude'], duration_class),
            ),
            coords=dict(time=times, latitude=fw_lats, longitude=fw_lons),
            attrs=dict(
                description=f'Baseline prediction results for {year}',
                created=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                fire_threshold=thresholds['fire_threshold'],
                obe_threshold =thresholds['obe_threshold'],
                h24_threshold =thresholds['h24_threshold'],
            )
        )

        out_nc = os.path.join(results_dir, f"baseline_{year}_pred.nc")
        ds_out.to_netcdf(out_nc)
        yearly_results[year] = out_nc
        print(f"[SAVE] {out_nc}")

        # Quick summary statistics
        mean_fire = np.nanmean(fire_probability)
        mean_obe  = np.nanmean(obe_probability)
        mean_h24  = np.nanmean(h24_probability)
        print(f"[Stats] year={year} fire={mean_fire:.4f} obe={mean_obe:.4f} h24={mean_h24:.4f}")

        for m in range(1, 13):
            mask = pd.DatetimeIndex(times).month == m
            if mask.sum() == 0: 
                continue
            monthly_statistics[m]['fire'].append(np.nanmean(fire_probability[mask]))
            monthly_statistics[m]['obe'].append(np.nanmean(obe_probability[mask]))
            monthly_statistics[m]['h24'].append(np.nanmean(h24_probability[mask]))

        # Plot annual means
        annual_fire = np.nanmean(fire_probability, axis=0)
        annual_obe  = np.nanmean(obe_probability,  axis=0)
        annual_h24  = np.nanmean(h24_probability,  axis=0)

        da_fire = xr.DataArray(annual_fire, coords={'latitude':fw_lats,'longitude':fw_lons}, dims=['latitude','longitude'])
        da_obe  = xr.DataArray(annual_obe,  coords={'latitude':fw_lats,'longitude':fw_lons}, dims=['latitude','longitude'])
        da_h24  = xr.DataArray(annual_h24,  coords={'latitude':fw_lats,'longitude':fw_lons}, dims=['latitude','longitude'])

        plot_spatial_predictions(da_fire, f"Baseline Fire Probability {year}",
                                 os.path.join(plots_dir, f"baseline_fireprob_{year}.png"),
                                 cmap='fire', vmin=0, vmax=1)
        plot_spatial_predictions(da_obe,  f"Baseline OBE Probability {year}",
                                 os.path.join(plots_dir, f"baseline_obeprob_{year}.png"),
                                 cmap='viridis', vmin=0, vmax=1)
        plot_spatial_predictions(da_h24,  f"Baseline H24 Probability {year}",
                                 os.path.join(plots_dir, f"baseline_h24prob_{year}.png"),
                                 cmap='viridis', vmin=0, vmax=1)

        merged_ds.close()

    # Merge all years
    stats_rows = []
    for m in range(1, 13):
        if len(monthly_statistics[m]['fire']) == 0:
            continue
        stats_rows.append({
            'month': m,
            'fire_mean': float(np.nanmean(monthly_statistics[m]['fire'])),
            'obe_mean':  float(np.nanmean(monthly_statistics[m]['obe'])),
            'h24_mean':  float(np.nanmean(monthly_statistics[m]['h24'])),
        })
    stats_df = pd.DataFrame(stats_rows)
    stats_csv = os.path.join(results_dir, f"baseline_monthly_stats_{current_time}.csv")
    stats_df.to_csv(stats_csv, index=False)
    print(f"[SAVE] {stats_csv}")

    biome_data.close()

    elapsed = time.time() - start_time
    print(f"\nAll done. Total time: {elapsed/60:.1f} minutes.")

if __name__ == "__main__":
    main()
