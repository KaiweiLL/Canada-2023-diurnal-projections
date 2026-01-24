# -*- coding: utf-8 -*-
"""
Created on Sat Jan 24 11:45:18 2026

@author: Kaiwei Luo
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

# ============ Paths and years (edit as needed) ============
MODEL_DIR = r"<MODEL_ROOT>\models_skl161"       # New model
BASE_PATH = r"<DATA_ROOT>\FireWeather79-16"
BIOME_NC  = r"<DATA_ROOT>\biome_hourmodeling.nc"

# Pick any future input/output NetCDF as the template (ensure it is 237×512; if it does not exist yet, point to a future input BUI file).
FUTURE_TEMPLATE_NC = r"<DATA_ROOT>\CMIP6_final\2040 245\2040_CANESM245_BUI.nc"
# Or use an already-generated prediction output: r"<OUTPUT_ROOT>\outputs_future\2040_245_CANESM_pred.nc"

RESULTS_DIR = r"<OUTPUT_ROOT>\outputs_single_year"
PLOTS_DIR   = r"<OUTPUT_ROOT>\plots_single_year"

# Years to run (example: [2021, 2023]).
YEARS = [2021, 2023]
# =============================================

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR,   exist_ok=True)

# ================== Original helper functions (logic unchanged) ==================
def get_variable_values(dataset, var_name):
    """Extract a fire-weather index variable from a dataset, handling different file structures."""
    if var_name in ['BUI', 'DMC', 'DC', 'FWI']:
        if 'time' in dataset.data_vars:
            data = dataset['time'].copy()
            data.name = var_name
            return data
    else:
        if var_name in dataset.data_vars:
            return dataset[var_name]
    for var in ['time', 'variable']:
        try:
            return dataset[var]
        except:
            pass
    for var in dataset.data_vars:
        if var != 'crs':
            return dataset[var]
    return None

def extend_biome_data(biome_data, target_lons, target_lats):
    """Nearest-neighbor regrid the biome to the target (lat, lon) grid (2D), consistent with the future logic."""
    b = biome_data
    if 'latitude' not in b.coords or 'longitude' not in b.coords:
        rename_map = {}
        if 'lat' in b.coords: rename_map['lat'] = 'latitude'
        if 'lon' in b.coords: rename_map['lon'] = 'longitude'
        b = b.rename(rename_map)
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
    """Load all variables for a given year and merge into a (time, lat, lon) Dataset."""
    datasets = {}
    reference_coords = {'time': None, 'latitude': None, 'longitude': None}
    for var_name, folder in var_folders.items():
        file_pattern = os.path.join(base_path, folder, f"*_{year}*.nc*")
        matching_files = glob.glob(file_pattern)
        if not matching_files:
            print(f"Warning: No file found for {var_name} in year {year}")
            return None
        file_path = matching_files[0]
        try:
            ds = xr.open_dataset(file_path)
            var_data = get_variable_values(ds, var_name)
            if var_data is None:
                print(f"Failed to extract {var_name} data"); return None
            ren = {}
            if 'lat' in ds.coords and 'latitude' not in ds.coords: ren['lat'] = 'latitude'
            if 'lon' in ds.coords and 'longitude' not in ds.coords: ren['lon'] = 'longitude'
            if 'day' in ds.coords: ren['day'] = 'time'
            if ren: ds = ds.rename(ren)
            if 'longitude' in ds.coords:
                lon_values = ds.longitude.values
                if np.any(lon_values > 180):
                    new_lon = np.where(lon_values > 180, lon_values - 360, lon_values)
                    ds = ds.assign_coords(longitude=new_lon).sortby('longitude')
            if reference_coords['time'] is None:
                reference_coords['time'] = ds.time.values
                reference_coords['latitude'] = ds.latitude.values
                reference_coords['longitude'] = ds.longitude.values
                if not np.issubdtype(reference_coords['time'].dtype, np.datetime64):
                    try:
                        reference_coords['time'] = pd.to_datetime(reference_coords['time'])
                    except:
                        reference_coords['time'] = np.arange(len(reference_coords['time']))
                        print("Warning: Using numeric indices for time dimension")
            vals = var_data.values
            if len(vals.shape) == 3:
                new_var = xr.DataArray(
                    vals, dims=['time','latitude','longitude'],
                    coords={'time': reference_coords['time'],
                            'latitude': reference_coords['latitude'],
                            'longitude': reference_coords['longitude']},
                    name=var_name
                )
                datasets[var_name] = xr.Dataset({var_name: new_var})
            else:
                print(f"Warning: Unexpected shape {vals.shape} for {var_name}"); return None
        except Exception as e:
            print(f"Error loading {var_name}: {e}"); return None
    if len(datasets) != len(var_folders):
        print(f"Warning: Only loaded {len(datasets)}/{len(var_folders)} variables"); return None
    try:
        return xr.merge(list(datasets.values()), compat='identical')
    except Exception as e:
        print(f"Error merging datasets: {e}"); return None
# ================================================================

def main():
    start_time = time.time()

    # 1) Load new models and thresholds
    print("Loading models & preprocessors from:", MODEL_DIR)
    encoder        = joblib.load(os.path.join(MODEL_DIR, 'encoder_biome_month.pkl'))
    scaler         = joblib.load(os.path.join(MODEL_DIR, 'scaler_features.pkl'))
    fire_model     = joblib.load(os.path.join(MODEL_DIR, 'fire_model.pkl'))
    duration_model = joblib.load(os.path.join(MODEL_DIR, 'duration_model.pkl'))
    obe_model      = joblib.load(os.path.join(MODEL_DIR, 'obe_model.pkl'))
    h24_model      = joblib.load(os.path.join(MODEL_DIR, 'h24event_model.pkl'))
    thresholds     = joblib.load(os.path.join(MODEL_DIR, 'thresholds.pkl'))
    print(f"Thresholds: fire={thresholds['fire_threshold']:.4f}, "
          f"obe={thresholds['obe_threshold']:.4f}, h24={thresholds['h24_threshold']:.4f}")

    # 2) Variable folder mapping (unchanged)
    var_folders = {
        'BUI': 'Daily_BUI_NA',
        'DMC': 'Daily_DMC_NA',
        'DC':  'Daily_DC_NA',
        'FWI': 'Daily_FWI_NA',
        'FFMC':'Daily_FFMC_forProjection',
        'ISI': 'Daily_ISI_forProjection'
    }
    feature_vars = ['BUI','DMC','DC','FWI','FFMC','ISI']

    # 3) Load biome
    biome_da = xr.open_dataset(BIOME_NC)['gez_code']

    # 4) Read template grid (to align to 237×512)
    try:
        tmpl = xr.open_dataset(FUTURE_TEMPLATE_NC, decode_times=False)
        if 'lat' in tmpl.coords and 'latitude' not in tmpl.coords: tmpl = tmpl.rename({'lat':'latitude'})
        if 'lon' in tmpl.coords and 'longitude' not in tmpl.coords: tmpl = tmpl.rename({'lon':'longitude'})
        tmpl_lat = tmpl['latitude'].values
        tmpl_lon = tmpl['longitude'].values
        tmpl.close()
        print(f"[TEMPLATE] grid -> lat={tmpl_lat.size}, lon={tmpl_lon.size} (expect 237×512)")
        use_template = True
    except Exception as e:
        print(f"[TEMPLATE WARNING] fail to open template: {e}\nProceed without regridding.")
        tmpl_lat = tmpl_lon = None
        use_template = False

    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 5) Process year(s) (single year / a few years)
    for year in YEARS:
        print(f"\n{'='*30} Processing Year {year} {'='*30}")
        merged_ds = load_annual_data(BASE_PATH, year, var_folders)
        if merged_ds is None:
            print(f"[SKIP] cannot load inputs for {year}"); continue

        # --- Regrid to template grid --- #
        if use_template:
            try:
                merged_ds = merged_ds.interp(latitude=tmpl_lat, longitude=tmpl_lon, method='nearest')
                print(f"[Regrid] {year} -> lat={merged_ds.dims['latitude']}, lon={merged_ds.dims['longitude']}")
            except Exception as e:
                print(f"[Regrid WARNING] {year}: {e}")

        fw_lats = merged_ds.latitude.values
        fw_lons = merged_ds.longitude.values
        times   = merged_ds.time.values
        num_days= len(times)

        # Biome on the same grid
        extended_biome = extend_biome_data(biome_da, fw_lons, fw_lats)

        # Output containers
        shape = (num_days, fw_lats.size, fw_lons.size)
        fire_prob = np.full(shape, np.nan, dtype=np.float32)
        fire_pred = np.full(shape, np.nan, dtype=np.float32)
        dur_pred  = np.full(shape, np.nan, dtype=np.float32)
        obe_prob  = np.full(shape, np.nan, dtype=np.float32)
        obe_pred  = np.full(shape, np.nan, dtype=np.float32)
        h24_prob  = np.full(shape, np.nan, dtype=np.float32)
        h24_pred  = np.full(shape, np.nan, dtype=np.float32)

        # Month indexing (handles non-datetime time)
        monthly_indices = [[] for _ in range(12)]
        july_1st = []
        for i, t in enumerate(times):
            try:
                dt = pd.Timestamp(t); m = dt.month; d = dt.day
            except Exception:
                day_of_year = (i % 365) + 1
                edges = [0,31,59,90,120,151,181,212,243,273,304,334,365]
                m = next(k for k, e in enumerate(edges[1:], start=1) if day_of_year <= e)
                d = day_of_year - edges[m-1]
            monthly_indices[m-1].append(i)
            if m == 7 and d == 1: july_1st.append(i)

        # Inference
        pbar = tqdm(total=num_days, desc=f"{year} days")
        batch_size = 5
        for s in range(0, num_days, batch_size):
            e = min(s + batch_size, num_days)
            for tidx in range(s, e):
                try:
                    try:
                        month = pd.Timestamp(times[tidx]).month
                    except Exception:
                        for m in range(12):
                            if tidx in monthly_indices[m]: month = m+1; break

                    # Extract the six variables for this day
                    day = {v: merged_ds[v].isel(time=tidx).values for v in feature_vars}

                    miss = np.zeros((fw_lats.size, fw_lons.size), dtype=bool)
                    for v in feature_vars: miss |= np.isnan(day[v])
                    miss |= np.isnan(extended_biome.values)
                    valid = ~miss

                    if valid.any():
                        idx = np.where(valid.ravel())[0]

                        df = pd.DataFrame({v: day[v].ravel()[idx] for v in feature_vars})
                        df['biome'] = extended_biome.values.ravel()[idx]
                        df['month'] = month

                        oh = encoder.transform(df[['biome','month']])
                        X  = pd.concat([df[feature_vars].reset_index(drop=True),
                                        pd.DataFrame(oh, columns=encoder.get_feature_names_out(['biome','month']))],
                                       axis=1)
                        X[feature_vars] = scaler.transform(X[feature_vars])

                        p_fire = fire_model.predict_proba(X)[:,1]
                        y_fire = (p_fire >= thresholds['fire_threshold']).astype(np.float32)

                        p_obe  = obe_model.predict_proba(X)[:,1]
                        y_obe  = (p_obe >= thresholds['obe_threshold']).astype(np.float32)

                        p_h24  = h24_model.predict_proba(X)[:,1]
                        y_h24  = (p_h24 >= thresholds['h24_threshold']).astype(np.float32)

                        y_dur  = np.zeros_like(y_fire)
                        fidx   = np.where(y_fire == 1)[0]
                        if fidx.size > 0:
                            y_dur[fidx] = duration_model.predict(X.iloc[fidx])

                        # Write back to grid
                        tmp_fire_prob = np.full((fw_lats.size, fw_lons.size), np.nan, np.float32)
                        tmp_fire_pred = np.full((fw_lats.size, fw_lons.size), np.nan, np.float32)
                        tmp_obe_prob  = np.full((fw_lats.size, fw_lons.size), np.nan, np.float32)
                        tmp_obe_pred  = np.full((fw_lats.size, fw_lons.size), np.nan, np.float32)
                        tmp_h24_prob  = np.full((fw_lats.size, fw_lons.size), np.nan, np.float32)
                        tmp_h24_pred  = np.full((fw_lats.size, fw_lons.size), np.nan, np.float32)
                        tmp_dur       = np.full((fw_lats.size, fw_lons.size), np.nan, np.float32)

                        vflat = valid.ravel()
                        tmp_fire_prob.ravel()[vflat] = p_fire
                        tmp_fire_pred.ravel()[vflat] = y_fire
                        tmp_obe_prob.ravel()[vflat]  = p_obe
                        tmp_obe_pred.ravel()[vflat]  = y_obe
                        tmp_h24_prob.ravel()[vflat]  = p_h24
                        tmp_h24_pred.ravel()[vflat]  = y_h24
                        tmp_dur.ravel()[vflat]       = y_dur

                        fire_prob[tidx] = tmp_fire_prob
                        fire_pred[tidx] = tmp_fire_pred
                        obe_prob[tidx]  = tmp_obe_prob
                        obe_pred[tidx]  = tmp_obe_pred
                        h24_prob[tidx]  = tmp_h24_prob
                        h24_pred[tidx]  = tmp_h24_pred
                        dur_pred[tidx]  = tmp_dur

                        # Optional: plot July 1 maps
                        if tidx in july_1st:
                            for v in feature_vars:
                                da = xr.DataArray(day[v], coords={'latitude': fw_lats, 'longitude': fw_lons},
                                                  dims=['latitude','longitude'])
                                plot_spatial_predictions(
                                    da, f"{v} on July 1, {year} (Baseline)",
                                    os.path.join(PLOTS_DIR, f"Baseline_{year}_July1_{v}.png"),
                                    cmap='viridis'
                                )
                    pbar.update(1)
                except Exception as e:
                    print(f"[WARN] Day {tidx} error: {e}"); pbar.update(1)
        pbar.close()

        # Save annual results (grid is already 237×512)
        ds_out = xr.Dataset(
            data_vars=dict(
                fire_probability=(['time','latitude','longitude'], fire_prob),
                fire_prediction =(['time','latitude','longitude'], fire_pred),
                duration_class  =(['time','latitude','longitude'], dur_pred),
                obe_probability =(['time','latitude','longitude'], obe_prob),
                obe_prediction  =(['time','latitude','longitude'], obe_pred),
                h24_probability =(['time','latitude','longitude'], h24_prob),
                h24_prediction  =(['time','latitude','longitude'], h24_pred),
            ),
            coords=dict(time=times, latitude=fw_lats, longitude=fw_lons),
            attrs=dict(
                description=f'Baseline single-year prediction ({year}) with models_skl161',
                created=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                year=year,
                fire_threshold=thresholds['fire_threshold'],
                obe_threshold =thresholds['obe_threshold'],
                h24_threshold =thresholds['h24_threshold'],
            )
        )
        out_nc = os.path.join(RESULTS_DIR, f"baseline_{year}_{current_time}.nc")
        enc = {v: {'zlib': True, 'complevel': 4} for v in ds_out.data_vars}
        ds_out.to_netcdf(out_nc, encoding=enc)
        print(f"[SAVE] {out_nc}")

        # Quick stats
        print("Basic stats:",
              f"fire={np.nanmean(fire_pred):.4f},",
              f"obe={np.nanmean(obe_pred):.4f},",
              f"h24={np.nanmean(h24_pred):.4f}")

    # Finish
    dt = time.time() - start_time
    print(f"\nDone. Total time: {dt:.2f}s ({dt/60:.2f} min)")

if __name__ == "__main__":
    main()
