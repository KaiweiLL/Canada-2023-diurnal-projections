# -*- coding: utf-8 -*-
"""
Created on Sat Jan 24 11:26:13 2026

@author: Kaiwei Luo
"""

# -*- coding: utf-8 -*-
"""
Future projection script for ABDp and OBEp using CMIP6 climate models.
- Processes multiple GCMs and SSP scenarios.
- Implements batch processing to optimize memory usage.
- Includes optional compression for output NetCDF files.
"""
import os, re, glob, time, warnings, argparse
import numpy as np
import pandas as pd
import xarray as xr
import joblib
from tqdm import tqdm
from datetime import datetime

# ================== Paths (same structure as before; outputs go to a dedicated output root) ==================
ROOT      = r"<DATA_ROOT>\CMIP6_final"
MODEL_DIR = r"<MODEL_ROOT>\models_skl161"
OUT_DIR   = r"<OUTPUT_ROOT>\outputs_future"
PLOTS_DIR = r"<OUTPUT_ROOT>\projection_plots"   # Placeholder; not required to be used
BIOME_NC  = r"<DATA_ROOT>\biome_hourmodeling.nc"  # Kept as a separate data root
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# ================== Constants & default filtering switches (can be overridden by CLI) ==================
FWI_VARS = ['BUI','DMC','DC','FWI','FFMC','ISI']

# You can change defaults here; or override via command line arguments
YEAR_ONLY_DEFAULT   = None          # None = run both years; or set "2040" / "2070"
SSP_ONLY_DEFAULT    = None          # None = run both 245/370; or set like {"245","370"} / {"245"}
MODELS_ONLY_DEFAULT = None          # None = run all; or set like {"CANESM","UKESM"}

# Whether to compress outputs (for speed, recommended False; compress later in a separate pass)
COMPRESS_OUT = False

# How many "days" to read per batch (I/O batching; still predicts day-by-day)
BATCH_DAYS = 10

# ================== Utility functions ==================
def log(msg): print(msg, flush=True)

def infer_scene_model(path):
    # Filename pattern: 2040_CANESM245_BUI.nc or 2070_ECEARTH370_ISI.nc
    name = os.path.basename(path)
    m = re.match(r'^(?P<yr>2040|2070)_(?P<m>[A-Za-z]+)(?P<ssn>245|370)_(?P<var>[A-Za-z]+)\.nc$', name, re.I)
    if not m:
        return None
    return dict(scene=m['yr']+' '+m['ssn'], model=m['m'].upper(), var=m['var'].upper())

def month_from_time_index(time_index_len):
    day = np.arange(time_index_len) % 365 + 1
    edges = np.array([0,31,59,90,120,151,181,212,243,273,304,334,365])
    month = np.searchsorted(edges, day, side='right')  # 1..12
    return month.astype(np.int16), day.astype(np.int16)

def open_first_dataarray(nc_path):
    ds = xr.open_dataset(nc_path, decode_times=False)
    for v in ds.data_vars:
        if v.lower() == 'crs':
            continue
        da = ds[v]
        break
    else:
        raise RuntimeError(f"{nc_path}: 未找到数据变量（只有 crs?）")
    need = ['time','latitude','longitude']
    if any(k not in da.dims for k in need):
        raise RuntimeError(f"{os.path.basename(nc_path)} 维度不是 {need}，而是 {list(da.dims)}")
    return da.transpose('time','latitude','longitude')

def extend_biome_to_grid(biome_da, target_lats, target_lons):
    # Standardize biome coordinate names to latitude/longitude and nearest-neighbor interpolate to the target grid
    b = biome_da
    if 'latitude' not in b.coords or 'longitude' not in b.coords:
        rename_map = {}
        if 'lat' in b.coords: rename_map['lat'] = 'latitude'
        if 'lon' in b.coords: rename_map['lon'] = 'longitude'
        b = b.rename(rename_map)
    b2 = b.interp(latitude=target_lats, longitude=target_lons, method='nearest')
    return b2.astype(np.int16)

def load_models(model_dir):
    log(f"[Load] models from {model_dir}")
    def _L(n):
        p = os.path.join(model_dir, n)
        if not os.path.exists(p): raise FileNotFoundError(p)
        o = joblib.load(p); log(f"  - {n} loaded"); return o
    return dict(
        encoder=_L('encoder_biome_month.pkl'),
        scaler =_L('scaler_features.pkl'),
        fire   =_L('fire_model.pkl'),
        duration=_L('duration_model.pkl'),
        obe    =_L('obe_model.pkl'),
        h24    =_L('h24event_model.pkl'),
        thr    =_L('thresholds.pkl'),
    )

def gather_files(root):
    """Return {(scene, model): {var: path}}; keep only combinations that include all six required variables."""
    files = glob.glob(os.path.join(root, '*', '*.nc'))
    buckets = {}
    keep = set(FWI_VARS) | {'FWI','PREC','RH','TEMP','WS'}
    for fp in files:
        meta = infer_scene_model(fp)
        if not meta:
            continue
        if meta['var'] not in keep:
            continue
        key = (meta['scene'], meta['model'])
        buckets.setdefault(key, {})
        buckets[key][meta['var']] = fp
    ok = {k:v for k,v in buckets.items() if set(FWI_VARS).issubset(v)}
    return ok

def filter_buckets(buckets, year=None, ssp=None, models=None):
    """Filter {(scene, model): paths} returned by gather_files()."""
    out = {}
    for (scene, model), paths in buckets.items():
        yr, ss = scene.split()   # scene = "2040 245"
        if year   is not None and yr != str(year):   # Year filter
            continue
        if ssp    is not None and ss not in set(map(str, ssp)):  # Scenario filter
            continue
        if models is not None and model not in set(models):      # Model filter
            continue
        out[(scene, model)] = paths
    return out

# ================== Core: stable day-by-day projection (keeps the original logic) ==================
def project_one(scene, model, paths, models):
    t0 = time.time()
    log(f"\n[RUN] {scene} × {model}")

    # Read six variable DataArrays and standardize to (time, lat, lon)
    das = {v: open_first_dataarray(paths[v]) for v in FWI_VARS}
    time_coord = das['BUI']['time']
    lat = das['BUI']['latitude']
    lon = das['BUI']['longitude']
    T, Ny, Nx = time_coord.size, lat.size, lon.size
    log(f"  [Grid] T={T}, lat={Ny}, lon={Nx}")

    # Month vector (1..12)
    month_vec, _ = month_from_time_index(T)
    log("  [Feat] month ready")

    # Biome (read once, interpolate once)
    with xr.open_dataset(BIOME_NC) as bds:
        if 'gez_code' not in bds.data_vars:
            raise SystemExit(f"biome 文件里没找到变量 'gez_code'：{BIOME_NC}")
        biome2d = extend_biome_to_grid(bds['gez_code'], lat, lon)  # (Ny, Nx)
    log("  [Feat] biome grid ready")

    # Output containers (keep variable names consistent with the original script)
    shape = (T, Ny, Nx)
    fire_prob = np.full(shape, np.nan, dtype=np.float32)
    fire_pred = np.full(shape, np.nan, dtype=np.float32)
    dur_pred  = np.full(shape, np.nan, dtype=np.float32)
    obe_prob  = np.full(shape, np.nan, dtype=np.float32)
    obe_pred  = np.full(shape, np.nan, dtype=np.float32)
    h24_prob  = np.full(shape, np.nan, dtype=np.float32)
    h24_pred  = np.full(shape, np.nan, dtype=np.float32)

    # Model objects and preprocessors
    enc  = models['encoder']
    scl  = models['scaler']
    thr  = models['thr']
    mdl_fire = models['fire']
    mdl_dur  = models['duration']
    mdl_obe  = models['obe']
    mdl_h24  = models['h24']

    # Day-wise progress bar
    pbar = tqdm(total=T, desc=f"{scene} {model}", dynamic_ncols=True, mininterval=2.0, leave=True)

    # Batch reading (I/O only; still predicts day-by-day)
    for s in range(0, T, BATCH_DAYS):
        e = min(T, s + BATCH_DAYS)
        # (b, Ny, Nx)
        batch = {v: das[v].isel(time=slice(s, e)).load().values for v in FWI_VARS}  # Load this small batch into memory
        b = e - s

        for i in range(b):
            # --- 1) Daily six variables (Ny, Nx)
            day_data = {v: batch[v][i] for v in FWI_VARS}

            # --- 2) Valid mask ---
            missing = np.zeros((Ny, Nx), dtype=bool)
            for v in FWI_VARS:
                missing |= np.isnan(day_data[v])
            missing |= np.isnan(biome2d.values)
            valid = ~missing

            if not valid.any():
                # Entire day is missing
                fire_prob[s+i] = np.nan; fire_pred[s+i] = np.nan
                obe_prob[s+i]  = np.nan; obe_pred[s+i]  = np.nan
                h24_prob[s+i]  = np.nan; h24_pred[s+i]  = np.nan
                dur_pred[s+i]  = np.nan
                pbar.update(1)
                continue

            # --- 3) Build per-day valid-pixel features (strictly day-by-day, stable) ---
            idx = np.where(valid.ravel())[0]  # length K
            K = idx.size

            # Numeric features (K, 6)
            X_num = np.column_stack([day_data[v].ravel()[idx] for v in FWI_VARS]).astype(np.float32)

            # Categorical features: biome, month (K, 1)
            biome_col = biome2d.values.ravel()[idx].astype(np.int64).reshape(-1,1)
            month_val = np.full((K,1), month_vec[s+i], dtype=np.int64)

            # Encode + standardize
            X_cat_enc = enc.transform(np.column_stack([biome_col, month_val]))
            X_num_std = scl.transform(X_num)
            X = np.concatenate([X_num_std, X_cat_enc], axis=1)   # Correct: numeric first, one-hot after


            # --- 4) Predict (same logic as the original) ---
            p_fire = mdl_fire.predict_proba(X)[:, 1].astype(np.float32)
            y_fire = (p_fire >= thr['fire_threshold']).astype(np.uint8)

            p_obe  = mdl_obe.predict_proba(X)[:, 1].astype(np.float32)
            y_obe  = (p_obe >= thr['obe_threshold']).astype(np.uint8)

            p_h24  = mdl_h24.predict_proba(X)[:, 1].astype(np.float32)
            y_h24  = (p_h24 >= thr['h24_threshold']).astype(np.uint8)

            y_dur = np.zeros_like(y_fire, dtype=np.uint8)
            if y_fire.any():
                fire_idx = np.where(y_fire == 1)[0]
                if fire_idx.size > 0:
                    y_dur[fire_idx] = mdl_dur.predict(X[fire_idx]).astype(np.uint8)

            # --- 5) Write back to grid ---
            fire_prob[s+i].ravel()[idx] = p_fire
            fire_pred[s+i].ravel()[idx] = y_fire
            obe_prob[s+i].ravel()[idx]  = p_obe
            obe_pred[s+i].ravel()[idx]  = y_obe
            h24_prob[s+i].ravel()[idx]  = p_h24
            h24_pred[s+i].ravel()[idx]  = y_h24
            dur_pred[s+i].ravel()[idx]  = y_dur

            pbar.update(1)

    pbar.close()

    # Package & write output (variable names kept consistent with the original code)
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
        coords=dict(time=time_coord, latitude=lat, longitude=lon),
        attrs=dict(
            description=f'Fire prediction results for {scene} {model}',
            created=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            scene=scene, model=model,
            fire_threshold=models['thr']['fire_threshold'],
            obe_threshold =models['thr']['obe_threshold'],
            h24_threshold =models['thr']['h24_threshold'],
        )
    )
    out_nc = os.path.join(OUT_DIR, f"{scene.replace(' ','_')}_{model}_pred.nc")
    encoding = None if not COMPRESS_OUT else {v: {'zlib': True, 'complevel': 4} for v in ds_out.data_vars}
    ds_out.to_netcdf(out_nc, encoding=encoding)
    log(f"  [SAVE] {out_nc}  ({time.time()-t0:.1f}s)")
    return out_nc

# ================== Main flow (supports CLI overrides for filtering) ==================
def parse_args():
    ap = argparse.ArgumentParser(description="Stable CMIP6 future projection (day-by-day).")
    ap.add_argument("--year",   choices=["2040","2070"], nargs="*", help="只跑这些年份（可多选）")
    ap.add_argument("--ssp",    choices=["245","370"],   nargs="*", help="只跑这些情景（可多选）")
    ap.add_argument("--model",  choices=["CANESM","ECEARTH","GFDL","UKESM"], nargs="*", help="只跑这些气候模型（可多选）")
    return ap.parse_args()

def main():
    warnings.filterwarnings("ignore")
    xr.set_options(keep_attrs=True)

    # 1) Load models
    models = load_models(MODEL_DIR)

    # 2) Gather all available combinations
    buckets_all = gather_files(ROOT)
    if not buckets_all:
        raise SystemExit(f"[EMPTY] 没在 {ROOT} 找到匹配的六变量组合")

    # 3) Parse CLI args and override default filters
    args = parse_args()
    # Year: if specified on CLI, use a set; otherwise use default
    year_only = set(args.year) if args.year else (set([YEAR_ONLY_DEFAULT]) if YEAR_ONLY_DEFAULT else None)
    # SSP
    ssp_only  = set(args.ssp) if args.ssp else (set(SSP_ONLY_DEFAULT) if SSP_ONLY_DEFAULT else None)
    # Model
    models_only = set(args.model) if args.model else (set(MODELS_ONLY_DEFAULT) if MODELS_ONLY_DEFAULT else None)

    # If year_only is a set, expand over years; filter_buckets expects a single year or None.
    if isinstance(year_only, set):
        filtered = {}
        for yr in year_only:
            filtered.update(filter_buckets(buckets_all, year=yr, ssp=ssp_only, models=models_only))
        buckets = filtered
    else:
        buckets = filter_buckets(buckets_all, year=year_only, ssp=ssp_only, models=models_only)

    print(f"[Found] 总组合: {len(buckets_all)}  |  过滤后: {len(buckets)}")
    if not buckets:
        raise SystemExit("[EMPTY] 过滤条件过严，已无可运行组合")
    for (scene, model) in sorted(buckets.keys()):
        vars_short = ", ".join(sorted(buckets[(scene,model)].keys()))
        print(f"  - {scene} × {model} | {vars_short}")

    # 4) Run each combination
    logs=[]
    for key, paths in sorted(buckets.items()):
        try:
            out_nc = project_one(key[0], key[1], paths, models)
            logs.append({'scene':key[0],'model':key[1],'status':'ok','out':out_nc})
        except Exception as e:
            print(f"[FAIL] {key}: {e}")
            logs.append({'scene':key[0],'model':key[1],'status':'fail','err':str(e)})

    pd.DataFrame(logs).to_csv(os.path.join(OUT_DIR, "run_log.csv"), index=False)
    print(f"[DONE] logs -> {os.path.join(OUT_DIR, 'run_log.csv')}")

if __name__ == "__main__":
    main()






import os
import glob
from tqdm import tqdm
import xarray as xr

# Input/output
input_dir  = r"<OUTPUT_ROOT>\outputs_future"
output_dir = r"<OUTPUT_ROOT>\outputs_future_compressed"
os.makedirs(output_dir, exist_ok=True)

# Find nc files
nc_files = glob.glob(os.path.join(input_dir, "*.nc"))
print(f"找到 {len(nc_files)} 个 NetCDF 文件")

def detect_time_dim(ds):
    # Prefer common names first
    for cand in ("time", "day", "Time"):
        if cand in ds.dims:
            return cand
    # Fallback: guess from the first dimension of the first data variable (unlikely to be needed)
    for v in ds.data_vars:
        if ds[v].dims:
            return ds[v].dims[0]
    raise ValueError("找不到时间维度；请检查文件结构")

def pick_chunk(n, target):
    # Choose a chunk size not exceeding n
    return int(min(max(1, target), int(n)))

for file_path in tqdm(nc_files, desc="压缩文件"):
    file_name   = os.path.basename(file_path)
    output_path = os.path.join(output_dir, file_name)

    # Skip if already exists
    if os.path.exists(output_path):
        print(f"跳过已存在文件: {file_name}")
        continue

    try:
        # Open once to get dimension info (no computation triggered)
        with xr.open_dataset(file_path) as ds_head:
            dims = {d: int(ds_head.dims[d]) for d in ds_head.dims}
            time_dim = detect_time_dim(ds_head)

        # Reasonable default chunk sizes (designed for ~10950×237×512, but adapts to other sizes)
        t_chunk  = pick_chunk(dims.get(time_dim, 1), 90)   # 90 days along time
        y_chunk  = pick_chunk(dims.get("latitude",  1), 64)
        x_chunk  = pick_chunk(dims.get("longitude", 1), 128)

        # Build encoding: compress data variables only; do not compress coordinates
        encoding = {}
        with xr.open_dataset(file_path, chunks={time_dim: t_chunk}) as ds:
            for var in ds.data_vars:
                var_dims = ds[var].dims
                # Set per-dimension chunksize (dims order may differ across variables)
                chunksizes = []
                for d in var_dims:
                    if d == time_dim:
                        chunksizes.append(t_chunk)
                    elif d == "latitude":
                        chunksizes.append(y_chunk)
                    elif d == "longitude":
                        chunksizes.append(x_chunk)
                    else:
                        # Other uncommon dims: full-dim chunk (or moderate chunk)
                        chunksizes.append(pick_chunk(ds.dims[d], ds.dims[d]))

                enc = {
                    "zlib": True,
                    "complevel": 5,
                    "shuffle": True,
                    "dtype": ds[var].dtype,                  # Preserve original dtype
                    "chunksizes": tuple(chunksizes),         # Ensure not exceeding each dim length
                }

                # Preserve existing _FillValue if present; otherwise do not force one (avoids dtype conflicts)
                fv = ds[var].encoding.get("_FillValue", None)
                if fv is not None:
                    enc["_FillValue"] = fv

                encoding[var] = enc

            # Write compressed file (netcdf4 recommended; switch engine if needed)
            print(f"开始保存压缩文件: {output_path}")
            ds.to_netcdf(output_path, encoding=encoding, engine="netcdf4")  # compute=True by default
            print(f"完成保存: {output_path}")

        # Report compression ratio
        orig_gb = os.path.getsize(file_path)  / (1024**3)
        comp_gb = os.path.getsize(output_path) / (1024**3)
        red = (1 - comp_gb / orig_gb) * 100 if orig_gb > 0 else 0
        print(f"{file_name}: 原始 {orig_gb:.2f} GB → 压缩 {comp_gb:.2f} GB（↓{red:.2f}%）")

    except Exception as e:
        print(f"处理文件 {file_path} 时出错: {e}")
        import traceback; traceback.print_exc()

print("完成所有文件压缩")
