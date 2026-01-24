# -*- coding: utf-8 -*-
"""
新分析脚本（符合你的最新口径）
- 仅分析 fire_probability / obe_probability（h24 已移除）
- biome 使用带小数 ID 的 NC（不能自行切分 east/west）
- baseline 时间轴按“旧逻辑”：若日序列比目标日期多，则剔除 2/29 以对齐 365×N
- 阈值强制来自新训练产物 thresholds.pkl（若数据 attrs 也有则校验一致，否则报错）
- 未来情景初始支持 245/370，可扩展到 126/585
- 强检空间对齐（lat/lon 坐标与长度完全一致，否则报错）
- 输出表格、变化图（nc）与 png 静态图

作者：你（整合 by ChatGPT）
"""

import os, re, glob, json, warnings, gc
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from datetime import datetime

warnings.filterwarnings('ignore')

# ========= 路径配置（按需修改） =========
MODEL_DIR      = r"E:\Projection paper\models_skl161"               # 新训练产物目录（必须含 thresholds.pkl）
BASELINE_DIR   = r"E:\Projection paper\outputs_baseline"            # 新 baseline 输出目录
FUTURE_DIR     = r"E:\Projection paper\outputs_future"              # 新 future 输出目录
BIOME_DEC_NC   = r"D:\000_collections\020_Chapter2\US_CAN_biome.nc" # 带“小数 ID”的 biome nc（变量名见 BIO_VAR）
BIO_VAR        = "gez_code_id"                                      # 带小数的变量名（按你的旧分析）
OUT_ROOT       = r"E:\Projection paper\analysis"                     # 分析输出根目录

os.makedirs(os.path.join(OUT_ROOT, "tables"), exist_ok=True)
os.makedirs(os.path.join(OUT_ROOT, "maps"),   exist_ok=True)
os.makedirs(os.path.join(OUT_ROOT, "plots"),  exist_ok=True)
os.makedirs(os.path.join(OUT_ROOT, "logs"),   exist_ok=True)

# ========= 基本设置 =========
BASE_START = "1991-01-01"  # baseline 起始日期（与旧逻辑一致）
FUTURE_SCENARIOS = ['245','370']   # 初始情景；后续可扩展到 ['126','245','370','585']
FUTURE_YEARS     = ['2040','2070'] # 2040→2041-2070, 2070→2071-2100（文件名中的“2040/2070”语义）
FUTURE_MODELS    = ['CANESM','ECEARTH','GFDL','UKESM']
VARS             = ['fire_probability','obe_probability']  # 仅两项

# biome 名称映射（带小数 key）——完全使用你给的映射
biome_name_map = {
    41.1: "Boreal coniferous forest east",
    41.2: "Boreal coniferous forest west",
    43.1: "Boreal mountain system",
    42.1: "Boreal tundra woodland east",
    42.2: "Boreal tundra woodland west",
    50.1: "Polar",
    24.1: "Subtropical desert",
    22.1: "Subtropical dry forest",
    21.1: "Subtropical humid forest",
    25.1: "Subtropical mountain system",
    23.1: "Subtropical steppe",
    32.1: "Temperate continental forest",
    34.1: "Temperate desert",
    35.1: "Temperate mountain system east",
    35.2: "Temperate mountain system west",
    31.1: "Temperate oceanic forest",
    33.1: "Temperate steppe",
    13.1: "Tropical dry forest",
    12.1: "Tropical moist forest",
    90.1: "Water",
}
EXCLUDE_BIOMES = {50.1, 90.1}  # 排除 Polar / Water

# ========= 工具函数 =========
def log(msg):
    print(msg, flush=True)

import joblib  # 确保已在顶部 import

def _read_thresholds_from_model_dir(model_dir: str) -> dict:
    """
    从训练产物目录读取 thresholds.pkl，必须包含：
    - fire_threshold
    - obe_threshold
    """
    p = os.path.join(model_dir, "thresholds.pkl")
    if not os.path.exists(p):
        raise FileNotFoundError(
            f"[ERROR] 找不到阈值文件：{p}\n"
            f"请确认训练脚本已在该目录保存 thresholds.pkl，或修改 MODEL_DIR 指向正确位置。"
        )
    try:
        th = joblib.load(p)
    except Exception as e:
        raise RuntimeError(f"[ERROR] 读取 {p} 失败：{e}")
    # 校验键是否齐全
    required = ["fire_threshold", "obe_threshold"]
    missing = [k for k in required if k not in th]
    if missing:
        raise KeyError(f"[ERROR] thresholds.pkl 缺少字段：{missing}，请检查训练输出。")
    # 转成 float，避免后续类型问题
    return {k: float(th[k]) for k in required}
def _read_thresholds_from_attrs(ds):
    # 若 ds.attrs 含阈值，读出来
    keys = ['fire_threshold','obe_threshold']
    found = {}
    for k in keys:
        if k in ds.attrs:
            try:
                found[k] = float(ds.attrs[k])
            except:
                pass
    return found

def get_thresholds_or_error(ds_list, model_dir: str) -> dict:
    """
    阈值以模型产物 thresholds.pkl 为准。
    若传入的数据集中 attrs 也带阈值，则进行一致性校验，不一致报错。
    """
    th_model = _read_thresholds_from_model_dir(model_dir)
    # 校验数据集 attrs（如果存在）
    for ds in ds_list:
        for k, v_model in th_model.items():
            if k in ds.attrs:
                try:
                    v_attr = float(ds.attrs[k])
                    if not np.isfinite(v_attr):
                        continue
                    if abs(v_attr - v_model) > 1e-6:
                        raise ValueError(
                            f"[ERROR] 数据集 attrs 中 {k}={v_attr} 与模型阈值 {v_model} 不一致；"
                            f"请统一口径后再跑分析。"
                        )
                except Exception:
                    # attrs 不是数字就忽略校验
                    pass
    return th_model
def rebuild_time_baseline(ds, start_date=BASE_START):
    """
    旧逻辑：
    - 以 start_date 构造按日的 DatetimeIndex；
    - 若新时间长度 > ds 原 time 长度，则剔除 2/29 使其匹配（保证 365×N）
    """
    nday = ds.sizes['time']
    newt = pd.date_range(start_date, periods=nday, freq='D')
    if len(newt) > nday:
        # 剔除闰日（2/29）
        newt = newt[~((newt.month == 2) & (newt.day == 29))]
    if len(newt) != nday:
        # 若仍不等，按旧脚本策略：再次剔除所有 2/29 并截断
        newt2 = pd.date_range(start_date, periods=nday+60, freq='D')
        newt2 = newt2[~((newt2.month == 2) & (newt2.day == 29))]
        newt = newt2[:nday]
    return ds.assign_coords(time=newt)

def rebuild_time_future(ds, file_year_tag):
    """
    未来时间轴重建：
    - file_year_tag: "2040"→从 2041-01-01 开始；"2070"→从 2071-01-01 开始
    - 强制 365×N：如果构造长度比 ds 短或长，优先按 2/29 剔除以对齐
    """
    nday = ds.sizes['time']
    if str(file_year_tag) == '2040':
        start_date = '2041-01-01'
    elif str(file_year_tag) == '2070':
        start_date = '2071-01-01'
    else:
        raise ValueError(f"[ERROR] 未知 future year 标签: {file_year_tag}")
    newt = pd.date_range(start_date, periods=nday, freq='D')
    if len(newt) > nday:
        newt = newt[~((newt.month == 2) & (newt.day == 29))]
    if len(newt) != nday:
        newt2 = pd.date_range(start_date, periods=nday+60, freq='D')
        newt2 = newt2[~((newt2.month == 2) & (newt2.day == 29))]
        newt = newt2[:nday]
    return ds.assign_coords(time=newt)

def open_baseline_dataset():
    # 优先合并文件
    cand = sorted(glob.glob(os.path.join(BASELINE_DIR, "baseline_1991_2020_*.nc")))
    if cand:
        log(f"[BASELINE] 使用合并文件: {os.path.basename(cand[0])}")
        return xr.open_dataset(cand[0])
    # 否则拼 per-year
    years = sorted(int(re.findall(r'baseline_(\d{4})_', os.path.basename(p))[0])
                   for p in glob.glob(os.path.join(BASELINE_DIR, "baseline_????_*.nc")))
    if not years:
        raise FileNotFoundError("[ERROR] baseline 文件不存在")
    # 串联（沿 time）
    parts = []
    for y in years:
        f = sorted(glob.glob(os.path.join(BASELINE_DIR, f"baseline_{y}_*.nc")))
        if not f:
            continue
        parts.append(xr.open_dataset(f[0]))
    ds = xr.concat(parts, dim='time')
    log(f"[BASELINE] 拼接 per-year: {years[0]}–{years[-1]} 共 {len(parts)} 个文件")
    return ds

def open_future_datasets():
    """
    返回 dict[(year_tag, ssp, model)] = xr.Dataset
    文件名约定：{2040|2070}_{SSP}_{MODEL}_pred_lite.nc
    """
    out = {}
    patt = re.compile(r'^(?P<yr>2040|2070)_(?P<ssp>126|245|370|585)_(?P<m>[A-Za-z]+)_pred_lite\.nc$', re.I)
    for fp in glob.glob(os.path.join(FUTURE_DIR, "*.nc")):
        name = os.path.basename(fp)
        m = patt.match(name)
        if not m:
            continue
        yr = m.group('yr')
        ssp = m.group('ssp')
        mod = m.group('m').upper()
        if yr not in FUTURE_YEARS:
            continue
        if ssp not in FUTURE_SCENARIOS:
            continue
        if mod not in FUTURE_MODELS:
            continue
        out[(yr, ssp, mod)] = xr.open_dataset(fp)
    if not out:
        raise FileNotFoundError("[ERROR] 未找到任何符合命名的 future 文件")
    log(f"[FUTURE] 发现 {len(out)} 组 (year_tag,ssp,model)")
    return out

def load_biome_decimal_to_grid(lat, lon):
    """
    读取带小数的 biome nc（BIO_VAR），最近邻插值到目标网格（基线网格），返回：
    - biome_da: DataArray (lat,lon) 小数 ID
    - masks: dict[biome_id_decimal] -> boolean mask
    """
    if not os.path.exists(BIOME_DEC_NC):
        raise FileNotFoundError(f"[ERROR] 生物群系NC不存在: {BIOME_DEC_NC}")
    dsb = xr.open_dataset(BIOME_DEC_NC)
    if BIO_VAR not in dsb:
        raise KeyError(f"[ERROR] {BIOME_DEC_NC} 中找不到变量 {BIO_VAR}")
    # 维度/坐标名兼容
    ren = {}
    if 'lat' in dsb.dims and 'latitude' not in dsb.dims: ren['lat'] = 'latitude'
    if 'lon' in dsb.dims and 'longitude' not in dsb.dims: ren['lon'] = 'longitude'
    if ren: dsb = dsb.rename(ren)
    biome_da = dsb[BIO_VAR].interp(latitude=lat, longitude=lon, method='nearest')
    vals = biome_da.values
    uniq = np.unique(vals[~np.isnan(vals)])
    masks = {}
    for bid in uniq:
        if float(bid) in EXCLUDE_BIOMES:
            continue
        masks[float(bid)] = (vals == bid)
    log(f"[BIOME] 有效 biome 数: {len(masks)}（排除了 {EXCLUDE_BIOMES}）")
    return biome_da, masks

def assert_same_grid(a, b, tag):
    # 强检：长度与坐标值
    for dim in ['latitude','longitude']:
        if a[dim].size != b[dim].size:
            raise ValueError(f"[ERROR] {tag} 网格 {dim} 长度不一致: {a[dim].size} vs {b[dim].size}")
        if not np.allclose(a[dim].values, b[dim].values, atol=1e-9):
            raise ValueError(f"[ERROR] {tag} 网格 {dim} 坐标不一致")
    log(f"[CHECK] {tag} 网格一致")

def mean_exceed_days_30yr(ds, var, thr):
    """
    返回 DataArray (lat,lon)：30 年均 exceed-days（逐年计数后取均值）
    """
    idx = pd.DatetimeIndex(ds.time.values)
    years = sorted(np.unique(idx.year))
    acc = None
    for y in years:
        da = ds[var].sel(time=str(y))
        cnt = (da > thr).sum(dim='time', skipna=True)
        acc = cnt if acc is None else acc + cnt
        del da, cnt
        gc.collect()
    mean_da = acc / len(years)
    return mean_da

def annual_exceed_days_by_biome(ds, var, thr, biome_masks, epoch, scenario, model):
    """
    生成长表记录：每 biome × 每年 的 exceed-days（像素平均）
    """
    idx = pd.DatetimeIndex(ds.time.values)
    years = sorted(np.unique(idx.year))
    rows = []
    for y in years:
        da = ds[var].sel(time=str(y))
        days_by_px = (da > thr).sum(dim='time', skipna=True).values  # (lat,lon)
        for bid, mask in biome_masks.items():
            arr = days_by_px[mask]
            if arr.size == 0: 
                continue
            rows.append({
                'biome_id': bid,
                'biome_name': biome_name_map.get(bid, f"Unknown_{bid}"),
                'epoch': epoch,                # baseline / future
                'scenario': scenario,          # baseline 或 ssp code
                'model': model,                # baseline 或 模式名
                'year': int(y),
                'variable': var,
                'threshold': float(thr),
                'exceed_days': float(np.nanmean(arr))
            })
    return rows

def save_change_nc(var, baseline_mean, future_mean, out_dir, scenario, model):
    abs_change = future_mean - baseline_mean
    with np.errstate(divide='ignore', invalid='ignore'):
        rel_change = (abs_change / baseline_mean) * 100.0
    rel_change = rel_change.where(~np.isinf(rel_change))
    ds_out = xr.Dataset(
        data_vars=dict(
            absolute_change=abs_change,
            relative_change=rel_change
        ),
        coords=dict(
            latitude=baseline_mean.latitude,
            longitude=baseline_mean.longitude
        ),
        attrs=dict(
            description=f"Change in 30-yr mean exceed-days for {var} "
                        f"(future: {scenario} {model} vs baseline 1991-2020)",
            created=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
    )
    out_nc = os.path.join(out_dir, f"{var}_days_{scenario}_{model}_change.nc")
    ds_out.to_netcdf(out_nc)
    log(f"[SAVE] 变化图: {out_nc}")
    return out_nc

def quick_map(da, title, out_png, vmin=None, vmax=None):
    plt.figure(figsize=(10,6))
    # 不引入 cartopy，直接经纬网格渲染
    plt.pcolormesh(da['longitude'], da['latitude'], da, shading='auto')
    plt.colorbar(label=title)
    plt.title(title)
    plt.xlabel("Longitude"); plt.ylabel("Latitude")
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()
    log(f"[PLOT] {out_png}")

def biome_barplot(df, title, out_png, topn=12):
    # 选取最近一年或平均值的前若干 biome 可视化
    plt.figure(figsize=(10,6))
    df2 = df.copy()
    g = df2.groupby(['biome_name'], as_index=False)['exceed_days'].mean()
    g = g.sort_values('exceed_days', ascending=False).head(topn)
    plt.barh(g['biome_name'], g['exceed_days'])
    plt.gca().invert_yaxis()
    plt.title(title)
    plt.xlabel("Exceed-days (mean)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()
    log(f"[PLOT] {out_png}")

# ========= 主流程 =========
def main():
    t0 = datetime.now()
    log(f"=== 分析开始 {t0.strftime('%Y-%m-%d %H:%M:%S')} ===")
    # 1) 打开 baseline & future
    ds_base_raw = open_baseline_dataset()
    fut_dict_raw = open_future_datasets()

    # 2) 时间轴重建
    ds_base = rebuild_time_baseline(ds_base_raw, BASE_START)
    fut_dict = {}
    for (yr, ssp, mod), ds in fut_dict_raw.items():
        fut_dict[(yr, ssp, mod)] = rebuild_time_future(ds, yr)

    # 3) 阈值（强制来自 MODEL_DIR；若数据 attrs 也有则校验一致）
    thresholds = get_thresholds_or_error([ds_base] + list(fut_dict.values()), MODEL_DIR)
    thr_fire = float(thresholds['fire_threshold'])
    thr_obe  = float(thresholds['obe_threshold'])
    log(f"[THR] fire={thr_fire:.4f}, obe={thr_obe:.4f}")

    # 4) 网格与 biome 掩膜（以 baseline 为基准）
    lat = ds_base['latitude'].values
    lon = ds_base['longitude'].values
    # 确认未来网格完全一致
    for key, ds in fut_dict.items():
        assert_same_grid(ds_base, ds, tag=f"baseline vs future {key}")
    biome_da, biome_masks = load_biome_decimal_to_grid(lat, lon)

    # 5) 计算 baseline 的 30 年均 exceed-days（像素级）与年度 biome 长表
    base_means = {}
    base_rows = []
    for var in VARS:
        if var not in ds_base:
            raise KeyError(f"[ERROR] baseline 缺变量 {var}")
        thr = thr_fire if var.startswith('fire_') else thr_obe
        base_means[var] = mean_exceed_days_30yr(ds_base, var, thr)
        base_rows += annual_exceed_days_by_biome(ds_base, var, thr, biome_masks,
                                                 epoch='baseline', scenario='baseline', model='baseline')

        # 画基线 30 年均值地图
        quick_map(base_means[var], f"Baseline (1991-2020) mean exceed-days: {var}",
                  os.path.join(OUT_ROOT, "plots", f"baseline_{var}_mean30.png"))

    # 6) 计算每个未来组合的 30 年均值、年度 biome 表与变化图
    fut_rows = []
    for (yr, ssp, mod), ds in sorted(fut_dict.items()):
        for var in VARS:
            if var not in ds:
                raise KeyError(f"[ERROR] future {yr}-{ssp}-{mod} 缺变量 {var}")
            thr = thr_fire if var.startswith('fire_') else thr_obe
            fut_mean = mean_exceed_days_30yr(ds, var, thr)
            # 保存变化 nc
            save_change_nc(var, base_means[var], fut_mean,
                           out_dir=os.path.join(OUT_ROOT, "maps"),
                           scenario=f"{yr}_{ssp}", model=mod)
            # 画变化 quick map
            quick_map((fut_mean - base_means[var]),
                      f"Change in mean exceed-days ({var}) — {yr}_{ssp} {mod}",
                      os.path.join(OUT_ROOT, "plots", f"chg_{var}_{yr}_{ssp}_{mod}.png"))

            # 年度 biome 表
            fut_rows += annual_exceed_days_by_biome(ds, var, thr, biome_masks,
                                                    epoch='future', scenario=f"{yr}_{ssp}", model=mod)

    # 7) 表格落盘
    tbl_dir = os.path.join(OUT_ROOT, "tables")
    annual_df = pd.DataFrame(base_rows + fut_rows)
    annual_csv = os.path.join(tbl_dir, "annual_exceed_days.csv")
    annual_df.to_csv(annual_csv, index=False, encoding="utf-8-sig")
    log(f"[SAVE] 年度长表: {annual_csv}")

    # 30 年平均（按 biome/epoch/scenario/model/var 聚合）
    avg30_df = (annual_df
                .groupby(['biome_id','biome_name','epoch','scenario','model','variable','threshold'], as_index=False)
                .agg(mean_30yr=('exceed_days','mean')))
    avg30_csv = os.path.join(tbl_dir, "avg30_exceed_days.csv")
    avg30_df.to_csv(avg30_csv, index=False, encoding="utf-8-sig")
    log(f"[SAVE] 30年平均: {avg30_csv}")

    # 8) 一些可视化（biome 层面）
    #   - 基线各 biome 的 exceed-days 平均 TopN（两个变量各画一张）
    for var in VARS:
        base_var_df = annual_df[(annual_df['epoch']=='baseline') & (annual_df['variable']==var)]
        if not base_var_df.empty:
            biome_barplot(base_var_df, f"Baseline mean exceed-days (Top) — {var}",
                          os.path.join(OUT_ROOT, "plots", f"baseline_biome_top_{var}.png"))

    t1 = datetime.now()
    log(f"=== 分析完成 {t1.strftime('%Y-%m-%d %H:%M:%S')} | 用时 {(t1-t0).total_seconds()/60:.1f} 分钟 ===")

if __name__ == "__main__":
    main()












#%% individual years


# -*- coding: utf-8 -*-
"""
单年出表（只出 CSV）
- 读取 YEAR_FILES 指定的单年 baseline 结果（time 可能是 int32/0..N，也可能是 datetime）
- 变量：fire_probability / obe_probability
- 阈值：MODEL_DIR/thresholds.pkl（若数据 attrs 也有则校验一致）
- 掩膜：US+Canada ∩ biome-decimal（小数 ID，不切分 east/west），轻微不一致则最近邻
- 时间：单年去闰日 2/29 后必须为 365 天
- 输出：E:\Projection paper\analysis\tables\annual_exceed_days_selected_years.csv
"""

import os, warnings, gc
import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
import regionmask
import joblib
from typing import Optional, Dict, List
from datetime import datetime

warnings.filterwarnings("ignore")

# ========= 路径 =========
MODEL_DIR = r"E:\Projection paper\models_skl161"   # 必须包含 thresholds.pkl

YEAR_FILES = {
    2020: r"E:\Projection paper\outputs_baseline\baseline_2020_20251021_192751.nc",
    2021: r"E:\Projection paper\outputs_baseline\baseline_2021_20251021_234549.nc",
    2023: r"E:\Projection paper\outputs_baseline\baseline_2023_20251021_234549.nc",
}

# 生物群系（小数 ID）与区域边界（都在 D 盘）
BIOME_DEC_NC = r"D:\000_collections\020_Chapter2\US_CAN_biome.nc"   # decimal id
BIO_VAR      = "gez_code_id"                                        # 不能自己切分 east/west
USCAN_SHP    = r"D:\000_collections\010_Nighttime Burning\011_Data\013_Biome_wwf2017\US_Canada_merged.shp"

# 输出
OUT_DIR = r"E:\Projection paper\analysis\tables"
OUT_CSV = os.path.join(OUT_DIR, "annual_exceed_days_selected_years.csv")
os.makedirs(OUT_DIR, exist_ok=True)

# ========= 常量 =========
VARS = ["fire_probability", "obe_probability"]
EXCLUDE_BIOMES = {50.1, 90.1}  # Polar / Water

# biome 名称映射（保留小数）
biome_name_map = {
    41.1: "Boreal coniferous forest east",
    41.2: "Boreal coniferous forest west",
    43.1: "Boreal mountain system",
    42.1: "Boreal tundra woodland east",
    42.2: "Boreal tundra woodland west",
    50.1: "Polar",
    24.1: "Subtropical desert",
    22.1: "Subtropical dry forest",
    21.1: "Subtropical humid forest",
    25.1: "Subtropical mountain system",
    23.1: "Subtropical steppe",
    32.1: "Temperate continental forest",
    34.1: "Temperate desert",
    35.1: "Temperate mountain system east",
    35.2: "Temperate mountain system west",
    31.1: "Temperate oceanic forest",
    33.1: "Temperate steppe",
    13.1: "Tropical dry forest",
    12.1: "Tropical moist forest",
    90.1: "Water",
}

# ========= 小工具 =========
def log(msg: str) -> None:
    print(msg, flush=True)

def read_thresholds(model_dir: str) -> Dict[str, float]:
    p = os.path.join(model_dir, "thresholds.pkl")
    if not os.path.exists(p):
        raise FileNotFoundError(f"[ERROR] 缺少阈值文件：{p}")
    th = joblib.load(p)
    for k in ("fire_threshold","obe_threshold"):
        if k not in th:
            raise KeyError(f"[ERROR] thresholds.pkl 缺少 {k}")
    return {"fire_threshold": float(th["fire_threshold"]),
            "obe_threshold":  float(th["obe_threshold"])}

def check_attrs_thresholds(ds: xr.Dataset, th: Dict[str,float], fname: str) -> None:
    for k in ("fire_threshold","obe_threshold"):
        if k in ds.attrs:
            try:
                v = float(ds.attrs[k])
                if np.isfinite(v) and abs(v - th[k]) > 1e-6:
                    raise ValueError(
                        f"[ERROR] {os.path.basename(fname)} attrs {k}={v} "
                        f"与模型阈值 {th[k]} 不一致。"
                    )
            except Exception:
                pass

def rebuild_single_year_365(ds: xr.Dataset, year: int) -> Optional[xr.Dataset]:
    """
    返回该年的 365 天数据集：
    - 若 time 是 datetime64：按年切片 → 去闰日（2/29）→ 必须 365
    - 若 time 是整数（你的文件）：用 attrs['year'] 校验；长度 365/366；若 366 去掉 index=59（0-based 的第 60 天）；
      然后赋上真实日期索引（去闰日后 365 天）
    """
    if "time" not in ds.dims or ds.sizes["time"] == 0:
        log(f"[WARN] {year}: 文件缺少有效 time 维，跳过。")
        return None

    # datetime64 分支
    if np.issubdtype(ds["time"].dtype, np.datetime64):
        try:
            sub = ds.sel(time=slice(f"{year}-01-01", f"{year}-12-31"))
        except Exception:
            log(f"[WARN] {year}: time 切片失败（datetime 分支），跳过。"); return None
        if sub.sizes["time"] == 0:
            log(f"[WARN] {year}: 切片后为空（datetime 分支），跳过。"); return None
        t = pd.DatetimeIndex(sub.time.values)
        keep = ~((t.month == 2) & (t.day == 29))
        sub = sub.isel(time=keep)
        if sub.sizes["time"] != 365:
            log(f"[WARN] {year}: 去闰日后 time={sub.sizes['time']} ≠ 365，跳过。"); return None
        newt = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D")
        newt = newt[~((newt.month == 2) & (newt.day == 29))]
        return sub.assign_coords(time=newt)

    # 整数分支（你的文件：0..365/364）
    file_year = None
    if "year" in ds.attrs:
        try:
            file_year = int(ds.attrs["year"])
        except Exception:
            file_year = None
    if file_year != year:
        log(f"[WARN] {year}: 文件 attrs.year={file_year} 与目标年不一致，跳过。")
        return None

    n = ds.sizes["time"]
    if n not in (365, 366):
        log(f"[WARN] {year}: time 长度={n}（应为 365 或 366），跳过。")
        return None

    sub = ds
    # 如果是闰年 366 → 删除 2/29 的那一天（0-based index=59）
    if n == 366:
        drop_idx = 59  # 2/29 的第 60 天（0-based）
        sub = sub.isel(time=[i for i in range(366) if i != drop_idx])

    # 赋上真实日期索引（365 天）
    newt = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D")
    newt = newt[~((newt.month == 2) & (newt.day == 29))]
    if sub.sizes["time"] != 365:
        log(f"[WARN] {year}: 去闰日后 time={sub.sizes['time']} ≠ 365（整数分支），跳过。")
        return None
    return sub.assign_coords(time=newt)

def _grid_summary(lon: np.ndarray, lat: np.ndarray):
    return dict(
        lon_min=float(np.nanmin(lon)), lon_max=float(np.nanmax(lon)),
        lat_min=float(np.nanmin(lat)), lat_max=float(np.nanmax(lat)),
        dlon=float(np.nanmedian(np.diff(lon))), dlat=float(np.nanmedian(np.diff(lat))),
        lon_asc=bool(lon[1] > lon[0]), lat_asc=bool(lat[1] > lat[0]),
        nlon=len(lon), nlat=len(lat),
    )

def roughly_compatible(src_lon: np.ndarray, src_lat: np.ndarray,
                       dst_lon: np.ndarray, dst_lat: np.ndarray,
                       tol_edge_deg: float = 0.75,
                       tol_step_frac: float = 0.25,
                       tol_extra_rc: int = 2) -> bool:
    s, d = _grid_summary(src_lon, src_lat), _grid_summary(dst_lon, dst_lat)
    if s["lon_asc"] != d["lon_asc"] or s["lat_asc"] != d["lat_asc"]:
        return False
    if (abs(s["lon_min"] - d["lon_min"]) > tol_edge_deg or
        abs(s["lon_max"] - d["lon_max"]) > tol_edge_deg or
        abs(s["lat_min"] - d["lat_min"]) > tol_edge_deg or
        abs(s["lat_max"] - d["lat_max"]) > tol_edge_deg):
        return False
    if (abs(s["dlon"] - d["dlon"]) / max(1e-6, abs(d["dlon"])) > tol_step_frac or
        abs(s["dlat"] - d["dlat"]) / max(1e-6, abs(d["dlat"])) > tol_step_frac):
        return False
    if abs(s["nlon"] - d["nlon"]) > tol_extra_rc or abs(s["nlat"] - d["nlat"]) > tol_extra_rc:
        return False
    return True

def build_masks_on_data_grid(data_lon: np.ndarray, data_lat: np.ndarray):
    """返回 combined_mask, region_mask, biome_masks（每个 biome 已与 combined 相交）"""
    # 区域：US+Canada
    LON2D, LAT2D = np.meshgrid(data_lon, data_lat)
    shp = gpd.read_file(USCAN_SHP)
    region_mask = ~np.isnan(regionmask.mask_geopandas(shp, LON2D, LAT2D, overlap=False))

    # biome：decimal id（小数不能切分）；必要时最近邻到数据网格
    dsb = xr.open_dataset(BIOME_DEC_NC)
    ren = {}
    if "lon" in dsb.coords and "longitude" not in dsb.coords: ren["lon"] = "longitude"
    if "lat" in dsb.coords and "latitude"  not in dsb.coords: ren["lat"] = "latitude"
    if ren: dsb = dsb.rename(ren)
    if BIO_VAR not in dsb:
        raise KeyError(f"[ERROR] {BIOME_DEC_NC} 缺少变量 {BIO_VAR}")

    bio_lon = dsb["longitude"].values
    bio_lat = dsb["latitude"].values
    if (len(bio_lon) == len(data_lon) and len(bio_lat) == len(data_lat) and
        np.allclose(bio_lon, data_lon) and np.allclose(bio_lat, data_lat)):
        bio_on = dsb[BIO_VAR]
    else:
        if not roughly_compatible(bio_lon, bio_lat, data_lon, data_lat):
            raise RuntimeError("[ERROR] biome 与数据网格差异过大，拒绝自动插值（请先重网格）。")
        bio_on = dsb[BIO_VAR].interp(
            longitude=xr.DataArray(data_lon, dims=("longitude",)),
            latitude =xr.DataArray(data_lat,  dims=("latitude",)),
            method="nearest"
        )

    bio_vals = bio_on.values
    valid_bio = (~np.isnan(bio_vals)).copy()
    for v in EXCLUDE_BIOMES:
        valid_bio &= (bio_vals != v)

    combined_mask = region_mask & valid_bio

    biome_masks: Dict[float, np.ndarray] = {}
    uniq = np.unique(bio_vals[combined_mask])
    for bid in uniq:
        bid_f = float(bid)
        if bid_f in EXCLUDE_BIOMES:
            continue
        m = (bio_vals == bid) & combined_mask
        if m.any():
            biome_masks[bid_f] = m

    return combined_mask, region_mask, biome_masks

def one_year_rows(ds: xr.Dataset, year: int, biome_masks: Dict[float, np.ndarray],
                  var: str, thr: float) -> List[dict]:
    """单年 exceed-days（像素天数）→ biome 平均"""
    rows: List[dict] = []
    da = ds[var]  # (time, lat, lon)
    cnt = (da > thr).sum(dim="time", skipna=True).values  # (lat, lon)
    for bid, mask in biome_masks.items():
        arr = cnt[mask]
        if arr.size == 0:
            continue
        rows.append({
            "biome_id": float(bid),
            "biome_name": biome_name_map.get(float(bid), f"Unknown_{bid}"),
            "year": int(year),
            "variable": var,
            "threshold": float(thr),
            "exceed_days": float(np.nanmean(arr)),
            "epoch": "single-year",
            "scenario": "baseline",
            "model": "baseline",
        })
    return rows

# ========= 主程序 =========
def main() -> None:
    t0 = datetime.now()
    log(f"=== 单年出表开始 {t0.strftime('%Y-%m-%d %H:%M:%S')} ===")

    # 阈值
    th = read_thresholds(MODEL_DIR)
    log(f"[THRESHOLDS] fire={th['fire_threshold']:.4f}, obe={th['obe_threshold']:.4f}")

    all_rows: List[dict] = []
    mask_cache = {}  # 按 (nlat,nlon, hash(lats),hash(lons)) 缓存掩膜，避免重复构建

    for year in sorted(YEAR_FILES.keys()):
        fpath = YEAR_FILES[year]
        if not os.path.exists(fpath):
            log(f"[WARN] {year}: 文件不存在 → {fpath}（跳过）")
            continue

        ds = xr.open_dataset(fpath)
        check_attrs_thresholds(ds, th, fpath)

        # 统一坐标名
        ren = {}
        if "lon" in ds.coords: ren["lon"] = "longitude"
        if "lat" in ds.coords: ren["lat"] = "latitude"
        if ren: ds = ds.rename(ren)

        # 单年 365 天
        sub = rebuild_single_year_365(ds, int(year))
        ds.close()
        if sub is None:
            continue

        # 掩膜（按该数据网格；缓存）
        lon = sub["longitude"].values
        lat = sub["latitude"].values
        key = (len(lat), len(lon), float(lat[0]), float(lat[-1]), float(lon[0]), float(lon[-1]))
        if key in mask_cache:
            biome_masks = mask_cache[key]
        else:
            _, _, biome_masks = build_masks_on_data_grid(lon, lat)
            mask_cache[key] = biome_masks

        # 两个变量
        for var in VARS:
            if var not in sub:
                log(f"[WARN] {year}: 缺变量 {var}（跳过该变量）")
                continue
            thr_use = th["fire_threshold"] if var.startswith("fire_") else th["obe_threshold"]
            all_rows.extend(one_year_rows(sub, year, biome_masks, var, thr_use))

        sub.close(); gc.collect()

    # 落盘
    if not all_rows:
        log("[WARN] 没有任何记录可写出（可能年份文件 time 异常或缺变量）。")
    else:
        df = pd.DataFrame(all_rows)
        cols = ["biome_id","biome_name","year","variable","threshold",
                "exceed_days","epoch","scenario","model"]
        df = df[cols]
        df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
        log(f"[SAVE] 单年长表: {OUT_CSV}")

    t1 = datetime.now()
    log(f"=== 结束 {t1.strftime('%Y-%m-%d %H:%M:%S')} | 用时 {(t1 - t0).total_seconds():.1f}s ===")

if __name__ == "__main__":
    main()




#%% canada 2023 us 2020/2021

"""
Regional exceed-days analysis (Canada 2023, Western US 2020/2021)
Now supports:
- Toggles to run ONLY baseline (single-year and/or 1991–2020), ONLY future, or BOTH
- Writes 1991–2020 baseline CSV into E:\Projection paper\regional_outputs\<group>\baseline\
- Keeps future discovery + both windows (2040/2070) behavior
"""

import os
import re
import glob
import gc
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import xarray as xr

# ──────────────────────────────────────────────────────────────────────────────
# 0) RUN TOGGLES  ←←← 这里控制运行范围
# ──────────────────────────────────────────────────────────────────────────────
DO_BASELINE_SINGLE_YEARS = False          # 计算单年基线（2023/2020/2021）→ 写入 annual_fire_days_*_YYYY.csv
DO_BASELINE_LONG_1991_2020 = True        # 计算 1991–2020 基线 → 写入 baseline/annual_fire_days_*_1991_2020_baseline.csv
DO_FUTURE = False                         # 处理未来 2040 / 2070 窗口

# 可选：限制只跑某些组，留空表示全部
FILTER_GROUPS = []  # e.g. ["canada_2023"] or ["us_2020", "us_2021"]

# ──────────────────────────────────────────────────────────────────────────────
# 1) CONFIG
# ──────────────────────────────────────────────────────────────────────────────
E_BASE = r"E:\\Projection paper"
OUT_BASE = os.path.join(E_BASE, "regional_outputs")  # parent folder

# Baselines on E (single years)
BASELINES = {
    "canada_2023": os.path.join(E_BASE, "outputs_baseline", "baseline_2023_*.nc"),
    "us_2020":     os.path.join(E_BASE, "outputs_baseline", "baseline_2020_*.nc"),
    "us_2021":     os.path.join(E_BASE, "outputs_baseline", "baseline_2021_*.nc"),
}

BASELINE_START_DATES = {
    "canada_2023": "2023-01-01",
    "us_2020": "2020-01-01",
    "us_2021": "2021-01-01",
}

# 1991–2020 combined baseline NetCDF
BASELINE_1991_2020 = os.path.join(E_BASE, "outputs_baseline", "baseline_1991_2020_*.nc")
BASELINE_1991_2020_START = "1991-01-01"

# Future files dir (auto-discovery)
FUT_DIR = os.path.join(E_BASE, "outputs_future")
FUT_FILE_GLOB = os.path.join(FUT_DIR, "*.nc")

# Thresholds
THRESHOLDS_PKL = os.path.join(E_BASE, "models_skl161", "thresholds.pkl")
FALLBACK_THRESHOLDS = {
    "fire_probability": 0.4171,
    "obe_probability": 0.4233,
    # "h24_probability": 0.5028,
}

# Region masks sources
US_STATE_PATH = r"D:\\000_collections\\020_Chapter2\\US_state_name.nc"
CANADA_PROVINCE_PATH = r"D:\\000_collections\\020_Chapter2\\Canada_PR_name.nc"
BIOME_PATH = r"D:\\000_collections\\020_Chapter2\\US_CAN_biome.nc"

WESTERN_US_STATES = [
    'Arizona', 'California', 'Colorado', 'Hawaii', 'Idaho', 'Montana',
    'Nevada', 'New Mexico', 'Oregon', 'Utah', 'Washington', 'Wyoming'
]
EXCLUDE_BIOMES = [50.1, 90.1, 12.1, 13.1, 21.1, 22.1, 31.1, 35.1]

# Variables to process (与可视化一致)
VARIABLES = ["fire_probability", "obe_probability"]

WINDOWS = {
    2040: ("2041-01-01", "2070-12-31"),
    2070: ("2071-01-01", "2100-12-31"),
}

CSV_OUT_NAMES = {
    "canada_2023": "annual_fire_days_canada_2023.csv",
    "us_2020": "annual_fire_days_us_2020.csv",
    "us_2021": "annual_fire_days_us_2021.csv",
}

# ──────────────────────────────────────────────────────────────────────────────
# 2) Helpers
# ──────────────────────────────────────────────────────────────────────────────

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)
    return path

def pick_one_nc(pattern: str) -> str:
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matched: {pattern}")
    files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return files[0]

def rebuild_time_coords(ds: xr.Dataset, start_date: str) -> xr.Dataset:
    nday = ds.sizes['time']
    newt = pd.date_range(start_date, periods=nday, freq='D')
    if len(newt) > nday:
        newt = newt[~((newt.month == 2) & (newt.day == 29))]
    return ds.assign_coords(time=newt)

def load_thresholds() -> dict:
    thr = dict(FALLBACK_THRESHOLDS)
    try:
        import pickle
        if os.path.exists(THRESHOLDS_PKL):
            with open(THRESHOLDS_PKL, 'rb') as f:
                obj = pickle.load(f)
            if 'fire_threshold' in obj:
                thr['fire_probability'] = float(obj['fire_threshold'])
            if 'obe_threshold' in obj:
                thr['obe_probability'] = float(obj['obe_threshold'])
            if 'h24_threshold' in obj and 'h24_probability' in VARIABLES:
                thr['h24_probability'] = float(obj['h24_threshold'])
    except Exception as e:
        print(f"Warning: failed to read thresholds.pkl, using fallback. Error: {e}")
    return thr

def parse_future_filename(fn: str):
    base = os.path.basename(fn)
    m = re.match(r"(2040|2070)[_\-](\d{3})[_\-]([A-Za-z0-9]+).*\.nc$", base)
    if not m:
        return None
    tag = int(m.group(1))
    ssp = m.group(2)
    model = m.group(3).upper()
    return tag, ssp, model

def discover_future_files():
    fut = {}
    for fn in glob.glob(FUT_FILE_GLOB):
        parsed = parse_future_filename(fn)
        if not parsed:
            continue
        tag, ssp, model = parsed
        fut.setdefault(tag, {}).setdefault(ssp, {})[model] = fn
    if not fut:
        raise FileNotFoundError(f"No future files discovered under {FUT_DIR}")
    return fut

def open_align_mask_source(ds_path, var_name):
    ds = xr.open_dataset(ds_path)
    if 'lon' in ds.dims:
        ds = ds.rename({"lon": "longitude", "lat": "latitude"})
    return ds[var_name]

def create_regional_masks(region_type: str, baseline_ds: xr.Dataset):
    print(f"\nCreating {region_type} masks…")

    if region_type == 'canada':
        region_da = open_align_mask_source(CANADA_PROVINCE_PATH, 'PRENAME')
        unique_regions = np.unique(region_da.values)
        unique_regions = unique_regions[unique_regions != '']
    else:
        region_da = open_align_mask_source(US_STATE_PATH, 'NAME')
        all_states = np.unique(region_da.values)
        unique_regions = [s for s in all_states if s in WESTERN_US_STATES]

    biome_da = open_align_mask_source(BIOME_PATH, 'gez_code_id')

    biome_mask = np.ones_like(biome_da.values, dtype=bool)
    for v in EXCLUDE_BIOMES:
        biome_mask &= (biome_da.values != v)
    biome_mask &= ~np.isnan(biome_da.values)

    region_masks = {}
    overall = np.zeros_like(region_da.values, dtype=bool)
    targets = unique_regions

    for r in targets:
        rm = (region_da.values == r) & biome_mask
        region_masks[str(r)] = rm
        overall |= rm
    region_masks['overall'] = overall

    # align to baseline grid
    base_lons = baseline_ds.longitude.values
    base_lats = baseline_ds.latitude.values
    reg_lons = region_da.longitude.values
    reg_lats = region_da.latitude.values

    aligned = {}
    for name, mask in region_masks.items():
        tmp = xr.DataArray(mask.astype(float),
                           coords={'latitude': reg_lats, 'longitude': reg_lons},
                           dims=['latitude', 'longitude'])
        am = tmp.interp(longitude=base_lons, latitude=base_lats, method='nearest').values > 0.5
        aligned[name] = am
        print(f"  {name}: {am.sum()} pixels")
    return aligned, list([k for k in region_masks.keys() if k != 'overall'])

def compute_yearly_exceed_days(da: xr.DataArray, thr: float) -> xr.DataArray:
    return (da > thr).sum(dim='time', skipna=True)

def avg_over_mask(arr2d: np.ndarray, mask: np.ndarray) -> float:
    vals = arr2d[mask]
    if vals.size == 0:
        return np.nan
    return float(np.nanmean(vals))

# ──────────────────────────────────────────────────────────────────────────────
# 3) Core analytics
# ──────────────────────────────────────────────────────────────────────────────

def analyze_baseline_single_year(baseline_ds: xr.Dataset, region_masks: dict, region_names: list,
                                 baseline_year: int, thresholds: dict):
    rows = []
    for region in ['overall'] + region_names:
        mask = region_masks[region]
        for var in VARIABLES:
            thr = thresholds[var]
            year_vals = baseline_ds[var].values  # [time, lat, lon]
            fire_days_by_px = np.sum(year_vals > thr, axis=0)
            rows.append({
                'region': region,
                'scenario': 'baseline',
                'model': 'baseline',
                'year': baseline_year,
                'variable': var,
                'threshold': thr,
                'fire_days': avg_over_mask(fire_days_by_px, mask)
            })
    return rows

def analyze_future_windows(fut_ds: xr.Dataset, region_masks: dict, region_names: list,
                           thresholds: dict, window_years: np.ndarray,
                           tag: int, ssp: str, model: str):
    rows = []
    times = pd.DatetimeIndex(fut_ds.time.values)
    for region in ['overall'] + region_names:
        mask = region_masks[region]
        for year in window_years:
            idx = np.where(times.year == year)[0]
            if idx.size == 0:
                continue
            for var in VARIABLES:
                thr = thresholds[var]
                year_data = fut_ds[var].isel(time=idx).values
                fire_days_by_px = np.sum(year_data > thr, axis=0)
                rows.append({
                    'region': region,
                    'scenario': f'SSP{ssp}',
                    'model': model,
                    'year': int(year),
                    'variable': var,
                    'threshold': thr,
                    'fire_days': avg_over_mask(fire_days_by_px, mask)
                })
    return rows

def precompute_future_exceed_means(fut_ds: xr.Dataset, thresholds: dict, window_years: np.ndarray):
    out = {}
    for var in VARIABLES:
        thr = thresholds[var]
        yearly = []
        for y in window_years:
            sel = fut_ds[var].sel(time=str(int(y)))
            yearly.append((sel > thr).sum(dim='time', skipna=True))
        out[var] = sum(yearly) / len(yearly)
        del yearly
        gc.collect()
    return out

def change_maps_against_baseline(baseline_ds: xr.Dataset, fut_means: dict, thresholds: dict,
                                 region_masks: dict, out_dir: str,
                                 tag: int, ssp: str, model: str, baseline_year: int):
    ensure_dir(out_dir)
    base_exceed = {}
    for var in VARIABLES:
        thr = thresholds[var]
        base_exceed[var] = (baseline_ds[var] > thr).sum(dim='time', skipna=True)

    for var in VARIABLES:
        fut_mean = fut_means[var]
        abs_change = fut_mean - base_exceed[var]
        with np.errstate(divide='ignore', invalid='ignore'):
            rel_change = (abs_change / base_exceed[var]) * 100.0
        rel_change = rel_change.where(~np.isinf(rel_change))

        fn_overall = f"{var}_days_window{tag}_SSP{ssp}_{model}_overall_change.nc"
        xr.Dataset(
            data_vars=dict(absolute_change=abs_change, relative_change=rel_change),
            coords=dict(latitude=baseline_ds.latitude, longitude=baseline_ds.longitude),
            attrs=dict(description=(
                f"Change in mean annual exceed-days ({var}>{thresholds[var]}) "
                f"between baseline ({baseline_year}) and window {tag} SSP{ssp} {model}"
            ), created_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        ).to_netcdf(os.path.join(out_dir, fn_overall))

        for region, mask in region_masks.items():
            if region == 'overall':
                continue
            xr.Dataset(
                data_vars=dict(
                    absolute_change=abs_change.where(mask),
                    relative_change=rel_change.where(mask),
                ),
                coords=dict(latitude=baseline_ds.latitude, longitude=baseline_ds.longitude),
                attrs=dict(description=(
                    f"Change in mean annual exceed-days ({var}>{thresholds[var]}) "
                    f"between baseline ({baseline_year}) and window {tag} SSP{ssp} {model} for {region}"
                ), created_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            ).to_netcdf(os.path.join(out_dir, f"{var}_days_window{tag}_SSP{ssp}_{model}_{region}_change.nc"))

# ──────────────────────────────────────────────────────────────────────────────
# 4) New: write 1991–2020 baseline CSV to E:\...\baseline\
# ──────────────────────────────────────────────────────────────────────────────

def write_long_baseline_csv(group_key: str, long_base_nc: str):
    """
    For the 1991–2020 combined baseline, compute annual exceed-days (per region)
    and write CSV to: E:\Projection paper\regional_outputs\<group>\baseline\annual_fire_days_<short>_1991_2020_baseline.csv
    """
    region_type = 'canada' if group_key == 'canada_2023' else 'us'
    short = 'canada' if group_key == 'canada_2023' else 'us'

    print(f"\n== Baseline 1991–2020 for {group_key} ==")
    ds = rebuild_time_coords(xr.open_dataset(long_base_nc), BASELINE_1991_2020_START)

    masks, region_names = create_regional_masks(region_type, ds)
    thresholds = load_thresholds()

    times = pd.DatetimeIndex(ds.time.values)
    years = np.unique(times.year)

    rows = []
    for yr in years:
        idx = np.where(times.year == yr)[0]
        for region in ['overall'] + region_names:
            mask = masks[region]
            for var in VARIABLES:
                thr = thresholds[var]
                year_data = ds[var].isel(time=idx).values
                fire_days = np.sum(year_data > thr, axis=0)
                rows.append({
                    'region': region,
                    'scenario': 'baseline_1991_2020',
                    'model': 'baseline',
                    'year': int(yr),
                    'variable': var,
                    'threshold': thr,
                    'fire_days': avg_over_mask(fire_days, mask)
                })

    out_dir = ensure_dir(os.path.join(OUT_BASE, group_key, "baseline"))
    out_csv = os.path.join(out_dir, f"annual_fire_days_{short}_1991_2020_baseline.csv")
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")

# ──────────────────────────────────────────────────────────────────────────────
# 5) Pipelines per region group
# ──────────────────────────────────────────────────────────────────────────────

def run_for_group(group_key: str, future_files: dict | None):
    print("="*80)
    print(f"Processing {group_key} …")

    thresholds = load_thresholds()
    region_type = 'canada' if group_key == 'canada_2023' else 'us'
    group_out_dir = ensure_dir(os.path.join(OUT_BASE, group_key))

    rows = []

    # ----- A) 单年基线 -----
    if DO_BASELINE_SINGLE_YEARS:
        base_nc = pick_one_nc(BASELINES[group_key])
        base_start = BASELINE_START_DATES[group_key]
        baseline_ds = rebuild_time_coords(xr.open_dataset(base_nc), base_start)

        region_masks, region_names = create_regional_masks(region_type, baseline_ds)
        baseline_year = int(base_start[:4])
        rows.extend(analyze_baseline_single_year(baseline_ds, region_masks, region_names, baseline_year, thresholds))

    # ----- B) 未来窗口 -----
    if DO_FUTURE:
        if future_files is None:
            future_files = discover_future_files()

        if not DO_BASELINE_SINGLE_YEARS:
            # 如果没跑单年基线，这里也需要 baseline_ds 用来掩膜与变更图
            base_nc = pick_one_nc(BASELINES[group_key])
            base_start = BASELINE_START_DATES[group_key]
            baseline_ds = rebuild_time_coords(xr.open_dataset(base_nc), base_start)
            region_masks, region_names = create_regional_masks(region_type, baseline_ds)
            baseline_year = int(base_start[:4])

        for tag, ssp_dict in sorted(future_files.items()):
            if tag not in WINDOWS:
                print(f"  Skip unknown window tag: {tag}")
                continue
            w_start, w_end = WINDOWS[tag]
            win_dir = ensure_dir(os.path.join(group_out_dir, f"window_{tag}"))

            for ssp, models in sorted(ssp_dict.items()):
                for model, path in sorted(models.items()):
                    print(f"  Future {tag} SSP{ssp} {model}: {os.path.basename(path)}")
                    fut_ds = xr.open_dataset(path)
                    # 对齐经度维（老逻辑）
                    if fut_ds.dims.get('longitude', None) != baseline_ds.dims.get('longitude', None):
                        if 'longitude' in fut_ds.dims and fut_ds.dims['longitude'] >= 3:
                            fut_ds = fut_ds.isel(longitude=slice(2, None))

                    fut_len = fut_ds.sizes['time']
                    new_time = pd.date_range(w_start, w_end, freq='D')
                    new_time = new_time[~((new_time.month == 2) & (new_time.day == 29))]
                    if len(new_time) != fut_len:
                        new_time = pd.date_range(w_start, periods=fut_len, freq='D')
                        new_time = new_time[~((new_time.month == 2) & (new_time.day == 29))]
                        new_time = new_time[:fut_len]
                    fut_ds = fut_ds.assign_coords(time=new_time)

                    years = np.unique(pd.DatetimeIndex(fut_ds.time.values).year)
                    rows.extend(analyze_future_windows(fut_ds, region_masks, region_names, thresholds, years, tag, ssp, model))

                    fut_means = precompute_future_exceed_means(fut_ds, thresholds, years)
                    change_maps_against_baseline(baseline_ds, fut_means, thresholds, region_masks,
                                                 out_dir=win_dir, tag=tag, ssp=ssp, model=model,
                                                 baseline_year=baseline_year)

                    del fut_ds
                    gc.collect()

    # ----- C) 写单年基线+未来的综合 CSV -----
    if DO_BASELINE_SINGLE_YEARS or DO_FUTURE:
        csv_path = os.path.join(group_out_dir, CSV_OUT_NAMES[group_key])
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        print(f"\nSaved: {csv_path}")

    # ----- D) 1991–2020 baseline -----
    if DO_BASELINE_LONG_1991_2020:
        long_base_nc = pick_one_nc(BASELINE_1991_2020)
        write_long_baseline_csv(group_key, long_base_nc)

# ──────────────────────────────────────────────────────────────────────────────
# 6) Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    print("Discovering future files…")
    future = None
    if DO_FUTURE:
        future = discover_future_files()

    ensure_dir(OUT_BASE)
    groups = ["canada_2023", "us_2020", "us_2021"]
    if FILTER_GROUPS:
        groups = [g for g in groups if g in FILTER_GROUPS]

    for key in groups:
        run_for_group(key, future)

    print("\nAll done.")

if __name__ == "__main__":
    import pandas as pd
    main()
