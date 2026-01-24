# -*- coding: utf-8 -*-
# plot_change_maps_new.py
# --------------------------------------------------------------
# 批量绘制 exceed_days *_change.nc：Absolute / Relative 两套图
# 掩膜：按“原有逻辑” + 最近邻插值对齐（仅当网格轻微不一致）
# 文件名解析：<var>_days_<yearTag>_<ssp>_<model>_change.nc
# 情景：当前 245/370；兼容未来 126/585；yearTag 任意4位数（常用 2040/2070）
# 面板图：按每个 yearTag 输出一张或多张，标题/文件名都含时段窗口
# 额外新增：
#   - DO_SINGLE / DO_PANEL：控制只出单图、只出面板或二者都出
#   - PANEL_CHANGE_TYPES：("relative",) or ("absolute",) or ("relative","absolute")
# --------------------------------------------------------------

# import os, re, sys, gc
# from glob import glob

# import numpy as np
# import xarray as xr
# import matplotlib.pyplot as plt
# from matplotlib.colors import BoundaryNorm
# import cartopy.crs as ccrs
# import cartopy.feature as cfeature
# import geopandas as gpd
# import regionmask

# # ========== 路径配置 ==========
# # 注意：新的 *_change.nc 在 E 盘，其余矢量/biome 在 D 盘（按你说的）
# INPUT_DIR  = r"E:\Projection paper\analysis\maps"                      # *_change.nc 的目录（E盘）
# OUTPUT_DIR = os.path.join(INPUT_DIR, "figures")                        # 输出在 E 盘
# BIOME_NC   = r"D:\000_collections\020_Chapter2\US_CAN_biome.nc"        # 生物群系（D盘）
# USCAN_SHP  = r"D:\000_collections\010_Nighttime Burning\011_Data\013_Biome_wwf2017\US_Canada_merged.shp"  # 边界（D盘）
# os.makedirs(OUTPUT_DIR, exist_ok=True)

# # ========== 外观 ==========
# plt.rcParams['font.family'] = 'Arial'
# plt.rcParams.update({'font.size': 8})

# MAP_EXTENT = [-170, -50, 15, 80]
# PROJ = ccrs.LambertConformal(
#     central_longitude=-100, central_latitude=50, standard_parallels=(35, 65)
# )

# # ========== 选择器（留空 None 表示不过滤、全处理）==========
# SELECT_YEAR_TAGS = ('2040', '2070')   # 或 None；yearTag 用四位字符串
# SELECT_SSPS      = ('245', '370')     # 或 None；注意这里用 3 位数字码
# SELECT_MODELS    = ('CANESM','ECEARTH','GFDL','UKESM')  # 或 None；大写
# SELECT_VARS      = ('fire_probability','obe_probability')  # 或 None

# # ========== 生成什么 ==========
# DO_SINGLE = False     # True=出单图（每个文件 absolute & relative）
# DO_PANEL  = True     # True=出面板
# PANEL_CHANGE_TYPES = ("relative",)  # 可选：("relative",) 或 ("absolute",) 或 ("relative","absolute")

# # ========== 文件名解析（支持更多情景 & 年份） ==========
# # 例：fire_probability_days_2070_370_GFDL_change.nc
# NAME_RE = re.compile(
#     r'^(?P<var>.+?)_days_(?P<year>\d{4})_(?P<ssp>126|245|370|585)_(?P<model>[A-Za-z0-9]+)_change\.nc$',
#     re.IGNORECASE
# )

# def parse_change_filename(path):
#     m = NAME_RE.match(os.path.basename(path))
#     if not m:
#         return None
#     meta = m.groupdict()
#     meta["scenario"]   = f"SSP{meta['ssp']}"
#     meta["model_disp"] = meta["model"].upper()
#     meta["year_tag"]   = meta["year"]
#     return meta

# # ========== yearTag → 30年窗口文本 ==========
# def year_tag_to_window(tag: str) -> str:
#     if tag == '2040':
#         return '2041–2070'
#     if tag == '2070':
#         return '2071–2100'
#     # 未知 yearTag 时退化为友好文本
#     return f'future period ({tag})'

# # ========== 网格一致性检查：是否“轻微不一致” ==========
# def _grid_summary(lon, lat):
#     return dict(
#         lon_min=float(np.nanmin(lon)), lon_max=float(np.nanmax(lon)),
#         lat_min=float(np.nanmin(lat)), lat_max=float(np.nanmax(lat)),
#         dlon=float(np.nanmedian(np.diff(lon))) if len(lon) > 1 else np.nan,
#         dlat=float(np.nanmedian(np.diff(lat))) if len(lat) > 1 else np.nan,
#         lon_asc=bool(lon[1] > lon[0]) if len(lon) > 1 else True,
#         lat_asc=bool(lat[1] > lat[0]) if len(lat) > 1 else True
#     )

# def _roughly_compatible(src_lon, src_lat, dst_lon, dst_lat,
#                         tol_edge_deg=0.75, tol_step_frac=0.25, tol_extra_rc=2):
#     """判断是否只是轻微不一致：边界差≤tol_edge_deg、步长相对差≤tol_step_frac，
#        且行列数量至多多/少 tol_extra_rc。"""
#     s, d = _grid_summary(src_lon, src_lat), _grid_summary(dst_lon, dst_lat)
#     if s["lon_asc"] != d["lon_asc"] or s["lat_asc"] != d["lat_asc"]:
#         return False
#     if (abs(s["lon_min"] - d["lon_min"]) > tol_edge_deg or
#         abs(s["lon_max"] - d["lon_max"]) > tol_edge_deg or
#         abs(s["lat_min"] - d["lat_min"]) > tol_edge_deg or
#         abs(s["lat_max"] - d["lat_max"]) > tol_edge_deg):
#         return False
#     if (np.isfinite(s["dlon"]) and np.isfinite(d["dlon"]) and
#         abs(s["dlon"] - d["dlon"]) / max(1e-6, abs(d["dlon"])) > tol_step_frac):
#         return False
#     if (np.isfinite(s["dlat"]) and np.isfinite(d["dlat"]) and
#         abs(s["dlat"] - d["dlat"]) / max(1e-6, abs(d["dlat"])) > tol_step_frac):
#         return False
#     if (abs(len(src_lat) - len(dst_lat)) > tol_extra_rc or
#         abs(len(src_lon) - len(dst_lon)) > tol_extra_rc):
#         return False
#     return True

# # ========== 掩膜（原有逻辑） + 最近邻插值对齐 ==========
# EXCLUDE_BIOMES = [50.1, 90.1, 12.1, 13.1, 21.1, 22.1, 31.1, 32.1, 35.1]

# def build_combined_mask_on_data_grid(data_lon, data_lat):
#     """
#     返回 combined_mask（在数据网格上）、以及区域/生物群系两个子掩膜。
#     区域掩膜：直接在数据网格上 rasterize；生物群系：若网格轻微不一致，则最近邻到数据网格。
#     """
#     # 1) 区域掩膜
#     LON2D, LAT2D = np.meshgrid(data_lon, data_lat)
#     shp = gpd.read_file(USCAN_SHP)
#     region = ~np.isnan(regionmask.mask_geopandas(shp, LON2D, LAT2D, overlap=False))

#     # 2) 生物群系（允许轻微不一致，必要时最近邻）
#     bds = xr.open_dataset(BIOME_NC)
#     ren = {}
#     if "lon" in bds.coords and "longitude" not in bds.coords: ren["lon"] = "longitude"
#     if "lat" in bds.coords and "latitude"  not in bds.coords: ren["lat"] = "latitude"
#     if ren: bds = bds.rename(ren)

#     bio_var = "gez_code_id" if "gez_code_id" in bds.data_vars else ("gez_code" if "gez_code" in bds.data_vars else None)
#     if bio_var is None:
#         raise RuntimeError(f"Biome 文件缺少 gez_code(_id)：{BIOME_NC}")

#     bio_lon = bds["longitude"].values
#     bio_lat = bds["latitude"].values

#     if (len(bio_lon) == len(data_lon) and len(bio_lat) == len(data_lat) and
#         np.allclose(bio_lon, data_lon) and np.allclose(bio_lat, data_lat)):
#         bio_on_data = bds[bio_var]
#     else:
#         if not _roughly_compatible(bio_lon, bio_lat, data_lon, data_lat):
#             raise RuntimeError("生物群系列网格与数据网格差异过大，拒绝自动插值（请先显式重网格）。")
#         bio_on_data = bds[bio_var].interp(
#             latitude=xr.DataArray(data_lat, dims=("latitude",)),
#             longitude=xr.DataArray(data_lon, dims=("longitude",)),
#             method="nearest"
#         )

#     bio_arr = bio_on_data.values
#     bio_mask = (~np.isnan(bio_arr)).copy()
#     for v in EXCLUDE_BIOMES:
#         bio_mask &= (bio_arr != v)

#     combined = region & bio_mask
#     return combined, region, bio_mask

# # ========== 色阶工具 ==========
# def abs_levels(masked, pos_pct=95, neg_pct=95, round_to=5):
#     arr = masked[np.isfinite(masked)]
#     if arr.size == 0:
#         lv = np.linspace(-1, 1, 9); return -1, 1, lv, "RdBu_r", "both"
#     pos = arr[arr > 0]; neg = arr[arr < 0]
#     ppos = np.nanpercentile(pos, pos_pct) if pos.size else 0.0
#     pneg = -np.nanpercentile(-neg, neg_pct) if neg.size else 0.0
#     vmax = np.ceil(max(abs(ppos), abs(pneg)) / round_to) * round_to or round_to
#     vmin = -vmax
#     levels = np.linspace(vmin, vmax, 9)
#     return vmin, vmax, levels, "RdBu_r", "both"

# def rel_levels(masked, pos_pct=95, neg_pct=95, cap=500, round_to=10):
#     arr = masked[np.isfinite(masked)]
#     if arr.size == 0:
#         lv = np.linspace(-10, 10, 9); return -10, 10, lv, "RdBu_r", "both"
#     pos = arr[arr > 0]; neg = arr[arr < 0]
#     ppos = np.nanpercentile(pos, pos_pct) if pos.size else 0.0
#     pneg = -np.nanpercentile(-neg, neg_pct) if neg.size else 0.0
#     ppos = min(ppos, cap); pneg = max(pneg, -cap)

#     if pos.size and not neg.size:
#         vmin, vmax = 0, np.ceil(ppos / round_to) * round_to or round_to
#         lv = np.linspace(vmin, vmax, 9); return vmin, vmax, lv, "Reds", "max"
#     if neg.size and not pos.size:
#         vmin, vmax = np.floor(pneg / round_to) * round_to, 0
#         lv = np.linspace(vmin, vmax, 9); return vmin, vmax, lv, "Blues_r", "min"

#     vmax = np.ceil(max(abs(ppos), abs(pneg)) / round_to) * round_to or round_to
#     vmin = -vmax
#     lv = np.linspace(vmin, vmax, 9)
#     return vmin, vmax, lv, "RdBu_r", "both"

# # ========== 底图 ==========
# def add_base(ax):
#     ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
#     ax.add_feature(cfeature.BORDERS, linewidth=0.4, linestyle=":")
#     states = cfeature.NaturalEarthFeature(
#         "cultural","admin_1_states_provinces_lines","50m",facecolor="none"
#     )
#     ax.add_feature(states, linewidth=0.3, edgecolor="gray")

# # ========== 单个文件绘图 ==========
# def draw_one(nc_path, combined_mask=None, lon=None, lat=None, out_dir=OUTPUT_DIR):
#     meta = parse_change_filename(nc_path)
#     if meta is None:
#         print("Skip malformed:", os.path.basename(nc_path)); return

#     ds = xr.open_dataset(nc_path)
#     ren = {}
#     if "lon" in ds.coords: ren["lon"] = "longitude"
#     if "lat" in ds.coords: ren["lat"] = "latitude"
#     if ren: ds = ds.rename(ren)

#     data_lon = ds["longitude"].values
#     data_lat = ds["latitude"].values

#     if combined_mask is None or lon is None or lat is None:
#         combined_mask, _, _ = build_combined_mask_on_data_grid(data_lon, data_lat)
#         lon, lat = data_lon, data_lat
#     else:
#         if (len(lon)!=len(data_lon) or len(lat)!=len(data_lat) or
#             (not np.allclose(lon,data_lon) or not np.allclose(lat,data_lat))):
#             combined_mask, _, _ = build_combined_mask_on_data_grid(data_lon, data_lat)
#             lon, lat = data_lon, data_lat

#     LON, LAT = np.meshgrid(lon, lat)

#     for key, label, lvl_fn in [
#         ("absolute_change", "Absolute change (days/yr)", abs_levels),
#         ("relative_change", "Relative change (%)",        rel_levels),
#     ]:
#         if key not in ds:
#             print(f"  Missing var '{key}' in", os.path.basename(nc_path)); continue

#         arr = ds[key].values.astype(float)
#         if arr.shape != combined_mask.shape:
#             raise RuntimeError(f"数组与掩膜尺寸不一致：{arr.shape} vs {combined_mask.shape}")
#         arr[~combined_mask] = np.nan

#         vmin, vmax, levels, cmap_name, extend = lvl_fn(arr)

#         fig = plt.figure(figsize=(10,6), dpi=300)
#         ax = fig.add_subplot(1,1,1, projection=PROJ)
#         ax.set_extent(MAP_EXTENT, crs=ccrs.PlateCarree())
#         add_base(ax)

#         cmap = plt.get_cmap(cmap_name); cmap.set_bad(alpha=0)
#         norm = BoundaryNorm(levels, cmap.N)
#         im = ax.pcolormesh(LON, LAT, arr, transform=ccrs.PlateCarree(),
#                            cmap=cmap, norm=norm, shading="auto")

#         cbar = plt.colorbar(im, ax=ax, orientation="horizontal",
#                             pad=0.03, shrink=0.85, extend=extend)
#         cbar.set_label(label); cbar.set_ticks(levels)
#         cbar.set_ticklabels([f"{x:.0f}" for x in levels])

#         title = (f"{meta['var']} — {label}\n"
#                  f"{meta['scenario']}  {meta['model_disp']}  "
#                  f"({year_tag_to_window(meta['year_tag'])} vs 1991–2020)")
#         ax.set_title(title, fontsize=10)

#         stem = os.path.splitext(os.path.basename(nc_path))[0]
#         png = os.path.join(out_dir, f"{stem}_{key}.png")
#         pdf = os.path.join(out_dir, f"{stem}_{key}.pdf")
#         plt.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
#         plt.savefig(pdf, bbox_inches="tight", facecolor="white")
#         plt.close(fig)
#         print("✓ saved", os.path.basename(png))

#     ds.close(); gc.collect()

# # ========== 面板图（按 year_tag 分张；可选 absolute/relative） ==========
# def _panel_core(change_type, nc_dir, vars_list, scenarios, models,
#                 out_name, pos_pct=90, neg_pct=90, year_tags=None):
#     files = glob(os.path.join(nc_dir, "*.nc"))
#     meta_all = []
#     for p in files:
#         meta = parse_change_filename(p)
#         if meta is None:
#             continue
#         meta['path'] = p
#         meta_all.append(meta)
#     if not meta_all:
#         print("No usable *_change.nc files."); return

#     yt_list = sorted({m['year_tag'] for m in meta_all}) if year_tags is None else list(year_tags)

#     # 标签+等级函数
#     if change_type == "relative":
#         label = "Relative change (%)"
#         lvl_fn = lambda arr: rel_levels(arr, pos_pct=pos_pct, neg_pct=neg_pct, cap=500, round_to=10)
#     else:
#         label = "Absolute change (days/yr)"
#         lvl_fn = lambda arr: abs_levels(arr, pos_pct=pos_pct, neg_pct=neg_pct, round_to=5)

#     for yt in yt_list:
#         metas = [m for m in meta_all if m['year_tag'] == yt]
#         if not metas:
#             continue

#         index = {}
#         for m in metas:
#             key = (m['var'], m['scenario'], m['model_disp'])
#             index[key] = m['path']

#         # 基准掩膜/网格
#         first_ds = xr.open_dataset(next(iter(index.values())))
#         ren = {}
#         if "lon" in first_ds.coords: ren["lon"] = "longitude"
#         if "lat" in first_ds.coords: ren["lat"] = "latitude"
#         if ren: first_ds = first_ds.rename(ren)
#         lon0 = first_ds["longitude"].values; lat0 = first_ds["latitude"].values
#         combined, _, _ = build_combined_mask_on_data_grid(lon0, lat0)
#         LON0, LAT0 = np.meshgrid(lon0, lat0)
#         first_ds.close()

#         fig = plt.figure(figsize=(16, 9), dpi=300)
#         fig.suptitle(f"{label}  {year_tag_to_window(yt)} vs 1991–2020",
#                      fontsize=14, y=0.99)

#         order = [(vars_list[0], scenarios[0]), (vars_list[1], scenarios[0]),
#                  (vars_list[0], scenarios[1]), (vars_list[1], scenarios[1])]

#         sub = 1
#         for mdl in models:
#             for (v, scn) in order:
#                 ax = fig.add_subplot(len(models), len(order), sub, projection=PROJ)
#                 ax.set_extent(MAP_EXTENT, crs=ccrs.PlateCarree()); add_base(ax)
#                 key = (v, scn, mdl.upper())
#                 if key not in index:
#                     ax.text(0.5,0.5,'No Data',transform=ax.transAxes,
#                             ha='center',va='center',color='gray')
#                 else:
#                     ds = xr.open_dataset(index[key])
#                     ren = {}
#                     if "lon" in ds.coords: ren["lon"] = "longitude"
#                     if "lat" in ds.coords: ren["lat"] = "latitude"
#                     if ren: ds = ds.rename(ren)
#                     lon = ds["longitude"].values; lat = ds["latitude"].values
#                     # 对齐掩膜（掩膜在所画数据网格上）
#                     if (len(lon)!=len(lon0) or len(lat)!=len(lat0) or
#                         not np.allclose(lon, lon0) or not np.allclose(lat, lat0)):
#                         comb_i, _, _ = build_combined_mask_on_data_grid(lon, lat)
#                         LONi, LATi = np.meshgrid(lon, lat)
#                     else:
#                         comb_i, LONi, LATi = combined, LON0, LAT0

#                     key_name = "relative_change" if change_type=="relative" else "absolute_change"
#                     if key_name not in ds:
#                         ax.text(0.5,0.5,'Missing var',transform=ax.transAxes,
#                                 ha='center',va='center',color='gray')
#                     else:
#                         arr = ds[key_name].values.astype(float)
#                         arr[~comb_i] = np.nan
#                         vmin, vmax, levels, cmap_name, extend = lvl_fn(arr)
#                         cmap = plt.get_cmap(cmap_name); cmap.set_bad(alpha=0)
#                         norm = BoundaryNorm(levels, cmap.N)
#                         im = ax.pcolormesh(LONi, LATi, arr, transform=ccrs.PlateCarree(),
#                                            cmap=cmap, norm=norm, shading='auto')
#                         cbar = plt.colorbar(im, ax=ax, orientation='horizontal',
#                                             pad=0.01, fraction=0.045, extend=extend)
#                         cbar.set_ticks(levels[::2])
#                         cbar.set_ticklabels([f"{x:.0f}" for x in levels[::2]])
#                         cbar.ax.tick_params(labelsize=6)
#                         cbar.set_label(label, fontsize=6)
#                     ds.close()

#                 # 栏头与行标
#                 if sub <= len(order):
#                     ax.set_title(f"{scn}  {'ABDp' if v=='fire_probability' else 'OBEp'}",
#                                  fontsize=10)
#                 if (sub-1) % len(order) == 0:
#                     ax.text(-0.1, 0.5, mdl, transform=ax.transAxes, rotation=90,
#                             va='center', ha='center', fontsize=11)
#                 sub += 1

#         plt.subplots_adjust(left=0.05, right=0.99, top=0.95, bottom=0.03,
#                             wspace=0.06, hspace=0.10)
#         win_for_name = year_tag_to_window(yt).replace('–','-')
#         png = os.path.join(OUTPUT_DIR, f"{out_name}_{change_type}_{yt}_{win_for_name}_vs_1991-2020.png")
#         pdf = os.path.join(OUTPUT_DIR, f"{out_name}_{change_type}_{yt}_{win_for_name}_vs_1991-2020.pdf")
#         plt.savefig(png, dpi=300, bbox_inches='tight', facecolor='white')
#         plt.savefig(pdf, bbox_inches='tight', facecolor='white')
#         plt.close(fig)
#         print("✓ panel saved:", png)

# # ========== 主程序 ==========
# def main():
#     nc_files = glob(os.path.join(INPUT_DIR, "*.nc"))
#     if not nc_files:
#         print("No NetCDF files in:", INPUT_DIR); sys.exit(0)

#     # 解析 + 过滤
#     metas = []
#     for p in nc_files:
#         m = parse_change_filename(p)
#         if not m:
#             continue
#         if SELECT_YEAR_TAGS and m['year_tag'] not in SELECT_YEAR_TAGS:
#             continue
#         if SELECT_SSPS and m['ssp'] not in SELECT_SSPS:
#             continue
#         if SELECT_MODELS and m['model_disp'] not in SELECT_MODELS:
#             continue
#         if SELECT_VARS and m['var'] not in SELECT_VARS:
#             continue
#         m['path'] = p
#         metas.append(m)

#     if not metas:
#         print("No files after filters."); sys.exit(0)

#     # —— 单图：逐文件输出（Absolute/Relative 两套图）——
#     if DO_SINGLE:
#         for m in metas:
#             draw_one(m['path'], combined_mask=None, lon=None, lat=None, out_dir=OUTPUT_DIR)

#     # —— 面板：每个 yearTag 出图（可选 absolute/relative）——
#     if DO_PANEL:
#         year_tags = sorted({m['year_tag'] for m in metas})
#         vars_list = tuple(SELECT_VARS) if SELECT_VARS else ('fire_probability','obe_probability')
#         scenarios = tuple(f"SSP{s}" for s in (SELECT_SSPS if SELECT_SSPS else ('370','245')))
#         models = tuple(SELECT_MODELS) if SELECT_MODELS else ('CANESM','ECEARTH','GFDL','UKESM')
#         out_name = f"ABDp_OBEp_{'_'.join(scenarios)}"
#         for change_type in PANEL_CHANGE_TYPES:
#             _panel_core(
#                 change_type=change_type,
#                 nc_dir=INPUT_DIR,
#                 vars_list=vars_list,
#                 scenarios=scenarios,
#                 models=models,
#                 out_name=out_name,
#                 pos_pct=90, neg_pct=90,
#                 year_tags=year_tags
#             )

# if __name__ == "__main__":
#     main()

# -*- coding: utf-8 -*-
import os, re, sys, gc
from glob import glob

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geopandas as gpd
import regionmask
from cartopy.io import shapereader

# ========== 路径配置 ==========
INPUT_DIR  = r"E:\Projection paper\analysis\maps"                      # *_change.nc 的目录（E盘）
OUTPUT_DIR = os.path.join(INPUT_DIR, "figures")                        # 输出目录
BIOME_NC   = r"D:\000_collections\020_Chapter2\US_CAN_biome.nc"        # 生物群系（D盘）
USCAN_SHP  = r"D:\000_collections\010_Nighttime Burning\011_Data\013_Biome_wwf2017\US_Canada_merged.shp"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ========== 全局外观 ==========
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size']   = 9

# 北美投影
PROJ = ccrs.LambertConformal(
    central_longitude=-100, central_latitude=50, standard_parallels=(35, 65)
)

# 只画北美大陆范围（US+Canada）
MAP_EXTENT = [-135, -67.5, 24.8, 82]

# ========== 选择器（None 表示不过滤）==========
SELECT_YEAR_TAGS = ('2040', '2070')   # or None
SELECT_SSPS      = ('245', '370')     # or None
SELECT_MODELS    = ('CANESM','ECEARTH','GFDL','UKESM')  # or None
SELECT_VARS      = ('fire_probability','obe_probability')  # or None

# ========== 生成什么 ==========
DO_SINGLE = True           # 每个文件单图（absolute + relative）
DO_PANEL  = True           # 4×4 面板 + 单列图
PANEL_CHANGE_TYPES = ("relative", "absolute")   # 面板：相对 & 绝对 都画

# ========== 文件名解析 ==========
# 示例: fire_probability_days_2070_370_GFDL_change.nc
NAME_RE = re.compile(
    r'^(?P<var>.+?)_days_(?P<year>\d{4})_(?P<ssp>126|245|370|585)_(?P<model>[A-Za-z0-9]+)_change\.nc$',
    re.IGNORECASE
)

def parse_change_filename(path):
    m = NAME_RE.match(os.path.basename(path))
    if not m:
        return None
    meta = m.groupdict()
    meta["scenario"]   = f"SSP{meta['ssp']}"
    meta["model_disp"] = meta["model"].upper()
    meta["year_tag"]   = meta["year"]
    return meta

def year_tag_to_window(tag: str) -> str:
    if tag == '2040':
        return '2041–2070'
    if tag == '2070':
        return '2071–2100'
    return f'future period ({tag})'

# ========== 网格一致性辅助 ==========
def _grid_summary(lon, lat):
    return dict(
        lon_min=float(np.nanmin(lon)), lon_max=float(np.nanmax(lon)),
        lat_min=float(np.nanmin(lat)), lat_max=float(np.nanmax(lat)),
        dlon=float(np.nanmedian(np.diff(lon))) if len(lon) > 1 else np.nan,
        dlat=float(np.nanmedian(np.diff(lat))) if len(lat) > 1 else np.nan,
        lon_asc=bool(lon[1] > lon[0]) if len(lon) > 1 else True,
        lat_asc=bool(lat[1] > lat[0]) if len(lat) > 1 else True
    )

def _roughly_compatible(src_lon, src_lat, dst_lon, dst_lat,
                        tol_edge_deg=0.75, tol_step_frac=0.25, tol_extra_rc=2):
    s, d = _grid_summary(src_lon, src_lat), _grid_summary(dst_lon, dst_lat)
    if s["lon_asc"] != d["lon_asc"] or s["lat_asc"] != d["lat_asc"]:
        return False
    if (abs(s["lon_min"] - d["lon_min"]) > tol_edge_deg or
        abs(s["lon_max"] - d["lon_max"]) > tol_edge_deg or
        abs(s["lat_min"] - d["lat_min"]) > tol_edge_deg or
        abs(s["lat_max"] - d["lat_max"]) > tol_edge_deg):
        return False
    if (np.isfinite(s["dlon"]) and np.isfinite(d["dlon"]) and
        abs(s["dlon"] - d["dlon"]) / max(1e-6, abs(d["dlon"])) > tol_step_frac):
        return False
    if (abs(len(src_lat) - len(dst_lat)) > tol_extra_rc or
        abs(len(src_lon) - len(dst_lon)) > tol_extra_rc):
        return False
    return True

# ========== 掩膜：region（US+CAN）+ biome ==========
EXCLUDE_BIOMES = [50.1, 90.1, 12.1, 13.1, 21.1, 22.1, 31.1, 32.1, 35.1]

def build_combined_mask_on_data_grid(data_lon, data_lat):
    """
    返回:
      combined_mask: region & bio
      region_mask  : 只有 US+Canada 区域掩膜
      bio_mask     : 生物群系掩膜
    """
    LON2D, LAT2D = np.meshgrid(data_lon, data_lat)

    # 区域掩膜（US+CAN shapefile）
    shp = gpd.read_file(USCAN_SHP)
    region_mask = ~np.isnan(regionmask.mask_geopandas(shp, LON2D, LAT2D, overlap=False))

    # 生物群系掩膜（必要时最近邻插值到数据网格）
    bds = xr.open_dataset(BIOME_NC)
    ren = {}
    if "lon" in bds.coords and "longitude" not in bds.coords: ren["lon"] = "longitude"
    if "lat" in bds.coords and "latitude"  not in bds.coords: ren["lat"] = "latitude"
    if ren: bds = bds.rename(ren)

    bio_var = "gez_code_id" if "gez_code_id" in bds.data_vars else ("gez_code" if "gez_code" in bds.data_vars else None)
    if bio_var is None:
        raise RuntimeError(f"Biome 文件缺少 gez_code(_id)：{BIOME_NC}")

    bio_lon = bds["longitude"].values
    bio_lat = bds["latitude"].values

    if (len(bio_lon) == len(data_lon) and len(bio_lat) == len(data_lat) and
        np.allclose(bio_lon, data_lon) and np.allclose(bio_lat, data_lat)):
        bio_on_data = bds[bio_var]
    else:
        if not _roughly_compatible(bio_lon, bio_lat, data_lon, data_lat):
            raise RuntimeError("生物群系列网格与数据网格差异过大，拒绝自动插值。")
        bio_on_data = bds[bio_var].interp(
            latitude=xr.DataArray(data_lat, dims=("latitude",)),
            longitude=xr.DataArray(data_lon, dims=("longitude",)),
            method="nearest"
        )

    bio_arr = bio_on_data.values
    bio_mask = (~np.isnan(bio_arr)).copy()
    for v in EXCLUDE_BIOMES:
        bio_mask &= (bio_arr != v)

    combined = region_mask & bio_mask
    return combined, region_mask, bio_mask

# ========== 色阶函数 ==========
def abs_levels(masked, pos_pct=95, neg_pct=95, round_to=5):
    arr = masked[np.isfinite(masked)]
    if arr.size == 0:
        lv = np.linspace(-1, 1, 9); return -1, 1, lv, "RdBu_r", "both"
    pos = arr[arr > 0]; neg = arr[arr < 0]
    ppos = np.nanpercentile(pos, pos_pct) if pos.size else 0.0
    pneg = -np.nanpercentile(-neg, neg_pct) if neg.size else 0.0
    vmax = np.ceil(max(abs(ppos), abs(pneg)) / round_to) * round_to or round_to
    vmin = -vmax
    levels = np.linspace(vmin, vmax, 9)
    return vmin, vmax, levels, "RdBu_r", "both"

def rel_levels(masked, pos_pct=95, neg_pct=95, cap=500, round_to=10):
    arr = masked[np.isfinite(masked)]
    if arr.size == 0:
        lv = np.linspace(-10, 10, 9); return -10, 10, lv, "RdBu_r", "both"
    pos = arr[arr > 0]; neg = arr[arr < 0]
    ppos = np.nanpercentile(pos, pos_pct) if pos.size else 0.0
    pneg = -np.nanpercentile(-neg, neg_pct) if neg.size else 0.0
    ppos = min(ppos, cap); pneg = max(pneg, -cap)

    if pos.size and not neg.size:
        vmin, vmax = 0, np.ceil(ppos / round_to) * round_to or round_to
        lv = np.linspace(vmin, vmax, 9); return vmin, vmax, lv, "Reds", "max"
    if neg.size and not pos.size:
        vmin, vmax = np.floor(pneg / round_to) * round_to, 0
        lv = np.linspace(vmin, vmax, 9); return vmin, vmax, lv, "Blues_r", "min"

    vmax = np.ceil(max(abs(ppos), abs(pneg)) / round_to) * round_to or round_to
    vmin = -vmax
    lv = np.linspace(vmin, vmax, 9)
    return vmin, vmax, lv, "RdBu_r", "both"

# ========== 底图：只画 USA + CAN，轻量 ==========
# 国家边界：110m
_country_reader_110 = shapereader.Reader(
    shapereader.natural_earth(
        resolution='110m',
        category='cultural',
        name='admin_0_countries'
    )
)
COUNTRY_GEOMS_USCAN = [
    rec.geometry
    for rec in _country_reader_110.records()
    if rec.attributes.get('ADM0_A3') in ('USA', 'CAN')
]

# 州 / 省界：50m
_states_reader_50 = shapereader.Reader(
    shapereader.natural_earth(
        resolution='50m',
        category='cultural',
        name='admin_1_states_provinces_lines'
    )
)
STATE_GEOMS_USCAN = []
for rec in _states_reader_50.records():
    attrs_lower = {k.lower(): v for k, v in rec.attributes.items()}
    code = attrs_lower.get('adm0_a3', '')
    if code in ('USA', 'CAN'):
        STATE_GEOMS_USCAN.append(rec.geometry)

print("US/CAN states found:", len(STATE_GEOMS_USCAN))

def add_base(ax):
    ax.add_geometries(
        COUNTRY_GEOMS_USCAN,
        crs=ccrs.PlateCarree(),
        edgecolor='0.35',
        facecolor='none',
        linewidth=0.5,
        zorder=2,
    )
    ax.add_geometries(
        STATE_GEOMS_USCAN,
        crs=ccrs.PlateCarree(),
        edgecolor='0.6',
        facecolor='none',
        linewidth=0.4,
        alpha=0.9,
        zorder=3,
    )

# ========== 单个文件绘图 ==========
def draw_one(nc_path, combined_mask=None, region_mask=None, lon=None, lat=None, out_dir=OUTPUT_DIR):
    meta = parse_change_filename(nc_path)
    if meta is None:
        print("Skip malformed:", os.path.basename(nc_path)); return

    ds = xr.open_dataset(nc_path)
    ren = {}
    if "lon" in ds.coords: ren["lon"] = "longitude"
    if "lat" in ds.coords: ren["lat"] = "latitude"
    if ren: ds = ds.rename(ren)

    data_lon = ds["longitude"].values
    data_lat = ds["latitude"].values

    if combined_mask is None or region_mask is None or lon is None or lat is None:
        combined_mask, region_mask, _ = build_combined_mask_on_data_grid(data_lon, data_lat)
        lon, lat = data_lon, data_lat
    else:
        if (len(lon)!=len(data_lon) or len(lat)!=len(data_lat) or
            (not np.allclose(lon,data_lon) or not np.allclose(lat,data_lat))):
            combined_mask, region_mask, _ = build_combined_mask_on_data_grid(data_lon, data_lat)
            lon, lat = data_lon, data_lat

    LON, LAT = np.meshgrid(lon, lat)

    for key, label, lvl_fn in [
        ("absolute_change", "Absolute change (days/yr)", abs_levels),
        ("relative_change", "Relative change (%)",        rel_levels),
    ]:
        if key not in ds:
            print(f"  Missing var '{key}' in", os.path.basename(nc_path)); continue

        arr = ds[key].values.astype(float)
        if arr.shape != combined_mask.shape:
            raise RuntimeError(f"数组与掩膜尺寸不一致：{arr.shape} vs {combined_mask.shape}")
        arr[~combined_mask] = np.nan

        vmin, vmax, levels, cmap_name, extend = lvl_fn(arr)

        fig = plt.figure(figsize=(5, 3.75), dpi=300)
        ax = fig.add_subplot(1,1,1, projection=PROJ)
        ax.set_extent(MAP_EXTENT, crs=ccrs.PlateCarree())
        add_base(ax)
        if hasattr(ax, "outline_patch"):
            ax.outline_patch.set_visible(False)

        cmap = plt.get_cmap(cmap_name); cmap.set_bad(alpha=0)
        norm = BoundaryNorm(levels, cmap.N)
        im = ax.pcolormesh(LON, LAT, arr, transform=ccrs.PlateCarree(),
                           cmap=cmap, norm=norm, shading="auto")

        cbar = plt.colorbar(im, ax=ax, orientation="vertical",
                            pad=0.02, shrink=0.9, extend=extend)
        cbar.set_label(label)
        cbar.set_ticks(levels)
        cbar.set_ticklabels([f"{x:.0f}" for x in levels])

        title = (f"{meta['var']} — {label}\n"
                 f"{meta['scenario']}  {meta['model_disp']}  "
                 f"({year_tag_to_window(meta['year_tag'])} vs 1991–2020)")
        ax.set_title(title)

        stem = os.path.splitext(os.path.basename(nc_path))[0]
        png = os.path.join(out_dir, f"{stem}_{key}.png")
        pdf = os.path.join(out_dir, f"{stem}_{key}.pdf")
        plt.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
        plt.savefig(pdf, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print("✓ saved", os.path.basename(png))

    ds.close(); gc.collect()

# ========== 面板图（整图 + 单列图） ==========
def _panel_core(change_type, nc_dir, vars_list, scenarios, models,
                out_name, pos_pct=90, neg_pct=90, year_tags=None):

    files = glob(os.path.join(nc_dir, "*.nc"))
    meta_all = []
    for p in files:
        meta = parse_change_filename(p)
        if meta is None:
            continue
        meta['path'] = p
        meta_all.append(meta)
    if not meta_all:
        print("No usable *_change.nc files."); return

    yt_list = sorted({m['year_tag'] for m in meta_all}) if year_tags is None else list(year_tags)

    if change_type == "relative":
        label = "Relative change (%)"
        lvl_fn = lambda arr: rel_levels(arr, pos_pct=pos_pct, neg_pct=neg_pct, cap=500, round_to=10)
    else:
        label = "Absolute change (days/yr)"
        lvl_fn = lambda arr: abs_levels(arr, pos_pct=pos_pct, neg_pct=neg_pct, round_to=5)

    for yt in yt_list:
        metas = [m for m in meta_all if m['year_tag'] == yt]
        if not metas:
            continue

        # 路径索引
        index = {}
        for m in metas:
            key = (m['var'], m['scenario'], m['model_disp'])
            index[key] = m['path']

        # 基准网格与掩膜
        first_ds = xr.open_dataset(next(iter(index.values())))
        ren = {}
        if "lon" in first_ds.coords: ren["lon"] = "longitude"
        if "lat" in first_ds.coords: ren["lat"] = "latitude"
        if ren: first_ds = first_ds.rename(ren)
        lon0 = first_ds["longitude"].values; lat0 = first_ds["latitude"].values
        combined0, region0, _ = build_combined_mask_on_data_grid(lon0, lat0)
        LON0, LAT0 = np.meshgrid(lon0, lat0)
        first_ds.close()

        panel_extent = MAP_EXTENT
        yt_window_str = year_tag_to_window(yt)
        order = [(vars_list[0], scenarios[0]), (vars_list[1], scenarios[0]),
                 (vars_list[0], scenarios[1]), (vars_list[1], scenarios[1])]

        # ---------- 1) 整体 4×4 面板（宽度减半：8×9） ----------
        fig = plt.figure(figsize=(8, 8), dpi=300)
        fig.suptitle(f"{label}  {yt_window_str} vs 1991–2020",
                     fontsize=13, y=0.98)

        sub = 1
        for mdl in models:
            for (v, scn) in order:
                ax = fig.add_subplot(len(models), len(order), sub, projection=PROJ)
                ax.set_extent(panel_extent, crs=ccrs.PlateCarree())
                add_base(ax)
                if hasattr(ax, "outline_patch"):
                    ax.outline_patch.set_visible(False)

                key = (v, scn, mdl.upper())
                if key not in index:
                    ax.text(0.5,0.5,'No Data',transform=ax.transAxes,
                            ha='center',va='center',color='gray')
                else:
                    ds = xr.open_dataset(index[key])
                    ren = {}
                    if "lon" in ds.coords: ren["lon"] = "longitude"
                    if "lat" in ds.coords: ren["lat"] = "latitude"
                    if ren: ds = ds.rename(ren)
                    lon = ds["longitude"].values; lat = ds["latitude"].values

                    if (len(lon)!=len(lon0) or len(lat)!=len(lat0) or
                        not np.allclose(lon, lon0) or not np.allclose(lat, lat0)):
                        combined_i, region_i, _ = build_combined_mask_on_data_grid(lon, lat)
                        LONi, LATi = np.meshgrid(lon, lat)
                        local_mask = combined_i
                    else:
                        LONi, LATi = LON0, LAT0
                        local_mask = combined0

                    key_name = "relative_change" if change_type=="relative" else "absolute_change"
                    if key_name not in ds:
                        ax.text(0.5,0.5,'Missing var',transform=ax.transAxes,
                                ha='center',va='center',color='gray')
                    else:
                        arr = ds[key_name].values.astype(float)
                        arr[~local_mask] = np.nan
                        vmin, vmax, levels, cmap_name, extend = lvl_fn(arr)
                        cmap = plt.get_cmap(cmap_name); cmap.set_bad(alpha=0)
                        norm = BoundaryNorm(levels, cmap.N)
                        im = ax.pcolormesh(
                            LONi, LATi, arr,
                            transform=ccrs.PlateCarree(),
                            cmap=cmap, norm=norm, shading='auto'
                        )
                        cbar = plt.colorbar(
                            im, ax=ax, orientation='horizontal',
                            pad=0.02, fraction=0.04, extend=extend
                        )
                        cbar.set_ticks(levels[::2])
                        cbar.set_ticklabels([f"{x:.0f}" for x in levels[::2]])
                        cbar.ax.tick_params(labelsize=6)
                        cbar.set_label(label, fontsize=6)
                    ds.close()

                # 顶部列标题 & 行标签
                if sub <= len(order):
                    ax.set_title(f"{scn}  {'ABDp' if v=='fire_probability' else 'OBEp'}",
                                 fontsize=10)
                if (sub-1) % len(order) == 0:
                    ax.text(-0.08, 0.5, mdl, transform=ax.transAxes, rotation=90,
                            va='center', ha='center', fontsize=10)
                sub += 1

        plt.subplots_adjust(
            left=0.06, right=0.99,
            top=0.94, bottom=0.06,
            wspace=0.06, hspace=0.12
        )
        win_for_name = yt_window_str.replace('–','-')
        png = os.path.join(
            OUTPUT_DIR,
            f"{out_name}_{change_type}_{yt}_{win_for_name}_vs_1991-2020.png"
        )
        pdf = os.path.join(
            OUTPUT_DIR,
            f"{out_name}_{change_type}_{yt}_{win_for_name}_vs_1991-2020.pdf"
        )
        plt.savefig(png, dpi=300, bbox_inches='tight', facecolor='white')
        plt.savefig(pdf, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print("✓ panel saved:", png)

        # ---------- 2) 单列图：每列一个 4×1 面板（宽度再减半：4×9） ----------
        for col_idx, (v_col, scn_col) in enumerate(order):
            fig_col = plt.figure(figsize=(4, 8), dpi=300)
            var_label = 'ABDp' if v_col == 'fire_probability' else 'OBEp'
            fig_col.suptitle(
                f"{label}  {yt_window_str} vs 1991–2020\n{scn_col} {var_label}",
                fontsize=12, y=0.97
            )

            sub = 1
            for mdl in models:
                ax = fig_col.add_subplot(len(models), 1, sub, projection=PROJ)
                ax.set_extent(panel_extent, crs=ccrs.PlateCarree())
                add_base(ax)
                if hasattr(ax, "outline_patch"):
                    ax.outline_patch.set_visible(False)

                key = (v_col, scn_col, mdl.upper())
                if key not in index:
                    ax.text(0.5,0.5,'No Data',transform=ax.transAxes,
                            ha='center',va='center',color='gray')
                else:
                    ds = xr.open_dataset(index[key])
                    ren = {}
                    if "lon" in ds.coords: ren["lon"] = "longitude"
                    if "lat" in ds.coords: ren["lat"] = "latitude"
                    if ren: ds = ds.rename(ren)
                    lon = ds["longitude"].values; lat = ds["latitude"].values

                    if (len(lon)!=len(lon0) or len(lat)!=len(lat0) or
                        not np.allclose(lon, lon0) or not np.allclose(lat, lat0)):
                        combined_i, region_i, _ = build_combined_mask_on_data_grid(lon, lat)
                        LONi, LATi = np.meshgrid(lon, lat)
                        local_mask = combined_i
                    else:
                        LONi, LATi = LON0, LAT0
                        local_mask = combined0

                    key_name = "relative_change" if change_type=="relative" else "absolute_change"
                    if key_name not in ds:
                        ax.text(0.5,0.5,'Missing var',transform=ax.transAxes,
                                ha='center',va='center',color='gray')
                    else:
                        arr = ds[key_name].values.astype(float)
                        arr[~local_mask] = np.nan
                        vmin, vmax, levels, cmap_name, extend = lvl_fn(arr)
                        cmap = plt.get_cmap(cmap_name); cmap.set_bad(alpha=0)
                        norm = BoundaryNorm(levels, cmap.N)
                        im = ax.pcolormesh(
                            LONi, LATi, arr,
                            transform=ccrs.PlateCarree(),
                            cmap=cmap, norm=norm, shading='auto'
                        )
                        cbar = plt.colorbar(
                            im, ax=ax, orientation='horizontal',
                            pad=0.02, fraction=0.06, extend=extend
                        )
                        cbar.set_ticks(levels[::2])
                        cbar.set_ticklabels([f"{x:.0f}" for x in levels[::2]])
                        cbar.ax.tick_params(labelsize=6)
                        cbar.set_label(label, fontsize=6)
                    ds.close()

                # 行标签
                ax.text(-0.08, 0.5, mdl, transform=ax.transAxes, rotation=90,
                        va='center', ha='center', fontsize=10)
                sub += 1

            plt.subplots_adjust(
                left=0.15, right=0.98,
                top=0.93, bottom=0.06,
                hspace=0.15
            )
            col_name = f"{scn_col}_{var_label}"
            png_col = os.path.join(
                OUTPUT_DIR,
                f"{out_name}_{change_type}_{yt}_{win_for_name}_{col_name}_column.png"
            )
            pdf_col = os.path.join(
                OUTPUT_DIR,
                f"{out_name}_{change_type}_{yt}_{win_for_name}_{col_name}_column.pdf"
            )
            plt.savefig(png_col, dpi=300, bbox_inches='tight', facecolor='white')
            plt.savefig(pdf_col, bbox_inches='tight', facecolor='white')
            plt.close(fig_col)
            print("✓ column panel saved:", png_col)

# ========== 主程序 ==========
def main():
    nc_files = glob(os.path.join(INPUT_DIR, "*.nc"))
    if not nc_files:
        print("No NetCDF files in:", INPUT_DIR); sys.exit(0)

    metas = []
    for p in nc_files:
        m = parse_change_filename(p)
        if not m:
            continue
        if SELECT_YEAR_TAGS and m['year_tag'] not in SELECT_YEAR_TAGS:
            continue
        if SELECT_SSPS and m['ssp'] not in SELECT_SSPS:
            continue
        if SELECT_MODELS and m['model_disp'] not in SELECT_MODELS:
            continue
        if SELECT_VARS and m['var'] not in SELECT_VARS:
            continue
        m['path'] = p
        metas.append(m)

    if not metas:
        print("No files after filters."); sys.exit(0)

    # 单图
    if DO_SINGLE:
        for m in metas:
            draw_one(
                m['path'],
                combined_mask=None,
                region_mask=None,
                lon=None, lat=None,
                out_dir=OUTPUT_DIR
            )

    # 面板 + 单列
    if DO_PANEL:
        year_tags = sorted({m['year_tag'] for m in metas})
        vars_list = tuple(SELECT_VARS) if SELECT_VARS else ('fire_probability','obe_probability')
        scenarios = tuple(f"SSP{s}" for s in (SELECT_SSPS if SELECT_SSPS else ('370','245')))
        models = tuple(SELECT_MODELS) if SELECT_MODELS else ('CANESM','ECEARTH','GFDL','UKESM')
        out_name = f"ABDp_OBEp_{'_'.join(scenarios)}"
        for change_type in PANEL_CHANGE_TYPES:
            _panel_core(
                change_type=change_type,
                nc_dir=INPUT_DIR,
                vars_list=vars_list,
                scenarios=scenarios,
                models=models,
                out_name=out_name,
                pos_pct=90, neg_pct=90,
                year_tags=year_tags
            )

if __name__ == "__main__":
    main()













# -*- coding: utf-8 -*-
import os, re, sys, gc
from glob import glob

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
import cartopy.crs as ccrs
import geopandas as gpd
import regionmask
from cartopy.io import shapereader

# =============================================================================
# 0) PATHS
# =============================================================================
INPUT_DIR_MAIN = r"E:\Projection paper\analysis\maps"
INPUT_DIR_SUP  = None  # 可选：r"E:\Projection paper_sup_scens\analysis\maps"

BIOME_NC  = r"D:\000_collections\020_Chapter2\US_CAN_biome.nc"
USCAN_SHP = r"D:\000_collections\010_Nighttime Burning\011_Data\013_Biome_wwf2017\US_Canada_merged.shp"

def make_unique_outdir(base_dir, folder_name="figures_ensemble_ssp370_2070_single"):
    v = 1
    while True:
        out = os.path.join(base_dir, f"{folder_name}_v{v}")
        if not os.path.exists(out):
            os.makedirs(out, exist_ok=False)
            return out
        v += 1

OUTPUT_DIR = make_unique_outdir(INPUT_DIR_MAIN)

# =============================================================================
# 1) STYLE (match your Part-1 single plots)
# =============================================================================
plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"]   = 9
plt.rcParams["axes.unicode_minus"] = False

# =============================================================================
# 2) MAP SETTINGS
# =============================================================================
PROJ = ccrs.LambertConformal(
    central_longitude=-100, central_latitude=50, standard_parallels=(35, 65)
)
MAP_EXTENT = [-135, -67.5, 24.8, 82]

# =============================================================================
# 3) TARGETS
# =============================================================================
YEAR_TAG   = "2070"
SSP_ONLY   = "370"
SCENARIO   = f"SSP{SSP_ONLY}"
MODELS     = ("CANESM", "ECEARTH", "GFDL", "UKESM")
VARS       = ("fire_probability", "obe_probability")  # ABDp, OBEp
CHANGE_TYPES = ("absolute", "relative")               # 4 plots total

EXCLUDE_BIOMES = [50.1, 90.1, 12.1, 13.1, 21.1, 22.1, 31.1, 32.1, 35.1]

# =============================================================================
# 4) FILENAME PARSING
# =============================================================================
NAME_RE = re.compile(
    r'^(?P<var>.+?)_days_(?P<year>\d{4})_(?P<ssp>126|245|370|585)_(?P<model>[A-Za-z0-9]+)_change\.nc$',
    re.IGNORECASE
)

def parse_change_filename(path):
    m = NAME_RE.match(os.path.basename(path))
    if not m:
        return None
    d = m.groupdict()
    d["scenario"]   = f"SSP{d['ssp']}"
    d["model_disp"] = d["model"].upper()
    d["year_tag"]   = d["year"]
    d["path"]       = path
    return d

def year_tag_to_window(tag: str) -> str:
    return "2071–2100" if tag == "2070" else ("2041–2070" if tag == "2040" else tag)

# =============================================================================
# 5) GRID COMPATIBILITY (reuse your logic)
# =============================================================================
def _grid_summary(lon, lat):
    return dict(
        lon_min=float(np.nanmin(lon)), lon_max=float(np.nanmax(lon)),
        lat_min=float(np.nanmin(lat)), lat_max=float(np.nanmax(lat)),
        dlon=float(np.nanmedian(np.diff(lon))) if len(lon) > 1 else np.nan,
        dlat=float(np.nanmedian(np.diff(lat))) if len(lat) > 1 else np.nan,
        lon_asc=bool(lon[1] > lon[0]) if len(lon) > 1 else True,
        lat_asc=bool(lat[1] > lat[0]) if len(lat) > 1 else True
    )

def _roughly_compatible(src_lon, src_lat, dst_lon, dst_lat,
                        tol_edge_deg=0.75, tol_step_frac=0.25, tol_extra_rc=2):
    s, d = _grid_summary(src_lon, src_lat), _grid_summary(dst_lon, dst_lat)
    if s["lon_asc"] != d["lon_asc"] or s["lat_asc"] != d["lat_asc"]:
        return False
    if (abs(s["lon_min"] - d["lon_min"]) > tol_edge_deg or
        abs(s["lon_max"] - d["lon_max"]) > tol_edge_deg or
        abs(s["lat_min"] - d["lat_min"]) > tol_edge_deg or
        abs(s["lat_max"] - d["lat_max"]) > tol_edge_deg):
        return False
    if (np.isfinite(s["dlon"]) and np.isfinite(d["dlon"]) and
        abs(s["dlon"] - d["dlon"]) / max(1e-6, abs(d["dlon"])) > tol_step_frac):
        return False
    if (abs(len(src_lat) - len(dst_lat)) > tol_extra_rc or
        abs(len(src_lon) - len(dst_lon)) > tol_extra_rc):
        return False
    return True

# =============================================================================
# 6) MASK
# =============================================================================
_mask_cache = {}

def build_combined_mask_on_data_grid(data_lon, data_lat):
    key = (len(data_lon), len(data_lat), float(data_lon[0]), float(data_lon[-1]),
           float(data_lat[0]), float(data_lat[-1]))
    if key in _mask_cache:
        return _mask_cache[key]

    LON2D, LAT2D = np.meshgrid(data_lon, data_lat)

    shp = gpd.read_file(USCAN_SHP)
    region_mask = ~np.isnan(regionmask.mask_geopandas(shp, LON2D, LAT2D, overlap=False))

    bds = xr.open_dataset(BIOME_NC)
    ren = {}
    if "lon" in bds.coords and "longitude" not in bds.coords: ren["lon"] = "longitude"
    if "lat" in bds.coords and "latitude"  not in bds.coords: ren["lat"] = "latitude"
    if ren: bds = bds.rename(ren)

    bio_var = "gez_code_id" if "gez_code_id" in bds.data_vars else ("gez_code" if "gez_code" in bds.data_vars else None)
    if bio_var is None:
        raise RuntimeError(f"Biome file missing gez_code(_id): {BIOME_NC}")

    bio_lon = bds["longitude"].values
    bio_lat = bds["latitude"].values

    if (len(bio_lon) == len(data_lon) and len(bio_lat) == len(data_lat) and
        np.allclose(bio_lon, data_lon) and np.allclose(bio_lat, data_lat)):
        bio_on_data = bds[bio_var]
    else:
        if not _roughly_compatible(bio_lon, bio_lat, data_lon, data_lat):
            raise RuntimeError("Biome grid too different from data grid; refuse to interp.")
        bio_on_data = bds[bio_var].interp(
            latitude=xr.DataArray(data_lat, dims=("latitude",)),
            longitude=xr.DataArray(data_lon, dims=("longitude",)),
            method="nearest"
        )

    bio_arr = bio_on_data.values
    bio_mask = (~np.isnan(bio_arr)).copy()
    for v in EXCLUDE_BIOMES:
        bio_mask &= (bio_arr != v)

    combined = region_mask & bio_mask
    bds.close()

    _mask_cache[key] = combined
    return combined

# =============================================================================
# 7) LEVELS
# =============================================================================
def abs_levels(masked, pos_pct=95, neg_pct=95, round_to=5):
    arr = masked[np.isfinite(masked)]
    if arr.size == 0:
        lv = np.linspace(-1, 1, 9); return -1, 1, lv, "RdBu_r", "both"
    pos = arr[arr > 0]; neg = arr[arr < 0]
    ppos = np.nanpercentile(pos, pos_pct) if pos.size else 0.0
    pneg = -np.nanpercentile(-neg, neg_pct) if neg.size else 0.0
    vmax = np.ceil(max(abs(ppos), abs(pneg)) / round_to) * round_to or round_to
    vmin = -vmax
    levels = np.linspace(vmin, vmax, 9)
    return vmin, vmax, levels, "RdBu_r", "both"

def rel_levels(masked, pos_pct=95, neg_pct=95, cap=500, round_to=10):
    arr = masked[np.isfinite(masked)]
    if arr.size == 0:
        lv = np.linspace(-10, 10, 9); return -10, 10, lv, "RdBu_r", "both"
    pos = arr[arr > 0]; neg = arr[arr < 0]
    ppos = np.nanpercentile(pos, pos_pct) if pos.size else 0.0
    pneg = -np.nanpercentile(-neg, neg_pct) if neg.size else 0.0
    ppos = min(ppos, cap); pneg = max(pneg, -cap)

    if pos.size and not neg.size:
        vmin, vmax = 0, np.ceil(ppos / round_to) * round_to or round_to
        lv = np.linspace(vmin, vmax, 9); return vmin, vmax, lv, "Reds", "max"
    if neg.size and not pos.size:
        vmin, vmax = np.floor(pneg / round_to) * round_to, 0
        lv = np.linspace(vmin, vmax, 9); return vmin, vmax, lv, "Blues_r", "min"

    vmax = np.ceil(max(abs(ppos), abs(pneg)) / round_to) * round_to or round_to
    vmin = -vmax
    lv = np.linspace(vmin, vmax, 9)
    return vmin, vmax, lv, "RdBu_r", "both"

# =============================================================================
# 8) BASEMAP (US+CAN only)
# =============================================================================
_country_reader_110 = shapereader.Reader(
    shapereader.natural_earth(resolution="110m", category="cultural", name="admin_0_countries")
)
COUNTRY_GEOMS_USCAN = [
    rec.geometry for rec in _country_reader_110.records()
    if rec.attributes.get("ADM0_A3") in ("USA", "CAN")
]

_states_reader_50 = shapereader.Reader(
    shapereader.natural_earth(resolution="50m", category="cultural", name="admin_1_states_provinces_lines")
)
STATE_GEOMS_USCAN = []
for rec in _states_reader_50.records():
    attrs_lower = {k.lower(): v for k, v in rec.attributes.items()}
    if attrs_lower.get("adm0_a3", "") in ("USA", "CAN"):
        STATE_GEOMS_USCAN.append(rec.geometry)

def add_base(ax):
    ax.add_geometries(COUNTRY_GEOMS_USCAN, crs=ccrs.PlateCarree(),
                      edgecolor="0.35", facecolor="none", linewidth=0.5, zorder=2)
    ax.add_geometries(STATE_GEOMS_USCAN, crs=ccrs.PlateCarree(),
                      edgecolor="0.6", facecolor="none", linewidth=0.4, alpha=0.9, zorder=3)

# =============================================================================
# 9) Helpers
# =============================================================================
def _open_and_standardize(path):
    ds = xr.open_dataset(path)
    ren = {}
    if "lon" in ds.coords: ren["lon"] = "longitude"
    if "lat" in ds.coords: ren["lat"] = "latitude"
    if ren: ds = ds.rename(ren)
    return ds

def _regrid_to_ref_if_needed(ds, lon_ref, lat_ref):
    lon = ds["longitude"].values
    lat = ds["latitude"].values
    if (len(lon) == len(lon_ref) and len(lat) == len(lat_ref) and
        np.allclose(lon, lon_ref) and np.allclose(lat, lat_ref)):
        return ds
    if not _roughly_compatible(lon, lat, lon_ref, lat_ref):
        raise RuntimeError("Grid too different from ref grid; refuse to interp for ensemble.")
    return ds.interp(
        latitude=xr.DataArray(lat_ref, dims=("latitude",)),
        longitude=xr.DataArray(lon_ref, dims=("longitude",)),
        method="nearest"
    )

def _plot_single_ensemble(var_name, change_type, metas):
    key_name = "absolute_change" if change_type == "absolute" else "relative_change"
    label = "Absolute change (days/yr)" if change_type == "absolute" else "Relative change (%)"
    lvl_fn = abs_levels if change_type == "absolute" else rel_levels

    # pick a reference file for grid
    ref = None
    for mdl in MODELS:
        hit = [m for m in metas if (m["year_tag"] == YEAR_TAG and m["scenario"] == SCENARIO and
                                    m["var"] == var_name and m["model_disp"] == mdl)]
        if hit:
            ref = hit[0]; break
    if ref is None:
        print(f"[skip] no ref file for {var_name} {SCENARIO} {YEAR_TAG}")
        return

    ds0 = _open_and_standardize(ref["path"])
    lon_ref = ds0["longitude"].values
    lat_ref = ds0["latitude"].values
    mask_ref = build_combined_mask_on_data_grid(lon_ref, lat_ref)
    LON, LAT = np.meshgrid(lon_ref, lat_ref)
    ds0.close()

    stacks = []
    used_models = []
    for mdl in MODELS:
        hit = [m for m in metas if (m["year_tag"] == YEAR_TAG and m["scenario"] == SCENARIO and
                                    m["var"] == var_name and m["model_disp"] == mdl)]
        if not hit:
            continue
        ds = _open_and_standardize(hit[0]["path"])
        ds = _regrid_to_ref_if_needed(ds, lon_ref, lat_ref)

        if key_name not in ds:
            ds.close()
            continue

        arr = ds[key_name].values.astype(float)
        arr[~mask_ref] = np.nan
        stacks.append(arr)
        used_models.append(mdl)
        ds.close()

    if not stacks:
        print(f"[skip] no data stacks for {var_name} {change_type}")
        return

    ens = np.nanmean(np.stack(stacks, axis=0), axis=0)

    # ---- Plot: same as Part-1 single size
    fig = plt.figure(figsize=(5, 3.75), dpi=300)
    ax = fig.add_subplot(1, 1, 1, projection=PROJ)
    ax.set_extent(MAP_EXTENT, crs=ccrs.PlateCarree())
    add_base(ax)
    if hasattr(ax, "outline_patch"):
        ax.outline_patch.set_visible(False)

    vmin, vmax, levels, cmap_name, extend = lvl_fn(ens)
    cmap = plt.get_cmap(cmap_name); cmap.set_bad(alpha=0)
    norm = BoundaryNorm(levels, cmap.N)

    im = ax.pcolormesh(LON, LAT, ens, transform=ccrs.PlateCarree(),
                       cmap=cmap, norm=norm, shading="auto")

    cbar = plt.colorbar(im, ax=ax, orientation="vertical",
                        pad=0.02, shrink=0.9, extend=extend)
    cbar.set_label(label)
    cbar.set_ticks(levels)
    cbar.set_ticklabels([f"{x:.0f}" for x in levels])

    var_lab = "ABDp" if var_name == "fire_probability" else "OBEp"
    window = year_tag_to_window(YEAR_TAG)
    ax.set_title(
        f"{var_lab} — {label}\n"
        f"Ensemble mean (n={len(used_models)}): {', '.join(used_models)}\n"
        f"{SCENARIO}  ({window} vs 1991–2020)"
    )

    out_stem = f"Ensemble_{SCENARIO}_{YEAR_TAG}_{var_lab}_{change_type}"
    out_png = os.path.join(OUTPUT_DIR, f"{out_stem}.png")
    out_pdf = os.path.join(OUTPUT_DIR, f"{out_stem}.pdf")
    plt.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
    plt.savefig(out_pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    gc.collect()
    print("✓ saved:", os.path.basename(out_png))

# =============================================================================
# 10) MAIN
# =============================================================================
def main():
    nc_files = glob(os.path.join(INPUT_DIR_MAIN, "*.nc"))
    if INPUT_DIR_SUP:
        nc_files += glob(os.path.join(INPUT_DIR_SUP, "*.nc"))

    metas = []
    for p in nc_files:
        m = parse_change_filename(p)
        if not m:
            continue
        if m["year_tag"] != YEAR_TAG:
            continue
        if m["ssp"] != SSP_ONLY:
            continue
        if m["model_disp"] not in MODELS:
            continue
        if m["var"] not in VARS:
            continue
        metas.append(m)

    if not metas:
        print("No files after filters (2070 + SSP370 + ABDp/OBEp + target models).")
        sys.exit(0)

    print("OUTPUT_DIR:", OUTPUT_DIR)

    # 4 figures total
    for v in VARS:
        for ct in CHANGE_TYPES:
            _plot_single_ensemble(v, ct, metas)

if __name__ == "__main__":
    main()























# all sceanrios included
# -*- coding: utf-8 -*-
"""
Ensemble (multi-model mean) change-map panels, using:
  - MAIN maps: E:\Projection paper\analysis\maps  (typically 245/370)
  - SUP  maps: E:\Projection paper_sup_scens\analysis\maps (typically 126/585)

Outputs 4 figures (2x4 panels each):
  1) Late-century absolute change (2070 tag)
  2) Late-century relative change (2070 tag)
  3) Mid-century  absolute change (2040 tag)
  4) Mid-century  relative change (2040 tag)

Layout per figure: 2 rows (ABDp, OBEp) × 4 columns (SSP126/245/370/585)
Each panel is the ensemble mean across available models (CANESM/ECEARTH/GFDL/UKESM).
"""
# -*- coding: utf-8 -*-
import os, re, sys, gc
from glob import glob

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
import cartopy.crs as ccrs
import geopandas as gpd
import regionmask
from cartopy.io import shapereader

# =============================================================================
# 0) PATHS
# =============================================================================
INPUT_DIR_MAIN = r"E:\Projection paper\analysis\maps"
INPUT_DIR_SUP  = r"E:\Projection paper_sup_scens\analysis\maps"

OUTPUT_DIR = os.path.join(INPUT_DIR_MAIN, "figures_panels_4ssp_single_models_45pct")
os.makedirs(OUTPUT_DIR, exist_ok=True)

BIOME_NC  = r"D:\000_collections\020_Chapter2\US_CAN_biome.nc"
USCAN_SHP = r"D:\000_collections\010_Nighttime Burning\011_Data\013_Biome_wwf2017\US_Canada_merged.shp"

# =============================================================================
# 1) GLOBAL STYLE  (max 6, min 5)
# =============================================================================
plt.rcParams.update({
    "font.family": "Arial",
    "font.sans-serif": ["Arial"],
    "axes.unicode_minus": False,

    "font.size": 5,
    "axes.titlesize": 5.5,
    "axes.labelsize": 5,
    "xtick.labelsize": 5,
    "ytick.labelsize": 5,
    "legend.fontsize": 5,
})

# =============================================================================
# 2) MAP SETTINGS
# =============================================================================
PROJ = ccrs.LambertConformal(
    central_longitude=-100, central_latitude=50, standard_parallels=(35, 65)
)
MAP_EXTENT = [-135, -67.5, 24.8, 82]

# =============================================================================
# 3) SELECTORS
# =============================================================================
SELECT_YEAR_TAGS = ("2040", "2070")
SELECT_SSPS      = ("126", "245", "370", "585")
SELECT_MODELS    = ("CANESM", "ECEARTH", "GFDL", "UKESM")
SELECT_VARS      = ("fire_probability", "obe_probability")

# only draw panels (NO single-file plots, NO column plots)
DO_PANEL = True
PANEL_CHANGE_TYPES = ("absolute", "relative")  # output both

# =============================================================================
# 4) FILENAME PARSING
# =============================================================================
# example: fire_probability_days_2070_370_GFDL_change.nc
NAME_RE = re.compile(
    r'^(?P<var>.+?)_days_(?P<year>\d{4})_(?P<ssp>126|245|370|585)_(?P<model>[A-Za-z0-9]+)_change\.nc$',
    re.IGNORECASE
)

def parse_change_filename(path):
    m = NAME_RE.match(os.path.basename(path))
    if not m:
        return None
    meta = m.groupdict()
    meta["scenario"]   = f"SSP{meta['ssp']}"
    meta["model_disp"] = meta["model"].upper()
    meta["year_tag"]   = meta["year"]
    return meta

def year_tag_to_window(tag: str) -> str:
    if tag == "2040":
        return "2041–2070"
    if tag == "2070":
        return "2071–2100"
    return f"future period ({tag})"

# =============================================================================
# 5) GRID COMPATIBILITY + MASKS (same as your original)
# =============================================================================
def _grid_summary(lon, lat):
    return dict(
        lon_min=float(np.nanmin(lon)), lon_max=float(np.nanmax(lon)),
        lat_min=float(np.nanmin(lat)), lat_max=float(np.nanmax(lat)),
        dlon=float(np.nanmedian(np.diff(lon))) if len(lon) > 1 else np.nan,
        dlat=float(np.nanmedian(np.diff(lat))) if len(lat) > 1 else np.nan,
        lon_asc=bool(lon[1] > lon[0]) if len(lon) > 1 else True,
        lat_asc=bool(lat[1] > lat[0]) if len(lat) > 1 else True
    )

def _roughly_compatible(src_lon, src_lat, dst_lon, dst_lat,
                        tol_edge_deg=0.75, tol_step_frac=0.25, tol_extra_rc=2):
    s, d = _grid_summary(src_lon, src_lat), _grid_summary(dst_lon, dst_lat)
    if s["lon_asc"] != d["lon_asc"] or s["lat_asc"] != d["lat_asc"]:
        return False
    if (abs(s["lon_min"] - d["lon_min"]) > tol_edge_deg or
        abs(s["lon_max"] - d["lon_max"]) > tol_edge_deg or
        abs(s["lat_min"] - d["lat_min"]) > tol_edge_deg or
        abs(s["lat_max"] - d["lat_max"]) > tol_edge_deg):
        return False
    if (np.isfinite(s["dlon"]) and np.isfinite(d["dlon"]) and
        abs(s["dlon"] - d["dlon"]) / max(1e-6, abs(d["dlon"])) > tol_step_frac):
        return False
    if (abs(len(src_lat) - len(dst_lat)) > tol_extra_rc or
        abs(len(src_lon) - len(dst_lon)) > tol_extra_rc):
        return False
    return True

EXCLUDE_BIOMES = [50.1, 90.1, 12.1, 13.1, 21.1, 22.1, 31.1, 32.1, 35.1]

_mask_cache = {}

def build_combined_mask_on_data_grid(data_lon, data_lat):
    """
    combined_mask = (US+CAN shapefile region) & (biome exclude mask), on the data grid
    """
    key = (len(data_lon), len(data_lat), float(data_lon[0]), float(data_lon[-1]), float(data_lat[0]), float(data_lat[-1]))
    if key in _mask_cache:
        return _mask_cache[key]

    LON2D, LAT2D = np.meshgrid(data_lon, data_lat)

    shp = gpd.read_file(USCAN_SHP)
    region_mask = ~np.isnan(regionmask.mask_geopandas(shp, LON2D, LAT2D, overlap=False))

    bds = xr.open_dataset(BIOME_NC)
    ren = {}
    if "lon" in bds.coords and "longitude" not in bds.coords: ren["lon"] = "longitude"
    if "lat" in bds.coords and "latitude"  not in bds.coords: ren["lat"] = "latitude"
    if ren: bds = bds.rename(ren)

    bio_var = "gez_code_id" if "gez_code_id" in bds.data_vars else ("gez_code" if "gez_code" in bds.data_vars else None)
    if bio_var is None:
        raise RuntimeError(f"Biome 文件缺少 gez_code(_id)：{BIOME_NC}")

    bio_lon = bds["longitude"].values
    bio_lat = bds["latitude"].values

    if (len(bio_lon) == len(data_lon) and len(bio_lat) == len(data_lat) and
        np.allclose(bio_lon, data_lon) and np.allclose(bio_lat, data_lat)):
        bio_on_data = bds[bio_var]
    else:
        if not _roughly_compatible(bio_lon, bio_lat, data_lon, data_lat):
            raise RuntimeError("生物群系列网格与数据网格差异过大，拒绝自动插值。")
        bio_on_data = bds[bio_var].interp(
            latitude=xr.DataArray(data_lat, dims=("latitude",)),
            longitude=xr.DataArray(data_lon, dims=("longitude",)),
            method="nearest"
        )

    bio_arr = bio_on_data.values
    bio_mask = (~np.isnan(bio_arr)).copy()
    for v in EXCLUDE_BIOMES:
        bio_mask &= (bio_arr != v)

    combined = region_mask & bio_mask
    bds.close()

    _mask_cache[key] = combined
    return combined

# =============================================================================
# 6) LEVELS
# =============================================================================
def abs_levels(masked, pos_pct=90, neg_pct=90, round_to=5):
    arr = masked[np.isfinite(masked)]
    if arr.size == 0:
        lv = np.linspace(-1, 1, 9); return -1, 1, lv, "RdBu_r", "both"
    pos = arr[arr > 0]; neg = arr[arr < 0]
    ppos = np.nanpercentile(pos, pos_pct) if pos.size else 0.0
    pneg = -np.nanpercentile(-neg, neg_pct) if neg.size else 0.0
    vmax = np.ceil(max(abs(ppos), abs(pneg)) / round_to) * round_to or round_to
    vmin = -vmax
    levels = np.linspace(vmin, vmax, 9)
    return vmin, vmax, levels, "RdBu_r", "both"

def rel_levels(masked, pos_pct=90, neg_pct=90, cap=500, round_to=10):
    arr = masked[np.isfinite(masked)]
    if arr.size == 0:
        lv = np.linspace(-10, 10, 9); return -10, 10, lv, "RdBu_r", "both"
    pos = arr[arr > 0]; neg = arr[arr < 0]
    ppos = np.nanpercentile(pos, pos_pct) if pos.size else 0.0
    pneg = -np.nanpercentile(-neg, neg_pct) if neg.size else 0.0
    ppos = min(ppos, cap); pneg = max(pneg, -cap)

    if pos.size and not neg.size:
        vmin, vmax = 0, np.ceil(ppos / round_to) * round_to or round_to
        lv = np.linspace(vmin, vmax, 9); return vmin, vmax, lv, "Reds", "max"
    if neg.size and not pos.size:
        vmin, vmax = np.floor(pneg / round_to) * round_to, 0
        lv = np.linspace(vmin, vmax, 9); return vmin, vmax, lv, "Blues_r", "min"

    vmax = np.ceil(max(abs(ppos), abs(pneg)) / round_to) * round_to or round_to
    vmin = -vmax
    lv = np.linspace(vmin, vmax, 9)
    return vmin, vmax, lv, "RdBu_r", "both"

# =============================================================================
# 7) BASEMAP (US+CAN only)
# =============================================================================
_country_reader_110 = shapereader.Reader(
    shapereader.natural_earth(
        resolution="110m",
        category="cultural",
        name="admin_0_countries"
    )
)
COUNTRY_GEOMS_USCAN = [
    rec.geometry
    for rec in _country_reader_110.records()
    if rec.attributes.get("ADM0_A3") in ("USA", "CAN")
]

_states_reader_50 = shapereader.Reader(
    shapereader.natural_earth(
        resolution="50m",
        category="cultural",
        name="admin_1_states_provinces_lines"
    )
)
STATE_GEOMS_USCAN = []
for rec in _states_reader_50.records():
    attrs_lower = {k.lower(): v for k, v in rec.attributes.items()}
    code = attrs_lower.get("adm0_a3", "")
    if code in ("USA", "CAN"):
        STATE_GEOMS_USCAN.append(rec.geometry)

def add_base(ax):
    ax.add_geometries(
        COUNTRY_GEOMS_USCAN,
        crs=ccrs.PlateCarree(),
        edgecolor="0.35",
        facecolor="none",
        linewidth=0.45,
        zorder=2,
    )
    ax.add_geometries(
        STATE_GEOMS_USCAN,
        crs=ccrs.PlateCarree(),
        edgecolor="0.6",
        facecolor="none",
        linewidth=0.35,
        alpha=0.9,
        zorder=3,
    )

# =============================================================================
# 8) PANEL CORE (single-model, expanded to 4 SSPs, 45% size)
# =============================================================================
def _panel_core_single_models(year_tag: str, change_type: str, meta_all: list,
                              vars_list, ssps, models, out_stem,
                              pos_pct=90, neg_pct=90):
    """
    Output one big panel per (year_tag, change_type):
      rows: models (4)
      cols: (SSP126 ABDp, SSP126 OBEp, ..., SSP585 ABDp, SSP585 OBEp) => 8 columns
    """
    assert change_type in ("relative", "absolute")
    key_name = "relative_change" if change_type == "relative" else "absolute_change"
    label = "Relative change (%)" if change_type == "relative" else "Absolute change (days/yr)"
    if change_type == "relative":
        lvl_fn = lambda arr: rel_levels(arr, pos_pct=pos_pct, neg_pct=neg_pct, cap=500, round_to=10)
    else:
        lvl_fn = lambda arr: abs_levels(arr, pos_pct=pos_pct, neg_pct=neg_pct, round_to=5)

    # filter metas for this year_tag
    metas = [m for m in meta_all if m["year_tag"] == year_tag]
    if not metas:
        print("No metas for year_tag:", year_tag)
        return

    # build index (var, scenario, model) -> path
    index = {}
    for m in metas:
        index[(m["var"], m["scenario"], m["model_disp"])] = m["path"]

    # establish a reference grid/mask using first available file
    first_path = next(iter(index.values()))
    ds0 = xr.open_dataset(first_path)
    ren = {}
    if "lon" in ds0.coords: ren["lon"] = "longitude"
    if "lat" in ds0.coords: ren["lat"] = "latitude"
    if ren: ds0 = ds0.rename(ren)
    lon0 = ds0["longitude"].values
    lat0 = ds0["latitude"].values
    mask0 = build_combined_mask_on_data_grid(lon0, lat0)
    LON0, LAT0 = np.meshgrid(lon0, lat0)
    ds0.close()

    # column order: for each SSP, two columns (ABDp then OBEp)
    col_order = []
    for s in ssps:
        scn = f"SSP{s}"
        col_order.append((vars_list[0], scn))
        col_order.append((vars_list[1], scn))

    nrows = len(models)
    ncols = len(col_order)

    # ---- FIG SIZE: scale 45% ----
    # your previous 4×4 was (8,8). Now columns doubled => width doubles.
    base_w, base_h = 12.0,8.0   # for 4x8 panel
    figsize = (base_w * 0.45, base_h * 0.45)

    fig = plt.figure(figsize=figsize, dpi=300)
    fig.suptitle(f"{label}  {year_tag_to_window(year_tag)} vs 1991–2020", fontsize=6, y=0.98)

    sub = 1
    for mdl in models:
        for (v, scn) in col_order:
            ax = fig.add_subplot(nrows, ncols, sub, projection=PROJ)
            ax.set_extent(MAP_EXTENT, crs=ccrs.PlateCarree())
            add_base(ax)
            if hasattr(ax, "outline_patch"):
                ax.outline_patch.set_visible(False)

            key = (v, scn, mdl.upper())
            if key not in index:
                ax.text(0.5, 0.5, "No Data", transform=ax.transAxes,
                        ha="center", va="center", color="0.6", fontsize=5)
            else:
                ds = xr.open_dataset(index[key])
                ren = {}
                if "lon" in ds.coords: ren["lon"] = "longitude"
                if "lat" in ds.coords: ren["lat"] = "latitude"
                if ren: ds = ds.rename(ren)

                lon = ds["longitude"].values
                lat = ds["latitude"].values

                if (len(lon) != len(lon0) or len(lat) != len(lat0) or
                    (not np.allclose(lon, lon0)) or (not np.allclose(lat, lat0))):
                    # rebuild mask on this grid
                    local_mask = build_combined_mask_on_data_grid(lon, lat)
                    LONi, LATi = np.meshgrid(lon, lat)
                else:
                    local_mask = mask0
                    LONi, LATi = LON0, LAT0

                if key_name not in ds:
                    ax.text(0.5, 0.5, "Missing var", transform=ax.transAxes,
                            ha="center", va="center", color="0.6", fontsize=5)
                else:
                    arr = ds[key_name].values.astype(float)
                    arr[~local_mask] = np.nan

                    vmin, vmax, levels, cmap_name, extend = lvl_fn(arr)
                    cmap = plt.get_cmap(cmap_name)
                    cmap.set_bad(alpha=0)
                    norm = BoundaryNorm(levels, cmap.N)
                    im = ax.pcolormesh(
                        LONi, LATi, arr,
                        transform=ccrs.PlateCarree(),
                        cmap=cmap, norm=norm, shading="auto"
                    )

                    # tiny horizontal cbar per panel (same as you used before)
                    cbar = plt.colorbar(im, ax=ax, orientation="horizontal",
                                        pad=0.02, fraction=0.04, extend=extend)
                    cbar.set_ticks(levels[::2])
                    cbar.set_ticklabels([f"{x:.0f}" for x in levels[::2]])
                    cbar.ax.tick_params(labelsize=5)
                    # cbar.set_label(label, fontsize=5)

                ds.close()

            # top row column titles
            if sub <= ncols:
                var_lab = "ABDp" if v == "fire_probability" else "OBEp"
                ax.set_title(f"{scn} {var_lab}", fontsize=6, pad=1)

            # leftmost col row label (model)
            if (sub - 1) % ncols == 0:
                ax.text(-0.10, 0.5, mdl, transform=ax.transAxes, rotation=90,
                        va="center", ha="center", fontsize=6)

            sub += 1

    plt.subplots_adjust(
        left=0.05, right=0.995,
        top=0.93, bottom=0.06,
        wspace=0.04, hspace=0.12
    )

    win_for_name = year_tag_to_window(year_tag).replace("–", "-").replace(" ", "")
    out_png = os.path.join(OUTPUT_DIR, f"{out_stem}_{change_type}_{year_tag}_{win_for_name}.png")
    out_pdf = os.path.join(OUTPUT_DIR, f"{out_stem}_{change_type}_{year_tag}_{win_for_name}.pdf")
    plt.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
    plt.savefig(out_pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("✓ panel saved:", out_png)

    gc.collect()

# =============================================================================
# 9) MAIN
# =============================================================================
def main():
    # load from BOTH dirs
    nc_files = []
    nc_files += glob(os.path.join(INPUT_DIR_MAIN, "*.nc"))
    nc_files += glob(os.path.join(INPUT_DIR_SUP, "*.nc"))

    if not nc_files:
        print("No NetCDF files in:", INPUT_DIR_MAIN, "and", INPUT_DIR_SUP)
        sys.exit(0)

    metas = []
    for p in nc_files:
        m = parse_change_filename(p)
        if not m:
            continue
        if SELECT_YEAR_TAGS and m["year_tag"] not in SELECT_YEAR_TAGS:
            continue
        if SELECT_SSPS and m["ssp"] not in SELECT_SSPS:
            continue
        if SELECT_MODELS and m["model_disp"] not in SELECT_MODELS:
            continue
        if SELECT_VARS and m["var"] not in SELECT_VARS:
            continue
        m["path"] = p
        metas.append(m)

    if not metas:
        print("No files after filters.")
        sys.exit(0)

    if DO_PANEL:
        vars_list = tuple(SELECT_VARS)
        ssps = tuple(SELECT_SSPS)
        models = tuple(SELECT_MODELS)

        out_stem = f"ABDp_OBEp_SSP{'-'.join(ssps)}_single_models"

        # 4 figures total:
        # late abs, late rel, mid abs, mid rel
        for year_tag in ("2070", "2040"):
            for ct in PANEL_CHANGE_TYPES:
                _panel_core_single_models(
                    year_tag=year_tag,
                    change_type=("relative" if ct == "relative" else "absolute"),
                    meta_all=metas,
                    vars_list=vars_list,
                    ssps=ssps,
                    models=models,
                    out_stem=out_stem,
                    pos_pct=90, neg_pct=90
                )

if __name__ == "__main__":
    main()























# -*- coding: utf-8 -*-
"""
Ensemble 统计（修正版）
- 读取 *_change.nc：<var>_days_<yearTag>_<ssp>_<model>_change.nc
- 掩膜 = (US+CAN 区域) ∩ (biome 排除列表)，biome 若网格轻微不一致 → 最近邻插值对齐
- 统计：按 (Variable, YearTag, Scenario, Model) 出各项；另算 ensemble 和一致性
- 面向 245/370，兼容 126/585
- 网格面积：从实际坐标推断，用球面带状公式（更稳，不写死 0.25°）
"""

import os, re, sys
from glob import glob
import numpy as np
import pandas as pd
import xarray as xr

import geopandas as gpd
import regionmask

# ========= 路径 =========
INPUT_DIR  = r"E:\Projection paper\analysis\maps"  # *_change.nc 在 E 盘
OUTPUT_DIR = r"E:\Projection paper\analysis\stats" # 输出统计在 E 盘
BIOME_NC   = r"D:\000_collections\020_Chapter2\US_CAN_biome.nc"
USCAN_SHP  = r"D:\000_collections\010_Nighttime Burning\011_Data\013_Biome_wwf2017\US_Canada_merged.shp"

os.makedirs(OUTPUT_DIR, exist_ok=True)
for p, name in [(INPUT_DIR,"INPUT_DIR"), (BIOME_NC,"BIOME_NC"), (USCAN_SHP,"USCAN_SHP")]:
    if name=="INPUT_DIR" and not os.path.isdir(p):
        print(f"[ERROR] {name} 不存在: {p}"); sys.exit(1)
    if name!="INPUT_DIR" and not os.path.exists(p):
        print(f"[ERROR] {name} 不存在: {p}"); sys.exit(1)

# ========= 文件名解析 =========
# 例：fire_probability_days_2070_370_GFDL_change.nc
NAME_RE = re.compile(
    r'^(?P<var>.+?)_days_(?P<year>\d{4})_(?P<ssp>126|245|370|585)_(?P<model>[A-Za-z0-9]+)_change\.nc$',
    re.IGNORECASE
)

def parse_name(path):
    m = NAME_RE.match(os.path.basename(path))
    if not m: return None
    g = m.groupdict()
    return {
        "var": g["var"],
        "year_tag": g["year"],               # '2040' / '2070'
        "ssp_code": g["ssp"],                # '245' / '370' / '126' / '585'
        "scenario": f"SSP{g['ssp']}",
        "model": g["model"].upper(),
        "path": path
    }

# ========= “轻微不一致”判断 + 掩膜构建 =========
EXCLUDE_BIOMES = [50.1, 90.1, 12.1, 13.1, 21.1, 22.1, 31.1, 32.1, 35.1]

def _grid_summary(lon, lat):
    return dict(
        lon_min=float(np.nanmin(lon)), lon_max=float(np.nanmax(lon)),
        lat_min=float(np.nanmin(lat)), lat_max=float(np.nanmax(lat)),
        dlon=float(np.nanmedian(np.diff(lon))) if len(lon)>1 else np.nan,
        dlat=float(np.nanmedian(np.diff(lat))) if len(lat)>1 else np.nan,
    )

def _roughly_compatible(src_lon, src_lat, dst_lon, dst_lat,
                        tol_edge_deg=0.75, tol_step_frac=0.25, tol_extra_rc=2):
    s, d = _grid_summary(src_lon, src_lat), _grid_summary(dst_lon, dst_lat)
    if (abs(s["lon_min"]-d["lon_min"])>tol_edge_deg or
        abs(s["lon_max"]-d["lon_max"])>tol_edge_deg or
        abs(s["lat_min"]-d["lat_min"])>tol_edge_deg or
        abs(s["lat_max"]-d["lat_max"])>tol_edge_deg):
        return False
    # 步长允许一定差异
    for a,b in [(s["dlon"],d["dlon"]), (s["dlat"],d["dlat"])]:
        if np.isfinite(a) and np.isfinite(b):
            if abs(a-b)/max(1e-6,abs(b))>tol_step_frac:
                return False
    if abs(len(src_lat)-len(dst_lat))>tol_extra_rc or abs(len(src_lon)-len(dst_lon))>tol_extra_rc:
        return False
    return True

_MASK_CACHE = {}
def _grid_key(lon, lat):
    return (len(lat), len(lon), float(lat[0]), float(lat[-1]), float(lon[0]), float(lon[-1]))

def build_combined_mask_on_grid(data_lon, data_lat):
    """返回 combined_mask（在数据网格上）。biome 允许“轻微不一致”时最近邻到数据网格。"""
    gk = _grid_key(data_lon, data_lat)
    if gk in _MASK_CACHE:
        return _MASK_CACHE[gk]

    LON2D, LAT2D = np.meshgrid(data_lon, data_lat)
    # 区域掩膜
    shp = gpd.read_file(USCAN_SHP)
    region = ~np.isnan(regionmask.mask_geopandas(shp, LON2D, LAT2D, overlap=False))

    # biome 掩膜
    bds = xr.open_dataset(BIOME_NC)
    ren = {}
    if "lon" in bds.coords and "longitude" not in bds.coords: ren["lon"]="longitude"
    if "lat" in bds.coords and "latitude"  not in bds.coords: ren["lat"]="latitude"
    if ren: bds = bds.rename(ren)

    bio_var = "gez_code_id" if "gez_code_id" in bds.data_vars else ("gez_code" if "gez_code" in bds.data_vars else None)
    if bio_var is None:
        raise RuntimeError(f"Biome 文件缺少 gez_code(_id)：{BIOME_NC}")

    bio_lon = bds["longitude"].values
    bio_lat = bds["latitude"].values

    if (len(bio_lon)==len(data_lon) and len(bio_lat)==len(data_lat) and
        np.allclose(bio_lon, data_lon) and np.allclose(bio_lat, data_lat)):
        bio_on = bds[bio_var]
    else:
        if not _roughly_compatible(bio_lon, bio_lat, data_lon, data_lat):
            raise RuntimeError("生物群系列网格与数据网格差异过大，拒绝自动插值。")
        bio_on = bds[bio_var].interp(
            latitude=xr.DataArray(data_lat, dims=("latitude",)),
            longitude=xr.DataArray(data_lon, dims=("longitude",)),
            method="nearest"
        )

    bio_arr = bio_on.values
    bio_mask = (~np.isnan(bio_arr)).copy()
    for v in EXCLUDE_BIOMES:
        bio_mask &= (bio_arr != v)

    combined = region & bio_mask
    _MASK_CACHE[gk] = combined
    return combined

# ========= 网格面积（从坐标推断；球面带状面积） =========
def gridcell_area_km2(lat, lon):
    """
    lat/lon 为中心格点一维坐标（规则网格）。用边界计算：
    A = R^2 * (lon_e - lon_w) * (sin(lat_n) - sin(lat_s))
    """
    R = 6371.0  # km
    lat = np.asarray(lat); lon = np.asarray(lon)
    # 步长
    dlat = np.median(np.diff(lat)) if len(lat)>1 else 0.25
    dlon = np.median(np.diff(lon)) if len(lon)>1 else 0.25
    # 边界（中心±半格）
    lat_edges = np.concatenate(([lat[0]-dlat/2], (lat[:-1]+lat[1:])/2, [lat[-1]+dlat/2]))
    lon_edges = np.concatenate(([lon[0]-dlon/2], (lon[:-1]+lon[1:])/2, [lon[-1]+dlon/2]))
    # 弧度
    lat_e_rad = np.deg2rad(lat_edges)
    lon_e_rad = np.deg2rad(lon_edges)
    # 网格面积
    d_sin = np.abs(np.sin(lat_e_rad[1:]) - np.sin(lat_e_rad[:-1]))[:, None]
    d_lon = np.abs(lon_e_rad[1:] - lon_e_rad[:-1])[None, :]
    area = (R**2) * d_lon * d_sin  # (nlat, nlon)
    return area

# ========= 主过程 =========
def main():
    files = glob(os.path.join(INPUT_DIR, "*.nc"))
    if not files:
        print(f"[ERROR] 目录无 nc：{INPUT_DIR}"); sys.exit(1)

    metas = []
    bad = 0
    for p in files:
        m = parse_name(p)
        if m: metas.append(m)
        else: bad += 1
    if bad:
        print(f"[INFO] 跳过无法解析文件名：{bad} 个")
    if not metas:
        print("[ERROR] 没有可用的 *_change.nc"); sys.exit(1)

    # 用第一份文件确定掩膜和面积
    ds0 = xr.open_dataset(metas[0]["path"])
    ren = {}
    if "lon" in ds0.coords: ren["lon"] = "longitude"
    if "lat" in ds0.coords: ren["lat"] = "latitude"
    if ren: ds0 = ds0.rename(ren)
    lat0 = ds0["latitude"].values; lon0 = ds0["longitude"].values
    combined0 = build_combined_mask_on_grid(lon0, lat0)
    area0 = gridcell_area_km2(lat0, lon0)
    ds0.close()

 # ===== 统计循环（✅ 合并为一个循环；同时填充 rows / bucket / bucket_area）=====
    rows = []
    bucket = {}        # (var, year_tag, scenario) -> list of rel_masked arrays
    bucket_area = {}   # (var, year_tag, scenario) -> area array（用于一致性面积）
    
    for m in metas:
        ds = xr.open_dataset(m["path"])
        ren = {}
        if "lon" in ds.coords: ren["lon"] = "longitude"
        if "lat" in ds.coords: ren["lat"] = "latitude"
        if ren: ds = ds.rename(ren)
    
        lat = ds["latitude"].values
        lon = ds["longitude"].values
    
        # 若网格与基准不同，按该网格重建掩膜与面积
        if (len(lat)!=len(lat0) or len(lon)!=len(lon0) or
            not np.allclose(lat, lat0) or not np.allclose(lon, lon0)):
            combined = build_combined_mask_on_grid(lon, lat)
            area     = gridcell_area_km2(lat, lon)
        else:
            combined = combined0
            area     = area0
    
        # 读变化变量
        if ("relative_change" not in ds) or ("absolute_change" not in ds):
            print(f"[WARN] 变量缺失，跳过：{os.path.basename(m['path'])}")
            ds.close()
            continue
        rel = ds["relative_change"].values.astype(float)
        abs_ = ds["absolute_change"].values.astype(float)
        ds.close()
    
        if rel.shape != combined.shape:
            print(f"[WARN] 尺寸不匹配，跳过：{os.path.basename(m['path'])}")
            continue
    
        # 掩膜
        rel_masked = rel.copy(); rel_masked[~combined] = np.nan
        abs_masked = abs_.copy();  abs_masked[~combined] = np.nan
    
        # 单模型统计
        valid      = np.isfinite(rel_masked)
        n_valid    = int(valid.sum())
        total_area = float(np.nansum(area[valid])) if n_valid>0 else 0.0
        mean_rel   = float(np.nanmean(rel_masked))
        mean_abs   = float(np.nanmean(abs_masked))
        pos_mask   = (rel_masked > 0) & valid
        pos_ratio  = float(pos_mask.sum()/n_valid) if n_valid>0 else 0.0
        pos_grids  = int(pos_mask.sum())
    
        rows.append({
            "Variable": m["var"],
            "YearTag":  m["year_tag"],                # 2040 / 2070
            "Scenario": m["scenario"],                # 'SSP245' / ...
            "SSP":      m["ssp_code"],                # '245' / ...
            "Model":    m["model"],
            "Valid_Grids": n_valid,
            "Total_Area_km2": total_area,
            "Mean_Relative_Change_%": mean_rel,
            "Mean_Absolute_Change_days": mean_abs,
            "Positive_Change_Ratio": pos_ratio,
            "Positive_Grids": pos_grids
        })
    
        # 供“一致性”部分使用的堆叠材料
        k = (m["var"], m["year_tag"], m["scenario"])
        bucket.setdefault(k, []).append(rel_masked)
    
        # 记录该组合的面积栅格（首次记录；后续若发现不一致就置 None 跳过面积占比）
        if k not in bucket_area:
            bucket_area[k] = area
        else:
            if isinstance(bucket_area[k], np.ndarray) and bucket_area[k].shape != area.shape:
                print(f"[WARN] {k} 各模型网格不一致，跳过一致性面积统计。")
                bucket_area[k] = None


    # ===== 个体模型表 =====
    df_ind = pd.DataFrame(rows).sort_values(["Variable","YearTag","Scenario","Model"])
    ind_csv = os.path.join(OUTPUT_DIR, "individual_model_stats.csv")
    df_ind.to_csv(ind_csv, index=False, encoding="utf-8-sig")
    print("[SAVE]", ind_csv)

    # ===== Ensemble 汇总 =====
    if df_ind.empty:
        print("[WARN] 无记录，结束。"); return

    ensemble = (df_ind
        .groupby(["Variable","YearTag","Scenario"], as_index=False)
        .agg(Mean_Relative_Change_mean = ("Mean_Relative_Change_%","mean"),
             Mean_Relative_Change_std  = ("Mean_Relative_Change_%","std"),
             Mean_Relative_Change_min  = ("Mean_Relative_Change_%","min"),
             Mean_Relative_Change_max  = ("Mean_Relative_Change_%","max"),
             Mean_Absolute_Change_mean = ("Mean_Absolute_Change_days","mean"),
             Mean_Absolute_Change_std  = ("Mean_Absolute_Change_days","std"),
             Mean_Absolute_Change_min  = ("Mean_Absolute_Change_days","min"),
             Mean_Absolute_Change_max  = ("Mean_Absolute_Change_days","max"),
             Positive_Change_Ratio_mean= ("Positive_Change_Ratio","mean"),
             Positive_Change_Ratio_std = ("Positive_Change_Ratio","std"))
        .round(3)
    )
    ens_csv = os.path.join(OUTPUT_DIR, "ensemble_stats.csv")
    ensemble.to_csv(ens_csv, index=False, encoding="utf-8-sig")
    print("[SAVE]", ens_csv)

    # ===== 一致性分析（面向“相对变化 > 0”）=====
    # ===== 一致性分析（面向“相对变化 > 0”）含面积 =====
    cons_rows = []
    for (var, year_tag, scenario), stack_list in bucket.items():
        if not stack_list:
            continue
        # 若该组合没有可靠的面积栅格，跳过面积占比（仍可给出网格占比）
        area = bucket_area.get((var, year_tag, scenario), None)
    
        # stack
        try:
            arr = np.stack(stack_list, axis=0)  # (n_model, nlat, nlon)
        except Exception as e:
            print(f"[WARN] 不能 stack {var}-{year_tag}-{scenario}: {e}")
            continue
    
        valid = np.all(np.isfinite(arr), axis=0)
        if not np.any(valid):
            continue
    
        pos_count = np.sum(arr > 0, axis=0)
        n_models  = arr.shape[0]
        pc_valid  = pos_count[valid]
        total_valid = pc_valid.size
    
        # 面积基准
        if area is not None and isinstance(area, np.ndarray) and area.shape == valid.shape:
            total_area_valid = float(np.nansum(area[valid]))
        else:
            total_area_valid = None  # 无面积
    
        def _emit(level_name, mask_logic):
            mask = mask_logic(pc_valid, n_models)
            n = int(mask.sum())
            row = {
                "Variable": var,
                "YearTag": year_tag,
                "Scenario": scenario,
                "Consistency_Level": level_name,
                "N_Models": n_models,
                "N_Grids": n,
                "Grid_Ratio": float(n/total_valid) if total_valid>0 else np.nan,
                "Total_Valid_Grids": int(total_valid),
            }
            # 面积
            if total_area_valid is not None:
                # 把 pc_valid 的布尔掩膜还原回全场：valid & condition
                full_mask = np.zeros_like(valid, dtype=bool)
                full_mask[valid] = mask
                area_km2 = float(np.nansum(area[full_mask]))
                row.update({
                    "Total_Area_km2": total_area_valid,
                    "Area_km2": area_km2,
                    "Area_Ratio": (area_km2/total_area_valid) if total_area_valid>0 else np.nan
                })
            cons_rows.append(row)
    
        _emit("Unanimous", lambda x, m: x ==  m)
        _emit(">=3/4",     lambda x, m: x >= int(np.ceil(0.75*m)))
        _emit(">=1/2",     lambda x, m: x >= int(np.ceil(0.50*m)))
        _emit("Any",       lambda x, m: x >  0)
    
    df_cons = pd.DataFrame(cons_rows)
    cons_csv = os.path.join(OUTPUT_DIR, "model_consistency.csv")
    df_cons.to_csv(cons_csv, index=False, encoding="utf-8-sig")
    print("[SAVE]", cons_csv)
    
    # 透视保持不变，可选把 Area_Ratio 也透视一份


    # 透视（便于快速浏览）
    if not df_cons.empty:
        df_piv = (df_cons
                  .pivot_table(index=["Variable","YearTag","Scenario"],
                               columns="Consistency_Level",
                               values="Grid_Ratio",
                               aggfunc="first")
                  .round(3)
                 )
        piv_csv = os.path.join(OUTPUT_DIR, "model_consistency_pivot.csv")
        df_piv.to_csv(piv_csv, encoding="utf-8-sig")
        print("[SAVE]", piv_csv)

    # ===== 场景对比（举例：SSP245 vs SSP585；若不存在则跳过）=====
    for var in df_cons["Variable"].unique() if not df_cons.empty else []:
        for year_tag in df_cons["YearTag"].unique():
            q = df_cons[(df_cons.Variable==var) & (df_cons.YearTag==year_tag) & (df_cons.Consistency_Level=="Unanimous")]
            if q.empty: continue
            a = q[q.Scenario=="SSP245"]["Grid_Ratio"].values
            b = q[q.Scenario=="SSP585"]["Grid_Ratio"].values
            if a.size and b.size and a[0]>0:
                ratio = b[0]/a[0]
                print(f"[COMPARE] {var} {year_tag}: SSP585 unanimous grid-ratio is {ratio:.2f}× SSP245")

    print("\nDone. All outputs in:", OUTPUT_DIR)

if __name__ == "__main__":
    main()



















# -*- coding: utf-8 -*-
import os
import re
import numpy as np
import pandas as pd
import xarray as xr

import geopandas as gpd
import regionmask

# ========= 路径 =========
CSV_PATH      = r"E:\Projection paper\analysis\tables\annual_exceed_days.csv"  # 新流水线长表
BIOME_NC      = r"D:\000_collections\020_Chapter2\US_CAN_biome.nc"             # 带小数 ID 的 biome
USCAN_SHP     = r"D:\000_collections\010_Nighttime Burning\011_Data\013_Biome_wwf2017\US_Canada_merged.shp"
OUTPUT_DIR    = r"E:\Projection paper\analysis\stats"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ========= 配置 =========
VARS_KEEP     = ("fire_probability", "obe_probability")  # 只保留 ABDp / OBEp
SCENARIOS_WANT= ("SSP245","SSP370")                      # 可扩展到 'SSP126','SSP585'
EXCLUDE_BIOMES= [50.1, 90.1, 12.1, 13.1, 21.1, 22.1, 31.1, 32.1, 35.1]  # 与绘图脚本一致

def year_window_from_yeartag(tag: str):
    """yearTag -> (start_year, end_year)"""
    if tag == "2040":
        return 2041, 2070
    if tag == "2070":
        return 2071, 2100
    # 兜底：不识别就返回 None
    return None

def parse_future_scenario(s: str):
    """
    CSV 中 future 的 scenario 形如 '2040_245' / '2070_370'
    返回 (SSP245/SSP370/..., (start,end))
    """
    m = re.match(r'^(?P<yr>\d{4})_(?P<ssp>\d{3})$', str(s))
    if not m:
        return None, None
    yr = m.group("yr")
    ssp = f"SSP{m.group('ssp')}"
    win = year_window_from_yeartag(yr)
    return ssp, win

def build_biome_grid_counts_with_region():
    """
    计算“US+CAN ∩ (biome not in EXCLUDE)”条件下，每个 biome_id 的有效格点数量（在 biome 自身网格上）
    这与绘图脚本掩膜口径一致（区别：这里不需要插值到数据网格）
    """
    ds = xr.open_dataset(BIOME_NC)
    ren = {}
    if 'lon' in ds.coords and 'longitude' not in ds.coords: ren['lon'] = 'longitude'
    if 'lat' in ds.coords and 'latitude'  not in ds.coords: ren['lat'] = 'latitude'
    if ren: ds = ds.rename(ren)
    if 'gez_code_id' in ds.data_vars:
        bio = ds['gez_code_id']
    elif 'gez_code' in ds.data_vars:
        bio = ds['gez_code']
    else:
        raise RuntimeError("Biome NC 缺少 gez_code(_id)")

    lon = ds['longitude'].values
    lat = ds['latitude'].values
    LON2D, LAT2D = np.meshgrid(lon, lat)

    shp = gpd.read_file(USCAN_SHP)
    region = ~np.isnan(regionmask.mask_geopandas(shp, LON2D, LAT2D, overlap=False))

    bio_arr = bio.values
    valid_bio = (~np.isnan(bio_arr)).copy()
    for v in EXCLUDE_BIOMES:
        valid_bio &= (bio_arr != v)

    combined = region & valid_bio

    uniq = np.unique(bio_arr[combined])
    counts = {}
    for bid in uniq:
        counts[float(bid)] = int(np.sum((bio_arr == bid) & combined))
    ds.close()
    return counts

def load_and_filter_csv():
    """
    读取新 CSV（annual_exceed_days.csv），筛选变量与字段，拆分 baseline/future
    """
    df = pd.read_csv(CSV_PATH)
    # 只保留我们关心的变量
    df = df[df['variable'].isin(VARS_KEEP)].copy()

    # baseline 与 future 拆分
    df_base = df[df['epoch'] == 'baseline'].copy()
    df_fut  = df[df['epoch'] == 'future'].copy()

    # baseline 年份（一般 1991–2020）
    base_years = sorted(df_base['year'].unique())

    # future：解析 scenario → (SSPxxx, 年窗)
    df_fut['SSP'] = None
    df_fut['win_start'] = None
    df_fut['win_end'] = None
    parsed = df_fut['scenario'].apply(parse_future_scenario)
    df_fut['SSP'] = parsed.apply(lambda x: x[0])
    df_fut['win'] = parsed.apply(lambda x: x[1])
    df_fut['win_start'] = df_fut['win'].apply(lambda w: w[0] if isinstance(w, tuple) else None)
    df_fut['win_end']   = df_fut['win'].apply(lambda w: w[1] if isinstance(w, tuple) else None)

    # 只保留识别成功且在关心 SSP 里的记录
    df_fut = df_fut[df_fut['SSP'].isin(SCENARIOS_WANT)].copy()

    return df_base, df_fut, base_years

def calc_30yr_total_overall(df_sub, years, biome_counts):
    """
    给定（某一时期）的子表：按 variable，把所有保留 biome 的
    “年均 exceed_days × 该 biome 有效格点数 × 年数(=30)” 求和
    """
    # 过滤到指定年
    df_sub = df_sub[df_sub['year'].isin(years)].copy()

    results = {}
    for var in df_sub['variable'].unique():
        dfv = df_sub[df_sub['variable']==var]
        total = 0.0
        for bid, cnt in biome_counts.items():
            x = dfv[dfv['biome_id']==bid]['exceed_days']
            if x.empty:
                continue
            annual_mean = x.mean()           # 这个 biome 的“每网格每年的平均 exceed-days”
            total += float(annual_mean) * cnt * 30.0
        results[var] = total
    return results

def calc_30yr_total_single_biome(df_sub, years, biome_id, biome_counts):
    df_sub = df_sub[(df_sub['year'].isin(years)) & (df_sub['biome_id']==biome_id)].copy()
    results = {}
    cnt = biome_counts.get(biome_id, 0)
    for var in df_sub['variable'].unique():
        dfv = df_sub[df_sub['variable']==var]
        if dfv.empty or cnt==0:
            results[var] = 0.0
            continue
        annual_mean = dfv['exceed_days'].mean()
        results[var] = float(annual_mean) * cnt * 30.0
    return results

def overall_pipeline():
    print("=== 简单火灾变化分析（新口径） ===")
    print("[1] 读取 CSV 并按口径筛选…")
    df_base, df_fut, base_years = load_and_filter_csv()
    if df_base.empty or df_fut.empty:
        raise RuntimeError("baseline 或 future 子表为空，请检查 CSV 路径/列名/过滤条件。")

    print("[2] 构建 US+CAN ∩ biome 掩膜的格点计数（与绘图口径一致）…")
    biome_counts = build_biome_grid_counts_with_region()
    total_grids  = int(sum(biome_counts.values()))
    print(f"    有效总格点: {total_grids:,}")

    print("[3] baseline 30 年总量（合并所有保留 biome）…")
    base_totals_overall = calc_30yr_total_overall(df_base, base_years, biome_counts)

    # baseline 每 grid 每年（用于换算 d/grid/yr）
    base_per_grid_per_year = {
        var: (base_totals_overall[var] / 30.0) / total_grids if total_grids>0 else np.nan
        for var in base_totals_overall
    }

    print("[4] future：按 (SSP, 年窗) + 模型 计算 30 年总量，并做变化（整体 & 各 biome）…")
    records_overall = []
    records_biome  = []

    # —— 先整体（合并所有 biome）——
    for ssp in sorted(df_fut['SSP'].dropna().unique()):
        if ssp not in SCENARIOS_WANT: 
            continue
        # 同一个 SSP 下，可能有两个 yearTag（2040/2070）
        for yrwin in sorted(df_fut[df_fut['SSP']==ssp]['win'].dropna().unique()):
            y1, y2 = yrwin
            # 各模型
            for mdl in sorted(df_fut[df_fut['SSP']==ssp]['model'].dropna().unique()):
                df_sub = df_fut[(df_fut['SSP']==ssp) & (df_fut['model']==mdl) &
                                (df_fut['year']>=y1) & (df_fut['year']<=y2)]
                if df_sub.empty: 
                    continue
                fut_tot = calc_30yr_total_overall(df_sub, range(y1, y2+1), biome_counts)
                for var, fval in fut_tot.items():
                    bval = base_totals_overall.get(var, 0.0)
                    abschg = fval - bval
                    relchg = (abschg / bval * 100.0) if bval>0 else np.nan
                    base_dpy = base_per_grid_per_year.get(var, np.nan)
                    fut_dpy  = (fval/30.0)/total_grids if total_grids>0 else np.nan
                    chg_dpy  = fut_dpy - base_dpy
                    records_overall.append({
                        "Variable": var,
                        "SSP": ssp,
                        "YearStart": y1, "YearEnd": y2,
                        "Scenario": f"{ssp}",
                        "Model": mdl,
                        "Baseline_30yr_total": bval,
                        "Future_30yr_total":   fval,
                        "Absolute_change":     abschg,
                        "Relative_change_%":   relchg,
                        "Baseline_days_per_grid_per_year": base_dpy,
                        "Future_days_per_grid_per_year":   fut_dpy,
                        "Change_days_per_grid_per_year":   chg_dpy,
                        "Total_grids": total_grids,
                        "Type": "Individual_Model"
                    })

            # —— Ensemble（跨模型平均）——
            for var in VARS_KEEP:
                # 收集该 SSP+窗 下所有模型的 future 总量
                vals = []
                for mdl in sorted(df_fut[df_fut['SSP']==ssp]['model'].dropna().unique()):
                    df_sub = df_fut[(df_fut['SSP']==ssp) & (df_fut['model']==mdl) &
                                    (df_fut['year']>=y1) & (df_fut['year']<=y2) &
                                    (df_fut['variable']==var)]
                    if df_sub.empty: 
                        continue
                    fut_tot = calc_30yr_total_overall(df_sub, range(y1, y2+1), biome_counts)
                    vals.append(fut_tot.get(var, 0.0))
                if not vals:
                    continue
                fmean = float(np.mean(vals))
                bval  = base_totals_overall.get(var, 0.0)
                abschg = fmean - bval
                relchg = (abschg / bval * 100.0) if bval>0 else np.nan
                base_dpy = base_per_grid_per_year.get(var, np.nan)
                fut_dpy  = (fmean/30.0)/total_grids if total_grids>0 else np.nan
                chg_dpy  = fut_dpy - base_dpy
                records_overall.append({
                    "Variable": var,
                    "SSP": ssp,
                    "YearStart": y1, "YearEnd": y2,
                    "Scenario": f"{ssp}",
                    "Model": "Ensemble_Mean",
                    "Baseline_30yr_total": bval,
                    "Future_30yr_total":   fmean,
                    "Absolute_change":     abschg,
                    "Relative_change_%":   relchg,
                    "Baseline_days_per_grid_per_year": base_dpy,
                    "Future_days_per_grid_per_year":   fut_dpy,
                    "Change_days_per_grid_per_year":   chg_dpy,
                    "Total_grids": total_grids,
                    "Type": "Ensemble_Mean",
                    "N_Models": len(vals)
                })

    # —— 各 biome ——（只做常见窗口/SSP；如不需要可关掉这段）
    biome_counts_sorted = dict(sorted(biome_counts.items()))
    for ssp in sorted(df_fut['SSP'].dropna().unique()):
        if ssp not in SCENARIOS_WANT: 
            continue
        for yrwin in sorted(df_fut[df_fut['SSP']==ssp]['win'].dropna().unique()):
            y1, y2 = yrwin
            for biome_id, cnt in biome_counts_sorted.items():
                # baseline
                base_tot_b = calc_30yr_total_single_biome(df_base, base_years, biome_id, biome_counts)
                # 各模型
                for mdl in sorted(df_fut[df_fut['SSP']==ssp]['model'].dropna().unique()):
                    df_sub = df_fut[(df_fut['SSP']==ssp) & (df_fut['model']==mdl) &
                                    (df_fut['year']>=y1) & (df_fut['year']<=y2)]
                    if df_sub.empty:
                        continue
                    fut_tot_b = calc_30yr_total_single_biome(df_sub, range(y1, y2+1), biome_id, biome_counts)
                    for var in VARS_KEEP:
                        bval = base_tot_b.get(var, 0.0)
                        fval = fut_tot_b.get(var, 0.0)
                        abschg = fval - bval
                        relchg = (abschg / bval * 100.0) if bval>0 else np.nan
                        base_dpy = (bval/30.0)/cnt if cnt>0 else np.nan
                        fut_dpy  = (fval/30.0)/cnt if cnt>0 else np.nan
                        chg_dpy  = fut_dpy - base_dpy
                        records_biome.append({
                            "Biome_ID": biome_id,
                            "Variable": var,
                            "SSP": ssp,
                            "YearStart": y1, "YearEnd": y2,
                            "Scenario": f"{ssp}",
                            "Model": mdl,
                            "Baseline_30yr_total": bval,
                            "Future_30yr_total":   fval,
                            "Absolute_change":     abschg,
                            "Relative_change_%":   relchg,
                            "Baseline_days_per_grid_per_year": base_dpy,
                            "Future_days_per_grid_per_year":   fut_dpy,
                            "Change_days_per_grid_per_year":   chg_dpy,
                            "Biome_grids": cnt,
                            "Type": "Individual_Model"
                        })
                # ensemble（该 biome）
                for var in VARS_KEEP:
                    vals = []
                    for mdl in sorted(df_fut[df_fut['SSP']==ssp]['model'].dropna().unique()):
                        df_sub = df_fut[(df_fut['SSP']==ssp) & (df_fut['model']==mdl) &
                                        (df_fut['year']>=y1) & (df_fut['year']<=y2) &
                                        (df_fut['variable']==var)]
                        if df_sub.empty:
                            continue
                        fut_tot_b = calc_30yr_total_single_biome(df_sub, range(y1, y2+1), biome_id, biome_counts)
                        vals.append(fut_tot_b.get(var, 0.0))
                    if not vals:
                        continue
                    fmean = float(np.mean(vals))
                    bval  = base_tot_b.get(var, 0.0)
                    abschg = fmean - bval
                    relchg = (abschg / bval * 100.0) if bval>0 else np.nan
                    base_dpy = (bval/30.0)/cnt if cnt>0 else np.nan
                    fut_dpy  = (fmean/30.0)/cnt if cnt>0 else np.nan
                    chg_dpy  = fut_dpy - base_dpy
                    records_biome.append({
                        "Biome_ID": biome_id,
                        "Variable": var,
                        "SSP": ssp,
                        "YearStart": y1, "YearEnd": y2,
                        "Scenario": f"{ssp}",
                        "Model": "Ensemble_Mean",
                        "Baseline_30yr_total": bval,
                        "Future_30yr_total":   fmean,
                        "Absolute_change":     abschg,
                        "Relative_change_%":   relchg,
                        "Baseline_days_per_grid_per_year": base_dpy,
                        "Future_days_per_grid_per_year":   fut_dpy,
                        "Change_days_per_grid_per_year":   chg_dpy,
                        "Biome_grids": cnt,
                        "Type": "Ensemble_Mean",
                        "N_Models": len(vals)
                    })

    # —— 写出 —— 
    df_overall = pd.DataFrame(records_overall)
    df_biome   = pd.DataFrame(records_biome)
    out1 = os.path.join(OUTPUT_DIR, "overall_fire_change_analysis.csv")
    out2 = os.path.join(OUTPUT_DIR, "individual_biome_fire_change_analysis.csv")
    df_overall.to_csv(out1, index=False, encoding="utf-8-sig")
    df_biome.to_csv(out2,   index=False, encoding="utf-8-sig")
    print("[SAVE]", out1)
    print("[SAVE]", out2)

    # —— 简短摘要（文章可用句）——
    def quick_line(ssp, var):
        dd = df_overall[(df_overall['Model']=='Ensemble_Mean') &
                        (df_overall['Scenario']==ssp) &
                        (df_overall['Variable']==var)]
        if dd.empty: 
            return None
        # 若有两个年窗，挑 2071–2100；否则取单个
        dd = dd.sort_values(['YearEnd']).iloc[-1]
        return (f"{var} {ssp} {int(dd['YearStart'])}-{int(dd['YearEnd'])}: "
                f"{dd['Relative_change_%']:+.1f}% "
                f"({dd['Change_days_per_grid_per_year']:+.1f} d/grid/yr)")

    print("\n=== Ensemble 关键数字（整体，优先 2071–2100）===")
    for ssp in SCENARIOS_WANT:
        for var in VARS_KEEP:
            line = quick_line(ssp, var)
            if line: print(" -", line)

    return df_overall, df_biome

if __name__ == "__main__":
    overall_pipeline()



#%% 这个是可选的部分
# -*- coding: utf-8 -*-
"""
Consistency maps & summaries (aligned to the new pipeline)
- Reads: <var>_days_<yearTag>_<ssp>_<model>_change.nc from E:\Projection paper\analysis\maps
- Variables: fire_probability (ABDp), obe_probability (OBEp)
- Scenarios: support 126/245/370/585, default comparisons show 245 vs 370
- Mask: (US+CAN) ∩ (exclude biomes); if biome grid ~ data grid "slightly different", NN-align to data grid
- Outputs: categorical consistency maps, continuous agreement maps, SSP comparison (245 vs 370),
           magnitude × consistency overlays, biome-group summaries (csv+bar), latitude-band gradients
"""

import os, re, gc
from glob import glob

import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.gridspec import GridSpec
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geopandas as gpd
import regionmask

# ---------- Paths ----------
INPUT_DIR  = r"E:\Projection paper\analysis\maps"
OUTPUT_DIR = r"E:\Projection paper\analysis\figures\consistency"
BIOME_NC   = r"D:\000_collections\020_Chapter2\US_CAN_biome.nc"
USCAN_SHP  = r"D:\000_collections\010_Nighttime Burning\011_Data\013_Biome_wwf2017\US_Canada_merged.shp"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------- Look & map ----------
plt.rcParams['font.family'] = 'Arial'
plt.rcParams.update({'font.size': 8})

MAP_EXTENT = [-170, -50, 25, 85]
PROJ = ccrs.LambertConformal(central_longitude=-95, central_latitude=49,
                             standard_parallels=(49, 77))

# ---------- Filename parser (new) ----------
# e.g. fire_probability_days_2070_370_GFDL_change.nc
NAME_RE = re.compile(
    r'^(?P<var>.+?)_days_(?P<year>\d{4})_(?P<ssp>126|245|370|585)_(?P<model>[A-Za-z0-9\-]+)_change\.nc$',
    re.IGNORECASE
)

def parse_change_filename(path):
    m = NAME_RE.match(os.path.basename(path))
    if not m:
        return None
    g = m.groupdict()
    var = g['var'].lower()
    # accept legacy aliases
    if var in ('abd_exceed','abd','active_burning_days'): var = 'fire_probability'
    if var in ('obe_exceed','obe','overnight_burning_events'): var = 'obe_probability'
    return {
        'var': var,                         # fire_probability / obe_probability
        'year_tag': g['year'],              # '2040' / '2070'
        'ssp': g['ssp'],                    # '245' / '370' / '126' / '585'
        'scenario': f"SSP{g['ssp']}",
        'model': g['model'].upper(),
        'path': path
    }

# ---------- Biome sets ----------
BOREAL_BIOMES       = [41.1, 41.2, 42.1, 42.2, 43.1]
SUBTROPICAL_BIOMES  = [23.1, 24.1, 25.1]
TEMPERATE_BIOMES    = [33.1, 34.1, 35.2]
BIOME_GROUPS = {
    'Boreal': BOREAL_BIOMES,
    'Subtropical/Temperate': SUBTROPICAL_BIOMES + TEMPERATE_BIOMES
}
EXCLUDE_BIOMES = [50.1, 90.1, 12.1, 13.1, 21.1, 22.1, 31.1, 32.1, 35.1]

# ---------- Mask utilities ----------
def _rename_lonlat(ds):
    ren = {}
    if 'lon' in ds.coords and 'longitude' not in ds.coords: ren['lon'] = 'longitude'
    if 'lat' in ds.coords and 'latitude'  not in ds.coords: ren['lat'] = 'latitude'
    if ren: ds = ds.rename(ren)
    return ds

def _grid_summary(lon, lat):
    return dict(
        lon_min=float(np.nanmin(lon)), lon_max=float(np.nanmax(lon)),
        lat_min=float(np.nanmin(lat)), lat_max=float(np.nanmax(lat)),
        dlon=float(np.nanmedian(np.diff(lon))) if len(lon)>1 else np.nan,
        dlat=float(np.nanmedian(np.diff(lat))) if len(lat)>1 else np.nan,
    )

def _roughly_compatible(src_lon, src_lat, dst_lon, dst_lat,
                        tol_edge_deg=0.75, tol_step_frac=0.25, tol_extra_rc=2):
    s, d = _grid_summary(src_lon, src_lat), _grid_summary(dst_lon, dst_lat)
    if (abs(s["lon_min"]-d["lon_min"])>tol_edge_deg or
        abs(s["lon_max"]-d["lon_max"])>tol_edge_deg or
        abs(s["lat_min"]-d["lat_min"])>tol_edge_deg or
        abs(s["lat_max"]-d["lat_max"])>tol_edge_deg):
        return False
    for a,b in [(s["dlon"],d["dlon"]), (s["dlat"],d["dlat"])]:
        if np.isfinite(a) and np.isfinite(b):
            if abs(a-b)/max(1e-6,abs(b))>tol_step_frac:
                return False
    if abs(len(src_lat)-len(dst_lat))>tol_extra_rc or abs(len(src_lon)-len(dst_lon))>tol_extra_rc:
        return False
    return True

_MASK_CACHE = {}
def _grid_key(lon, lat):
    return (len(lat), len(lon), float(lat[0]), float(lat[-1]), float(lon[0]), float(lon[-1]))

def build_combined_mask_on_grid(data_lon, data_lat):
    gk = _grid_key(data_lon, data_lat)
    if gk in _MASK_CACHE:
        return _MASK_CACHE[gk]

    # region mask on the data grid
    LON2D, LAT2D = np.meshgrid(data_lon, data_lat)
    shp = gpd.read_file(USCAN_SHP)
    region = ~np.isnan(regionmask.mask_geopandas(shp, LON2D, LAT2D, overlap=False))

    # biome to data grid (allow slight mismatch)
    bds = xr.open_dataset(BIOME_NC)
    bds = _rename_lonlat(bds)
    bio_var = 'gez_code_id' if 'gez_code_id' in bds.data_vars else ('gez_code' if 'gez_code' in bds.data_vars else None)
    if bio_var is None:
        raise RuntimeError(f"Biome file missing gez_code(_id): {BIOME_NC}")
    bio_lon = bds['longitude'].values
    bio_lat = bds['latitude'].values

    if (len(bio_lon)==len(data_lon) and len(bio_lat)==len(data_lat) and
        np.allclose(bio_lon, data_lon) and np.allclose(bio_lat, data_lat)):
        bio_on = bds[bio_var]
    else:
        if not _roughly_compatible(bio_lon, bio_lat, data_lon, data_lat):
            raise RuntimeError("Biome grid differs too much from data grid; please regrid explicitly.")
        bio_on = bds[bio_var].interp(
            latitude=xr.DataArray(data_lat, dims=('latitude',)),
            longitude=xr.DataArray(data_lon, dims=('longitude',)),
            method='nearest'
        )
    bio_arr = bio_on.values
    bio_mask = (~np.isnan(bio_arr)).copy()
    for v in EXCLUDE_BIOMES:
        bio_mask &= (bio_arr != v)

    combined = region & bio_mask
    _MASK_CACHE[gk] = (combined, bio_arr)
    return (combined, bio_arr)

# ---------- Load & organize ----------
def organize_data(nc_files, combined_mask):
    data = {}
    for p in nc_files:
        meta = parse_change_filename(p)
        if not meta:  # skip unmatched
            continue
        ds = xr.open_dataset(p)
        ds = _rename_lonlat(ds)
        if ('relative_change' not in ds) or ('absolute_change' not in ds):
            ds.close(); continue
        rel = ds['relative_change'].values
        abs_ = ds['absolute_change'].values
        ds.close()

        # apply mask
        if rel.shape != combined_mask.shape:
            print(f"[WARN] skip size mismatch: {os.path.basename(p)} {rel.shape} vs {combined_mask.shape}")
            continue
        rel = rel.astype(float); abs_ = abs_.astype(float)
        rel[~combined_mask] = np.nan; abs_[~combined_mask] = np.nan

        key = (meta['var'], meta['scenario'], meta['year_tag'])
        data.setdefault(key, {})[meta['model']] = {'relative': rel, 'absolute': abs_}
    return data

# ---------- Consistency calculators ----------
def consistency_categories(models_data, mask):
    n = len(models_data)
    stack = np.stack([m['relative'] for m in models_data.values()], axis=0)  # (n, ny, nx)
    pos = np.sum(stack > 0, axis=0)
    cat = np.full(pos.shape, np.nan, dtype=float)
    # 0 No data, 1 No increase, 2 Any>0, 3 ≥1/2, 4 ≥3/4, 5 Unanimous
    cat[~mask] = 0
    cat[mask & (pos == 0)] = 1
    cat[mask & (pos > 0)]  = 2
    cat[mask & (pos >= int(np.ceil(n/2.0)))] = 3
    cat[mask & (pos >= int(np.ceil(0.75*n)))] = 4
    cat[mask & (pos == n)] = 5
    return cat

def consistency_percent(models_data, mask):
    n = len(models_data)
    stack = np.stack([m['relative'] for m in models_data.values()], axis=0)
    pct = np.sum(stack > 0, axis=0) / n * 100.0
    pct[~mask] = np.nan
    return pct

# ---------- Map helpers ----------
def add_map_features(ax):
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor='black')
    ax.add_feature(cfeature.BORDERS,   linewidth=0.5, edgecolor='black')
    states = cfeature.NaturalEarthFeature(
        'cultural', 'admin_1_states_provinces_lines', '50m', facecolor='none'
    )
    ax.add_feature(states, linewidth=0.3, edgecolor='gray')
    ax.set_extent(MAP_EXTENT, crs=ccrs.PlateCarree())

# ---------- Plotters ----------
def plot_categorical(key, models_data, mask, lon, lat):
    var, scn, yt = key
    cats = ['No data', 'No increase', 'Any (>0)', '≥1/2', '≥3/4', 'Unanimous']
    colors = ['#ffffff', '#f0f0f0', '#abd9e9', '#74add1', '#4575b4', '#d73027']
    cmap = ListedColormap(colors)
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
    norm = BoundaryNorm(bounds, cmap.N)

    arr = consistency_categories(models_data, mask)
    LON, LAT = np.meshgrid(lon, lat)

    fig = plt.figure(figsize=(10,8), dpi=300)
    ax = fig.add_subplot(1,1,1, projection=PROJ)
    im = ax.pcolormesh(LON, LAT, arr, transform=ccrs.PlateCarree(), cmap=cmap, norm=norm, shading='auto')
    add_map_features(ax)
    cbar = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.05, fraction=0.04, shrink=0.8)
    cbar.set_ticks([0,1,2,3,4,5]); cbar.set_ticklabels(cats); cbar.ax.tick_params(rotation=45)
    title = f"{var} — Model consistency (categorical)\n{scn}  {yt}"
    ax.set_title(title, fontsize=11)
    fname = os.path.join(OUTPUT_DIR, f"consistency_categorical_{var}_{scn}_{yt}.png")
    plt.savefig(fname, dpi=300, bbox_inches='tight'); plt.close(fig); print("✓ saved", os.path.basename(fname))

def plot_continuous(key, models_data, mask, lon, lat):
    var, scn, yt = key
    pct = consistency_percent(models_data, mask)
    LON, LAT = np.meshgrid(lon, lat)

    fig = plt.figure(figsize=(10,8), dpi=300)
    ax = fig.add_subplot(1,1,1, projection=PROJ)
    im = ax.pcolormesh(LON, LAT, pct, transform=ccrs.PlateCarree(), cmap='YlOrRd', vmin=0, vmax=100, shading='auto')
    add_map_features(ax)
    cbar = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.05, fraction=0.04, shrink=0.8)
    cbar.set_label('Model agreement (%)')
    title = f"{var} — Model agreement (continuous)\n{scn}  {yt}"
    ax.set_title(title, fontsize=11)
    fname = os.path.join(OUTPUT_DIR, f"consistency_continuous_{var}_{scn}_{yt}.png")
    plt.savefig(fname, dpi=300, bbox_inches='tight'); plt.close(fig); print("✓ saved", os.path.basename(fname))

def plot_magnitude_consistency(key, models_data, mask, lon, lat,
                               thresh_abs=5.0, thresh_agree=75.0):
    var, scn, yt = key
    stack = np.stack([m['relative'] for m in models_data.values()], axis=0)
    ens = np.nanmean(stack, axis=0)
    agree = np.sum(stack>0, axis=0)/stack.shape[0]*100.0
    ens[~mask] = np.nan; agree[~mask] = np.nan
    LON, LAT = np.meshgrid(lon, lat)

    vmax = np.nanpercentile(np.abs(ens), 95)
    fig = plt.figure(figsize=(10,8), dpi=300)
    ax = fig.add_subplot(1,1,1, projection=PROJ)
    im = ax.pcolormesh(LON, LAT, ens, transform=ccrs.PlateCarree(),
                       cmap='RdBu_r', vmin=-vmax, vmax=vmax, shading='auto')
    sig = (np.abs(ens) > thresh_abs) & (agree > thresh_agree) & np.isfinite(ens)
    if np.any(sig):
        ax.scatter(LON[sig], LAT[sig], s=8, c='k', marker='.',
                   transform=ccrs.PlateCarree(), alpha=0.5, label=f'|Δ|>{thresh_abs} & agree>{thresh_agree}%')
    add_map_features(ax)
    cbar = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.05, fraction=0.04, shrink=0.8)
    cbar.set_label('Ensemble-mean relative change (%)')
    if np.any(sig): ax.legend(loc='lower left', frameon=True, framealpha=0.8)
    title = f"{var} — Magnitude × consistency\n{scn}  {yt}"
    ax.set_title(title, fontsize=11)
    fname = os.path.join(OUTPUT_DIR, f"magnitude_consistency_{var}_{scn}_{yt}.png")
    plt.savefig(fname, dpi=300, bbox_inches='tight'); plt.close(fig); print("✓ saved", os.path.basename(fname))

def plot_ssp_pair(var, year_tag, data, mask, lon, lat, ssp_pair=('SSP245','SSP370')):
    keys = []
    for scn in ssp_pair:
        k = (var, scn, year_tag)
        if k in data and data[k]:
            keys.append(k)
    if len(keys) != 2:
        return

    cons = [consistency_percent(data[k], mask) for k in keys]
    LON, LAT = np.meshgrid(lon, lat)

    fig = plt.figure(figsize=(16,8), dpi=300)
    gs = GridSpec(1, 3, width_ratios=[1,1,0.05], wspace=0.08)
    for i, (k, arr) in enumerate(zip(keys, cons)):
        ax = fig.add_subplot(gs[0,i], projection=PROJ)
        im = ax.pcolormesh(LON, LAT, arr, transform=ccrs.PlateCarree(),
                           cmap='YlOrRd', vmin=0, vmax=100, shading='auto')
        add_map_features(ax)
        ax.set_title(f"{k[0]} — {k[1]} ({k[2]})", fontsize=10)
    cax = fig.add_subplot(gs[0,2])
    sm = plt.cm.ScalarMappable(cmap='YlOrRd', norm=plt.Normalize(vmin=0, vmax=100))
    sm.set_array([])
    cbar = plt.colorbar(sm, cax=cax); cbar.set_label('Model agreement (%)')
    fig.suptitle(f"Model consistency comparison: {var} — {year_tag}", fontsize=12, y=0.98)
    fname = os.path.join(OUTPUT_DIR, f"consistency_compare_{var}_{year_tag}_{ssp_pair[0]}_vs_{ssp_pair[1]}.png")
    plt.savefig(fname, dpi=300, bbox_inches='tight'); plt.close(fig); print("✓ saved", os.path.basename(fname))

# ---------- Group summaries ----------
def biome_consistency_summary(data, mask, bio_arr, lon, lat):
    rows = []
    for (var, scn, yt), models in data.items():
        if not models: continue
        cons = consistency_percent(models, mask)  # 0-100
        for gname, blist in BIOME_GROUPS.items():
            bmask = np.isin(bio_arr, blist) & mask
            if np.any(bmask):
                vals = cons[bmask]
                vals = vals[np.isfinite(vals)]
                if vals.size:
                    rows.append(dict(
                        Variable=var, Scenario=scn, YearTag=yt, Biome_Group=gname,
                        Mean_Consistency=np.mean(vals),
                        Median_Consistency=np.median(vals),
                        High_Consistency_Area_Ratio=(np.sum(vals>75)/vals.size*100.0),
                        N_Grids=int(vals.size)
                    ))
    df = pd.DataFrame(rows)
    if not df.empty:
        csv = os.path.join(OUTPUT_DIR, "biome_consistency_stats.csv")
        df.to_csv(csv, index=False, encoding="utf-8-sig"); print("✓ saved", os.path.basename(csv))

        # quick bar plots for first 4 combos
        combos = df[['Variable','Scenario','YearTag']].drop_duplicates().head(4).to_records(index=False)
        fig, axes = plt.subplots(2,2, figsize=(12,10)); axes = axes.ravel()
        for i, (v, s, y) in enumerate(combos):
            ax = axes[i]
            sub = df[(df.Variable==v)&(df.Scenario==s)&(df.YearTag==y)]
            if sub.empty: continue
            x = np.arange(sub.shape[0]); w = 0.38
            ax.bar(x-w/2, sub['Mean_Consistency'],   width=w, label='Mean', color='steelblue')
            ax.bar(x+w/2, sub['Median_Consistency'], width=w, label='Median', color='lightcoral')
            ax.set_xticks(x); ax.set_xticklabels(sub['Biome_Group'], rotation=30)
            ax.set_ylim(0, 100); ax.grid(True, alpha=0.3)
            ax.set_title(f"{v} — {s} ({y})"); 
            if i==0: ax.legend()
        plt.tight_layout()
        png = os.path.join(OUTPUT_DIR, "biome_consistency_comparison.png")
        plt.savefig(png, dpi=300, bbox_inches='tight'); plt.close(fig); print("✓ saved", os.path.basename(png))

def latitude_gradient_summary(data, mask, lat):
    bands = np.arange(25, 85, 5)
    combos = list(data.keys())[:4]
    if not combos: return
    fig, axes = plt.subplots(2,2, figsize=(12,10)); axes = axes.ravel()
    for i, key in enumerate(combos):
        var, scn, yt = key
        cons = consistency_percent(data[key], mask)
        stats = []
        for j in range(len(bands)-1):
            sel = (lat >= bands[j]) & (lat < bands[j+1])
            sel2d = sel[:, None]
            valid = mask & sel2d
            if np.any(valid):
                vals = cons[valid]; vals = vals[np.isfinite(vals)]
                if vals.size:
                    stats.append(dict(
                        lat_center=(bands[j]+bands[j+1])/2.0,
                        mean=np.mean(vals), std=np.std(vals), n=vals.size
                    ))
        if stats:
            df = pd.DataFrame(stats)
            ax = axes[i]
            ax.errorbar(df['lat_center'], df['mean'], yerr=df['std'], marker='o', capsize=4)
            ax.set_ylim(0,100); ax.grid(True, alpha=0.3)
            ax.set_xlabel('Latitude (°N)'); ax.set_ylabel('Model agreement (%)')
            ax.axvspan(60, 85, alpha=0.15, color='blue')
            ax.axvspan(25, 45, alpha=0.15, color='orange')
            ax.set_title(f"{var} — {scn} ({yt})")
    plt.tight_layout()
    png = os.path.join(OUTPUT_DIR, "latitude_gradient_consistency.png")
    plt.savefig(png, dpi=300, bbox_inches='tight'); plt.close(fig); print("✓ saved", os.path.basename(png))

# ---------- Main ----------
def main():
    files = glob(os.path.join(INPUT_DIR, "*.nc"))
    if not files:
        print("[ERROR] No *_change.nc in", INPUT_DIR); return

    # establish grid & mask from the first usable file
    ds0 = None
    for p in files:
        try:
            ds0 = xr.open_dataset(p); ds0 = _rename_lonlat(ds0)
            lat0 = ds0['latitude'].values; lon0 = ds0['longitude'].values
            (combined0, bio_arr0) = build_combined_mask_on_grid(lon0, lat0)
            ds0.close(); break
        except Exception as e:
            if ds0: ds0.close()
            continue
    if ds0 is None:
        print("[ERROR] Could not establish base grid/mask."); return

    # collect only files that match base grid (others会自动落在 organize 里按各自网格处理)
    metas = [parse_change_filename(p) for p in files]
    metas = [m for m in metas if m is not None]
    if not metas:
        print("[ERROR] No usable files after parsing."); return

    # Organize data (per (var, scenario, year_tag))
    data = organize_data(files, combined0)

    # plot per key
    for key, models in data.items():
        if not models: continue
        # individual maps
        plot_categorical(key, models, combined0, lon0, lat0)
        plot_continuous(key,  models, combined0, lon0, lat0)
        plot_magnitude_consistency(key, models, combined0, lon0, lat0,
                                   thresh_abs=5.0, thresh_agree=75.0)

    # SSP comparison (245 vs 370) per variable & year_tag
    vars_list  = sorted({k[0] for k in data.keys()})
    year_tags  = sorted({k[2] for k in data.keys()})
    for v in vars_list:
        for yt in year_tags:
            plot_ssp_pair(v, yt, data, combined0, lon0, lat0, ssp_pair=('SSP245','SSP370'))

    # biome summaries & latitude gradient
    biome_consistency_summary(data, combined0, bio_arr0, lon0, lat0)
    latitude_gradient_summary(data, combined0, lat0)

    gc.collect()
    print("\nDone. Outputs ->", OUTPUT_DIR)

if __name__ == "__main__":
    main()












#%% addtional plots figure 1






import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# ========== 全局样式（尽量接近期刊风 & 使用 Arial） ==========
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Arial']  # 确保系统里有 Arial
mpl.rcParams['pdf.fonttype'] = 42           # 嵌入 TrueType
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['axes.unicode_minus'] = False

plt.rcParams['font.size'] = 6
plt.rcParams['axes.linewidth'] = 0.7
plt.rcParams['xtick.direction'] = 'out'
plt.rcParams['ytick.direction'] = 'out'

# 颜色与标签
color_map = {
    "fire_probability": "#1f78b4",  # ABDp 蓝
    "obe_probability": "#e66101",   # OBEp 橙
}
var_label = {
    "fire_probability": "ABDp",
    "obe_probability": "OBEp",
}

ssp_order = ["SSP126", "SSP245", "SSP370", "SSP585"]
vars_keep = ["fire_probability", "obe_probability"]

# ========== 1. 相对增幅数据 ==========

REL_FILES = [
    r"E:\Projection paper_sup_scens\analysis\stats\overall_fire_change_analysis.csv",  # SSP126 + SSP585
    r"E:\Projection paper\analysis\stats\overall_fire_change_analysis.csv",           # SSP245 + SSP370
]

rel_all = pd.concat([pd.read_csv(p) for p in REL_FILES], ignore_index=True)
rel_all = rel_all[rel_all["Type"] == "Individual_Model"].copy()

def assign_period(row):
    if row["YearStart"] == 2041 and row["YearEnd"] == 2070:
        return "mid-century (2041–2070)"
    elif row["YearStart"] == 2071 and row["YearEnd"] == 2100:
        return "late-century (2071–2100)"
    else:
        return "other"

rel_all["Period"] = rel_all.apply(assign_period, axis=1)

rel_plot = rel_all[
    rel_all["SSP"].isin(ssp_order)
    & rel_all["Variable"].isin(vars_keep)
].copy()

agg_rel = (
    rel_plot
    .groupby(["Period", "SSP", "Variable"])
    .agg(
        mean_change=("Relative_change_%", "mean"),
        min_change=("Relative_change_%", "min"),
        max_change=("Relative_change_%", "max"),
    )
    .reset_index()
)

# ========== 2. 空间一致性数据 ==========

CONS_FILES = [
    r"E:\Projection paper\analysis\stats\model_consistency_pivot.csv",            # SSP245 & SSP370
    r"E:\Projection paper_sup_scens\analysis\stats\model_consistency_pivot.csv",  # SSP126 & SSP585
]

cons_all = pd.concat([pd.read_csv(p) for p in CONS_FILES], ignore_index=True)

def assign_period_from_yeartag(year):
    if year == 2040:
        return "mid-century (2041–2070)"
    elif year == 2070:
        return "late-century (2071–2100)"
    else:
        return "other"

cons_all["Period"] = cons_all["YearTag"].apply(assign_period_from_yeartag)

cons_metrics = [">=3/4", "Unanimous"]
metric_linestyle = {
    ">=3/4": "-",
    "Unanimous": "--",
}

# ========== 3. 画 4 个 panel ==========

# 5 cm × 16 cm -> inch
fig_w = 5.0 / 2.54
fig_h = 16.0 / 2.54

fig, axes = plt.subplots(
    4, 1,
    figsize=(fig_w, fig_h),
    sharex=True,
    gridspec_kw={"height_ratios": [2, 1, 2, 1], "hspace": 0.4},
)

x_center = np.arange(len(ssp_order))
offset = 0.10  # ABDp / OBEp 在 x 上错开一点


# ---- 相对增幅 panel ----
def plot_relative_panel(ax, period):
    dfp = agg_rel[agg_rel["Period"] == period]
    all_vals = []

    for var in vars_keep:
        sub = (
            dfp[dfp["Variable"] == var]
            .set_index("SSP")
            .reindex(ssp_order)
        )
        mean = sub["mean_change"].values
        ymin = sub["min_change"].values
        ymax = sub["max_change"].values

        # 不同变量在 x / y 上稍微偏移，防止文字完全重叠
        if var == "fire_probability":
            xpos = x_center - offset
            text_dy = -8   # ABDp 文本往下偏
            text_dx = -4   # 往左一点
        else:
            xpos = x_center + offset
            text_dy = 6    # OBEp 文本往上偏
            text_dx = 4    # 往右一点

        all_vals.extend(list(ymin))
        all_vals.extend(list(ymax))

        yerr = np.vstack([mean - ymin, ymax - mean])

        ax.errorbar(
            xpos, mean,
            yerr=yerr,
            fmt="o-",
            capsize=2.5,
            markersize=3.0,
            linewidth=0.9,
            color=color_map[var],
            label=var_label[var],
        )

        # ====== 保留“所有点”的文字标注：你后续可以自己删减 ======
        for xi, ssp, m, lo, hi in zip(xpos, ssp_order, mean, ymin, ymax):
            # 两行文字：mean 在上，(min–max) 在下
            label = f"{m:.1f}\n({lo:.1f}–{hi:.1f})"
            ax.annotate(
                label,
                xy=(xi, m),
                xytext=(text_dx, text_dy),
                textcoords="offset points",
                ha="center",
                va="bottom" if text_dy > 0 else "top",
                fontsize=5,
                linespacing=0.9,
            )

    if all_vals:
        y_min = min(all_vals)
        y_max = max(all_vals)
        margin = max((y_max - y_min) * 0.15, 5)
        ax.set_ylim(y_min - margin, y_max + margin)

    ax.set_ylabel("Relative change\nin probability (%)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ---- 空间一致性 panel ----
def plot_consistency_panel(ax, period):
    dfp = cons_all[cons_all["Period"] == period]
    all_vals = []

    for var in vars_keep:
        for metric in cons_metrics:
            sub = (
                dfp[dfp["Variable"] == var]
                .set_index("Scenario")
                .reindex(ssp_order)
            )
            y = sub[metric].values * 100.0

            if var == "fire_probability":
                xpos = x_center - offset
            else:
                xpos = x_center + offset

            all_vals.extend(list(y))

            ax.plot(
                xpos, y,
                marker="o",
                markersize=3.0,
                linewidth=0.9,
                color=color_map[var],
                linestyle=metric_linestyle[metric],
            )

    if all_vals:
        ymin = min(all_vals)
        low = max(0, ymin - 5)
        ax.set_ylim(low, 100)

    ax.set_ylabel("Fraction of\ngrid cells (%)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ---- 填 4 个 panel ----
plot_relative_panel(axes[0], "late-century (2071–2100)")
plot_consistency_panel(axes[1], "late-century (2071–2100)")
plot_relative_panel(axes[2], "mid-century (2041–2070)")
plot_consistency_panel(axes[3], "mid-century (2041–2070)")

# x 轴（只在最下面画 tick + label）
axes[-1].set_xticks(x_center)
axes[-1].set_xticklabels(ssp_order, rotation=0)
axes[-1].set_xlabel("SSP scenario")

# ====== 图例：颜色（ABDp/OBEp） + 线型（>=3/4 vs Unanimous） ======
handles_color = [
    plt.Line2D([0], [0], color=color_map["fire_probability"], lw=1, marker="o", ms=3),
    plt.Line2D([0], [0], color=color_map["obe_probability"], lw=1, marker="o", ms=3),
]
labels_color = ["ABDp", "OBEp"]

handles_ls = [
    plt.Line2D([0], [0], color="black", lw=1, linestyle=metric_linestyle[">=3/4"]),
    plt.Line2D([0], [0], color="black", lw=1, linestyle=metric_linestyle["Unanimous"]),
]
labels_ls = [">= 3/4 models", "Unanimous"]

fig.legend(
    handles_color, labels_color,
    loc="upper center",
    ncol=2,
    frameon=False,
    bbox_to_anchor=(0.5, 1.03),
    columnspacing=0.8,
)

fig.legend(
    handles_ls, labels_ls,
    loc="upper center",
    ncol=2,
    frameon=False,
    bbox_to_anchor=(0.5, 1.12),
    columnspacing=0.8,
)

plt.tight_layout()

# ========== 保存成 PDF ==========
out_path = r"E:\Projection paper\analysis\maps\figures\AAAFig_projection_ABDp_OBEp_fulltext.pdf"
fig.savefig(out_path, bbox_inches="tight")

# 如果想在屏幕上看一眼：
plt.show()




#%% canada 2023 us 2020/2021



# -*- coding: utf-8 -*-
import os, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = r"E:\Projection paper"
REGIONAL_DIR = os.path.join(BASE_DIR, "regional_outputs")
FIG_DIR = os.path.join(REGIONAL_DIR, "figures")
PCT_DIR = os.path.join(FIG_DIR, "percentiles")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(PCT_DIR, exist_ok=True)

DATASETS = [
    {"name": "Canada", "region_tag": "canada_2023", "years": [2023]},
    {"name": "Western US (2020)", "region_tag": "us_2020", "years": [2020]},
    {"name": "Western US (2021)", "region_tag": "us_2021", "years": [2021]},
]
WESTERN_US_COMBINED = {"name": "Western US", "tags": ["us_2020","us_2021"], "years": [2020, 2021]}

WINDOWS = {
    "window_2040": {"label": "Mid-Century (2040–2070)"},
    "window_2070": {"label": "Late-Century (2071–2100)"},
}
WINDOW_YEAR_RANGE = {  # 用于从“总表”里筛出对应窗口
    "window_2040": (2041, 2070),
    "window_2070": (2071, 2100),
}

VARIABLES = ["fire_probability", "obe_probability"]
VAR_LABELS = {
    "fire_probability": "Active burning (ABDp)",
    "obe_probability": "Overnight burning extremes (OBEp)"
}
COLORS = {
    "SSP245": "#1f77b4",
    "SSP585": "#ff7f0e",  # 370 同 585 颜色
    "SSP370": "#ff7f0e",
    "1991_2020": "#808080",
    2020: "#8b0000",
    2021: "#ff1493",
    2023: "#dc143c",
}
plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.sans-serif"] = ["Arial"]
plt.rcParams["axes.unicode_minus"] = False

_df_cache = {}
def _read_csv_cached(path: str) -> pd.DataFrame:
    if path not in _df_cache:
        _df_cache[path] = pd.read_csv(path)
    return _df_cache[path]

def _find_annual_file(region_tag: str, window_tag: str, year: int) -> str:
    """优先找 window_XXXX 子目录；找不到就回退到区域根目录"""
    # 1) 子目录
    folder = os.path.join(REGIONAL_DIR, region_tag, window_tag)
    patt = os.path.join(folder, f"annual_fire_days_*_{year}.csv")
    m = glob.glob(patt)
    if m: return m[0]
    # 2) 区域根目录（分析脚本就是这里写的）
    folder2 = os.path.join(REGIONAL_DIR, region_tag)
    patt2 = os.path.join(folder2, f"annual_fire_days_*_{year}.csv")
    m2 = glob.glob(patt2)
    if m2: return m2[0]
    raise FileNotFoundError(f"No CSV found for {region_tag}-{window_tag}-{year}: {patt} OR {patt2}")

def _find_baseline_1991_2020(region_tag: str) -> str:
    folder = os.path.join(REGIONAL_DIR, region_tag)
    matches = glob.glob(os.path.join(folder, "**", "*1991_2020_baseline.csv"), recursive=True)
    if not matches:
        raise FileNotFoundError(f"No 1991_2020 baseline CSV found under {folder}")
    matches.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return matches[0]

def _precise_percentile(vals: np.ndarray, x: float) -> float:
    vals = np.asarray(vals); n = len(vals)
    if n == 0: return np.nan
    rank = np.sum(vals < x) + 0.5*np.sum(vals == x)
    return rank/n*100.0

def _scenario_color(s: str) -> str:
    return COLORS["SSP245"] if s.upper().startswith("SSP245") else COLORS["SSP585"]

def _subset_future_by_window(df_overall: pd.DataFrame, window_tag: str) -> pd.DataFrame:
    """从总表里按年份范围抽取对应窗口的 future 行"""
    y0, y1 = WINDOW_YEAR_RANGE[window_tag]
    df_future = df_overall[df_overall["scenario"] != "baseline"].copy()
    if "year" in df_future.columns:
        return df_future[(df_future["year"] >= y0) & (df_future["year"] <= y1)]
    return df_future  # 兜底（基本不会走到）

# ---------- 单区域（Canada 或 US单年） ----------
def plot_window(dataset_name: str, region_tag: str, years: list[int], window_tag: str):
    base1991 = _read_csv_cached(_find_baseline_1991_2020(region_tag))
    base1991_overall = base1991[base1991["region"]=="overall"].copy()

    # 读“总表/子表”，并切出该窗口的 future
    frames = {}
    baseline_lines = {}
    for y in years:
        f = _find_annual_file(region_tag, window_tag, y)
        df = _read_csv_cached(f)
        overall = df[df["region"]=="overall"].copy()
        base = overall[overall["scenario"]=="baseline"]
        fut  = _subset_future_by_window(overall, window_tag)
        frames[y] = (base, fut)
        for var in VARIABLES:
            v = base[base["variable"]==var]["fire_days"].values
            if v.size: baseline_lines.setdefault(var, {})[y] = float(v[0])

    fig, axes = plt.subplots(1, len(VARIABLES), figsize=(12,6), constrained_layout=True)
    if not isinstance(axes, np.ndarray): axes = np.array([axes])
    title_years = ", ".join(map(str, years))
    fig.suptitle(f"{dataset_name} {title_years} – {WINDOWS[window_tag]['label']} fire days comparison",
                 fontsize=16, y=1.02)

    for ax, var in zip(axes, VARIABLES):
        violin, pos, labels = [], [], []; x = 0
        vals1991 = base1991_overall[base1991_overall["variable"]==var]["fire_days"].values
        if vals1991.size:
            violin.append(vals1991); pos.append(x); labels.append("1991–2020\nBaseline"); x += 1.5
        fut_all = pd.concat([frames[y][1] for y in years], ignore_index=True)
        fut_var = fut_all[fut_all["variable"]==var]
        for ssp in sorted(fut_var["scenario"].dropna().unique()):
            sub = fut_var[fut_var["scenario"]==ssp]
            for model in sorted(sub["model"].dropna().unique()):
                vals = sub[sub["model"]==model]["fire_days"].values
                if vals.size: violin.append(vals); pos.append(x); labels.append(f"{ssp}\n{model}"); x += 1
            x += 0.5

        if not violin:
            ax.text(.5,.5,"No data", ha="center", va="center", transform=ax.transAxes); continue

        parts = ax.violinplot(violin, positions=pos, widths=0.7, showmeans=True, showmedians=True, showextrema=True)
        for i,(pc,lbl) in enumerate(zip(parts["bodies"], labels)):
            if i==0 and "Baseline" in lbl: pc.set_facecolor(COLORS["1991_2020"]); pc.set_alpha(.6)
            else: pc.set_facecolor(_scenario_color(lbl.split("\n")[0])); pc.set_alpha(.65)
        for k in ["cmeans","cmedians","cbars","cmaxes","cmins"]:
            if k in parts: parts[k].set_edgecolor("black"); parts[k].set_linewidth(1)

        if var in baseline_lines:
            for y,v in baseline_lines[var].items():
                ax.axhline(v, color=COLORS[y], linestyle="--", linewidth=2, label=f"{y}: {v:.1f}")
        if vals1991.size:
            m1991 = float(np.mean(vals1991))
            ax.axhline(m1991, color=COLORS["1991_2020"], linestyle=":", linewidth=2, label=f"1991–2020: {m1991:.1f}")

        ax.set_xticks(pos); ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
        ax.set_title(VAR_LABELS[var], fontsize=14, pad=10); ax.grid(True, axis="y", alpha=.3)
        ax.legend(loc="upper left", fontsize=9)
        if var==VARIABLES[0]: ax.set_ylabel("Annual potential days", fontsize=12)

    safe = dataset_name.replace(" ","_").lower()
    out_png = os.path.join(FIG_DIR, f"{safe}_{window_tag}_violin.png")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_png.replace(".png",".pdf"), bbox_inches="tight")
    print(f"Saved: {out_png}")
    plt.close(fig)

# ---------- Western US 合并 ----------
def plot_western_us_combined(window_tag: str):
    name = WESTERN_US_COMBINED["name"]; tags = WESTERN_US_COMBINED["tags"]; years = WESTERN_US_COMBINED["years"]
    base1991 = _read_csv_cached(_find_baseline_1991_2020(tags[0]))
    base1991_overall = base1991[base1991["region"]=="overall"].copy()

    baseline_lines = {}; futures_ref = None
    for i,(tag,y) in enumerate(zip(tags, years)):
        f = _find_annual_file(tag, window_tag, y)
        df = _read_csv_cached(f)
        overall = df[df["region"]=="overall"].copy()
        base = overall[overall["scenario"]=="baseline"]
        for var in VARIABLES:
            v = base[base["variable"]==var]["fire_days"].values
            if v.size: baseline_lines.setdefault(var, {})[y] = float(v[0])
        if i==0:
            futures_ref = _subset_future_by_window(overall, window_tag).copy()

    fig, axes = plt.subplots(1, len(VARIABLES), figsize=(12,6), constrained_layout=True)
    if not isinstance(axes, np.ndarray): axes = np.array([axes])
    fig.suptitle(f"{name} – {WINDOWS[window_tag]['label']} fire days comparison", fontsize=16, y=1.02)

    for ax, var in zip(axes, VARIABLES):
        violin, pos, labels = [], [], []; x = 0
        vals1991 = base1991_overall[base1991_overall["variable"]==var]["fire_days"].values
        if vals1991.size:
            violin.append(vals1991); pos.append(x); labels.append("1991–2020\nBaseline"); x += 1.5
        fut_var = futures_ref[futures_ref["variable"]==var]
        for ssp in sorted(fut_var["scenario"].dropna().unique()):
            sub = fut_var[fut_var["scenario"]==ssp]
            for model in sorted(sub["model"].dropna().unique()):
                vals = sub[sub["model"]==model]["fire_days"].values
                if vals.size: violin.append(vals); pos.append(x); labels.append(f"{ssp}\n{model}"); x += 1
            x += .5

        parts = ax.violinplot(violin, positions=pos, widths=0.7, showmeans=True, showmedians=True, showextrema=True)
        for i,(pc,lbl) in enumerate(zip(parts["bodies"], labels)):
            if i==0 and "Baseline" in lbl: pc.set_facecolor(COLORS["1991_2020"]); pc.set_alpha(.6)
            else: pc.set_facecolor(_scenario_color(lbl.split("\n")[0])); pc.set_alpha(.65)
        for k in ["cmeans","cmedians","cbars","cmaxes","cmins"]:
            if k in parts: parts[k].set_edgecolor("black"); parts[k].set_linewidth(1)

        if var in baseline_lines:
            for y,v in baseline_lines[var].items():
                ax.axhline(v, color=COLORS[y], linestyle="--", linewidth=2, label=f"{y}: {v:.1f}")
        if vals1991.size:
            m1991 = float(np.mean(vals1991))
            ax.axhline(m1991, color=COLORS["1991_2020"], linestyle=":", linewidth=2, label=f"1991–2020: {m1991:.1f}")

        ax.set_xticks(pos); ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
        ax.set_title(VAR_LABELS[var], fontsize=14, pad=10); ax.grid(True, axis="y", alpha=.3)
        ax.legend(loc="upper left", fontsize=9)
        if var==VARIABLES[0]: ax.set_ylabel("Annual potential days", fontsize=12)

    out_png = os.path.join(FIG_DIR, f"western_us_combined_{window_tag}.png")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_png.replace(".png",".pdf"), bbox_inches="tight")
    print(f"Saved: {out_png}")
    plt.close(fig)

# ---------- 百分位（同样按窗口筛年份） ----------
def percentile_analysis(dataset_name: str, region_tag: str, years: list[int]):
    rows = []
    base1991 = _read_csv_cached(_find_baseline_1991_2020(region_tag))
    base1991_overall = base1991[base1991["region"]=="overall"].copy()

    for window_tag in WINDOWS.keys():
        for y in years:
            f = _find_annual_file(region_tag, window_tag, y)
            dfy = _read_csv_cached(f)
            overall = dfy[dfy["region"]=="overall"].copy()
            base = overall[overall["scenario"]=="baseline"]
            fut  = _subset_future_by_window(overall, window_tag)

            for var in VARIABLES:
                v_ext = base[base["variable"]==var]["fire_days"].values
                if not v_ext.size: continue
                extreme = float(v_ext[0])
                basevals = base1991_overall[base1991_overall["variable"]==var]["fire_days"].values
                if not basevals.size: continue

                rows.append({
                    "dataset": dataset_name, "region_tag": region_tag, "window": window_tag,
                    "year": y, "variable": var, "scenario": "1991_2020", "model": "ALL",
                    "percentile_of_extreme": _precise_percentile(basevals, extreme),
                    "dist_mean": float(np.mean(basevals)), "dist_std": float(np.std(basevals)),
                    "extreme_value": extreme
                })

                dfv = fut[fut["variable"]==var]
                for ssp in sorted(dfv["scenario"].dropna().unique()):
                    sub = dfv[dfv["scenario"]==ssp]
                    ens = sub["fire_days"].values
                    if ens.size:
                        rows.append({
                            "dataset": dataset_name, "region_tag": region_tag, "window": window_tag,
                            "year": y, "variable": var, "scenario": ssp, "model": "ENSEMBLE",
                            "percentile_of_extreme": _precise_percentile(ens, extreme),
                            "dist_mean": float(np.mean(ens)), "dist_std": float(np.std(ens)),
                            "extreme_value": extreme
                        })
                    for model in sorted(sub["model"].dropna().unique()):
                        mv = sub[sub["model"]==model]["fire_days"].values
                        if mv.size:
                            rows.append({
                                "dataset": dataset_name, "region_tag": region_tag, "window": window_tag,
                                "year": y, "variable": var, "scenario": ssp, "model": model,
                                "percentile_of_extreme": _precise_percentile(mv, extreme),
                                "dist_mean": float(np.mean(mv)), "dist_std": float(np.std(mv)),
                                "extreme_value": extreme
                            })

    if not rows:
        print(f"[WARN] No percentile rows for {dataset_name} ({region_tag})"); return
    out = pd.DataFrame(rows)
    safe = dataset_name.replace(" ","_").lower().replace("(","").replace(")","")
    out_csv = os.path.join(PCT_DIR, f"percentiles_{safe}.csv")
    out.to_csv(out_csv, index=False)
    print(f"Saved percentile table: {out_csv}")

# ---------- Main ----------
def main():
    # Canada + US（单年）
    for ds in DATASETS:
        name, tag, yrs = ds["name"], ds["region_tag"], ds["years"]
        for wtag in WINDOWS.keys():
            try: plot_window(name, tag, yrs, wtag)
            except FileNotFoundError as e: print(f"[SKIP plot] {e}")
        try: percentile_analysis(name, tag, yrs)
        except FileNotFoundError as e: print(f"[SKIP pct] {e}")

    # Western US 合并图
    for wtag in WINDOWS.keys():
        try: plot_western_us_combined(wtag)
        except FileNotFoundError as e: print(f"[SKIP combined plot] {e}")

if __name__ == "__main__":
    main()
















# -*- coding: utf-8 -*-
import os, glob
from typing import Optional, Tuple, Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ============================================================
# 0) CONFIG
# ============================================================

BASE_DIR_MAIN = r"E:\Projection paper"
BASE_DIR_SUP  = r"E:\Projection paper_sup_scens"

REGIONAL_MAIN = os.path.join(BASE_DIR_MAIN, "regional_outputs")
REGIONAL_SUP  = os.path.join(BASE_DIR_SUP,  "regional_outputs")

OUT_DIR = os.path.join(REGIONAL_MAIN, "figures_fig4_2x2_violin_pooledmodels")
os.makedirs(OUT_DIR, exist_ok=True)

DATASETS = [
    {"name": "Canada 2023",          "tags": ["canada_2023"],         "years": [2023],       "mode": "single"},
    {"name": "Western US 2020/2021", "tags": ["us_2020","us_2021"],   "years": [2020, 2021], "mode": "combined"},
]

WINDOWS = {
    "window_2040": {"label": "Mid-Century (2041–2070)"},
    "window_2070": {"label": "Late-Century (2071–2100)"},
}
WINDOW_YEAR_RANGE = {
    "window_2040": (2041, 2070),
    "window_2070": (2071, 2100),
}

VARIABLES = ["fire_probability", "obe_probability"]
VAR_LABELS = {
    "fire_probability": "Active burning (ABDp)",
    "obe_probability": "Overnight burning extremes (OBEp)"
}

# ---- toggle SSP585 on/off ----
INCLUDE_SSP585 = True
SSP_ORDER_ALL = ["SSP126", "SSP245", "SSP370", "SSP585"]
SSP_ORDER = SSP_ORDER_ALL if INCLUDE_SSP585 else ["SSP126", "SSP245", "SSP370"]

# 颜色（你现有配色）
COLORS = {
    "SSP126": "#2ca02c",
    "SSP245": "#1f77b4",
    "SSP370": "#ff7f0e",
    "SSP585": "#d62728",
    "1991_2020": "#808080",
    2020: "#8b0000",
    2021: "#ff1493",
    2023: "#dc143c",
}

# 模型 + marker
MODEL_ORDER = ["CANESM", "ECEARTH", "GFDL", "UKESM"]
MODEL_MARKER = {"CANESM": "o", "ECEARTH": "^", "GFDL": "s", "UKESM": "x"}
MODEL_OFFSETS = {"CANESM": -0.18, "ECEARTH": -0.06, "GFDL": 0.06, "UKESM": 0.18}

# ---- 全局字体：最大不超过 7 ----
plt.rcParams.update({
    "font.family": "Arial",
    "font.sans-serif": ["Arial"],
    "axes.unicode_minus": False,

    "font.size": 7,
    "axes.titlesize": 7,
    "axes.labelsize": 7,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "legend.fontsize": 6,
})

_df_cache: Dict[str, pd.DataFrame] = {}

def _read_csv_cached(path: str) -> pd.DataFrame:
    if path not in _df_cache:
        _df_cache[path] = pd.read_csv(path)
    return _df_cache[path]

def _find_annual_file(regional_dir: str, region_tag: str, window_tag: str, year: int) -> str:
    folder = os.path.join(regional_dir, region_tag, window_tag)
    patt = os.path.join(folder, f"annual_fire_days_*_{year}.csv")
    m = glob.glob(patt)
    if m:
        return m[0]
    folder2 = os.path.join(regional_dir, region_tag)
    patt2 = os.path.join(folder2, f"annual_fire_days_*_{year}.csv")
    m2 = glob.glob(patt2)
    if m2:
        return m2[0]
    raise FileNotFoundError(f"No annual CSV for {region_tag}-{window_tag}-{year}")

def _find_baseline_1991_2020(regional_dir: str, region_tag: str) -> str:
    folder = os.path.join(regional_dir, region_tag)
    matches = glob.glob(os.path.join(folder, "**", "*1991_2020_baseline.csv"), recursive=True)
    if not matches:
        raise FileNotFoundError(f"No 1991_2020 baseline CSV under {folder}")
    matches.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return matches[0]

def _subset_future_by_window(df_overall: pd.DataFrame, window_tag: str) -> pd.DataFrame:
    y0, y1 = WINDOW_YEAR_RANGE[window_tag]
    df_future = df_overall[df_overall["scenario"] != "baseline"].copy()
    if "year" in df_future.columns:
        df_future = df_future[(df_future["year"] >= y0) & (df_future["year"] <= y1)]
    return df_future

def _norm_scenario(s: str) -> str:
    u = str(s).upper()
    for k in SSP_ORDER_ALL:
        if u.startswith(k):
            return k
    return u

def _norm_model(m: str) -> Optional[str]:
    if m is None:
        return None
    u = str(m).upper()
    if "CANESM" in u: return "CANESM"
    if "ECEARTH" in u or "EC-EARTH" in u or ("EC" in u and "EARTH" in u): return "ECEARTH"
    if "GFDL" in u: return "GFDL"
    if "UKESM" in u: return "UKESM"
    return None

def _scenario_color(ssp: str) -> str:
    return COLORS.get(ssp, "#999999")

def _load_overall_for_year(region_tag: str, window_tag: str, year: int) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    main = None
    sup = None
    try:
        f1 = _find_annual_file(REGIONAL_MAIN, region_tag, window_tag, year)
        df1 = _read_csv_cached(f1)
        main = df1[df1["region"] == "overall"].copy()
        main["__source__"] = "MAIN"
    except FileNotFoundError:
        pass
    try:
        f2 = _find_annual_file(REGIONAL_SUP, region_tag, window_tag, year)
        df2 = _read_csv_cached(f2)
        sup = df2[df2["region"] == "overall"].copy()
        sup["__source__"] = "SUP"
    except FileNotFoundError:
        pass
    return main, sup

def _merge_future(main: Optional[pd.DataFrame], sup: Optional[pd.DataFrame], window_tag: str) -> pd.DataFrame:
    frames = []
    if main is not None:
        frames.append(_subset_future_by_window(main, window_tag))
    if sup is not None:
        frames.append(_subset_future_by_window(sup, window_tag))
    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)

    keys = [c for c in ["scenario", "model", "year", "variable", "region"] if c in df.columns]
    if keys and "__source__" in df.columns:
        df["__prio__"] = df["__source__"].map({"MAIN": 0, "SUP": 1}).fillna(9)
        df = df.sort_values("__prio__").drop_duplicates(subset=keys, keep="first")
        df = df.drop(columns=["__prio__"], errors="ignore")

    df["scenario_norm"] = df["scenario"].apply(_norm_scenario) if "scenario" in df.columns else ""
    df["model_norm"] = df["model"].apply(_norm_model) if "model" in df.columns else None
    return df

def _load_baseline_dist(region_tag: str) -> pd.DataFrame:
    f = _find_baseline_1991_2020(REGIONAL_MAIN, region_tag)
    df = _read_csv_cached(f)
    return df[df["region"] == "overall"].copy()

def _collect_future_and_extreme_lines(dataset: dict):
    tags = dataset["tags"]
    years = dataset["years"]
    mode = dataset["mode"]

    future_by_window: Dict[str, pd.DataFrame] = {w: pd.DataFrame() for w in WINDOWS.keys()}
    extreme_lines: Dict[str, Dict[str, Dict[int, float]]] = {w: {v: {} for v in VARIABLES} for w in WINDOWS.keys()}

    for wtag in WINDOWS.keys():
        frames = []

        if mode == "single":
            region_tag, y = tags[0], years[0]
            m, s = _load_overall_for_year(region_tag, wtag, y)
            df_for_base = m if m is not None else s
            if df_for_base is not None:
                base = df_for_base[df_for_base["scenario"] == "baseline"]
                for var in VARIABLES:
                    vv = base[base["variable"] == var]["fire_days"].values
                    if vv.size:
                        extreme_lines[wtag][var][y] = float(vv[0])

            fut = _merge_future(m, s, wtag)
            if not fut.empty:
                frames.append(fut)

        else:
            # combined: keep both extreme lines (2020 & 2021), future only from first tag/year
            for i, (region_tag, y) in enumerate(zip(tags, years)):
                m, s = _load_overall_for_year(region_tag, wtag, y)
                df_for_base = m if m is not None else s
                if df_for_base is not None:
                    base = df_for_base[df_for_base["scenario"] == "baseline"]
                    for var in VARIABLES:
                        vv = base[base["variable"] == var]["fire_days"].values
                        if vv.size:
                            extreme_lines[wtag][var][y] = float(vv[0])

                if i == 0:
                    fut = _merge_future(m, s, wtag)
                    if not fut.empty:
                        frames.append(fut)

        future_by_window[wtag] = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    return future_by_window, extreme_lines

def _panel_violin_mergedmodels(ax, base_dist: pd.DataFrame, fut: pd.DataFrame,
                               var: str, title: str, extreme_lines: Dict[int, float]):

    ax.set_title(title, pad=6)

    violins: List[np.ndarray] = []
    pos: List[float] = []
    vcolors: List[str] = []
    xticklabels: List[str] = []

    x = 0.0

    # baseline violin
    vb = base_dist[base_dist["variable"] == var]["fire_days"].values.astype(float)
    if vb.size:
        violins.append(vb)
        pos.append(x)
        vcolors.append(COLORS["1991_2020"])
        xticklabels.append("Baseline")
        x += 1.4

    # each SSP: one violin pooled across models + years
    dfv = fut[fut["variable"] == var].copy()

    plotted_ssps = []
    for ssp in SSP_ORDER:
        sub = dfv[dfv["scenario_norm"] == ssp]
        if sub.empty:
            continue
        vals = sub["fire_days"].values.astype(float)
        if vals.size == 0:
            continue
        violins.append(vals)
        pos.append(x)
        vcolors.append(_scenario_color(ssp))
        xticklabels.append(ssp.replace("SSP", ""))
        plotted_ssps.append(ssp)
        x += 1.0

    if not violins:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return

    parts = ax.violinplot(
        violins,
        positions=pos,
        widths=0.65,          # 你说你改半宽仍大，这里再收一点
        showmeans=True,
        showmedians=True,
        showextrema=True
    )

    for body, c in zip(parts["bodies"], vcolors):
        body.set_facecolor(c)
        body.set_alpha(0.65)
        body.set_edgecolor("black")
        body.set_linewidth(0.5)

    for k in ["cmeans", "cmedians", "cbars", "cmaxes", "cmins"]:
        if k in parts:
            parts[k].set_edgecolor("black")
            parts[k].set_linewidth(0.9)

    # baseline mean line
    if vb.size:
        mbase = float(np.mean(vb))
        ax.axhline(mbase, color=COLORS["1991_2020"], linestyle=":", linewidth=1.3,
                   label=f"1991–2020 mean: {mbase:.1f}")

    # extreme lines (2023 / 2020 / 2021)
    for y, v in sorted(extreme_lines.items()):
        ax.axhline(v, color=COLORS.get(y, "black"), linestyle="--", linewidth=1.3,
                   label=f"{y}: {v:.1f}")

    # overlay model means as markers at each SSP position (skip baseline)
    if "model_norm" in dfv.columns and plotted_ssps:
        # baseline is pos[0] if present
        offset0 = 1 if vb.size else 0
        for j, ssp in enumerate(plotted_ssps):
            x0 = pos[offset0 + j]
            sub_ssp = dfv[dfv["scenario_norm"] == ssp]
            for m in MODEL_ORDER:
                sub_m = sub_ssp[sub_ssp["model_norm"] == m]
                if sub_m.empty:
                    continue
                mu = float(np.mean(sub_m["fire_days"].values.astype(float)))

                mk = MODEL_MARKER[m]
                if mk == "x":
                    # “x” 强制黑色（你说白色看不到）
                    ax.scatter(
                        x0 + MODEL_OFFSETS[m], mu,
                        marker="x", s=28,
                        color="black",
                        linewidths=1.0,
                        zorder=6
                    )
                else:
                    ax.scatter(
                        x0 + MODEL_OFFSETS[m], mu,
                        marker=mk, s=28,
                        facecolor="white",
                        edgecolor="black",
                        linewidth=0.9,
                        zorder=6
                    )

    ax.set_xticks(pos)
    ax.set_xticklabels(xticklabels)
    ax.grid(True, axis="y", alpha=0.22)

    # y label 只放左列（外面主函数会控制）
    # ax.set_ylabel("Annual potential days")

def plot_2x2_violin_pooledmodels(dataset: dict):
    base_dist = _load_baseline_dist(dataset["tags"][0])
    future_by_window, extreme_lines = _collect_future_and_extreme_lines(dataset)

    # —— 关键：图形“更高”，y 视觉更拉长 —— #
    # 你如果还嫌扁：把第二个数再加大，比如 (10, 9)
    fig, axes = plt.subplots(2, 2, figsize=(6, 5), constrained_layout=True)

    fig.suptitle(
        f"{dataset['name']} – Mid vs Late (violin; pooled models; SSP={'/'.join([s.replace('SSP','') for s in SSP_ORDER])})",
        fontsize=7, y=1.01
    )

    panels = [
        ("window_2070", "fire_probability", 0, 0, f"Late – {VAR_LABELS['fire_probability']}"),
        ("window_2070", "obe_probability",  0, 1, f"Late – {VAR_LABELS['obe_probability']}"),
        ("window_2040", "fire_probability", 1, 0, f"Mid – {VAR_LABELS['fire_probability']}"),
        ("window_2040", "obe_probability",  1, 1, f"Mid – {VAR_LABELS['obe_probability']}"),
    ]

    for wtag, var, r, c, title in panels:
        ax = axes[r, c]
        fut = future_by_window.get(wtag, pd.DataFrame())
        if fut is None or fut.empty:
            ax.set_title(title)
            ax.text(0.5, 0.5, "No future data", ha="center", va="center", transform=ax.transAxes)
            continue

        _panel_violin_mergedmodels(
            ax=ax,
            base_dist=base_dist,
            fut=fut,
            var=var,
            title=title,
            extreme_lines=extreme_lines[wtag][var],
        )

        if c == 0:
            ax.set_ylabel("Annual potential days")

    # legend（放图外更干净）
    scen_handles = [Line2D([0],[0], color=_scenario_color(s), lw=4, label=s.replace("SSP","SSP")) for s in SSP_ORDER]
    model_handles = [
        Line2D([0],[0], marker=MODEL_MARKER[m], color="black", lw=0,
               markerfacecolor=("white" if MODEL_MARKER[m] != "x" else "none"),
               markersize=6, label=m)
        for m in MODEL_ORDER
    ]
    extra = [
        Line2D([0],[0], color=COLORS["1991_2020"], lw=1.3, linestyle=":", label="1991–2020 mean"),
        Line2D([0],[0], color="black", lw=1.3, linestyle="--", label="Extreme-year line(s)"),
    ]

    fig.legend(
        handles=scen_handles + model_handles + extra,
        loc="upper left",
        bbox_to_anchor=(0.01, 0.99),
        frameon=True
    )

    safe = dataset["name"].replace(" ", "_").replace("/", "_").replace("(", "").replace(")", "").lower()
    out_png = os.path.join(OUT_DIR, f"{safe}_2x2_violin_pooledmodels.png")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_png.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_png}")

def main():
    for ds in DATASETS:
        try:
            plot_2x2_violin_pooledmodels(ds)
        except Exception as e:
            print(f"[SKIP] {ds['name']} -> {e}")

if __name__ == "__main__":
    main()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    # -*- coding: utf-8 -*-
import os, glob
from typing import Optional, Tuple, Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.transforms import blended_transform_factory

# ============================================================
# 0) CONFIG
# ============================================================

BASE_DIR_MAIN = r"E:\Projection paper"
BASE_DIR_SUP  = r"E:\Projection paper_sup_scens"

REGIONAL_MAIN = os.path.join(BASE_DIR_MAIN, "regional_outputs")
REGIONAL_SUP  = os.path.join(BASE_DIR_SUP,  "regional_outputs")

OUT_DIR = os.path.join(REGIONAL_MAIN, "figures_fig4_2x2_violin_pooledmodels_finalish_more")
os.makedirs(OUT_DIR, exist_ok=True)

DATASETS = [
    {"name": "Canada 2023",          "tags": ["canada_2023"],         "years": [2023],       "mode": "single"},
    {"name": "Western US 2020/2021", "tags": ["us_2020","us_2021"],   "years": [2020, 2021], "mode": "combined"},
]

WINDOWS = {
    "window_2040": {"label": "Mid-Century (2041–2070)"},
    "window_2070": {"label": "Late-Century (2071–2100)"},
}
WINDOW_YEAR_RANGE = {
    "window_2040": (2041, 2070),
    "window_2070": (2071, 2100),
}

VARIABLES = ["fire_probability", "obe_probability"]
VAR_LABELS = {
    "fire_probability": "Active burning (ABDp)",
    "obe_probability": "Overnight burning extremes (OBEp)"
}

# ---- toggle SSP585 on/off ----
INCLUDE_SSP585 = True
SSP_ORDER_ALL = ["SSP126", "SSP245", "SSP370", "SSP585"]
SSP_ORDER = SSP_ORDER_ALL if INCLUDE_SSP585 else ["SSP126", "SSP245", "SSP370"]

# 颜色（你现有配色）
COLORS = {
    "SSP126": "#2ca02c",
    "SSP245": "#1f77b4",
    "SSP370": "#ff7f0e",
    "SSP585": "#d62728",
    "1991_2020": "#808080",
    2020: "#8b0000",
    2021: "#ff1493",
    2023: "#dc143c",
}

# 模型 + marker（回到你原来的风格：白心黑边；x 纯黑）
MODEL_ORDER = ["CANESM", "ECEARTH", "GFDL", "UKESM"]
MODEL_MARKER = {"CANESM": "o", "ECEARTH": "^", "GFDL": "s", "UKESM": "x"}
MODEL_OFFSETS = {"CANESM": -0.24, "ECEARTH": -0.08, "GFDL": 0.08, "UKESM": 0.24}

# ---- 你要的行为开关 ----
SHOW_PERCENTILE_TEXT = True

# 这个就是你犹豫的：一个 pooled violin 叠加“每个模型/每年”的小点（可能 120 个/面板）
SHOW_INDIVIDUAL_POINTS = True   # <---- 想试就 True
POINT_SIZE  = 6                 # 很小
POINT_ALPHA = 0.12              # 很淡
POINT_JITTER = 0.025            # 横向抖动幅度（别太大，不然看着乱）
POINT_SEED = 1234               # 固定随机种子保证复现

# 极端年份 + baseline mean 的数值标注（放左侧）
LABEL_LINES_LEFT = False
LINE_LABEL_FS = 6
LINE_LABEL_BOX_ALPHA = 0.7

# percentile 文本字号
PCT_TEXT_FS = 5

# ---- 全局字体：最大不超过 7 ----
plt.rcParams.update({
    "font.family": "Arial",
    "font.sans-serif": ["Arial"],
    "axes.unicode_minus": False,

    "font.size": 7,
    "axes.titlesize": 7,
    "axes.labelsize": 7,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "legend.fontsize": 6,
})

_df_cache: Dict[str, pd.DataFrame] = {}

def _read_csv_cached(path: str) -> pd.DataFrame:
    if path not in _df_cache:
        _df_cache[path] = pd.read_csv(path)
    return _df_cache[path]

def _find_annual_file(regional_dir: str, region_tag: str, window_tag: str, year: int) -> str:
    folder = os.path.join(regional_dir, region_tag, window_tag)
    patt = os.path.join(folder, f"annual_fire_days_*_{year}.csv")
    m = glob.glob(patt)
    if m:
        return m[0]
    folder2 = os.path.join(regional_dir, region_tag)
    patt2 = os.path.join(folder2, f"annual_fire_days_*_{year}.csv")
    m2 = glob.glob(patt2)
    if m2:
        return m2[0]
    raise FileNotFoundError(f"No annual CSV for {region_tag}-{window_tag}-{year}")

def _find_baseline_1991_2020(regional_dir: str, region_tag: str) -> str:
    folder = os.path.join(regional_dir, region_tag)
    matches = glob.glob(os.path.join(folder, "**", "*1991_2020_baseline.csv"), recursive=True)
    if not matches:
        raise FileNotFoundError(f"No 1991_2020 baseline CSV under {folder}")
    matches.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return matches[0]

def _subset_future_by_window(df_overall: pd.DataFrame, window_tag: str) -> pd.DataFrame:
    y0, y1 = WINDOW_YEAR_RANGE[window_tag]
    df_future = df_overall[df_overall["scenario"] != "baseline"].copy()
    if "year" in df_future.columns:
        df_future = df_future[(df_future["year"] >= y0) & (df_future["year"] <= y1)]
    return df_future

def _norm_scenario(s: str) -> str:
    u = str(s).upper()
    for k in SSP_ORDER_ALL:
        if u.startswith(k):
            return k
    return u

def _norm_model(m: str) -> Optional[str]:
    if m is None:
        return None
    u = str(m).upper()
    if "CANESM" in u: return "CANESM"
    if "ECEARTH" in u or "EC-EARTH" in u or ("EC" in u and "EARTH" in u): return "ECEARTH"
    if "GFDL" in u: return "GFDL"
    if "UKESM" in u: return "UKESM"
    return None

def _scenario_color(ssp: str) -> str:
    return COLORS.get(ssp, "#999999")

def _precise_percentile(vals: np.ndarray, x: float) -> float:
    vals = np.asarray(vals, dtype=float)
    n = len(vals)
    if n == 0:
        return np.nan
    rank = np.sum(vals < x) + 0.5 * np.sum(vals == x)
    return rank / n * 100.0

def _load_overall_for_year(region_tag: str, window_tag: str, year: int) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    main = None
    sup = None
    try:
        f1 = _find_annual_file(REGIONAL_MAIN, region_tag, window_tag, year)
        df1 = _read_csv_cached(f1)
        main = df1[df1["region"] == "overall"].copy()
        main["__source__"] = "MAIN"
    except FileNotFoundError:
        pass
    try:
        f2 = _find_annual_file(REGIONAL_SUP, region_tag, window_tag, year)
        df2 = _read_csv_cached(f2)
        sup = df2[df2["region"] == "overall"].copy()
        sup["__source__"] = "SUP"
    except FileNotFoundError:
        pass
    return main, sup

def _merge_future(main: Optional[pd.DataFrame], sup: Optional[pd.DataFrame], window_tag: str) -> pd.DataFrame:
    frames = []
    if main is not None:
        frames.append(_subset_future_by_window(main, window_tag))
    if sup is not None:
        frames.append(_subset_future_by_window(sup, window_tag))
    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)

    keys = [c for c in ["scenario", "model", "year", "variable", "region"] if c in df.columns]
    if keys and "__source__" in df.columns:
        df["__prio__"] = df["__source__"].map({"MAIN": 0, "SUP": 1}).fillna(9)
        df = df.sort_values("__prio__").drop_duplicates(subset=keys, keep="first")
        df = df.drop(columns=["__prio__"], errors="ignore")

    df["scenario_norm"] = df["scenario"].apply(_norm_scenario) if "scenario" in df.columns else ""
    df["model_norm"] = df["model"].apply(_norm_model) if "model" in df.columns else None
    return df

def _load_baseline_dist(region_tag: str) -> pd.DataFrame:
    f = _find_baseline_1991_2020(REGIONAL_MAIN, region_tag)
    df = _read_csv_cached(f)
    return df[df["region"] == "overall"].copy()

def _collect_future_and_extreme_lines(dataset: dict):
    tags = dataset["tags"]
    years = dataset["years"]
    mode = dataset["mode"]

    future_by_window: Dict[str, pd.DataFrame] = {w: pd.DataFrame() for w in WINDOWS.keys()}
    extreme_lines: Dict[str, Dict[str, Dict[int, float]]] = {w: {v: {} for v in VARIABLES} for w in WINDOWS.keys()}

    for wtag in WINDOWS.keys():
        frames = []

        if mode == "single":
            region_tag, y = tags[0], years[0]
            m, s = _load_overall_for_year(region_tag, wtag, y)
            df_for_base = m if m is not None else s
            if df_for_base is not None:
                base = df_for_base[df_for_base["scenario"] == "baseline"]
                for var in VARIABLES:
                    vv = base[base["variable"] == var]["fire_days"].values
                    if vv.size:
                        extreme_lines[wtag][var][y] = float(vv[0])

            fut = _merge_future(m, s, wtag)
            if not fut.empty:
                frames.append(fut)

        else:
            # combined: keep both extreme lines (2020 & 2021), future only from first tag/year
            for i, (region_tag, y) in enumerate(zip(tags, years)):
                m, s = _load_overall_for_year(region_tag, wtag, y)
                df_for_base = m if m is not None else s
                if df_for_base is not None:
                    base = df_for_base[df_for_base["scenario"] == "baseline"]
                    for var in VARIABLES:
                        vv = base[base["variable"] == var]["fire_days"].values
                        if vv.size:
                            extreme_lines[wtag][var][y] = float(vv[0])

                if i == 0:
                    fut = _merge_future(m, s, wtag)
                    if not fut.empty:
                        frames.append(fut)

        future_by_window[wtag] = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    return future_by_window, extreme_lines

def _compute_percentile_summary_for_extreme(fut: pd.DataFrame, var: str, extreme_value: float):
    """
    返回：{ssp: (mean, min, max)}，percentile = extreme_value 在未来分布(按model分)中的 percentile
    """
    out = {}
    dfv = fut[fut["variable"] == var].copy()
    if dfv.empty:
        return out

    for ssp in SSP_ORDER:
        sub = dfv[dfv["scenario_norm"] == ssp]
        if sub.empty:
            continue

        pvals = []
        for m in MODEL_ORDER:
            sub_m = sub[sub["model_norm"] == m]
            vals = sub_m["fire_days"].values.astype(float)
            if vals.size == 0:
                continue
            pvals.append(_precise_percentile(vals, extreme_value))

        if not pvals:
            continue

        pvals = np.array(pvals, dtype=float)
        out[ssp] = (float(np.nanmean(pvals)), float(np.nanmin(pvals)), float(np.nanmax(pvals)))
    return out

def _annotate_lines_left(ax, baseline_mean: Optional[float], extreme_lines: Dict[int, float]):
    """
    左侧标注 baseline mean + 每条 extreme-year 值
    """
    trans = blended_transform_factory(ax.transAxes, ax.transData)

    # 为避免重叠：按 y 排序，近的稍微上移
    items = []
    if baseline_mean is not None and np.isfinite(baseline_mean):
        items.append(("BASE", baseline_mean))
    for y, v in extreme_lines.items():
        items.append((str(y), v))
    items.sort(key=lambda kv: kv[1])

    y0, y1 = ax.get_ylim()
    min_sep = 0.035 * (y1 - y0)
    last_y = None

    for key, v in items:
        yy = v
        if last_y is not None and abs(yy - last_y) < min_sep:
            yy = last_y + min_sep

        if key == "BASE":
            txt = f"1991–2020 mean: {v:.1f}"
            c = COLORS["1991_2020"]
        else:
            txt = f"{key}: {v:.1f}"
            c = COLORS.get(int(key), "black") if key.isdigit() else "black"

        ax.text(
            0.01, yy, txt,
            transform=trans,
            ha="left", va="bottom",
            fontsize=LINE_LABEL_FS,
            color=c,
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=LINE_LABEL_BOX_ALPHA),
            zorder=10
        )
        last_y = yy

def _overlay_individual_points(ax, dfv: pd.DataFrame, ssp: str, x0: float):
    """
    叠加每个模型的所有年度值：很小、很淡、带一点 jitter
    """
    rng = np.random.default_rng(POINT_SEED + hash(ssp) % 10000)
    ssp_c = _scenario_color(ssp)

    for m in MODEL_ORDER:
        sub_m = dfv[(dfv["scenario_norm"] == ssp) & (dfv["model_norm"] == m)]
        if sub_m.empty:
            continue
        yvals = sub_m["fire_days"].values.astype(float)
        if yvals.size == 0:
            continue

        jit = rng.uniform(-POINT_JITTER, POINT_JITTER, size=yvals.size)
        xx = x0 + MODEL_OFFSETS[m] + jit

        mk = MODEL_MARKER[m]
        if mk == "x":
            ax.scatter(xx, yvals, marker="x", s=POINT_SIZE, color="black",
                       alpha=POINT_ALPHA, linewidths=0.7, zorder=4)
        else:
            ax.scatter(xx, yvals, marker=mk, s=POINT_SIZE,
                       facecolor=ssp_c, edgecolor="black",
                       alpha=POINT_ALPHA, linewidth=0.5, zorder=4)

def _panel(ax, dataset: dict, base_dist: pd.DataFrame, fut: pd.DataFrame,
           window_tag: str, var: str, title: str, extreme_lines: Dict[int, float]):

    ax.set_title(title, pad=5)

    violins: List[np.ndarray] = []
    pos: List[float] = []
    vcolors: List[str] = []
    xticklabels: List[str] = []

    x = 0.0

    # baseline violin
    vb = base_dist[base_dist["variable"] == var]["fire_days"].values.astype(float)
    has_base = vb.size > 0
    baseline_mean = float(np.mean(vb)) if has_base else None

    if has_base:
        violins.append(vb)
        pos.append(x)
        vcolors.append(COLORS["1991_2020"])
        xticklabels.append("Baseline")
        x += 1.35

    # future pooled violins per SSP
    dfv = fut[fut["variable"] == var].copy()
    plotted_ssps = []

    for ssp in SSP_ORDER:
        sub = dfv[dfv["scenario_norm"] == ssp]
        if sub.empty:
            continue
        vals = sub["fire_days"].values.astype(float)
        if vals.size == 0:
            continue
        violins.append(vals)
        pos.append(x)
        vcolors.append(_scenario_color(ssp))
        xticklabels.append(ssp.replace("SSP", ""))
        plotted_ssps.append(ssp)
        x += 1.0

    if not violins:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return

    parts = ax.violinplot(
        violins,
        positions=pos,
        widths=0.60,
        showmeans=True,
        showmedians=True,
        showextrema=True
    )

    for body, c in zip(parts["bodies"], vcolors):
        body.set_facecolor(c)
        body.set_alpha(0.65)
        body.set_edgecolor("black")
        body.set_linewidth(0.5)

    for k in ["cmeans", "cmedians", "cbars", "cmaxes", "cmins"]:
        if k in parts:
            parts[k].set_edgecolor("black")
            parts[k].set_linewidth(0.9)

    # baseline mean line
    if has_base and baseline_mean is not None:
        ax.axhline(baseline_mean, color=COLORS["1991_2020"], linestyle=":", linewidth=1.3)

    # extreme-year lines
    for y, v in sorted(extreme_lines.items()):
        ax.axhline(v, color=COLORS.get(y, "black"), linestyle="--", linewidth=1.3)

    # —— 左侧写 baseline mean + extreme years 数值 —— #
    if LABEL_LINES_LEFT:
        _annotate_lines_left(ax, baseline_mean=baseline_mean, extreme_lines=extreme_lines)

    # —— 可选：叠加所有年度值的小点（很淡很小） —— #
    if SHOW_INDIVIDUAL_POINTS and plotted_ssps and "model_norm" in dfv.columns:
        offset0 = 1 if has_base else 0
        for j, ssp in enumerate(plotted_ssps):
            x0 = pos[offset0 + j]
            _overlay_individual_points(ax, dfv=dfv, ssp=ssp, x0=x0)

    # overlay model means as original markers (white fill, black edge; x black)
    if plotted_ssps and "model_norm" in dfv.columns:
        offset0 = 1 if has_base else 0
        for j, ssp in enumerate(plotted_ssps):
            x0 = pos[offset0 + j]
            sub_ssp = dfv[dfv["scenario_norm"] == ssp]

            for m in MODEL_ORDER:
                sub_m = sub_ssp[sub_ssp["model_norm"] == m]
                if sub_m.empty:
                    continue
                mu = float(np.mean(sub_m["fire_days"].values.astype(float)))
                mk = MODEL_MARKER[m]

                if mk == "x":
                    ax.scatter(x0 + MODEL_OFFSETS[m], mu, marker="x", s=28,
                               color="black", linewidths=1.0, zorder=6)
                else:
                    ax.scatter(x0 + MODEL_OFFSETS[m], mu, marker=mk, s=28,
                               facecolor="white", edgecolor="black",
                               linewidth=0.9, zorder=6)

    # percentile text: mean (min-max)
    # percentile text: for Western US put 2020 on first line and 2021 on second line (no horizontal crowding)
    if SHOW_PERCENTILE_TEXT and plotted_ssps and extreme_lines:
        y0, y1 = ax.get_ylim()
        offset0 = 1 if has_base else 0
        years_sorted = sorted(extreme_lines.keys())  # e.g., [2020, 2021] or [2023]
    
        # bottom text band (relative to y-range)
        base_y = y0 + 0.035 * (y1 - y0)
    
        # line spacing
        line_gap = 0.055 * (y1 - y0)  # distance between the 2020/2021 lines
    
        for j, ssp in enumerate(plotted_ssps):
            x0 = pos[offset0 + j]
    
            if dataset["mode"] == "combined":
                # Two fixed lines: one per extreme year
                for k, yr in enumerate(years_sorted):
                    extv = extreme_lines[yr]
                    pct_map = _compute_percentile_summary_for_extreme(fut=fut, var=var, extreme_value=extv)
                    if ssp not in pct_map:
                        continue
                    meanp, pmin, pmax = pct_map[ssp]
                    y_txt = base_y + (len(years_sorted) - 1 - k) * line_gap  # top line=first year
    
                    txt = f"{meanp:.0f} ({pmin:.0f}-{pmax:.0f})"

                    ax.text(
                        x0, y_txt, txt,
                        ha="center", va="bottom",
                        fontsize=PCT_TEXT_FS,
                        color="black",
                        bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=0.55),
                        zorder=10
                    )
            else:
                # Single extreme year (e.g., Canada 2023): keep one-line style
                yr = years_sorted[0]
                extv = extreme_lines[yr]
                pct_map = _compute_percentile_summary_for_extreme(fut=fut, var=var, extreme_value=extv)
                if ssp not in pct_map:
                    continue
                meanp, pmin, pmax = pct_map[ssp]
                txt = f"{meanp:.0f} ({pmin:.0f}-{pmax:.0f})"
                ax.text(
                    x0, base_y, txt,
                    ha="center", va="bottom",
                    fontsize=PCT_TEXT_FS,
                    color="black",
                    bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=0.55),
                    zorder=10
                )


    ax.set_xticks(pos)
    ax.set_xticklabels(xticklabels)
    ax.grid(True, axis="y", alpha=0.22)

def plot_2x2(dataset: dict):
    base_dist = _load_baseline_dist(dataset["tags"][0])
    future_by_window, extreme_lines = _collect_future_and_extreme_lines(dataset)

    fig, axes = plt.subplots(2, 2, figsize=(6,5), constrained_layout=True)
    fig.suptitle(
        f"{dataset['name']} – Mid vs Late (pooled-model violin; SSP={'/'.join([s.replace('SSP','') for s in SSP_ORDER])})",
        fontsize=7, y=1.01
    )

    panels = [
        ("window_2070", "fire_probability", 0, 0, f"Late – {VAR_LABELS['fire_probability']}"),
        ("window_2070", "obe_probability",  0, 1, f"Late – {VAR_LABELS['obe_probability']}"),
        ("window_2040", "fire_probability", 1, 0, f"Mid – {VAR_LABELS['fire_probability']}"),
        ("window_2040", "obe_probability",  1, 1, f"Mid – {VAR_LABELS['obe_probability']}"),
    ]

    for wtag, var, r, c, title in panels:
        ax = axes[r, c]
        fut = future_by_window.get(wtag, pd.DataFrame())
        if fut is None or fut.empty:
            ax.set_title(title)
            ax.text(0.5, 0.5, "No future data", ha="center", va="center", transform=ax.transAxes)
            continue

        _panel(
            ax=ax,
            dataset=dataset,
            base_dist=base_dist,
            fut=fut,
            window_tag=wtag,
            var=var,
            title=title,
            extreme_lines=extreme_lines[wtag][var],
        )

        if c == 0:
            ax.set_ylabel("Annual potential days")

    # ============================================================
    # Legend: extreme years in legend with their own colors (你要的)
    # ============================================================

    scen_handles = [Line2D([0],[0], color=_scenario_color(s), lw=4, label=s) for s in SSP_ORDER]

    model_handles = [
        Line2D([0],[0], marker=MODEL_MARKER[m], color="black", lw=0,
               markerfacecolor=("white" if MODEL_MARKER[m] != "x" else "none"),
               markersize=6, label=m)
        for m in MODEL_ORDER
    ]

    baseline_handle = Line2D([0],[0], color=COLORS["1991_2020"], lw=1.3, linestyle=":", label="1991–2020 mean")

    # extreme-year handles depend on dataset
    years_for_legend = sorted(dataset["years"])
    extreme_handles = [
        Line2D([0],[0], color=COLORS.get(y, "black"), lw=1.3, linestyle="--", label=str(y))
        for y in years_for_legend
    ]

    fig.legend(
        handles=scen_handles + model_handles + [baseline_handle] + extreme_handles,
        loc="upper left",
        bbox_to_anchor=(0.01, 0.99),
        frameon=True
    )

    safe = dataset["name"].replace(" ", "_").replace("/", "_").replace("(", "").replace(")", "").lower()
    tag = "pointsON" if SHOW_INDIVIDUAL_POINTS else "pointsOFF"
    out_png = os.path.join(OUT_DIR, f"{safe}_2x2_violin_{tag}.png")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_png.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_png}")

def main():
    for ds in DATASETS:
        try:
            plot_2x2(ds)
        except Exception as e:
            print(f"[SKIP] {ds['name']} -> {e}")

if __name__ == "__main__":
    main()

















#%% canada 2023 vs baseline -biome level
# -*- coding: utf-8 -*-
import os
import glob
import pickle
import warnings
warnings.filterwarnings("ignore")

from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

# ============================================================
# 0) CONFIG
# ============================================================
E_BASE = r"E:\Projection paper"
OUT_DIR = r"E:\Projection paper\regional_outputs_boreal_biome_baseline_vs_2023\boreal_only"
os.makedirs(OUT_DIR, exist_ok=True)

# Input NetCDF (pick latest)
BASELINE_2023_GLOB      = os.path.join(E_BASE, "outputs_baseline", "baseline_2023_*.nc")
BASELINE_1991_2020_GLOB = os.path.join(E_BASE, "outputs_baseline", "baseline_1991_2020_*.nc")

BASELINE_2023_START = "2023-01-01"
BASELINE_1991_2020_START = "1991-01-01"

# Thresholds
THRESHOLDS_PKL = os.path.join(E_BASE, "models_skl161", "thresholds.pkl")
FALLBACK_THRESHOLDS = {
    "fire_probability": 0.4171,
    "obe_probability": 0.4233,
}

VARIABLES = ["fire_probability", "obe_probability"]
VAR_LABELS = {
    "fire_probability": "Annual ABDp",
    "obe_probability": "Annual OBEp",
}

# Region masks sources
CANADA_PROVINCE_PATH = r"D:\000_collections\020_Chapter2\Canada_PR_name.nc"
BIOME_PATH           = r"D:\000_collections\020_Chapter2\US_CAN_biome.nc"

# Exclude biomes (keep your original list)
EXCLUDE_BIOMES = [50.1, 90.1, 12.1, 13.1, 21.1, 22.1, 31.1, 35.1]

# Boreal-only (your requirement)
BOREAL_BIOME_CODES = [41.1, 41.2, 42.1, 42.2, 43.1]

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

# Plot style (max font <= 7)
plt.rcParams.update({
    "font.family": "Arial",
    "axes.unicode_minus": False,
    "font.size": 7,
    "axes.titlesize": 7,
    "axes.labelsize": 7,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
})

# ============================================================
# 1) UTILS
# ============================================================
def pick_latest(pattern: str) -> str:
    m = sorted(glob.glob(pattern))
    if not m:
        raise FileNotFoundError(f"No file matched: {pattern}")
    m.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return m[0]

def rebuild_time_coords(ds: xr.Dataset, start_date: str) -> xr.Dataset:
    nday = ds.sizes["time"]
    t = pd.date_range(start_date, periods=nday, freq="D")
    # remove leap day if present
    if len(t) > nday:
        t = t[~((t.month == 2) & (t.day == 29))]
    # ensure exact length
    if len(t) != nday:
        t = pd.date_range(start_date, periods=nday, freq="D")
        t = t[~((t.month == 2) & (t.day == 29))]
        t = t[:nday]
    return ds.assign_coords(time=t)

def load_thresholds() -> Dict[str, float]:
    thr = dict(FALLBACK_THRESHOLDS)
    try:
        if os.path.exists(THRESHOLDS_PKL):
            with open(THRESHOLDS_PKL, "rb") as f:
                obj = pickle.load(f)
            if "fire_threshold" in obj:
                thr["fire_probability"] = float(obj["fire_threshold"])
            if "obe_threshold" in obj:
                thr["obe_probability"] = float(obj["obe_threshold"])
    except Exception as e:
        print(f"[WARN] failed reading thresholds.pkl; using fallback. {e}")
    return thr

def open_align_mask_source(path: str, var: str) -> xr.DataArray:
    ds = xr.open_dataset(path)
    if "lon" in ds.dims:
        ds = ds.rename({"lon": "longitude", "lat": "latitude"})
    return ds[var]

def align_numeric_da_to_baseline_grid(da: xr.DataArray, baseline_ds: xr.Dataset) -> xr.DataArray:
    """interp requires numeric; ensure da is numeric before calling."""
    base_lons = baseline_ds.longitude.values
    base_lats = baseline_ds.latitude.values
    return da.interp(longitude=base_lons, latitude=base_lats, method="nearest")

def precise_percentile(vals: np.ndarray, x: float) -> float:
    vals = np.asarray(vals, dtype=float)
    n = vals.size
    if n == 0 or not np.isfinite(x):
        return np.nan
    rank = np.sum(vals < x) + 0.5 * np.sum(vals == x)
    return rank / n * 100.0

def kde_density(x: np.ndarray, gridsize: int = 256):
    """Try scipy gaussian_kde; fallback to smoothed histogram."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 3:
        return None, None
    xmin, xmax = np.min(x), np.max(x)
    pad = 0.08 * (xmax - xmin + 1e-9)
    grid = np.linspace(xmin - pad, xmax + pad, gridsize)

    try:
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(x)
        dens = kde(grid)
        return grid, dens
    except Exception:
        # fallback: histogram density + simple gaussian smoothing
        hist, edges = np.histogram(x, bins=40, density=True)
        centers = 0.5 * (edges[:-1] + edges[1:])
        # naive smoothing
        k = 9
        w = np.exp(-0.5 * (np.linspace(-2, 2, k) ** 2))
        w = w / w.sum()
        sm = np.convolve(hist, w, mode="same")
        # resample to grid
        dens = np.interp(grid, centers, sm, left=0.0, right=0.0)
        return grid, dens

# ============================================================
# 2) MASKS: Canada -> biome exclude -> boreal -> biome-level
# ============================================================
def build_canada_boreal_biome_masks(baseline_ds: xr.Dataset) -> Tuple[Dict[float, np.ndarray], List[float]]:
    # 1) Canada mask from province names (object) -> convert to numeric BEFORE interp
    prov = open_align_mask_source(CANADA_PROVINCE_PATH, "PRENAME")  # object/string
    prov_vals = prov.values

    # numeric canada mask (True where province name is non-empty)
    canada_mask_src = np.array([(v is not None) and (str(v).strip() != "") for v in prov_vals.ravel()],
                               dtype=bool).reshape(prov_vals.shape)
    canada_da_num = xr.DataArray(
        canada_mask_src.astype(np.float32),
        coords={"latitude": prov.latitude.values, "longitude": prov.longitude.values},
        dims=["latitude", "longitude"],
        name="canada_mask_num"
    )
    canada_aligned = align_numeric_da_to_baseline_grid(canada_da_num, baseline_ds).values > 0.5

    # 2) Biome codes (numeric) aligned
    bio = open_align_mask_source(BIOME_PATH, "gez_code_id")
    bio_aligned = align_numeric_da_to_baseline_grid(bio.astype(np.float32), baseline_ds)
    biome_vals = bio_aligned.values.astype(float)

    # ★ robust: round to 1 decimal to avoid float artifacts (50.100002 etc)
    biome_codes = np.round(biome_vals, 1)

    # 3) biome exclude (on rounded codes)
    biome_mask = canada_aligned & (~np.isnan(biome_codes))
    for v in EXCLUDE_BIOMES:
        biome_mask &= (biome_codes != float(v))

    # 4) boreal mask
    boreal_mask = biome_mask & np.isin(biome_codes, np.array(BOREAL_BIOME_CODES, dtype=float))

    # 5) biome-level masks (boreal-only)
    masks: Dict[float, np.ndarray] = {}
    for code in BOREAL_BIOME_CODES:
        m = boreal_mask & (biome_codes == float(code))
        masks[float(code)] = m

    return masks, [float(c) for c in BOREAL_BIOME_CODES]

# ============================================================
# 3) COMPUTE: baseline 1991–2020 distribution + 2023 value
# ============================================================
def avg_over_mask(arr2d: np.ndarray, mask: np.ndarray) -> float:
    v = arr2d[mask]
    if v.size == 0:
        return np.nan
    return float(np.nanmean(v))

def compute_baseline_distribution(ds_1991_2020: xr.Dataset,
                                 masks: Dict[float, np.ndarray],
                                 biome_codes: List[float],
                                 thresholds: Dict[str, float]) -> pd.DataFrame:
    times = pd.DatetimeIndex(ds_1991_2020.time.values)
    years = np.unique(times.year)

    rows = []
    for yr in years:
        idx = np.where(times.year == yr)[0]
        if idx.size == 0:
            continue
        for var in VARIABLES:
            thr = thresholds[var]
            data = ds_1991_2020[var].isel(time=idx).values  # [t, lat, lon]
            exceed = np.sum(data > thr, axis=0)  # [lat, lon]
            for b in biome_codes:
                rows.append({
                    "year": int(yr),
                    "biome_code": float(b),
                    "variable": var,
                    "value": avg_over_mask(exceed, masks[b]),
                })
    return pd.DataFrame(rows)

def compute_2023_values(ds_2023: xr.Dataset,
                        masks: Dict[float, np.ndarray],
                        biome_codes: List[float],
                        thresholds: Dict[str, float]) -> pd.DataFrame:
    rows = []
    for var in VARIABLES:
        thr = thresholds[var]
        data = ds_2023[var].values  # [t, lat, lon]
        exceed = np.sum(data > thr, axis=0)
        for b in biome_codes:
            rows.append({
                "year": 2023,
                "biome_code": float(b),
                "variable": var,
                "value": avg_over_mask(exceed, masks[b]),
            })
    return pd.DataFrame(rows)

# ============================================================
# 4) PLOT: boreal-only, no overall
# ============================================================
def plot_boreal_baseline_vs_2023(df_base: pd.DataFrame, df_2023: pd.DataFrame, biome_codes: List[float]):
    n = len(biome_codes)
    fig, axes = plt.subplots(nrows=n, ncols=2, figsize=(7.2, 0.95 * n + 0.7), constrained_layout=True)

    fig.suptitle("Canada boreal biome-level baseline (1991–2020) vs 2023", y=1.01, fontsize=7)

    for i, code in enumerate(biome_codes):
        name = biome_name_map.get(code, f"biome_{code:.1f}")

        for j, var in enumerate(VARIABLES):
            ax = axes[i, j] if n > 1 else axes[j]
            vals = df_base[(df_base["biome_code"] == code) & (df_base["variable"] == var)]["value"].values.astype(float)
            v2023 = float(df_2023[(df_2023["biome_code"] == code) & (df_2023["variable"] == var)]["value"].values[0])

            grid, dens = kde_density(vals)
            if grid is not None:
                ax.fill_between(grid, dens, color="#d9d9d9", alpha=0.85, linewidth=0.0)
                ax.plot(grid, dens, color="#4d4d4d", linewidth=1.0)
            else:
                ax.text(0.5, 0.5, "No baseline data", transform=ax.transAxes,
                        ha="center", va="center", fontsize=7)

            ax.axvline(v2023, color="crimson", linewidth=1.8)

            pct = precise_percentile(vals, v2023)
            ax.text(0.98, 0.90, f"2023 percentile: {pct:.0f}",
                    transform=ax.transAxes, ha="right", va="top", fontsize=6)

            ax.set_title(f"{name} — {('ABDp' if var=='fire_probability' else 'OBEp')}", pad=2)
            ax.grid(True, axis="both", alpha=0.20)

            if i == n - 1:
                ax.set_xlabel(VAR_LABELS[var])
            else:
                ax.set_xlabel("")
            ax.set_ylabel("Density")

    out_png = os.path.join(OUT_DIR, "canada_boreal_biome_baseline_1991_2020_vs_2023.png")
    out_pdf = out_png.replace(".png", ".pdf")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved:\n  {out_png}\n  {out_pdf}")

# ============================================================
# 5) MAIN
# ============================================================
def main():
    f2023 = pick_latest(BASELINE_2023_GLOB)
    fbase = pick_latest(BASELINE_1991_2020_GLOB)

    print(f"[INFO] OUT_DIR: {OUT_DIR}")
    print(f"[INFO] Using baseline-2023:      {os.path.basename(f2023)}")
    print(f"[INFO] Using baseline-1991_2020: {os.path.basename(fbase)}")

    thr = load_thresholds()
    print(f"[INFO] thresholds: {thr}")

    ds2023 = rebuild_time_coords(xr.open_dataset(f2023), BASELINE_2023_START)
    dsbase = rebuild_time_coords(xr.open_dataset(fbase), BASELINE_1991_2020_START)

    print("[INFO] Building masks: Canada -> biome exclude -> boreal -> biome-level")
    masks, boreal_codes = build_canada_boreal_biome_masks(ds2023)  # use 2023 grid as baseline grid

    print(f"[INFO] Boreal biomes: {boreal_codes}")

    print("[INFO] Computing baseline (1991–2020) distributions…")
    df_base = compute_baseline_distribution(dsbase, masks, boreal_codes, thr)

    print("[INFO] Computing 2023 values…")
    df_2023 = compute_2023_values(ds2023, masks, boreal_codes, thr)

    # Save data
    df_base.to_csv(os.path.join(OUT_DIR, "baseline_1991_2020_boreal_biome_level.csv"), index=False)
    df_2023.to_csv(os.path.join(OUT_DIR, "year_2023_boreal_biome_level.csv"), index=False)

    print("[INFO] Plotting…")
    plot_boreal_baseline_vs_2023(df_base, df_2023, boreal_codes)

if __name__ == "__main__":
    main()




#%%heatmap

# -*- coding: utf-8 -*-
# Boreal biomes – KDE of exceed-days (ABDp & OBEp)
# Fixes:
# 1) 严格按窗口过滤（2040/2070），不再混用两个窗口
# 2) 规范 scenario → {BASELINE, SSP245, SSP370} + window_tag ∈ {2040, 2070}
# 3) 统一模型名，兜底 ECEARTH/EC-EARTH 等写法
# 4) 百分比 = (未来均值 - baseline均值) / baseline均值 * 100，仅用同一 csv

import os, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# ───────── Paths ─────────
BASE = r"E:\Projection paper"
PATH_SCEN = os.path.join(BASE, r"analysis\tables\annual_exceed_days.csv")
PATH_YSEL = os.path.join(BASE, r"analysis\tables\annual_exceed_days_selected_years.csv")
OUT_DIR   = os.path.join(BASE, r"regional_outputs\figures\boreal_kde")
os.makedirs(OUT_DIR, exist_ok=True)

# ───────── Targets ─────────
BIOMES = [
    'Boreal coniferous forest east',
    'Boreal coniferous forest west',
    'Boreal tundra woodland east',
    'Boreal tundra woodland west',
    'Boreal mountain system',
]
VARIABLES = {
    'fire_probability': 'Active burning (ABDp)',
    'obe_probability' : 'Overnight burning extremes (OBEp)',
}
MODELS = ['CanESM', 'ECEarth', 'GFDL', 'UKESM']  # 固定顺序

# ───────── Style ─────────
plt.rcParams.update({'font.family': 'Arial', 'font.size': 9})
COLORS = {
    'Baseline': '#666666',
    'CanESM'  : '#1f77b4',  # 蓝
    'ECEarth' : '#ff7f0e',  # 橙
    'GFDL'    : '#2ca02c',  # 绿
    'UKESM'   : '#bc80bd',  # 紫
    '2023'    : '#E74C3C',  # 红
}

# ───────── Helpers ─────────
def _pick_value_column(df: pd.DataFrame) -> str:
    """支持 fire_days / exceed_days / value 三选一"""
    for c in ['fire_days', 'exceed_days', 'value']:
        if c in df.columns:
            return c
    raise ValueError("No numeric column found (expect one of: fire_days / exceed_days / value).")

def _norm_model(m: str) -> str:
    """把各种写法统一到 {CanESM, ECEarth, GFDL, UKESM}"""
    if m is None:
        return ''
    s = re.sub(r'[^A-Za-z0-9]+', '', str(m).upper())
    if 'CANESM' in s:
        return 'CanESM'
    if 'ECEARTH' in s or s.startswith('ECE') or 'ECE' in s:  # 兜住 ECEARTH/EC-EARTH/EC_EARTH 等
        return 'ECEarth'
    if 'GFDL' in s:
        return 'GFDL'
    if 'UKESM' in s:
        return 'UKESM'
    return str(m)

def _parse_scenario_and_window(s: str):
    """
    接受诸如 'Baseline' / 'baseline' / '1991-2020' / '2040_245' / '2070_370' 等，
    返回 (scenario, window_tag) 其中 scenario ∈ {BASELINE, SSP245, SSP370}；window_tag ∈ {'2040','2070',None}
    """
    if s is None:
        return 'BASELINE', None
    u = str(s).upper()
    # baseline
    if 'BASE' in u or 'HIST' in u or '1991' in u or 'HISTORICAL' in u:
        return 'BASELINE', None
    # window
    win = None
    if '2040' in u:
        win = '2040'
    elif '2070' in u:
        win = '2070'
    # scenario code
    if '245' in u:
        return 'SSP245', win
    if '370' in u or '3-7' in u or '3_7' in u:
        return 'SSP370', win
    if '585' in u:
        return 'SSP585', win  # 以防后续扩展
    # 默认兜底
    return u, win

def kde_xy(arr, bw_adj=1.0, n=400):
    arr = np.asarray(arr, float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.r_[0.0], np.r_[0.0]
    kde = gaussian_kde(arr)
    kde.set_bandwidth(kde.factor * bw_adj)
    h   = kde.factor * arr.std(ddof=1)
    xs  = np.linspace(arr.min() - 3*h, arr.max() + 3*h, n)
    return xs, kde(xs)

def read_tables():
    # scenarios (baseline + futures)
    df_s = pd.read_csv(PATH_SCEN)
    val_col = _pick_value_column(df_s)
    df_s = df_s.rename(columns={val_col: 'exceed_days'})

    # 规范 scenario 与 window_tag
    scen_win = df_s['scenario'].apply(_parse_scenario_and_window)
    df_s['scenario']   = [sw[0] for sw in scen_win]
    df_s['window_tag'] = [sw[1] for sw in scen_win]

    # 模型名规范
    if 'model' in df_s.columns:
        df_s['model'] = df_s['model'].map(_norm_model)
    else:
        df_s['model'] = ''

    # selected years (仅提供 2023 竖线)
    df_y = pd.read_csv(PATH_YSEL)
    val_y = _pick_value_column(df_y)
    df_y = df_y.rename(columns={val_y: 'exceed_days'})
    if 'year' not in df_y.columns:
        raise ValueError("annual_exceed_days_selected_years.csv must have a 'year' column.")
    return df_s, df_y

def panel(ax, df_s, df_y, biome, var, scenario_code, window_tag):
    """画一个小面板：baseline + 4 模型 + 2023 竖线；右上角写 Δ%（相对 baseline 均值）"""
    # baseline 分布（1991–2020）
    base = df_s.query(
        "biome_name==@biome and scenario=='BASELINE' and variable==@var"
    )['exceed_days'].values
    xs, ys = kde_xy(base, bw_adj=1.0)
    ax.plot(xs, ys, color=COLORS['Baseline'], lw=2.5, label='Baseline')
    base_mean = float(np.nanmean(base)) if base.size else np.nan

    # future 分布（严格过滤指定窗口）
    pct_lines = []
    sub = df_s.query(
        "biome_name==@biome and scenario==@scenario_code and variable==@var"
    ).copy()
    sub = sub[sub['window_tag'] == window_tag]   # 关键修复：总是按 2040/2070 过滤

    for mdl in MODELS:
        vals = sub.loc[sub['model'] == mdl, 'exceed_days'].values
        if vals.size:
            xs, ys = kde_xy(vals, bw_adj=1.0)
            ax.plot(xs, ys, color=COLORS[mdl], lw=2.0, label=mdl)
            if np.isfinite(base_mean) and base_mean != 0:
                pct = (np.mean(vals) - base_mean) / base_mean * 100.0
                pct_lines.append(f"{mdl}: {pct:+.0f}%")

    # 2023 竖线（仅参考，不参与计算）
    x23 = df_y.query(
        "year==2023 and biome_name==@biome and variable==@var"
    )['exceed_days'].values
    if x23.size:
        ax.axvline(x23[0], color=COLORS['2023'], lw=2.2, zorder=3)

    # Δ% 文本
    for i, txt in enumerate(pct_lines):
        ax.text(0.98, 0.95 - i*0.12, txt,
                transform=ax.transAxes, ha='right', va='top', fontsize=7)

    # 美化
    ax.spines[['top','right']].set_visible(False)
    ax.grid(axis='y', alpha=.3, lw=.5)

def plot_grid(df_s, df_y, window_tag, scenario_code, out_png):
    nrow, ncol = len(BIOMES), 2
    fig, axes = plt.subplots(nrow, ncol, figsize=(10.5, 2.6*nrow), sharey=False)
    plt.subplots_adjust(hspace=0.55, wspace=0.08,
                        left=0.07, right=0.98, top=0.90, bottom=0.10)

    # 标题
    win_label = 'Mid-Century (2041–2070)' if window_tag=='2040' else 'Late-Century (2071–2100)'
    fig.suptitle(f"Boreal biomes – KDE of exceed-days | {win_label} {scenario_code}",
                 fontsize=14, weight='bold', y=0.97)

    # 绘图
    for r, biome in enumerate(BIOMES):
        for c, (var, vlabel) in enumerate(VARIABLES.items()):
            ax = axes[r, c]
            panel(ax, df_s, df_y, biome, var, scenario_code, window_tag)
            # 轴标签 & 小标题
            if r == nrow-1:
                xl = "Annual ABDp — exceed-days / year" if var=='fire_probability' \
                     else "Annual OBEp — exceed-days / year"
                ax.set_xlabel(xl)
            if c == 0:
                ax.set_ylabel("Density")
                ax.set_title(biome, loc='left', fontsize=10, weight='bold')
            else:
                ax.set_title("")

    # 全局图例（含 2023）
    axes[0,0].plot([], [], color=COLORS['2023'], lw=2.2, label='2023')
    handles, labels = axes[0,0].get_legend_handles_labels()
    want = ['Baseline'] + MODELS + ['2023']
    order = [labels.index(x) for x in want if x in labels]
    handles = [handles[i] for i in order]
    labels  = [labels[i]  for i in order]
    fig.legend(handles, labels, loc='upper center',
               ncol=6, frameon=False, fontsize=9, bbox_to_anchor=(0.5, 0.905))

    # 保存
    fig.savefig(out_png, dpi=300, bbox_inches='tight')
    fig.savefig(out_png.replace('.png', '.pdf'), bbox_inches='tight')
    print("Saved:", out_png)
    plt.close(fig)

# ───────── Main ─────────
def main():
    df_s, df_y = read_tables()

    # 简要检查
    print("\n[INFO] Models after normalization (overall):")
    print(df_s['model'].value_counts(dropna=False))
    print("\n[INFO] Scenario×Window counts:")
    print(df_s.groupby(['scenario','window_tag']).size())

    # 2040/2070 × SSP245/SSP370
    for wtag in ['2040', '2070']:
        for scn in ['SSP245', 'SSP370']:
            out = os.path.join(OUT_DIR, f"boreal_kde_window{wtag}_{scn}.png")
            plot_grid(df_s, df_y, wtag, scn, out)

if __name__ == "__main__":
    main()






# -*- coding: utf-8 -*-
"""
Biome heatmaps (two panels: SSP2-4.5 vs SSP3-7.0), per window (2040 / 2070).
- Preferred input: avg30_exceed_days.csv
- Fallback: annual_exceed_days.csv (auto-aggregate to 30-yr means)
Outputs (PNG+PDF):
  E:\Projection paper\analysis\tables\figures\heatmaps\biome_changes_<window>.png/pdf

Updates in this version:
1) Force BOTH windows to use the 2070-based color range (consistent colorbar).
2) Reduce figure height by ~30% (from 5 to 3.5 inches).
"""

import os, glob
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

# ======== Config ========
BASE = r"E:\Projection paper\analysis\tables"
OUT_DIR = os.path.join(BASE, "figures", "heatmaps_supplementary")
os.makedirs(OUT_DIR, exist_ok=True)

# 哪个窗口要画；都要就写 ["window_2040", "window_2070"]
WINDOWS = ["window_2040", "window_2070"]

VARIABLES = ["fire_probability", "obe_probability"]
VAR_LABEL = {"fire_probability": "ABDp", "obe_probability": "OBEp"}

SELECTED_BIOMES = [
    'Subtropical steppe', 'Subtropical desert', 'Subtropical mountain system',
    'Temperate steppe', 'Temperate desert', 'Temperate mountain system west',
    'Boreal coniferous forest east', 'Boreal coniferous forest west',
    'Boreal tundra woodland east', 'Boreal tundra woodland west',
    'Boreal mountain system'
]

MODELS = ["CanESM", "ECEarth", "GFDL", "UKESM"]

# ======== Helpers ========
def _find_csv(name):
    """在 BASE 下递归找指定文件名，取最近修改的一个"""
    matches = glob.glob(os.path.join(BASE, "**", name), recursive=True)
    if not matches:
        return None
    matches.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return matches[0]

def _load_avg30():
    p = _find_csv("avg30_exceed_days.csv")
    if p:
        df = pd.read_csv(p)
        rename = {c: c.strip() for c in df.columns}
        df = df.rename(columns=rename)
        return df
    return None

def _load_annual_and_aggregate():
    p = _find_csv("annual_exceed_days.csv")
    if not p:
        return None
    df = pd.read_csv(p)
    # baseline 30 年均
    base = (
        df[df["epoch"] == "baseline"]
        .groupby(["biome_name", "variable"], as_index=False)["exceed_days"].mean()
        .rename(columns={"exceed_days": "mean_30yr"})
    )
    base["epoch"] = "baseline"
    base["scenario"] = "baseline"
    base["model"] = "baseline"
    base["window"] = "baseline"

    # future 30 年均
    fut = (
        df[df["epoch"] == "future"]
        .groupby(["biome_name", "variable", "scenario", "model"], as_index=False)["exceed_days"].mean()
        .rename(columns={"exceed_days": "mean_30yr"})
    )
    fut["epoch"] = "future"
    fut["window"] = fut["scenario"].str.slice(0, 4).map({"2040": "window_2040", "2070": "window_2070"})
    out = pd.concat([base, fut], ignore_index=True)
    return out

def _simplify_biome_label(name: str) -> str:
    mapping = {
        'Subtropical steppe':'Subtrop.\nsteppe',
        'Subtropical desert':'Subtrop.\ndesert',
        'Subtropical mountain system':'Subtrop.\nmountain',
        'Temperate steppe':'Temp.\nsteppe',
        'Temperate desert':'Temp.\ndesert',
        'Temperate mountain system west':'Temp.\nmount. W',
        'Boreal coniferous forest east':'Boreal\nconif. E',
        'Boreal coniferous forest west':'Boreal\nconif. W',
        'Boreal tundra woodland east':'Boreal\ntundra E',
        'Boreal tundra woodland west':'Boreal\ntundra W',
        'Boreal mountain system':'Boreal\nmountain'
    }
    return mapping.get(name, name)

# ======== Color range utilities ========
def _collect_changes_for_window(df30: pd.DataFrame, window_tag: str):
    scen_left  = "2040_245" if window_tag == "window_2040" else "2070_245"
    scen_right = "2040_370" if window_tag == "window_2040" else "2070_370"
    vals = []
    for biome in SELECTED_BIOMES:
        for var in VARIABLES:
            b = df30[
                (df30["scenario"] == "baseline") &
                (df30["biome_name"] == biome) &
                (df30["variable"] == var)
            ]["mean_30yr"]
            if b.empty or b.iloc[0] == 0:
                continue
            bmean = float(b.iloc[0])
            for scen in (scen_left, scen_right):
                for m in MODELS:
                    f = df30[
                        (df30["scenario"] == scen) &
                        (df30["model"].str.upper() == m.upper()) &
                        (df30["biome_name"] == biome) &
                        (df30["variable"] == var)
                    ]["mean_30yr"]
                    if f.empty:
                        continue
                    vals.append((float(f.iloc[0]) - bmean) / bmean * 100.0)
    return vals

def compute_norm_from_2070(df30: pd.DataFrame) -> TwoSlopeNorm:
    """用 2070 的分布来确定统一色标"""
    vals_2070 = _collect_changes_for_window(df30, "window_2070")
    if len(vals_2070) == 0:
        return TwoSlopeNorm(vmin=-100, vcenter=0.0, vmax=100)
    vmin = np.percentile(vals_2070, 2)
    vmax = np.percentile(vals_2070, 98)
    vmax_abs = max(abs(vmin), abs(vmax))
    return TwoSlopeNorm(vmin=-vmax_abs, vcenter=0.0, vmax=+vmax_abs)

# ======== Core plotting ========
def make_heatmap(df30: pd.DataFrame, window_tag: str, out_path: str, norm_override: TwoSlopeNorm = None):
    scen_left  = "2040_245" if window_tag == "window_2040" else "2070_245"
    scen_right = "2040_370" if window_tag == "window_2040" else "2070_370"

    if norm_override is None:
        all_vals = _collect_changes_for_window(df30, window_tag)
        if len(all_vals) == 0:
            print(f"[WARN] No values for {window_tag}. Skip.")
            return
        vmin = np.percentile(all_vals, 2)
        vmax = np.percentile(all_vals, 98)
        vmax_abs = max(abs(vmin), abs(vmax))
        norm = TwoSlopeNorm(vmin=-vmax_abs, vcenter=0.0, vmax=+vmax_abs)
    else:
        norm = norm_override

    # 图高压缩 30%：原 (10, 5) -> (10, 3.5)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    scenarios = [(scen_left, "SSP2-4.5"), (scen_right, "SSP3-7.0")]

    for ax, (scen, title) in zip(axes, scenarios):
        matrix_vals, anno_vals, row_names = [], [], []

        for biome in SELECTED_BIOMES:
            row_v, row_a, ok = [], [], False
            for var in VARIABLES:
                b = df30[
                    (df30["scenario"] == "baseline") &
                    (df30["biome_name"] == biome) &
                    (df30["variable"] == var)
                ]["mean_30yr"]
                if b.empty or b.iloc[0] == 0:
                    row_v += [np.nan] * len(MODELS)
                    row_a += [""] * len(MODELS)
                    continue
                bmean = float(b.iloc[0])

                for m in MODELS:
                    fm = df30[
                        (df30["scenario"] == scen) &
                        (df30["model"].str.upper() == m.upper()) &
                        (df30["biome_name"] == biome) &
                        (df30["variable"] == var)
                    ]["mean_30yr"]
                    if fm.empty:
                        row_v.append(np.nan)
                        row_a.append("")
                    else:
                        fmean = float(fm.iloc[0])
                        rel = (fmean - bmean) / bmean * 100.0
                        abschg = fmean - bmean
                        row_v.append(rel)
                        ok = True
                        rel_txt = f"{int(round(rel))}%"
                        abs_txt = f"{int(round(abschg))}d" if abs(abschg) >= 1 else f"{abschg:.1f}d"
                        row_a.append(f"{rel_txt}\n({abs_txt})")
            if ok:
                matrix_vals.append(row_v)
                anno_vals.append(row_a)
                row_names.append(_simplify_biome_label(biome))

        cols = [m for _ in VARIABLES for m in MODELS]
        mat = pd.DataFrame(matrix_vals, index=row_names, columns=cols)
        ann = pd.DataFrame(anno_vals,   index=row_names, columns=cols)

        sns.heatmap(
            mat, annot=ann, fmt="", cmap="RdBu_r", norm=norm,
            cbar=(ax is axes[1]),
            cbar_kws={'label': 'Relative change (%)', 'shrink': 0.8, 'aspect': 25} if ax is axes[1] else None,
            linewidths=0.2, linecolor='gray',
            annot_kws={'fontsize': 5, 'ha': 'center', 'va': 'center'},
            square=True, ax=ax
        )

        ax.text(2, -0.5, VAR_LABEL['fire_probability'], ha='center', va='bottom', fontsize=8, fontweight='bold')
        ax.text(6, -0.5, VAR_LABEL['obe_probability'],  ha='center', va='bottom', fontsize=8, fontweight='bold')
        ax.axvline(x=4, color='black', linewidth=1.2)
        ax.set_title(title, fontsize=9, fontweight='bold', pad=8)
        ax.set_xlabel(""); ax.set_ylabel("")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=6)
        if ax is axes[0]:
            ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=6)
        else:
            ax.set_yticklabels([])

    plt.tight_layout()
    plt.subplots_adjust(top=0.95, wspace=0.05, bottom=0.15)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.savefig(out_path.replace(".png", ".pdf"), bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")
    print(f"       {out_path.replace('.png', '.pdf')}")

# ======== Main ========
def main():
    df30 = _load_avg30()
    if df30 is None:
        print("[INFO] avg30_exceed_days.csv not found, try annual_exceed_days.csv …")
        df30 = _load_annual_and_aggregate()
        if df30 is None:
            raise FileNotFoundError("No avg30_exceed_days.csv or annual_exceed_days.csv found under BASE.")

    # 只保留需要的行 & 规范模型名
    df30 = df30[df30["variable"].isin(VARIABLES)].copy()
    df30["model"] = df30["model"].astype(str).str.upper().replace({
        "CANESM": "CanESM",
        "ECEARTH": "ECEarth",
        "GFDL": "GFDL",
        "UKESM": "UKESM",
        "BASELINE": "baseline"
    })

    # 统一色标：用 2070 的分布
    norm_2070 = compute_norm_from_2070(df30)

    for window in WINDOWS:
        if window == "baseline":
            continue
        out = os.path.join(OUT_DIR, f"biome_changes_{window}.png")
        make_heatmap(df30, window, out, norm_override=norm_2070)

if __name__ == "__main__":
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.sans-serif'] = ['Arial']
    plt.rcParams['axes.unicode_minus'] = False
    main()

