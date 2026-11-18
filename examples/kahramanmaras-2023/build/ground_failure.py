"""Step 3b: ground-failure components along the routed links (liquefaction, landslide, fault crossing).

usage: python ground_failure.py <net dir with links.gpkg> <DEM tile dir> <turkey-latest.osm.pbf>
                                <gem_active_faults_harmonized.geojson> <USGS ground-failure dir> <out dir>

Writes <out>/gf_components.csv (one row per component: link, kind, s_m, length_m and its site inputs)
and <out>/gf_calibration.json (the regional constants and the screening statistics). Every model,
coefficient and assumption is listed in the example's README and network.json.

Stations every 100 m along each link carry:
  slope_deg     slope of the Copernicus GLO-90 DEM, mean over 3 x 3 cells (about 270 m)
  slope30       slope at 30 arc-seconds (about 900 m), m/m: the scale Allen & Wald (2007) use
  vs30          from slope30 by Allen & Wald (2007) Table 2, active tectonic, log-linear between breakpoints
  dw_km         distance to the nearest OSM river, canal, lake or coastline
Zhu et al. (2017) site term, without the shaking term:
  x0_zhu  = 8.801 - 1.918 ln(vs30) - 0.2054 dw + C_zhu            (C_zhu replaces 0.0005408 precip - 0.0333 wtd)
Nowicki Jessee et al. (2018) site term (USGS "slim" form, X0):
  x0_jes  = -6.30 + 0.06 slope_deg + C_jes                       (C_jes replaces rock + land cover + 0.03 CTI)
C_zhu and C_jes are calibrated as the median, over road stations, of the site term implied by the USGS
ground-failure maps of the two 2023 main shocks and the ShakeMap versions they were run on.
"""
import json
import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import geopandas as gpd
import numpy as np
import osmium
import pandas as pd
import rasterio
from rasterio.merge import merge
from scipy.ndimage import uniform_filter
from shapely import STRtree
from shapely.geometry import LineString, Point

net, dem_dir, pbf, faults_file, gf_dir, out = (Path(a) for a in sys.argv[1:7])
out.mkdir(parents=True, exist_ok=True)
REPORT = "--report" in sys.argv
W, S, E, N = 35.2, 35.8, 39.0, 38.8
UTM = 32637
STEP = 100.0                                      # station spacing along a link, m

# ---------------- model constants (sources in the README) ----------------
ZHU = dict(b0=8.801, b_pgv=0.334, b_vs30=-1.918, b_d=-0.2054, cov=(0.4915, 42.4, 9.165),
           vs30_max=620.0, slope_max=5.0, pgv_min=3.0, pga_min=0.10)
JES = dict(b0=-6.30, b_pgv=1.65, b_slope=0.06, b_int=0.01, cov=(-7.592, 5.237, -3.042, 4.035),
           slope_min=2.0, pga_min=0.02, pgv_clip=211.0)
WA07 = [(1.0e-4, 180.0), (2.2e-3, 240.0), (6.3e-3, 300.0), (0.018, 360.0), (0.05, 490.0), (0.10, 620.0), (0.138, 760.0)]
PGV_REF, M_REF = 50.0, 7.0                        # reference shaking for the exposure screen
SCREEN = {"liquefaction": 0.05, "landslide": 0.05}  # coverage at the reference shaking that makes a component
MAX_LEN = 1000.0                                  # a stretch longer than this is split
LIQ_CLASS = [(0.20, "vh"), (0.10, "h"), (0.05, "m"), (0.02, "l"), (0.0, "vl")]   # Zhu coverage at reference
SLOPE_SOIL = dict(c_kpa=20.0, phi_deg=32.0, gamma_kn=19.0, t_m=3.0, m=0.0)          # infinite-slope Newmark inputs


def msf(m):
    return 1.0 / (1.0 + math.exp(-2.0 * (m - 6.0)))


def zhu_cov(x):
    p = 1 / (1 + np.exp(-x))
    a, b, c = ZHU["cov"]
    return a / (1 + b * np.exp(-c * p)) ** 2


def jes_cov(x):
    p = 1 / (1 + np.exp(-x))
    a, b, c, d = JES["cov"]
    return np.exp(a + b * p + c * p ** 2 + d * p ** 3)


def zhu_x_from_cov(f):
    a, b, c = ZHU["cov"]
    f = np.asarray(f, float)
    p = -np.log((np.sqrt(a / f) - 1) / b) / c
    return np.log(p / (1 - p))


def jes_x_from_cov(f):
    a, b, c, d = JES["cov"]
    f = np.asarray(f, float)
    lo, hi = np.full(f.shape, 1e-9), np.full(f.shape, 1 - 1e-9)
    target = np.log(f)
    for _ in range(80):                              # the cubic is increasing in p, so bisect
        mid = (lo + hi) / 2
        val = a + b * mid + c * mid ** 2 + d * mid ** 3
        lo, hi = np.where(val < target, mid, lo), np.where(val < target, hi, mid)
    p = (lo + hi) / 2
    return np.log(p / (1 - p))


def wald_allen(slope_mm):
    s = np.clip(np.asarray(slope_mm, float), WA07[0][0], WA07[-1][0])
    xs, ys = np.log10([w[0] for w in WA07]), [w[1] for w in WA07]
    return np.interp(np.log10(s), xs, ys)


def critical_acc(slope_deg):
    """Infinite-slope critical acceleration (g), Newmark: (FS - 1) sin(alpha); FS dry (m = 0)."""
    a = np.radians(np.clip(slope_deg, 0.5, 89.0))
    p = SLOPE_SOIL
    fs = p["c_kpa"] / (p["gamma_kn"] * p["t_m"] * np.sin(a)) + np.tan(np.radians(p["phi_deg"])) / np.tan(a) \
        - p["m"] * 9.81 * np.tan(np.radians(p["phi_deg"])) / (p["gamma_kn"] * np.tan(a))
    return np.maximum((fs - 1.0) * np.sin(a), 0.01)


# ---------------- stations along the links ----------------
links = gpd.read_file(net / "links.gpkg", layer="links")
lines_utm = links.to_crs(UTM)
rows = []
for lk, line in zip(links["link"], lines_utm.geometry):
    L = line.length
    for s in np.arange(STEP / 2, L, STEP):
        p = line.interpolate(s)
        rows.append(dict(link=lk, s_m=s, x=p.x, y=p.y))
st = pd.DataFrame(rows)
ll = gpd.GeoSeries(gpd.points_from_xy(st.x, st.y), crs=UTM).to_crs(4326)
st["lon"], st["lat"] = ll.x.values, ll.y.values
print(f"{len(st)} stations on {len(links)} links")

# ---------------- DEM: slope at 90 m and at 30 arc-seconds ----------------
srcs = [rasterio.open(p) for p in sorted(dem_dir.glob("*.tif"))]
dem, tr = merge(srcs, bounds=(W - 0.05, S - 0.05, E + 0.05, N + 0.05), nodata=0)
dem = dem[0].astype("float64")
dem[dem < -100] = 0.0
res = tr.a                                                   # degrees per cell (GLO-90: 3 arc-seconds)
lat_rows = tr.f + res * 0.5 * -1 + np.arange(dem.shape[0]) * tr.e
dy = abs(tr.e) * 110574.0
dx = res * 111320.0 * np.cos(np.radians(lat_rows))[:, None]
gy, gx = np.gradient(dem)
slope90 = np.degrees(np.arctan(np.hypot(gx / dx, gy / dy)))
slope_s = uniform_filter(slope90, size=3, mode="nearest")
k = int(round((30 / 3600) / res))                            # cells per 30 arc-seconds
h, w = (dem.shape[0] // k) * k, (dem.shape[1] // k) * k
coarse = dem[:h, :w].reshape(h // k, k, w // k, k).mean(axis=(1, 3))
lat_c = tr.f + (np.arange(h // k) + 0.5) * k * tr.e
gyc, gxc = np.gradient(coarse)
slope30 = np.hypot(gxc / (k * res * 111320.0 * np.cos(np.radians(lat_c))[:, None]), gyc / (k * abs(tr.e) * 110574.0))
ci = ((st.lat.values - tr.f) / tr.e).astype(int)
cj = ((st.lon.values - tr.c) / tr.a).astype(int)
st["slope_deg"] = slope_s[ci.clip(0, dem.shape[0] - 1), cj.clip(0, dem.shape[1] - 1)]
st["slope30"] = slope30[(ci // k).clip(0, slope30.shape[0] - 1), (cj // k).clip(0, slope30.shape[1] - 1)]
st["slope30_deg"] = np.degrees(np.arctan(st.slope30))
st["vs30"] = wald_allen(st.slope30)

# ---------------- distance to water (OSM) ----------------
water = []
for o in osmium.FileProcessor(str(pbf)).with_locations().with_areas() \
        .with_filter(osmium.filter.KeyFilter("waterway", "natural", "water")):
    t = o.tags
    try:
        if o.is_way() and (t.get("waterway") in ("river", "canal") or t.get("natural") == "coastline"):
            pts = [(n.lon, n.lat) for n in o.nodes if n.location.valid()]
            if len(pts) > 1 and any(W - 0.2 <= x <= E + 0.2 and S - 0.2 <= y <= N + 0.2 for x, y in pts):
                water.append(LineString(pts))
        elif o.is_area() and (t.get("natural") == "water" or "water" in t):
            for ring in o.outer_rings():
                pts = [(n.lon, n.lat) for n in ring if n.location.valid()]
                if len(pts) > 3 and any(W - 0.2 <= x <= E + 0.2 and S - 0.2 <= y <= N + 0.2 for x, y in pts):
                    water.append(LineString(pts))
    except Exception:
        continue
wg = gpd.GeoSeries(water, crs=4326).to_crs(UTM)
tree = STRtree(wg.values)
pts = gpd.points_from_xy(st.x, st.y)
idx = tree.nearest(pts)
st["dw_km"] = np.array([p.distance(wg.values[i]) for p, i in zip(pts, idx)]) / 1000.0
print(f"water features: {len(wg)}; median distance to water {st.dw_km.median():.2f} km")

# ---------------- calibration to the USGS 2023 maps ----------------
ns = {"s": "http://earthquake.usgs.gov/eqcenter/shakemap"}


def grid_values(path, fields):
    r = ET.parse(path).getroot()
    spec = r.find("s:grid_specification", ns).attrib
    flds = {e.attrib["name"]: int(e.attrib["index"]) - 1 for e in r.findall("s:grid_field", ns)}
    a = np.array(r.find("s:grid_data", ns).text.split(), float).reshape(-1, len(flds))
    nx, ny = int(spec["nlon"]), int(spec["nlat"])
    lon0, lat1, dlon, dlat = float(spec["lon_min"]), float(spec["lat_max"]), float(spec["nominal_lon_spacing"]), float(spec["nominal_lat_spacing"])
    i = np.clip(np.round((lat1 - st.lat.values) / dlat).astype(int), 0, ny - 1)
    j = np.clip(np.round((st.lon.values - lon0) / dlon).astype(int), 0, nx - 1)
    return {f: a[i * nx + j, flds[f]] for f in fields}


def raster_values(path):
    with rasterio.open(path) as r:
        v = np.array([x[0] for x in r.sample(zip(st.lon.values, st.lat.values))], float)
        if r.nodata is not None:
            v[v == r.nodata] = np.nan
    return v


cal = {"zhu": [], "jes": []}
for ev, mag, grid in [("us6000jllz", 7.8, "us6000jllz_v17_grid.xml"), ("us6000jlqa", 7.5, "us6000jlqa_v11_grid.xml")]:
    g = grid_values(gf_dir / grid, ["PGV", "PGA"])
    pgv, pga = g["PGV"], g["PGA"] / 100.0
    fz = raster_values(gf_dir / f"{ev}_zhu_2017_general_model.tif")
    ok = np.isfinite(fz) & (fz > 1e-4) & (fz < ZHU["cov"][0] * 0.999)
    x0 = zhu_x_from_cov(fz[ok]) - ZHU["b_pgv"] * np.log(pgv[ok] * msf(mag))
    partial = ZHU["b0"] + ZHU["b_vs30"] * np.log(st.vs30.values[ok]) + ZHU["b_d"] * st.dw_km.values[ok]
    cal["zhu"] += list(x0 - partial)
    fj = raster_values(gf_dir / f"{ev}_jessee_2018_model.tif")
    ok = np.isfinite(fj) & (fj > 1e-6)
    pgvc = np.minimum(pgv[ok], JES["pgv_clip"])
    x0 = jes_x_from_cov(fj[ok]) - JES["b_pgv"] * np.log(pgvc) - JES["b_int"] * np.log(pgvc) * st.slope_deg.values[ok]
    cal["jes"] += list(x0 - (JES["b0"] + JES["b_slope"] * st.slope_deg.values[ok]))
C = {k: float(np.median(v)) for k, v in cal.items()}
calib = {k: {"constant": C[k], "n_stations": len(v), "iqr": [float(np.percentile(v, 25)), float(np.percentile(v, 75))]}
         for k, v in cal.items()}
print("calibration:", json.dumps(calib))

# ---------------- site terms and exposure screen ----------------
st["x0_zhu"] = ZHU["b0"] + ZHU["b_vs30"] * np.log(st.vs30) + ZHU["b_d"] * st.dw_km + C["zhu"]
st["x0_jes"] = JES["b0"] + JES["b_slope"] * st.slope_deg + C["jes"]
liq_ok = (st.vs30 <= ZHU["vs30_max"]) & (st.slope30_deg <= ZHU["slope_max"])
st["f_ref_liq"] = np.where(liq_ok, zhu_cov(st.x0_zhu + ZHU["b_pgv"] * np.log(PGV_REF * msf(M_REF))), 0.0)
ls_ok = st.slope_deg >= JES["slope_min"]
st["f_ref_ls"] = np.where(ls_ok, jes_cov(st.x0_jes + JES["b_pgv"] * np.log(PGV_REF)
                                         + JES["b_int"] * np.log(PGV_REF) * st.slope_deg), 0.0)
for kind, col in (("liquefaction", "f_ref_liq"), ("landslide", "f_ref_ls")):
    q = st[col].quantile([0.5, 0.9, 0.99]).round(4).to_dict()
    counts = {t: int((st[col] >= t).sum()) for t in (0.01, 0.02, 0.05, 0.1, 0.2)}
    print(f"{kind}: station coverage at reference quantiles {q}; stations >= threshold {counts}")

# ---------------- group flagged stations into components ----------------
comps = []
for kind, col in (("liquefaction", "f_ref_liq"), ("landslide", "f_ref_ls")):
    for lk, g in st[st[col] >= SCREEN[kind]].groupby("link"):
        g = g.sort_values("s_m")
        run = [g.iloc[0]]
        groups = []
        for _, r in list(g.iterrows())[1:]:
            if r.s_m - run[-1].s_m <= STEP + 1 and (r.s_m - run[0].s_m) < MAX_LEN:
                run.append(r)
            else:
                groups.append(run)
                run = [r]
        groups.append(run)
        for run in groups:
            d = pd.DataFrame(run)
            c = dict(link=lk, kind=kind, s_m=round(float(d.s_m.mean()), 1), length_m=float(len(d) * STEP))
            if kind == "liquefaction":
                fmean = float(d.f_ref_liq.mean())
                c.update(vs30=round(float(d.vs30.mean()), 1), dw_km=round(float(d.dw_km.mean()), 3),
                         x0=round(float(d.x0_zhu.mean()), 4), f_ref=round(fmean, 4),
                         liq_class=next(cls for t, cls in LIQ_CLASS if fmean >= t))
            else:
                sl = float(d.slope_deg.max())
                c.update(slope_deg=round(float(d.slope_deg.mean()), 2), slope_max_deg=round(sl, 2),
                         x0=round(float(d.x0_jes.mean()), 4), f_ref=round(float(d.f_ref_ls.mean()), 4),
                         ac_g=round(float(critical_acc(sl)), 4))
            comps.append(c)

# ---------------- fault crossings (GEM Global Active Faults) ----------------
fl = gpd.read_file(faults_file)
fl = fl.cx[W - 0.1:E + 0.1, S - 0.1:N + 0.1].to_crs(UTM)
n_fault = 0
for lk, line in zip(links["link"], lines_utm.geometry):
    for _, f in fl.iterrows():
        x = line.intersection(f.geometry)
        if x.is_empty:
            continue
        for p in getattr(x, "geoms", [x]):
            if p.geom_type != "Point":
                continue
            comps.append(dict(link=lk, kind="fault", s_m=round(float(line.project(p)), 1), length_m=0.0,
                              fault_id=str(f.get("catalog_id") or ""), fault_catalog=str(f.get("catalog_name") or ""),
                              slip_type=str(f.get("slip_type") or ""), net_slip_rate=str(f.get("net_slip_rate") or "")))
            n_fault += 1
cdf = pd.DataFrame(comps)
print("components:", cdf["kind"].value_counts().to_dict())
if not REPORT:
    cdf.to_csv(out / "gf_components.csv", index=False, encoding="utf-8")
    json.dump({"calibration": calib, "screen": SCREEN, "reference": {"pgv_cms": PGV_REF, "mw": M_REF},
               "liq_class_bands": LIQ_CLASS, "slope_soil": SLOPE_SOIL, "counts": cdf["kind"].value_counts().to_dict()},
              open(out / "gf_calibration.json", "w"), indent=2)
    print("wrote", out / "gf_components.csv")
