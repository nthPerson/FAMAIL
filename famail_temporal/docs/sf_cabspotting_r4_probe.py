#!/usr/bin/env python3
"""R4 probe: does the faithful 0.01deg SF grid carry usable demographic SIGNAL
for F_causal? Joins ACS 5-year (Census Reporter, keyless) to the active SF taxi
cells via tract centroids (Gazetteer internal points) and tests cross-cell
variance + collinearity (VIF) of the {housing, income, migrant} primary set
(+ log pop-density as the sensitivity feature).

Probe-grade aggregation: tract centroid -> grid cell, population-weighted mean,
nearest-tract fallback for empty active cells. Production (Phase 3) would use
proper areal interpolation. Vintage: ACS 2020-2024 (variance proxy; production
would use 2008-2012 to match the taxi data).
"""
import os, glob, json, math, tempfile, urllib.request
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))  # famail_temporal/docs/
DATA = os.environ.get(
    "SF_CAB_DIR",
    os.path.join(_HERE, "..", "source_data", "second_dataset", "cabspottingdata"),
)
SCRATCH = os.environ.get("SF_R4_CACHE", os.path.join(tempfile.gettempdir(), "sf_r4_cache"))
os.makedirs(SCRATCH, exist_ok=True)
GRID_SIZE_DEG = 0.01
COUNTIES = {"06075": "San Francisco", "06081": "San Mateo", "06001": "Alameda"}
TABLES = ["B19301", "B25077", "B05002", "B01003"]  # percap income, median home value, nativity, pop

# ---------------------------------------------------------------------------
# 1) Active SF taxi cells (re-derive same 32x30 grid + footprint as the de-risk)
# ---------------------------------------------------------------------------
files = sorted(glob.glob(os.path.join(DATA, "new_*.txt")))
lat_l, lon_l, cab_l = [], [], []
for ci, f in enumerate(files):
    with open(f, "rb") as fh:
        flat = np.array(fh.read().split(), dtype=np.float64)
    if flat.size < 4:
        continue
    a = flat[:(flat.size // 4) * 4].reshape(-1, 4)
    lat_l.append(a[:, 0]); lon_l.append(a[:, 1]); cab_l.append(np.full(a.shape[0], ci))
lat = np.concatenate(lat_l); lon = np.concatenate(lon_l); cab = np.concatenate(cab_l).astype(np.int32)
v = (lat > 36.5) & (lat < 38.8) & (lon > -123.2) & (lon < -121.2)
lat, lon, cab = lat[v], lon[v], cab[v]
b = (np.percentile(lat, 0.5), np.percentile(lat, 99.5), np.percentile(lon, 0.5), np.percentile(lon, 99.5))
GX = math.ceil((b[1] - b[0]) / GRID_SIZE_DEG); GY = math.ceil((b[3] - b[2]) / GRID_SIZE_DEG)
ix = np.clip(((lat - b[0]) / GRID_SIZE_DEG).astype(int), 0, GX - 1)
iy = np.clip(((lon - b[2]) / GRID_SIZE_DEG).astype(int), 0, GY - 1)
cellkey = ix.astype(np.int64) * GY + iy
# footprint cells + a "dense" subset (>=20 pings)
cells, counts = np.unique(cellkey, return_counts=True)
active_cells = cells                            # any taxi activity
dense_cells = cells[counts >= 20]
print(f"Grid {GX}x{GY} over bbox lat {b[0]:.3f}..{b[1]:.3f} lon {b[2]:.3f}..{b[3]:.3f}")
print(f"Active footprint cells: {active_cells.size}  | dense (>=20 pings): {dense_cells.size}")
# cell center lat/lon for nearest-tract fallback
def cell_center(k):
    cx, cy = k // GY, k % GY
    return b[0] + (cx + 0.5) * GRID_SIZE_DEG, b[2] + (cy + 0.5) * GRID_SIZE_DEG

# ---------------------------------------------------------------------------
# 2) ACS via Census Reporter (cached)
# ---------------------------------------------------------------------------
def fetch_acs():
    cache = os.path.join(SCRATCH, "acs_cache.json")
    if os.path.exists(cache):
        return json.load(open(cache))
    import subprocess
    out = {}
    for fips in COUNTIES:
        url = (f"https://api.censusreporter.org/1.0/data/show/latest?"
               f"table_ids={','.join(TABLES)}&geo_ids=140%7C05000US{fips}")
        txt = subprocess.run(["curl", "-sSL", "--max-time", "60", url],
                             capture_output=True, text=True).stdout
        d = json.loads(txt)
        out.update(d["data"])
        print(f"  ACS {COUNTIES[fips]} ({fips}): {len(d['data'])} tracts  release={d['release']['name']}")
    json.dump(out, open(cache, "w"))
    return out

print("\nFetching ACS (Census Reporter, keyless)...")
acs = fetch_acs()

# ---------------------------------------------------------------------------
# 3) Gazetteer tract centroids + land area (cached)
# ---------------------------------------------------------------------------
gaz = {}
import zipfile
gz = os.path.join(SCRATCH, "gaz.zip")
if not os.path.exists(gz):
    urllib.request.urlretrieve(
        "https://www2.census.gov/geo/docs/maps-data/data/gazetteer/2020_Gazetteer/2020_Gaz_tracts_national.zip", gz)
z = zipfile.ZipFile(gz)
for line in z.read(z.namelist()[0]).decode("latin-1").splitlines()[1:]:
    p = line.split("\t")
    geoid = p[1].strip()
    if geoid[:5] in COUNTIES:
        gaz[geoid] = (float(p[2]), float(p[6]), float(p[7]))  # ALAND(m2), INTPTLAT, INTPTLONG

# ---------------------------------------------------------------------------
# 4) Build per-tract feature rows
# ---------------------------------------------------------------------------
def est(rec, table, cell):
    try:
        val = rec[table]["estimate"][cell]
        return float(val) if val is not None else np.nan
    except Exception:
        return np.nan

tracts = []  # (lat, lon, housing, income, migrant, pop, aland)
for full_geoid, rec in acs.items():
    geoid = full_geoid.split("US")[-1]
    if geoid not in gaz:
        continue
    aland, tlat, tlon = gaz[geoid]
    pop = est(rec, "B01003", "B01003001")
    fb_tot = est(rec, "B05002", "B05002001"); fb = est(rec, "B05002", "B05002013")
    migrant = (fb / fb_tot) if (fb_tot and fb_tot > 0) else np.nan
    housing = est(rec, "B25077", "B25077001")   # median home value
    income = est(rec, "B19301", "B19301001")    # per capita income
    tracts.append((tlat, tlon, housing, income, migrant, pop, aland))
T = np.array(tracts, dtype=np.float64)
print(f"Tracts joined (ACS∩Gazetteer, 3 counties): {T.shape[0]}")

# assign each tract centroid to a grid cell
t_ix = np.clip(((T[:, 0] - b[0]) / GRID_SIZE_DEG).astype(int), 0, GX - 1)
t_iy = np.clip(((T[:, 1] - b[2]) / GRID_SIZE_DEG).astype(int), 0, GY - 1)
t_cell = t_ix.astype(np.int64) * GY + t_iy

# ---------------------------------------------------------------------------
# 5) Aggregate tracts -> active cells (pop-weighted; nearest-tract fallback)
# ---------------------------------------------------------------------------
def aggregate(cell_set, label):
    rows = []
    n_direct = 0
    for k in cell_set:
        sel = np.flatnonzero(t_cell == k)
        used_fallback = False
        if sel.size == 0:
            cy, cx = cell_center(k)
            d = (T[:, 0] - cy) ** 2 + (T[:, 1] - cx) ** 2
            sel = np.array([int(np.argmin(d))]); used_fallback = True
        else:
            n_direct += 1
        w = T[sel, 5].copy()  # pop weight
        w = np.where(np.isfinite(w) & (w > 0), w, 1.0)
        def wmean(col):
            vals = T[sel, col]; m = np.isfinite(vals)
            return np.average(vals[m], weights=w[m]) if m.any() else np.nan
        pop = np.nansum(T[sel, 5]); aland = np.nansum(T[sel, 6])
        dens = (pop / (aland / 1e6)) if aland > 0 else np.nan  # people / km^2
        rows.append((wmean(2), wmean(3), wmean(4),
                     math.log(dens) if (dens and dens > 0) else np.nan))
    R = np.array(rows, dtype=np.float64)
    print(f"\n--- {label}: {len(cell_set)} cells ({n_direct} with a tract centroid inside, "
          f"{len(cell_set)-n_direct} nearest-fallback) ---")
    names = ["housing($)", "income($)", "migrant(sh)", "logdensity"]
    finite_all = np.isfinite(R).all(axis=1)
    print(f"Cells with all 4 features finite: {finite_all.sum()}/{len(cell_set)}")
    for j, nm in enumerate(names):
        c = R[:, j][np.isfinite(R[:, j])]
        cv = np.std(c) / abs(np.mean(c)) if np.mean(c) != 0 else float('nan')
        print(f"  {nm:12s} n={c.size:4d}  mean={np.mean(c):11.3f}  std={np.std(c):11.3f}  "
              f"CV={cv:5.2f}  min={np.min(c):10.2f}  p50={np.median(c):10.2f}  max={np.max(c):10.2f}")
    # collinearity (VIF) on z-scored finite rows
    Z = R[finite_all]
    Zz = (Z - Z.mean(0)) / Z.std(0)
    corr = np.corrcoef(Zz, rowvar=False)
    try:
        vif = np.diag(np.linalg.inv(corr))
        print(f"  VIF: " + "  ".join(f"{names[j].split('(')[0]}={vif[j]:.2f}" for j in range(4)))
        print(f"  |corr| max off-diag = {np.abs(corr - np.eye(4)).max():.2f}  (Shenzhen primary set: max VIF 4.45)")
    except np.linalg.LinAlgError:
        print("  VIF: singular correlation matrix (DEGENERATE)")
    return R

R_active = aggregate(active_cells, "ACTIVE footprint cells")
R_dense = aggregate(dense_cells, "DENSE cells (>=20 pings)")

print("\n" + "=" * 70)
print("VERDICT: demographic features are non-degenerate over the active SF grid"
      "\n  if each has std>0, CV not ~0, coverage high, and VIF well below ~10.")
print("=" * 70)
