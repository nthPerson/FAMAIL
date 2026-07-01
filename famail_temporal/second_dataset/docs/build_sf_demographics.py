#!/usr/bin/env python3
"""Assemble the SF second-dataset DEMOGRAPHIC layer for FAMAIL.

Fetches the vintage-correct ACS 5-year (2006-2010, centered on the May-Jun 2008
taxi data) tract estimates for the SF taxi footprint (San Francisco + San Mateo +
Alameda counties) via the keyed US Census API, joins 2010-vintage tract geometry
(Gazetteer internal points + land area, and TIGER polygons for areal
interpolation), and writes a clean, documented source-data layer under
    famail_temporal/source_data/second_dataset/demographics/

It also emits a PROVISIONAL per-cell grid file (pop-weighted centroid aggregation)
for inspection ONLY — the production per-cell aggregation method (areal
interpolation vs centroid vs pop-weighted) is a Phase-2 decision under the
algorithm-change protocol and is NOT finalized here.

Key: read from env CENSUS_API_KEY, else from
     source_data/second_dataset/.census_api_key  (gitignored).

Run:  python famail_temporal/docs/build_sf_demographics.py
"""
import os, glob, json, math, csv, zipfile, urllib.request, subprocess
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
SD = os.path.normpath(os.path.join(_HERE, "..", "..", "source_data", "second_dataset"))
DEMO = os.path.join(SD, "demographics")
RAW = os.path.join(DEMO, "raw")
CAB = os.path.join(SD, "cabspottingdata")
os.makedirs(RAW, exist_ok=True)

GRID_SIZE_DEG = 0.01
ACS_YEAR = 2010          # 2006-2010 5-year, midpoint 2008 (closest to the taxi data)
ACS_DATASET = "acs/acs5"
STATE = "06"
COUNTIES = {"001": "Alameda", "075": "San Francisco", "081": "San Mateo"}
# ACS variables: primary {housing, comp, migrant} + population, plus robustness alts.
VARS = [
    "B25077_001E", "B25077_001M",   # median home value  -> housing (primary)
    "B19301_001E", "B19301_001M",   # per-capita income  -> comp (primary)
    "B05002_001E", "B05002_013E",   # total / foreign-born -> migrant share (primary)
    "B01003_001E",                  # total population   -> density + weighting
    "B19013_001E",                  # median household income (comp alt)
    "B25064_001E",                  # median gross rent (housing alt)
    "B25003_001E", "B25003_002E", "B25003_003E",  # tenure: total/owner/renter (housing alt)
]

def census_key():
    k = os.environ.get("CENSUS_API_KEY")
    if k:
        return k.strip()
    p = os.path.join(SD, ".census_api_key")
    if os.path.exists(p):
        return open(p).read().strip()
    raise SystemExit("No Census API key: set $CENSUS_API_KEY or create second_dataset/.census_api_key")

KEY = census_key()

# ---------------------------------------------------------------------------
# 1) ACS 2006-2010 tract estimates (keyed API), per county
# ---------------------------------------------------------------------------
def fetch_county(cc):
    url = (f"https://api.census.gov/data/{ACS_YEAR}/{ACS_DATASET}"
           f"?get=NAME,{','.join(VARS)}&for=tract:*&in=state:{STATE}+county:{cc}&key={KEY}")
    txt = subprocess.run(["curl", "-sSL", "--max-time", "90", url],
                         capture_output=True, text=True).stdout
    raw_path = os.path.join(RAW, f"acs{ACS_YEAR}_county{cc}.json")
    open(raw_path, "w").write(txt)
    rows = json.loads(txt)
    return rows

print(f"Fetching ACS {ACS_YEAR-4}-{ACS_YEAR} 5-year (keyed Census API)...")
header = None
records = {}   # GEOID -> dict
for cc, name in COUNTIES.items():
    rows = fetch_county(cc)
    header = rows[0]
    idx = {h: i for i, h in enumerate(header)}
    for r in rows[1:]:
        geoid = r[idx["state"]] + r[idx["county"]] + r[idx["tract"]]
        rec = {v: r[idx[v]] for v in VARS}
        rec["NAME"] = r[idx["NAME"]]
        rec["county_name"] = name
        records[geoid] = rec
    print(f"  {name} ({cc}): {len(rows)-1} tracts")
print(f"Total ACS tracts: {len(records)}")

def fnum(x):
    try:
        v = float(x)
        return v if v > -1e8 else np.nan   # ACS uses large negatives for null/jam values
    except (TypeError, ValueError):
        return np.nan

# ---------------------------------------------------------------------------
# 2) 2010-vintage tract geometry: Gazetteer internal points + land area
# ---------------------------------------------------------------------------
gaz_zip = os.path.join(RAW, "Gaz_tracts_2010_national.zip")
if not os.path.exists(gaz_zip):
    print("Downloading 2010 Gazetteer tracts...")
    urllib.request.urlretrieve(
        "https://www2.census.gov/geo/docs/maps-data/data/gazetteer/Gaz_tracts_national.zip", gaz_zip)
z = zipfile.ZipFile(gaz_zip)
glines = z.read(z.namelist()[0]).decode("latin-1").splitlines()
ghdr = [h.strip() for h in glines[0].split("\t")]
gidx = {h: i for i, h in enumerate(ghdr)}
geom = {}  # GEOID -> (lat, lon, aland_m2)
for line in glines[1:]:
    p = line.split("\t")
    g = p[gidx["GEOID"]].strip()
    if g[:5] in {STATE + cc for cc in COUNTIES}:
        geom[g] = (float(p[gidx["INTPTLAT"]]), float(p[gidx["INTPTLONG"]]), float(p[gidx["ALAND"]]))
print(f"Gazetteer 2010 tract geometry rows (3 counties): {len(geom)}")

# ---------------------------------------------------------------------------
# 3) TIGER 2010 CA tract polygons (for production areal interpolation)
# ---------------------------------------------------------------------------
tiger = os.path.join(DEMO, "tiger_2010_tracts_06_CA.zip")
if not os.path.exists(tiger):
    print("Downloading TIGER 2010 CA tract polygons (~29 MB)...")
    urllib.request.urlretrieve(
        "https://www2.census.gov/geo/tiger/TIGER2010/TRACT/2010/tl_2010_06_tract10.zip", tiger)
print(f"TIGER polygons: {os.path.relpath(tiger, SD)} ({os.path.getsize(tiger)//1024//1024} MB)")

# ---------------------------------------------------------------------------
# 4) Write the tidy tract-level CSV (the authoritative source layer)
# ---------------------------------------------------------------------------
tracts_csv = os.path.join(DEMO, "acs_2006_2010_tracts.csv")
n_written = 0
with open(tracts_csv, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow([
        "GEOID", "county", "name", "intptlat", "intptlon", "aland_m2",
        "pop", "housing_median_value", "housing_median_value_moe",
        "income_percapita", "income_percapita_moe", "income_median_hh", "rent_median",
        "tenure_total", "tenure_owner", "tenure_renter", "owner_share",
        "foreignborn", "nat_total", "migrant_share",
    ])
    for g, rec in records.items():
        if g not in geom:
            continue
        lat, lon, aland = geom[g]
        natt = fnum(rec["B05002_001E"]); fb = fnum(rec["B05002_013E"])
        ten_t = fnum(rec["B25003_001E"]); ten_o = fnum(rec["B25003_002E"]); ten_r = fnum(rec["B25003_003E"])
        w.writerow([
            g, rec["county_name"], rec["NAME"], lat, lon, aland,
            fnum(rec["B01003_001E"]),
            fnum(rec["B25077_001E"]), fnum(rec["B25077_001M"]),
            fnum(rec["B19301_001E"]), fnum(rec["B19301_001M"]),
            fnum(rec["B19013_001E"]), fnum(rec["B25064_001E"]),
            ten_t, ten_o, ten_r, (ten_o / ten_t) if ten_t and ten_t > 0 else "",
            fb, natt, (fb / natt) if natt and natt > 0 else "",
        ])
        n_written += 1
print(f"Wrote {tracts_csv}  ({n_written} tracts with geometry)")

# tract geometry CSV (subset, convenience)
geo_csv = os.path.join(DEMO, "tract_geometry_2010.csv")
with open(geo_csv, "w", newline="") as f:
    w = csv.writer(f); w.writerow(["GEOID", "intptlat", "intptlon", "aland_m2"])
    for g, (lat, lon, aland) in sorted(geom.items()):
        w.writerow([g, lat, lon, aland])
print(f"Wrote {geo_csv}")

# ---------------------------------------------------------------------------
# 5) SF grid + active cells (re-derive the faithful 32x30 from the cab data)
# ---------------------------------------------------------------------------
files = sorted(glob.glob(os.path.join(CAB, "new_*.txt")))
lat_l, lon_l = [], []
for fp in files:
    with open(fp, "rb") as fh:
        flat = np.array(fh.read().split(), dtype=np.float64)
    if flat.size < 4:
        continue
    a = flat[:(flat.size // 4) * 4].reshape(-1, 4)
    lat_l.append(a[:, 0]); lon_l.append(a[:, 1])
lat = np.concatenate(lat_l); lon = np.concatenate(lon_l)
v = (lat > 36.5) & (lat < 38.8) & (lon > -123.2) & (lon < -121.2)
lat, lon = lat[v], lon[v]
b = (np.percentile(lat, 0.5), np.percentile(lat, 99.5),
     np.percentile(lon, 0.5), np.percentile(lon, 99.5))
GX = math.ceil((b[1] - b[0]) / GRID_SIZE_DEG); GY = math.ceil((b[3] - b[2]) / GRID_SIZE_DEG)
ix = np.clip(((lat - b[0]) / GRID_SIZE_DEG).astype(int), 0, GX - 1)
iy = np.clip(((lon - b[2]) / GRID_SIZE_DEG).astype(int), 0, GY - 1)
cellkey = ix.astype(np.int64) * GY + iy
cells, counts = np.unique(cellkey, return_counts=True)
print(f"Grid {GX}x{GY}; active footprint cells: {cells.size}")

# tract centroids -> feature arrays (reload from CSV for clarity)
import numpy as _np
TR = []
with open(tracts_csv) as f:
    for row in csv.DictReader(f):
        def g(k):
            try: return float(row[k])
            except: return _np.nan
        TR.append((g("intptlat"), g("intptlon"), g("housing_median_value"),
                   g("income_percapita"), g("migrant_share"), g("pop"), g("aland_m2")))
TR = _np.array(TR, dtype=float)
t_ix = _np.clip(((TR[:, 0] - b[0]) / GRID_SIZE_DEG).astype(int), 0, GX - 1)
t_iy = _np.clip(((TR[:, 1] - b[2]) / GRID_SIZE_DEG).astype(int), 0, GY - 1)
t_cell = t_ix.astype(_np.int64) * GY + t_iy

# ---------------------------------------------------------------------------
# 6) PROVISIONAL per-cell grid (pop-weighted centroid; NOT the final method)
# ---------------------------------------------------------------------------
draft = os.path.join(DEMO, "draft_grid_features_PROVISIONAL.csv")
cnt_map = dict(zip(cells.tolist(), counts.tolist()))
with open(draft, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["cell_x", "cell_y", "n_pings", "method",
                "housing_median_value", "income_percapita", "migrant_share", "logdensity"])
    for k in cells:
        sel = _np.flatnonzero(t_cell == k); method = "centroid"
        if sel.size == 0:
            cx, cy = k // GY, k % GY
            clat = b[0] + (cx + 0.5) * GRID_SIZE_DEG; clon = b[2] + (cy + 0.5) * GRID_SIZE_DEG
            d = (TR[:, 0] - clat) ** 2 + (TR[:, 1] - clon) ** 2
            sel = _np.array([int(_np.argmin(d))]); method = "nearest_fallback"
        wt = TR[sel, 5].copy(); wt = _np.where(_np.isfinite(wt) & (wt > 0), wt, 1.0)
        def wm(col):
            vv = TR[sel, col]; m = _np.isfinite(vv)
            return float(_np.average(vv[m], weights=wt[m])) if m.any() else ""
        pop = _np.nansum(TR[sel, 5]); aland = _np.nansum(TR[sel, 6])
        dens = (pop / (aland / 1e6)) if aland > 0 else _np.nan
        w.writerow([int(k // GY), int(k % GY), cnt_map[int(k)], method,
                    wm(2), wm(3), wm(4),
                    round(math.log(dens), 4) if dens and dens > 0 else ""])
print(f"Wrote {draft}  (PROVISIONAL — pending Phase-2 aggregation-method decision)")

# ---------------------------------------------------------------------------
# 7) Manifest
# ---------------------------------------------------------------------------
manifest = {
    "grid": {"size_deg": GRID_SIZE_DEG, "dims": [GX, GY],
             "bbox_latmin": b[0], "bbox_latmax": b[1], "bbox_lonmin": b[2], "bbox_lonmax": b[3],
             "note": "faithful constant-0.01deg cells; matches Shenzhen GRID_SIZE_DEG; NOT 48x90"},
    "acs": {"release": f"{ACS_YEAR-4}-{ACS_YEAR} ACS 5-year", "dataset": ACS_DATASET,
            "counties": COUNTIES, "variables": VARS, "n_tracts": n_written},
    "geometry_vintage": "2010 (Gazetteer internal points + TIGER 2010 tract polygons)",
    "feature_mapping": {
        "housing": "B25077_001E median home value (alt: B25064 rent, B25003 owner share)",
        "comp": "B19301_001E per-capita income (alt: B19013 median HH income)",
        "migrant": "B05002_013E/B05002_001E foreign-born share (US analog of Shenzhen hukou/migrant)",
        "logdensity": "log(B01003_001E / ALAND_km2) — sensitivity feature",
    },
    "provisional_grid_caveat": "draft_grid_features_PROVISIONAL.csv uses pop-weighted centroid "
        "aggregation for inspection only; production method (areal interpolation) is a Phase-2 "
        "decision under the algorithm-change protocol.",
}
open(os.path.join(DEMO, "MANIFEST.json"), "w").write(json.dumps(manifest, indent=2))
print(f"Wrote {os.path.join(DEMO, 'MANIFEST.json')}")
print("\nDONE — demographic source layer assembled under source_data/second_dataset/demographics/")
