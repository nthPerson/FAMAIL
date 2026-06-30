#!/usr/bin/env python3
"""Phase-1 de-risk analysis of the SF Cabspotting dataset for FAMAIL.

Computes the TRUE risk profile vs Shenzhen under the geometrically-faithful
constant-0.01-degree gridding (matching source_generation/config.GRID_SIZE_DEG).

No FAMAIL imports — standalone, numpy-only.
"""
import os, glob, math, sys
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))  # famail_temporal/docs/
DATA = os.environ.get(
    "SF_CAB_DIR",
    os.path.join(_HERE, "..", "source_data", "second_dataset", "cabspottingdata"),
)
GRID_SIZE_DEG = 0.01          # matches Shenzhen source_generation/config.py
GAP_SEC = 300                 # split a trajectory if consecutive pings > 5 min apart
PDT_OFFSET = 7 * 3600         # May–Jun 2008 SF local = UTC-7 (PDT) for day bucketing
N_CTX = 5                     # discriminator batches N=5 trajs per driver(-day)

# Shenzhen reference (from TRAINING_RESULTS.md / project memory / the investigation doc)
SHEN = dict(drivers=50, days=66, grid="48x90", cell_deg=0.01,
            corpus=95297, n_active=34524, seeking=105401, driving=92429)

files = sorted(glob.glob(os.path.join(DATA, "new_*.txt")))
print(f"# SF Cabspotting de-risk  |  {len(files)} cab files\n")

# ---- Load all points; remember per-cab contiguous ranges -------------------
lat_l, lon_l, t_l, occ_l, cab_l = [], [], [], [], []
cab_names = []
for ci, f in enumerate(files):
    try:
        with open(f, "rb") as fh:
            raw = fh.read()
        if not raw.strip():
            continue
        flat = np.array(raw.split(), dtype=np.float64)
        if flat.size < 4:
            continue
        arr = flat[: (flat.size // 4) * 4].reshape(-1, 4)
    except Exception as e:
        print(f"  !! failed {os.path.basename(f)}: {e}"); continue
    if arr.shape[0] == 0:
        continue
    lat_l.append(arr[:, 0]); lon_l.append(arr[:, 1])
    occ_l.append(arr[:, 2].astype(np.int8)); t_l.append(arr[:, 3].astype(np.int64))
    cab_l.append(np.full(arr.shape[0], ci, dtype=np.int32))
    cab_names.append(os.path.basename(f)[4:-4])

lat = np.concatenate(lat_l); lon = np.concatenate(lon_l)
occ = np.concatenate(occ_l); t = np.concatenate(t_l); cab = np.concatenate(cab_l)
N = lat.size
ncabs = len(cab_names)
print(f"Total GPS points : {N:,}")
print(f"Cabs (drivers)   : {ncabs}   (Shenzhen: {SHEN['drivers']})")

# Drop obviously-invalid coords (0,0 or out of plausible Bay Area window)
valid = (lat > 36.5) & (lat < 38.8) & (lon > -123.2) & (lon < -121.2)
print(f"Invalid/stray pts dropped: {(~valid).sum():,}  ({100*(~valid).mean():.2f}%)")

# ---- Time span -------------------------------------------------------------
tmin, tmax = t[valid].min(), t[valid].max()
import datetime as dt
def ts(x): return datetime_utc(x)
def datetime_utc(x): return dt.datetime.utcfromtimestamp(int(x)).strftime("%Y-%m-%d %H:%M")
span_days = (tmax - tmin) / 86400.0
print(f"Time span        : {datetime_utc(tmin)} → {datetime_utc(tmax)} UTC  ({span_days:.1f} days)")
local_day = ((t - PDT_OFFSET) // 86400)
n_distinct_days = np.unique(local_day[valid]).size
print(f"Distinct local days: {n_distinct_days}   (Shenzhen: {SHEN['days']} weekdays)\n")

# ===========================================================================
# 1) GRIDDING GEOMETRY  (the user's concern)
# ===========================================================================
print("="*74)
print("1) GRIDDING GEOMETRY — constant 0.01° square cells (faithful) vs forced 48×90")
print("="*74)
vlat, vlon = lat[valid], lon[valid]
def bbox(p_lo, p_hi):
    return (np.percentile(vlat, p_lo), np.percentile(vlat, p_hi),
            np.percentile(vlon, p_lo), np.percentile(vlon, p_hi))
full = (vlat.min(), vlat.max(), vlon.min(), vlon.max())
trim = bbox(0.5, 99.5)
def grid_dims(b):
    la = math.ceil((b[1]-b[0]) / GRID_SIZE_DEG); lo = math.ceil((b[3]-b[2]) / GRID_SIZE_DEG)
    return la, lo
for label, b in [("FULL bbox (min/max)", full), ("TRIMMED bbox (0.5–99.5 pct)", trim)]:
    la, lo = grid_dims(b)
    print(f"{label}:")
    print(f"   lat {b[0]:.4f}..{b[1]:.4f} ({(b[1]-b[0]):.3f}°)   lon {b[2]:.4f}..{b[3]:.4f} ({(b[3]-b[2]):.3f}°)")
    print(f"   → 0.01° grid = {la} × {lo} cells  ({la*lo:,} cells; Shenzhen 48×90={48*90:,})")
phi = math.radians(float(np.median(vlat)))
km_lat = GRID_SIZE_DEG * 110.574
km_lon = GRID_SIZE_DEG * 111.320 * math.cos(phi)
print(f"\nPhysical cell @ SF lat {math.degrees(phi):.2f}°: {km_lat:.3f} km (N-S) × {km_lon:.3f} km (E-W)")
print(f"Physical cell @ Shenzhen lat 22.60°: {0.01*110.574:.3f} km (N-S) × {0.01*111.320*math.cos(math.radians(22.6)):.3f} km (E-W)")
print(f"ε-ball=2 (5×5 window) physical span @ SF ≈ {2*max(km_lat,km_lon):.2f} km (cf. Shenzhen ≈ {2*0.01*110.574:.2f} km)")
# Contrast: forcing the trimmed bbox into 48×90
la_t, lo_t = trim[1]-trim[0], trim[3]-trim[2]
print(f"\nIf FORCED into 48×90 over the trimmed bbox: cell = "
      f"{(la_t/48)*110.574:.3f} km × {(lo_t/90)*111.320*math.cos(phi):.3f} km  "
      f"→ aspect/size DIFFERS from Shenzhen (distorts ε-ball meaning)")

# Use TRIMMED bbox as the grid origin for the rest (keeps stray trips from inflating the grid)
b = trim
GX, GY = grid_dims(b)
ix = np.clip(((vlat - b[0]) / GRID_SIZE_DEG).astype(int), 0, GX-1)
iy = np.clip(((vlon - b[2]) / GRID_SIZE_DEG).astype(int), 0, GY-1)
vcab = cab[valid]; vt = t[valid]; vocc = occ[valid]; vday = local_day[valid]
vhour = ((vt - PDT_OFFSET) % 86400) // 3600

# ===========================================================================
# 2) OCCUPANCY SEGMENTATION → seeking / driving corpora  (R2/R3)
# ===========================================================================
print("\n" + "="*74)
print("2) TRAJECTORY CORPUS via occupancy flag (R2/R3) — split on occ-change or >5min gap")
print("="*74)
order = np.lexsort((vt, vcab))          # sort by cab, then time ascending
o_cab, o_t, o_occ = vcab[order], vt[order], vocc[order]
o_ix, o_iy, o_day = ix[order], iy[order], vhour[order]*0 + vday[order]
# segment breaks: cab change, occupancy change, or time gap
dt_prev = np.diff(o_t, prepend=o_t[0])
cab_change = np.diff(o_cab, prepend=o_cab[0]-1) != 0
occ_change = np.diff(o_occ, prepend=o_occ[0]+5) != 0
gap = dt_prev > GAP_SEC
# within a cab, dt across cab boundary is meaningless; force break there
seg_break = cab_change | occ_change | (gap & ~cab_change)
seg_id = np.cumsum(seg_break) - 1
n_seg = seg_id[-1] + 1
# per-segment occupancy + length
seg_first = np.flatnonzero(seg_break)
seg_occ = o_occ[seg_first]
seg_len = np.diff(np.append(seg_first, len(o_occ)))
seg_cab = o_cab[seg_first]
seg_day = o_day[seg_first]
seek_mask = seg_occ == 0
drive_mask = seg_occ == 1
# Keep only segments with >=2 points (a trajectory needs a sequence)
multi = seg_len >= 2
print(f"Raw segments       : {n_seg:,}  (seeking {seek_mask.sum():,} / driving {drive_mask.sum():,})")
print(f"Trajectories (≥2 pts): seeking {int((seek_mask&multi).sum()):,}  |  driving {int((drive_mask&multi).sum()):,}")
print(f"   Shenzhen retained : seeking {SHEN['seeking']:,}  |  driving {SHEN['driving']:,}")
def pct(a, qs=(50,90,99)):
    return ", ".join(f"p{q}={np.percentile(a,q):.0f}" for q in qs)
print(f"Seeking traj length (pts): {pct(seg_len[seek_mask&multi])}  max={seg_len[seek_mask&multi].max()}")
print(f"Driving traj length (pts): {pct(seg_len[drive_mask&multi])}  max={seg_len[drive_mask&multi].max()}")
# fares (pickups) = occ 0->1 transitions within same cab & small gap
trans = (np.diff(o_occ, prepend=o_occ[0]) == 1) & ~cab_change & ~(gap)
print(f"Fares (pickups, 0→1 transitions): {int(trans.sum()):,}   (Shenzhen corpus {SHEN['corpus']:,})")
med_dt = np.median(dt_prev[(~cab_change) & (dt_prev>0)])
print(f"Median inter-ping interval: {med_dt:.0f} s")

# ===========================================================================
# 3) PAIR FEASIBILITY for the same-driver Siamese retrain (R3)
# ===========================================================================
print("\n" + "="*74)
print("3) DISCRIMINATOR PAIR FEASIBILITY (R3) — same-driver/different-day positives")
print("="*74)
# per cab: # distinct days; per (cab,day): # seeking trajectories
seek_cab = seg_cab[seek_mask&multi]; seek_day = seg_day[seek_mask&multi]
from collections import defaultdict
days_per_cab = defaultdict(set)
trajs_per_cabday = defaultdict(int)
for c, d in zip(seek_cab, seek_day):
    days_per_cab[c].add(d); trajs_per_cabday[(c, d)] += 1
dpc = np.array([len(v) for v in days_per_cab.values()])
cabday_ge5 = sum(1 for v in trajs_per_cabday.values() if v >= N_CTX)
cabs_ge2days = int((dpc >= 2).sum())
print(f"Cabs with seeking data       : {len(days_per_cab)}")
print(f"Days/cab (seeking)           : {pct(dpc)}  min={dpc.min()} max={dpc.max()}")
print(f"Cabs with ≥2 days (→ can form same-driver/diff-day positives): {cabs_ge2days}/{len(days_per_cab)}")
print(f"(cab,day) cells with ≥{N_CTX} seeking trajs (→ valid N=5 batch): {cabday_ge5:,}")
# crude positive-pair capacity: sum over cabs of C(days,2) capped is huge; report trajs available
print(f"Total seeking trajs available for pairing: {len(seek_cab):,}  "
      f"(target ≈10k labeled PAIRS → need ≈10k trajs; {'OK' if len(seek_cab)>=10000 else 'TIGHT'})")

# ===========================================================================
# 4) ACTIVE-UNIT COUNT under faithful grid (R5)
# ===========================================================================
print("\n" + "="*74)
print("4) ACTIVE (cell,hour) UNITS (R5) — proxy for n_active")
print("="*74)
# distinct (cell, hour) that see any activity
cellhour = (o_ix.astype(np.int64)*GY + o_iy)*24 + o_day*0 + vhour[order]
uniq_ch = np.unique(cellhour).size
# stricter: (cell,hour) with >=1 distinct cab (supply proxy)
ch_key = (ix.astype(np.int64)*GY + iy)*24 + vhour
# count distinct cabs per (cell,hour)
order2 = np.argsort(ch_key)
ch_sorted = ch_key[order2]; cab_sorted = vcab[order2]
boundaries = np.flatnonzero(np.diff(ch_sorted, prepend=ch_sorted[0]-1) != 0)
# distinct cab count per group
distinct_cabs = []
for i in range(len(boundaries)):
    s = boundaries[i]; e = boundaries[i+1] if i+1 < len(boundaries) else len(ch_sorted)
    distinct_cabs.append(np.unique(cab_sorted[s:e]).size)
distinct_cabs = np.array(distinct_cabs)
ch_ge1 = (distinct_cabs >= 1).sum()
ch_ge2 = (distinct_cabs >= 2).sum()
total_units = GX*GY*24
print(f"Grid {GX}×{GY}×24 = {total_units:,} possible (cell,hour) units")
print(f"(cell,hour) with any activity        : {uniq_ch:,}  ({100*uniq_ch/total_units:.1f}% of grid)")
print(f"(cell,hour) with ≥1 distinct cab     : {ch_ge1:,}")
print(f"(cell,hour) with ≥2 distinct cabs    : {ch_ge2:,}")
print(f"   Shenzhen n_active (supply>0.5 over 48×90×24={48*90*24:,}): {SHEN['n_active']:,}")

# ===========================================================================
# 5) ACTION-SPACE INVARIANT at 0.01° (R6) — cGAIL max(|dx|,|dy|) <= 1
# ===========================================================================
print("\n" + "="*74)
print("5) cGAIL ACTION-SPACE compatibility (R6) at 0.01° cells")
print("="*74)
# consecutive within-segment displacement in cells
same_seg = ~seg_break
dx = np.abs(np.diff(o_ix, prepend=o_ix[0]))
dy = np.abs(np.diff(o_iy, prepend=o_iy[0]))
step = np.maximum(dx, dy)
in_seg_step = step[same_seg]
ok = (in_seg_step <= 1).mean()
print(f"Within-trajectory consecutive steps: {in_seg_step.size:,}")
print(f"  satisfy max(|dx|,|dy|) ≤ 1 (cGAIL-legal): {100*ok:.1f}%")
for thr in (1,2,3,5):
    print(f"  ≤{thr} cells: {100*(in_seg_step<=thr).mean():.1f}%")
print(f"  step distribution: {pct(in_seg_step, (50,90,99))} max={in_seg_step.max()}")
print(f"  (Shenzhen pipeline already drops ~38–50% of trajectories to such filters)")

print("\n" + "="*74)
print("DONE")
print("="*74)
