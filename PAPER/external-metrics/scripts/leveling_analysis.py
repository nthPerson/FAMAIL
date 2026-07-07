"""Why does the editor level-down? Empirical flow + leverage analysis.

Analyzes the headline Shenzhen PRIMARY edit (k=10000, causal-emphasis, cleaned):
1. FLOWS: pickup mass moved between migrant-axis groups (D=high-migrant, A=low-migrant, M=middle/excluded).
2. LEVERAGE: per-unit |dY/dD| by group (the gradient-asymmetry hypothesis).
3. ORACLE: best achievable lift of the disadvantaged group's mean Y by ANY
   demand-only edit of k pickups (upper bound), vs. the observed leveling-down.
"""
import sys, pickle, json
sys.path.insert(0, "/home/robert/FAMAIL")
import numpy as np

from famail_temporal.data.loader import DataBundle
from famail_temporal import config
from famail_temporal.baselines import external_fairness as ef
from famail_temporal.baselines import external_fairness_io as io
from famail_temporal.baselines.datasets import pickup_unit_of, pickup_mass

EDIT = "/home/robert/FAMAIL/famail_temporal/results/2026-06-29T12-06-55_k-10000_causal_emphasis_no-dedup_cleaned_hcm"
FLOOR = config.DEMAND_FLOOR

bundle = DataBundle.load()
mask = bundle.mask_3d
S3 = bundle.active_taxis_3d.astype(np.float64)
D3_before = bundle.pickup_3d.astype(np.float64)

# --- cell-level migrant grouping (mirrors region_extremes on the unit values) ---
sel = io._enriched_selected_grid()          # (48,90,3): housing, comp, migrant
migrant_cell = sel[:, :, 2]
groups_cell = ef.region_extremes(migrant_cell.ravel(), disadvantaged_high=True).reshape(migrant_cell.shape)
# 1 = D (high migrant, under-served), 0 = A (low migrant), -1 = middle/excl/nan

# unit-level labels identical to the report's grouping
demo = io.per_unit_demographics(bundle)
g_unit = ef.region_extremes(demo["MigrantRatio"], disadvantaged_high=True)

def group_mean_Y(pick3d):
    Y = io.service_ratio_Y(pick3d, bundle)
    return Y[g_unit == 1].mean(), Y[g_unit == 0].mean()

# --- 1. FLOWS ---
# Safe: histories.pkl is a trusted artifact produced by this repo's own
# pipeline (evaluation/persistence.py), same load pattern as localized_metrics.py.
with open(f"{EDIT}/histories.pkl", "rb") as f:
    hist = pickle.load(f)

after = D3_before.copy()
lab = {1: "D(poor/hi-migrant)", 0: "A(rich/lo-migrant)", -1: "M(middle/excl)"}
order = [1, 0, -1]
flow_mass = {(a, b): 0.0 for a in order for b in order}
flow_n = {(a, b): 0 for a in order for b in order}
n_moved_cell = 0
for h in hist:
    ox, oy, ot = pickup_unit_of(h.original)
    mx, my, mt = pickup_unit_of(h.modified)
    mo, mm = pickup_mass(bundle, ot), pickup_mass(bundle, mt)
    after[ox, oy, ot] = max(after[ox, oy, ot] - mo, FLOOR)
    after[mx, my, mt] += mm
    go, gm = int(groups_cell[ox, oy]), int(groups_cell[mx, my])
    flow_mass[(go, gm)] += mo
    flow_n[(go, gm)] += 1
    if (ox, oy) != (mx, my):
        n_moved_cell += 1

print(f"histories: {len(hist)}, moved to a different cell: {n_moved_cell}")
print("\nFLOW MATRIX (n edits, origin group -> destination group):")
print(f"{'':22s}" + "".join(f"{lab[b]:>22s}" for b in order))
for a in order:
    print(f"{lab[a]:22s}" + "".join(f"{flow_n[(a,b)]:>22d}" for b in order))

yd0, ya0 = group_mean_Y(D3_before)
yd1, ya1 = group_mean_Y(after)
print(f"\nmean Y  D: {yd0:.4f} -> {yd1:.4f} (delta {yd1-yd0:+.4f})")
print(f"mean Y  A: {ya0:.4f} -> {ya1:.4f} (delta {ya1-ya0:+.4f})")

# net demand change per group (units)
dD = after - D3_before
for g in order:
    cellmask3 = np.broadcast_to((groups_cell == g)[:, :, None], mask.shape) & mask
    print(f"net demand-mass change in {lab[g]:20s}: {dD[cellmask3].sum():+.4f}")

# --- 2. LEVERAGE: |dY/dD| = S/D^2 (0 below floor) per group ---
D_N = D3_before[mask]; S_N = S3[mask]
Dc = np.maximum(D_N, FLOOR)
lev = np.where(D_N > FLOOR, S_N / Dc**2, 0.0)      # marginal |dY/dD| for removal
lev_add = S_N / Dc**2                               # for addition (approx, ignores floor-crossing)
for g, name in ((1, "D"), (0, "A")):
    m = g_unit == g
    print(f"\ngroup {name}: units={m.sum()}, mean Y={S_N[m].sum()/Dc[m].sum():.2f} (ratio-of-sums)"
          f"\n  mean |dY/dD| add: {lev_add[m].mean():.3f}   removal: {lev[m].mean():.3f}"
          f"\n  median D: {np.median(D_N[m]):.3f}  median S: {np.median(S_N[m]):.3f}"
          f"\n  frac units at/below DEMAND_FLOOR (removal does nothing): {(D_N[m] <= FLOOR).mean():.2%}")

# --- 3. ORACLE: max lift of mean(Y|D) from removing k pickup events from D cells ---
# eligible = actual trajectory pickups located in D-group active cells
elig = []
for i, tr in enumerate(bundle.trajectories):
    cx, cy, tb = pickup_unit_of(tr)
    if groups_cell[cx, cy] == 1 and mask[cx, cy, tb]:
        elig.append((cx, cy, tb))
print(f"\neligible pickups located in D-group cells: {len(elig)} (edit budget was k=10000)")

# greedy: repeatedly remove one event-mass from the D-cell unit with max marginal gain
from collections import Counter
cnt = Counter(elig)
units = list(cnt.keys())
Dwork = {u: float(D3_before[u]) for u in units}
avail = dict(cnt)
K = 10000
total_gain, used = 0.0, 0
import heapq
def gain(u):
    d = Dwork[u]; m = pickup_mass(bundle, u[2]); s = float(S3[u])
    return s / max(d - m, FLOOR) - s / max(d, FLOOR)
heap = [(-gain(u), u) for u in units]
heapq.heapify(heap)
while heap and used < K:
    negg, u = heapq.heappop(heap)
    if avail[u] <= 0:
        continue
    g_now = gain(u)
    if g_now <= 0:
        continue
    if -negg - g_now > 1e-12:           # stale entry, reinsert with fresh gain
        heapq.heappush(heap, (-g_now, u))
        continue
    Dwork[u] = max(Dwork[u] - pickup_mass(bundle, u[2]), FLOOR)
    avail[u] -= 1
    total_gain += g_now
    used += 1
    if avail[u] > 0:
        heapq.heappush(heap, (-gain(u), u))

N_D = int((g_unit == 1).sum())
print(f"ORACLE (remove up to k={K} D-cell pickups, greedy): events used={used}, "
      f"max delta mean(Y|D) = +{total_gain / N_D:.4f}")
print(f"observed advantaged-side move:  delta mean(Y|A) = {ya1-ya0:+.4f}")
print(f"observed DP-gap close from leveling-down: {-(ya1-ya0):.4f} vs oracle lift-up ceiling: {total_gain / N_D:.4f}")
