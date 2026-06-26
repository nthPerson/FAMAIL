"""Pure, dependency-light helpers for runner enrichment (Plan 4).
No torch import. Every function is unit-tested; runners call these at write-sites."""
from __future__ import annotations
import math
import random
import numpy as np
from famail_temporal.baselines.transmission import terminal_cell_histogram


def t_ci(values, confidence: float = 0.95):
    """t-based confidence interval of the MEAN. (nan, nan) if fewer than 2 values."""
    vals = [float(v) for v in values]
    n = len(vals)
    if n < 2:
        return (float("nan"), float("nan"))
    from scipy.stats import t
    mean = float(np.mean(vals))
    sem = float(np.std(vals, ddof=1)) / math.sqrt(n)
    h = sem * float(t.ppf(0.5 + confidence / 2.0, n - 1))
    return (mean - h, mean + h)


def shannon_entropy_bits(hist) -> float:
    """Shannon entropy (base-2) of a non-negative vector; normalized internally."""
    p = np.asarray(hist, dtype=np.float64)
    total = p.sum()
    if total <= 0:
        return 0.0
    p = p[p > 0] / total
    return float(-np.sum(p * np.log2(p)))


def degeneracy_scalars(terminal_pickups, gen_cells, *, n_cells) -> dict:
    """E11 collapse check: terminal-cell entropy (bits) + trip-length mean/std.
    Low entropy or near-1 trip length => degenerate generator."""
    hist = terminal_cell_histogram(terminal_pickups, n_cells=n_cells)
    lengths = [len(seq) for seq in gen_cells]
    return {
        "terminal_cell_entropy_bits": shannon_entropy_bits(hist),
        "mean_trip_length": float(np.mean(lengths)) if lengths else 0.0,
        "std_trip_length": float(np.std(lengths, ddof=1)) if len(lengths) > 1 else 0.0,
    }


def effective_edited_fraction(n_edited, n_total, w) -> float:
    """E28: weight-adjusted edited mass = (n_edited*w) / (n_edited*w + n_unedited)."""
    n_edited = float(n_edited); n_unedited = float(n_total) - n_edited
    num = n_edited * float(w)
    denom = num + n_unedited
    return float(num / denom) if denom > 0 else 0.0


def dose_response_table(per_arm, paired_vs_raw, weights) -> list:
    """E10: flat rows w -> {delta_f_causal, wilcoxon_p, fidelity_b, fidelity_a}."""
    rows = []
    for w in weights:
        arm = f"edited_w{int(w)}"
        pc = paired_vs_raw.get("f_causal", {}).get(arm, {})
        a = per_arm.get(arm, {})
        rows.append({
            "w": int(w),
            "delta_f_causal": float(pc.get("mean", float("nan"))),
            "wilcoxon_p": pc.get("wilcoxon_p"),
            "fidelity_b": float(a.get("fidelity_b", {}).get("mean", float("nan"))),
            "fidelity_a": float(a.get("fidelity_a", {}).get("mean", float("nan"))),
        })
    return rows


def chosen_placebo_ids(raw_traj_ids, edited_id_set, placebo_seed, k=None) -> list:
    """E27: deterministic re-derivation of the placebo subset's trajectory_ids.
    Mirrors run_weighted_bc_smoke.random_subset_weight_vector's selection:
    sample k indices from the NON-edited positions with random.Random(placebo_seed),
    then map positions back to trajectory_ids."""
    edited = set(edited_id_set)
    non_edited_pos = [i for i, tid in enumerate(raw_traj_ids) if int(tid) not in edited]
    n = len(edited) if k is None else k
    chosen_pos = random.Random(placebo_seed).sample(non_edited_pos, n)
    return [int(raw_traj_ids[i]) for i in chosen_pos]
