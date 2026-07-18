"""Option A — supply-endogenous rollout evaluation (leveling-down follow-up).

Question: do BC policies trained on edited+upweighted data reposition SEEKING
SUPPLY (state-visits) and PICKUPS (terminal states) toward under-served areas,
relative to raw-trained policies?

Protocol matches the published weighted-BC sweep (run_weighted_bc_smoke /
cleaned_hcm_6seed): TrajectoryLSTM (driver-conditioned), train_mle 20 epochs,
lr 1e-3, batch 32, max_batch_tokens 8192, corpus-matched rollout contexts
(one rollout per real trajectory, same driver + start cell + start t_block,
identical across arms). Upweighting = loss weights (w x per-token CE on the
edited trajectories), NOT duplication.

Metrics per (arm, seed), per equity axis (migrant/comp/housing, district-
extremes grouping): share of generated state-visits and share of terminal
pickups landing in D (disadvantaged) / A (advantaged) / middle cells, plus a
supply-per-pickup ratio-of-sums per group. Paired per-seed deltas vs the raw
arm; Wilcoxon across seeds.

Outputs: JSON per policy (crash-resumable) + summary.json under
famail_temporal/baselines/external_fairness/results/option_a_rollout/
(gitignored). Reference rows (raw vs edited TRAINING CORPUS shares) via
--refs-only (no training, CPU-fast).

Usage:
  python PAPER/external-metrics/scripts/option_a_rollout_eval.py --refs-only
  python PAPER/external-metrics/scripts/option_a_rollout_eval.py --smoke
  python PAPER/external-metrics/scripts/option_a_rollout_eval.py            # full (GPU, ~3.5h)
"""
import argparse
import json
import pickle  # trusted repo-internal artifact (histories.pkl), same as localized_metrics.py
import sys
import time
from pathlib import Path

sys.path.insert(0, "/home/robert/FAMAIL")
import numpy as np
import torch

from famail_temporal import config
from famail_temporal.data.loader import DataBundle
from famail_temporal.baselines import external_fairness as ef
from famail_temporal.baselines import external_fairness_io as io
from famail_temporal.baselines.run_level2_table import (
    build_edited_corpus, traj_training_data,
)
from famail_temporal.baselines.gan import config as gc
from famail_temporal.baselines.gan.drivers import build_driver_index
from famail_temporal.baselines.gan.generator import TrajectoryLSTM
from famail_temporal.baselines.gan.train_mle import train_mle
from famail_temporal.baselines.gan.rollout import generate_trajectories
from famail_temporal.baselines.gan.sequences import unflat_cell
from famail_temporal.utils.seeding import set_all_seeds

# --edit-dir is REQUIRED: the old default silently pointed at the legacy
# pre-cleanup corpus (2026-06-29 hcm, edited=2455) and caused an era-wrong
# launch on 2026-07-17 (FB-ROLLOUT, aborted +25s; see run ledger).
DEFAULT_OUT_DIR = Path("/home/robert/FAMAIL/famail_temporal/baselines/external_fairness/"
               "results/option_a_rollout")
AXES = ["MigrantRatio", "CompPerCapita", "AvgHousingPricePerSqM"]


def cell_group_grids():
    """Per-axis (GX, GY) group grid: 1=D, 0=A, -1=middle/excl (district extremes)."""
    sel = io._enriched_selected_grid()  # (GX, GY, 3) housing, comp, migrant
    grids = {}
    for j, axis in enumerate(io.EQUITY_AXES):
        vals = sel[:, :, j].ravel()
        g = ef.region_extremes(vals, disadvantaged_high=io.DISADVANTAGED_HIGH[axis])
        grids[axis] = g.reshape(sel.shape[:2])
    return grids


def shares_from_cells(cells_xy, group_grid):
    """cells_xy: iterable of (x, y). Returns dict of visit shares per group."""
    n = {1: 0, 0: 0, -1: 0}
    for x, y in cells_xy:
        n[int(group_grid[x, y])] += 1
    tot = sum(n.values())
    if tot == 0:
        return {"share_D": float("nan"), "share_A": float("nan"),
                "share_mid": float("nan"), "n": 0}
    return {"share_D": n[1] / tot, "share_A": n[0] / tot,
            "share_mid": n[-1] / tot, "n": tot}


def rollout_metrics(sequences, grids):
    """sequences: list of flat-cell-id lists. Shares of states + terminals per axis,
    plus supply-per-pickup ratio-of-sums per group."""
    all_states, terminals = [], []
    for seq in sequences:
        if not seq:
            continue
        pts = [unflat_cell(c) for c in seq]
        all_states.extend(pts)
        terminals.append(pts[-1])
    out = {}
    for axis in AXES:
        gg = grids[axis]
        states = shares_from_cells(all_states, gg)
        picks = shares_from_cells(terminals, gg)
        # supply-per-pickup ratio of sums per group (endogenous-S proxy)
        spp = {}
        for glabel, gval in (("D", 1), ("A", 0)):
            s_cnt = states["n"] * states[f"share_{glabel}"]
            p_cnt = picks["n"] * picks[f"share_{glabel}"]
            spp[glabel] = (s_cnt / p_cnt) if p_cnt > 0 else float("nan")
        out[axis] = {"states": states, "pickups": picks, "supply_per_pickup": spp}
    return out


def corpus_metrics(trajs, grids):
    """Same metrics computed on real Trajectory objects (reference rows)."""
    seq_like = []
    for t in trajs:
        seq_like.append([int(s.x_grid) * config.GRID_DIMS[1] + int(s.y_grid)
                         for s in t.states])
    return rollout_metrics(seq_like, grids)


def weight_vector(trajs, edited_ids, w):
    return [float(w) if int(t.trajectory_id) in edited_ids else 1.0 for t in trajs]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refs-only", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seeds", type=str, default="0,1,2,3,4,5")
    ap.add_argument("--arms", type=str, default=None,
                     help="Comma-separated arm names (default: raw,edited,edited_w10,"
                          "edited_w30). --smoke normally narrows this to raw,edited_w30, "
                          "unless --arms is passed explicitly -- then it is respected as-is.")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--edit-dir", type=str, required=True,
                     help="Dir containing histories.pkl for the edited corpus "
                          "(default: legacy trim-only k-10000 causal-emphasis run).")
    ap.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT_DIR),
                     help="Output dir for policy_*.json / summary.json "
                          "(default: legacy option_a_rollout dir).")
    args = ap.parse_args()

    EDIT_DIR = Path(args.edit_dir)
    OUT_DIR = Path(args.out_dir)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log = open(OUT_DIR / "run.log", "a")

    def say(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        log.write(line + "\n")
        log.flush()

    grids = cell_group_grids()
    bundle = DataBundle.load()
    raw_trajs = bundle.trajectories
    with open(EDIT_DIR / "histories.pkl", "rb") as f:  # trusted repo artifact
        histories = pickle.load(f)
    edited_trajs = build_edited_corpus(raw_trajs, histories)
    edited_ids = {int(h.original.trajectory_id) for h in histories}
    say(f"corpus={len(raw_trajs)} edited={len(edited_ids)}")

    # ---- reference rows: the training data itself ----
    refs_path = OUT_DIR / "corpus_refs.json"
    if not refs_path.exists():
        refs = {"raw_corpus": corpus_metrics(raw_trajs, grids),
                "edited_corpus": corpus_metrics(edited_trajs, grids)}
        refs_path.write_text(json.dumps(refs, indent=1))
        say("wrote corpus_refs.json")
    if args.refs_only:
        return 0

    # ---- policy training + rollout ----
    seeds = [int(s) for s in args.seeds.split(",")]
    arms_explicit = args.arms is not None
    arms = (args.arms if arms_explicit
            else "raw,edited,edited_w10,edited_w30").split(",")
    slice_n = None
    epochs = args.epochs
    if args.smoke:
        seeds, epochs, slice_n = [0], 1, 1500
        if not arms_explicit:
            arms = ["raw", "edited_w30"]
        say(f"SMOKE MODE: 1 seed, {len(arms)} arm(s) ({','.join(arms)}), "
            f"1 epoch, 1500-traj slice")

    device = (torch.device("cuda" if torch.cuda.is_available() else "cpu")
              if args.device == "auto" else torch.device(args.device))
    say(f"device={device}")
    if device.type == "cpu" and not args.smoke:
        say("FATAL: full run requires GPU (CPU ~18h/policy). Aborting.")
        return 2

    r_slice = raw_trajs[:slice_n] if slice_n else raw_trajs
    e_slice = edited_trajs[:slice_n] if slice_n else edited_trajs
    d2i = build_driver_index(r_slice)
    n_drivers = len(d2i)
    D_raw = traj_training_data(r_slice, d2i)
    D_edited = traj_training_data(e_slice, d2i)

    # ---- fairness-baseline arm fixtures (arm-constant; built ONCE, no RNG) ----
    # Mirrors run_weighted_bc_smoke.py's --fairness-reweigh/--fairness-penalty wiring.
    # Index-aligned to r_slice/D_raw (not the unsliced raw_trajs), same convention as
    # edited_w's weight_vector(e_slice, ...) below -- required for --smoke's 1500-traj slice.
    _fair_reweigh_w = None
    if "fair_reweigh" in arms:
        from famail_temporal.baselines.fairness_baseline import (
            fairness_reweigh_weight_vector)
        _fair_reweigh_w = fairness_reweigh_weight_vector(r_slice, bundle)
        say("fair_reweigh: inverse-SDR instance weights on RAW corpus")

    _penalty_fn = None
    if any(a.startswith("fair_penalty_l") for a in arms):
        from famail_temporal.baselines.fairness_baseline import (
            unit_groups_and_sdr, cell_masks_for_vocab, dp_gap_penalty)
        from famail_temporal.baselines.gan.sequences import flat_cell
        cell_group, _sdr = unit_groups_and_sdr(bundle)
        m_d, m_a = cell_masks_for_vocab(
            cell_group, gc.VOCAB_SIZE, lambda cell: flat_cell(cell[0], cell[1]))
        assert int(m_d.sum()) > 0 and int(m_a.sum()) > 0, "empty group mask"
        m_d, m_a = m_d.to(device), m_a.to(device)
        _penalty_fn = lambda lg, tg: dp_gap_penalty(lg, tg, m_d, m_a, pad_id=gc.PAD)
        say("fair_penalty: DP-gap penalty fn built (masks non-empty)")

    def arm_spec(name):
        if name == "raw":
            return D_raw, None, 0.0
        if name == "edited":
            return D_edited, None, 0.0
        if name.startswith("edited_w"):
            w = float(name.split("edited_w")[1])
            return D_edited, weight_vector(e_slice, edited_ids, w), 0.0
        if name == "fair_reweigh":
            return D_raw, _fair_reweigh_w, 0.0
        if name.startswith("fair_penalty_l"):
            lam = float(name.split("fair_penalty_l")[1])
            return D_raw, None, lam
        raise ValueError(name)

    for seed in seeds:
        for arm in arms:
            tag = f"{arm}_seed{seed}" + ("_smoke" if args.smoke else "")
            out_path = OUT_DIR / f"policy_{tag}.json"
            if out_path.exists():
                say(f"skip {tag} (exists)")
                continue
            D, sw, plam = arm_spec(arm)
            say(f"train {tag} (epochs={epochs}) ...")
            t0 = time.time()
            set_all_seeds(seed)
            model = TrajectoryLSTM(n_drivers=n_drivers).to(device)
            train_mle(model, D["sequences"], D["contexts"], epochs=epochs,
                      lr=gc.MLE_LR, batch_size=gc.MLE_BATCH_SIZE, device=device,
                      driver_idxs=D["driver_idxs"], max_batch_tokens=8192,
                      sample_weights=sw,
                      penalty_fn=(_penalty_fn if plam else None),
                      penalty_lambda=plam)
            t_train = time.time() - t0
            say(f"  trained in {t_train:.0f}s; rolling out {len(D_raw['contexts'])} ...")
            t0 = time.time()
            seqs = generate_trajectories(
                model, D_raw["contexts"], max_len=gc.MAX_GEN_LEN, device=device,
                gen_batch_size=gc.GEN_BATCH_SIZE, driver_idxs=D_raw["driver_idxs"])
            t_roll = time.time() - t0
            res = {"arm": arm, "seed": seed, "smoke": args.smoke,
                   "epochs": epochs, "n_rollouts": len(seqs),
                   "t_train_s": round(t_train, 1), "t_rollout_s": round(t_roll, 1),
                   "metrics": rollout_metrics(seqs, grids)}
            out_path.write_text(json.dumps(res, indent=1))
            say(f"  wrote {out_path.name} (rollout {t_roll:.0f}s)")
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    # ---- aggregate: paired deltas vs raw ----
    if args.smoke:
        say("smoke complete (no aggregation)")
        return 0
    from scipy.stats import wilcoxon
    per = {}
    for p in OUT_DIR.glob("policy_*.json"):
        r = json.loads(p.read_text())
        if r.get("smoke"):
            continue
        per.setdefault(r["arm"], {})[r["seed"]] = r["metrics"]
    summary = {}
    for arm in per:
        if arm == "raw":
            continue
        summary[arm] = {}
        for axis in AXES:
            for kind in ("states", "pickups"):
                deltas = [per[arm][s][axis][kind]["share_D"]
                          - per["raw"][s][axis][kind]["share_D"]
                          for s in sorted(set(per[arm]) & set(per["raw"]))]
                if not deltas:
                    continue
                try:
                    pval = float(wilcoxon(deltas).pvalue) if len(deltas) >= 5 else None
                except ValueError:
                    pval = None
                summary[arm][f"{axis}.{kind}.share_D_delta"] = {
                    "mean": float(np.mean(deltas)), "std": float(np.std(deltas)),
                    "per_seed": deltas, "n_pos": int(sum(d > 0 for d in deltas)),
                    "wilcoxon_p": pval,
                }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=1))
    say("wrote summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
