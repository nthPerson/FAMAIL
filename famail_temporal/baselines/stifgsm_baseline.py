"""Baseline trajectory editors: vanilla ST-iFGSM, plain FGSM, random jitter.

Standalone Mission-3 module (Meeting-41 P0 #3). Attacks the frozen HuMID
discriminator on (original, perturbed) same-driver pairs over the CONTINUOUS
float-grid seeking states — the discriminator's native input space — with a
per-coordinate cumulative L-inf budget. Deliberately independent of
famail_temporal/algorithm/ (the frozen editor is untouched).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
import torch

from famail_temporal import config
from famail_temporal.utils.trajectory import Trajectory


@dataclass
class AttackOutcome:
    trajectory_id: Any
    perturbed_xy: np.ndarray   # (S, 2) float64 — attacked x,y per state
    final_p: float             # discriminator P(same driver) at the kept iterate
    iterations_run: int
    delta: np.ndarray          # (S, 2) applied perturbation (== perturbed - original)


def _features(traj: Trajectory) -> np.ndarray:
    return np.array(
        [[s.x_grid, s.y_grid, float(s.time_bucket), float(s.day_index)]
         for s in traj.states],
        dtype=np.float32,
    )


def _batch(trajs, profiles, device):
    """Pad to (B, 1, Lmax, 4) N=1 identity branches + masks + profiles."""
    lens = [len(t.states) for t in trajs]
    lmax = max(lens)
    b = len(trajs)
    x = torch.zeros(b, 1, lmax, 4, dtype=torch.float32, device=device)
    m = torch.zeros(b, 1, lmax, dtype=torch.bool, device=device)
    p = torch.zeros(b, 11, dtype=torch.float32, device=device)
    for i, t in enumerate(trajs):
        f = torch.from_numpy(_features(t)).to(device)
        x[i, 0, : lens[i]] = f
        m[i, 0, : lens[i]] = True
        p[i] = torch.from_numpy(np.asarray(profiles[t.driver_id], dtype=np.float32)).to(device)
    return x, m, p, lens


def attack_trajectories(
    trajectories: List[Trajectory],
    disc: torch.nn.Module,
    profiles: Dict[Any, np.ndarray],
    mode: str,
    *,
    epsilon: float | None = None,
    step: float | None = None,
    max_iterations: int | None = None,
    patience: int | None = None,
    convergence_tol: float | None = None,
    seed: int = 0,
    device: str = "cpu",
    batch_size: int = 256,
) -> List[AttackOutcome]:
    if mode not in ("ifgsm", "fgsm", "random"):
        raise ValueError(f"unknown mode '{mode}'")
    epsilon = config.EPSILON_BALL if epsilon is None else float(epsilon)
    step = config.STEP_SIZE_ALPHA if step is None else float(step)
    max_iterations = config.MAX_ITERATIONS if max_iterations is None else int(max_iterations)
    patience = config.PATIENCE if patience is None else int(patience)
    convergence_tol = (config.CONVERGENCE_TOL if convergence_tol is None
                       else float(convergence_tol))
    if mode == "fgsm":                    # single full-budget signed step
        max_iterations, step = 1, epsilon

    dev = torch.device(device)
    disc = disc.to(dev)
    disc.train(False)
    for prm in disc.parameters():
        prm.requires_grad_(False)

    outcomes: List[AttackOutcome] = []
    for start in range(0, len(trajectories), batch_size):
        chunk = trajectories[start : start + batch_size]
        x_orig, mask, prof, lens = _batch(chunk, profiles, dev)
        bsz, _, lmax, _ = x_orig.shape
        mask_f = mask.unsqueeze(-1).float()          # (B,1,L,1) freeze padding

        if mode == "random":
            g = torch.Generator(device="cpu").manual_seed(seed)
            signs = torch.randint(0, 2, (bsz, 1, lmax, 2), generator=g,
                                  dtype=torch.float32).mul_(2).sub_(1).to(dev)
            delta = (signs * epsilon) * mask_f
            x_adv = x_orig.clone()
            x_adv[..., :2] = x_orig[..., :2] + delta
            with torch.no_grad():
                p = disc(x_orig, x_adv, mask1=mask, mask2=mask,
                         profile_1=prof, profile_2=prof).reshape(-1)
            best_delta, best_p, iters = delta, p, torch.ones_like(p, dtype=torch.long)
        else:
            # Random start within the epsilon ball (per-trajectory, seed- and
            # global-index-derived so batch_size is invariant): a symmetric
            # loss landscape has zero gradient at delta=0, which would leave a
            # pure sign-gradient iterate stuck at the origin forever.
            delta_init = torch.zeros(bsz, 1, lmax, 2, device=dev)
            for i, ln in enumerate(lens):
                gi = torch.Generator(device="cpu").manual_seed(seed + start + i)
                noise = (torch.rand(ln, 2, generator=gi) * 2 - 1) * epsilon
                delta_init[i, 0, :ln, :] = noise.to(dev)
            delta = delta_init.clone().requires_grad_(True)
            best_p = torch.full((bsz,), float("inf"), device=dev)
            best_delta = torch.zeros_like(delta)
            iters = torch.zeros(bsz, dtype=torch.long, device=dev)
            stall = torch.zeros(bsz, dtype=torch.long, device=dev)
            for _ in range(max_iterations):
                x_adv = x_orig.clone()
                x_adv[..., :2] = x_orig[..., :2] + delta * mask_f
                p = disc(x_orig, x_adv, mask1=mask, mask2=mask,
                         profile_1=prof, profile_2=prof).reshape(-1)
                loss = p.sum()                      # descend P(same driver)
                grad = torch.autograd.grad(loss, delta)[0]
                with torch.no_grad():
                    live = stall < patience
                    improved = p < (best_p - convergence_tol)
                    upd = improved & live
                    best_delta[upd] = delta[upd]
                    best_p = torch.where(improved & live, p, best_p)
                    stall = torch.where(improved & live, torch.zeros_like(stall),
                                        stall + live.long())
                    iters += live.long()
                    d_new = (delta - step * grad.sign()).clamp_(-epsilon, epsilon)
                    delta.data = torch.where(live.view(-1, 1, 1, 1), d_new, delta.data)
                    delta.data *= mask_f            # padding stays zero
                if not bool((stall < patience).any()):
                    break
            best_delta = best_delta.detach() * mask_f

        x_final = x_orig.clone()
        x_final[..., :2] = x_orig[..., :2] + best_delta
        for i, traj in enumerate(chunk):
            s = lens[i]
            outcomes.append(AttackOutcome(
                trajectory_id=traj.trajectory_id,
                perturbed_xy=x_final[i, 0, :s, :2].detach().cpu().double().numpy(),
                final_p=float(best_p[i]),
                iterations_run=int(iters[i]),
                delta=best_delta[i, 0, :s].detach().cpu().double().numpy(),
            ))
    return outcomes
