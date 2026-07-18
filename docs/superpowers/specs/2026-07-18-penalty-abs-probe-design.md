# Absolute-Value Penalty Probe — Design (Robert-approved to run 2026-07-18)

**Question.** §4.5's in-processing penalty finding ("inert at every dose that trains
stably... and, in this formulation, destructive") is honestly scoped to the SIGNED
DP-gap formulation (`pen = mass_a − mass_d`, the adjudicated plan-governs deviation from
the 2026-07-16 spec's absolute-value text). This probe runs the ABSOLUTE formulation
(`pen = |mass_a − mass_d|`) so the scoping caveat can be closed either way.

**Why it can differ.** With the advantaged group over-served (`mass_a > mass_d`), the two
formulations have identical gradients — until an overshoot crosses the gap through zero,
where signed keeps pushing (the plausible mechanism of the λ=1000 collapse: F_causal
−0.2053, Fid-B 0.514) while absolute pushes back toward equality. The probe tests whether
absolute (a) stays inert at moderate λ like signed, and (b) avoids the catastrophic end —
i.e., whether a constructive operating range exists that the signed formulation hid.

## Build (Task 1 — implementer)

- `famail_temporal/baselines/fairness_baseline.py`: add `dp_gap_penalty_abs(...)` with the
  SAME signature as `dp_gap_penalty`, returning `torch.abs(dp_gap_penalty(...))` — a thin
  composition; the proven signed function is NOT modified.
- Tests (extend the module's existing test file): (1) equality `abs_pen == |signed_pen|` on
  the existing fixtures; (2) the DISCRIMINATING test — construct a fixture where
  `mass_d > mass_a` (negative signed gap) and assert the gradients of the two formulations
  point in OPPOSITE directions there (this is the entire behavioral difference; a test that
  only checks the positive-gap region proves nothing).
- `famail_temporal/baselines/run_weighted_bc_smoke.py`: `--fairness-penalty-abs` flag
  mirroring `--fairness-penalty`'s wiring EXACTLY (arm naming `fair_penalty_abs_l<λ>`,
  same λ plumbing, default-off untouched). No changes to `train_mle` (the existing
  `penalty_fn` kwarg carries the variant).
- Suites stay green (baselines + analysis). Default-off invariant needs no new gate run:
  the change is purely additive (new function + new flag); reviewer verifies by diff that
  no existing code path changed.

## Runs (Task 2 — controller, ledger-wrapped chain `fb_abs_chain.sh`)

1. **FB-PENALTY-ABS-PILOT**: seed 0, λ ∈ {1, 3.16, 10, 100, 1000} — the exact signed-suite
   grid for point-by-point comparability (~50 min).
2. **FB-PENALTY-ABS**: n=6 seeds at λ ∈ {10, 1000} — the inert-representative and the
   catastrophic-representative doses (~2.5h). Escalate to more λs only if the pilot
   disagrees with signed anywhere.

## Decision rule (pre-committed)

- Absolute ≈ signed at every tested dose (inert ≤10; destructive/unstable at 1000) →
  §4.5 drops the "in this formulation" scoping: state the penalty has no constructive
  operating range under EITHER formulation; retire the formulation-adjudication % NOTE.
- Absolute finds ANY constructive dose (ΔF_causal significantly positive with Fid-B cost
  bounded near w30's) → STOP, surface to Robert immediately (this would be a real
  competing-baseline finding; no slot-in without his read).
- Mixed/other → report as measured, surface.

**Metric firewall** unchanged (no F_causal anywhere in baseline training). Era: s10 α*
corpus `2026-07-10T02-06-37_alpha_sweep_s10_c80_f10_filtered` (SZ; the signed suite's
corpus — like-for-like). GPU idle; no chain gate needed, markers for restart-resume only.

*Scope note: this doc doubles as the plan (2 tasks); the build is a one-function thin
variant of reviewed code, executed SDD-style with an Opus implementer + reviewer.*
