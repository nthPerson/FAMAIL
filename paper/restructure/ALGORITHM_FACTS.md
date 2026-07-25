# ALGORITHM FACTS — ground truth for FATE (do not contradict these in prose)

Purpose: Dr. Zhang's directives are sometimes written from an imperfect model of the
algorithm. Any paper text produced during the restructure must stay consistent with
THIS sheet. When a directive conflicts, flag it in the request ledger
(`ZHANG_DIRECTIVES.md`) instead of bending the description. Verification sources:
`paper/sections/03_methodology.tex`, `paper/sections/appendix.tex`,
`famail_temporal/` (code), `PAPER/` (artifact index: `PAPER/DATA_INVENTORY.md`).

## Setting and quantities
- City discretized into 0.01° grid cells × hourly time blocks; a (cell, hour) pair with
  recorded activity is an **active unit**; N ≈ 34,500 on Shenzhen.
- Demand D_i = recorded pickups in unit i. Supply S_i = active-taxi presence aggregated
  over the 5×5 cell neighborhood. Service ratio Y_i = S_i / max(D_i, d0), demand floor
  d0 = 0.5.
- Demographic covariates x_i (housing price, per-capita compensation, migrant share)
  resolve at DISTRICT granularity (~10 districts) — the associational/ecological caveat
  is mandatory; no causal or individual-level claims.
- Corpus T = real per-driver passenger-seeking trajectories, each ending in a pickup.
  D and S are deterministic aggregations of T.

## Objective (single differentiable scalarization, gradient ASCENT)
- L = α_sp·F_spatial + α_de·F_demo + α_fi·F_fidelity, α = (0.1, 0.8, 0.1), selected by
  a three-class empirical criterion (sweep in §4/App C).
- **F_demo** = R^T(I−H)R / R^T M R = 1 − r²_demo. Stage 1: flexible power basis g0(D)
  fits demand→service-ratio; residual R_i = Y_i − g0(D_i). Stage 2 evaluated in closed
  form via hat matrix H of the z-scored demographic design matrix; M = centering
  matrix. r²_demo = share of demand-adjusted service variation explained by
  demographics; HIGHER F_demo = LOWER demographic dependence = fairer. 0 = fully
  unfair, 1 = fully fair. H is constant during editing (demographics fixed), so each
  edit step implicitly re-fits the regression exactly. O(N) evaluation identity +
  Frisch–Waugh–Lovell exactness: Appendix A.
- **F_spatial** = 1 − ½(Gini(DSR) + Gini(ASR)); DSR_i = D_i/S_i is the
  **departure-service ratio** (renamed 2026-07-24 to match its source, Su et al. 2018
  `su2018taxigini`; code keeps legacy name "demand-service ratio"), ASR_i = A_i/S_i the
  arrival-service ratio (A = dropoffs). Demographic-independent spatial regularizer.
- **F_fidelity**: frozen driver-identity discriminator (ST-SiameseNet family), trained
  once, never updated during editing. Under FATE's small bounded edits its gradient is
  near zero → it is a **guardrail, not a driver of edits**. It is an identity-level
  behavioral-fidelity signal, NOT a guarantee of full trajectory realism (that axis is
  Fidelity-B, measured at evaluation only, never optimized).

## The editor: three steps, two phases (this is the core to not misstate)
- **Attribution decides which trajectories are worth editing** (the objective is
  optimized one trajectory at a time). Two mechanisms, one per phase:
  1. **Demand deficit attribution** (drives trim / outcome-side): because M and I−H are
     idempotent, r²_demo admits an EXACT per-unit partition of the fairness deficit
     across active units (App A, Eq. 4), with a signed variant separating over- from
     under-served units. Trim selects trajectories whose pickups land in the
     highest-deficit units.
  2. **Supply-gradient attribution** (drives lift / resource-aware): v_i = ∂L/∂S_i
     evaluated by autograd at ΔS = 0 (one backward pass; F_demo component has a closed
     form, App A Eq. 5) = marginal fairness value of added taxi presence at each unit →
     a value-of-presence map. A linearized-offset screen then ranks trajectories by the
     best bounded tail translation (rigid integer offsets in [−ε, ε]²) — the screen
     only NOMINATES; each nominee is then optimized under the FULL objective.
- **Trim (outcome-side edit)**: relocates selected pickups within the ε-ball, padding
  recorded demand into over-served, supply-rich cells so their measured over-service
  falls. Supply is FROZEN during trim. Trim's optimization is identical to the
  demand-only editor (the scientific-control property).
- **Lift (resource-aware edit)**: reroutes the final seeking tail (pickup + up to 4
  prior seeking states; ~80k of ~95k SZ trajectories are long enough). Pickup moves by
  full offset δ; earlier tail states by linearly tapered fractions w_j = j/ℓ
  (0.25/0.5/0.75/1.0 at ℓ=4); the anchor state before the tail NEVER moves. Moved
  states carry supply with them differentiably (each contributes 1/12 hourly presence
  unit to each cell of its 5×5 neighborhood; objective evaluates clamp(S+ΔS, s0=0.1)).
  Supply is ENDOGENOUS during lift: the optimizer sees the supply consequences of its
  own moves. Each completed edit updates a shared running state (later edits optimize
  against earlier results). Fidelity term scores the actual rerouted tail every
  iteration.
- **Per-iteration update**: δ ← clip(η·sign(∇_δ L), −ε, ε); δ_total ← clip(δ_total + δ,
  −ε, ε). No pickup ever moves more than ε = 2 cells from its recorded location,
  regardless of iteration count. Discrete cells are bridged by temperature-annealed
  soft cell assignment (best iterate kept).
- **Phase order**: trim first; lift fills the remaining budget with positive-score
  nominees; supply gradient computed on the POST-TRIM state; lift never alters trim's
  edits → demand-only results carry over unchanged into the combined run, and the
  trim-only vs trim+lift ablation isolates lift.
- **Budget k**: Shenzhen k = 10,000 → 2,455 trim selected (118 ≈ 5% reverted post hoc,
  2,337 net) + 7,545 lift. San Francisco k = 2,000 → 1,330 trim + 629 lift. Per-edit
  bound ε = 2 grid cells (L∞).

## Validity and the accept/skip/revert logic (frequent misstatement hazard)
- King-move rule: consecutive states move at most one cell per axis (the source data is
  filtered on this). Discretized edits pass through an EXACT backward-reachability
  repair returning the king-compliant assignment nearest the tapered targets, or
  reporting none exists.
- Infeasible edits are NOT applied: lift skips them in-loop; the ~5% of trim edits with
  no compliant repair are reverted post hoc BEFORE fairness metrics are computed.
- **Fidelity is NOT a per-edit accept/reject threshold.** It is (a) a weighted term in
  L at every iteration and (b) a reported evaluation gate (Fidelity-A) on the finished
  corpus. The accept/skip/revert decision is driven by VALIDITY (king-move
  feasibility), not by a fidelity test. Zhang's suggested pipeline step "evaluate
  fidelity → accept, skip, or revert" must be worded so it does not claim a fidelity
  threshold gate. Accurate 6-step pipeline: propose gradient edit → clip to ε →
  discretize + repair continuity (skip/revert if infeasible) → objective (fairness +
  fidelity terms) evaluated each iteration, best iterate kept → accepted edit updates
  the shared corpus state → repeat until budget k is spent.

## Why demand-only editing cannot help the under-served (current §3.3; must survive
somewhere — it is one of the paper's strongest sections)
- Empirical: every one of the 2,455 SZ trim pickups originated in advantaged cells;
  none landed in a disadvantaged cell. Leveling-down ANALOGY ONLY (trim relocates
  under conservation; nothing destroyed).
- Structural reasons: (i) selection — deficit attribution concentrates in over-served
  high-residual cells; (ii) leverage — with supply frozen, ∂Y_i/∂D_i = −S_i/D_i²;
  93% of disadvantaged units sit AT the demand floor d0 where removing demand changes
  nothing; (iii) supply-side inequity — median taxi presence 1.8 (disadvantaged) vs
  17.6 (advantaged), untouchable by a demand-only editor.
- The one non-perverse lever: ∂Y_i/∂S_i = 1/max(D_i, d0) > 0 everywhere; at the floor,
  ΔY = 2ΔS → raising under-served service requires ADDING presence (lift).
- Demand endogeneity: recorded demand is suppressed where service has been thin, so a
  demand-adjusted metric under-detects inequity AND a demand-only editor has nothing to
  move in under-served areas. (Same assumption bounds metric and editor.)

## Downstream stage (edit-aware weighting)
- Edited slice ≈ a tenth of the corpus. Vanilla (uniform-weight) BC on the edited
  corpus is NULL (SZ +0.0016, n=12, p=0.11; predicted, not a failure). FATE upweights
  the edited demonstrations in the imitation loss (Kamiran–Calders instance reweighing
  transplanted to imitation learning).
- Dose-response monotone: +0.0217 (w10), +0.0267 (w20), +0.0297±0.0029 (w30, adopted,
  the knee; n=12, 12/12 positive, exact Wilcoxon p=.00049 in BOTH cities; SF w30
  +0.0333±0.0050). Controls at n=12 stay null/dominated: random size-matched subset
  and most-fair-selection.
- p = 0.031 is the n=6 sign-unanimity floor, NEVER an effect size on its own.

## Headline results (era-guarded; do not swap in stale numbers)
- Data-level SZ: F_demo 0.7988→0.8214 (+0.0226); F_spatial +0.0061. SF: 0.8752→0.9067
  (+0.0316); F_spatial +0.0139.
- External instruments (class iii): SZ DI +0.0162, DP gap −0.890, Theil −0.0087 (all
  CIs excl. 0). Gap closes from BOTH ends (advantaged falls, disadvantaged rises).
- Supply channel (the lift-up claim): SZ tier-1 +0.0176 / tier-2 distinct-taxi +0.0411;
  SF tier-1 +0.0209 / tier-2 +0.1027; SF total under tier-1 is net-negative (−0.0324)
  — read as demand-endogeneity observed in the wild; tier-2 total +0.0493 positive.
  Any single supply number states its accounting tier.
- Ablation: trim-only improves instruments but disadvantaged level is FLAT (+0.000);
  trim+lift raises it (+0.053, CI excl. 0); ΔF_spatial flips negative→positive.
- Three-ring metric firewall: (i) optimized (F_spatial/F_demo/F_fidelity);
  (ii) design-targeted not optimized (mean(Y|group), supply/demand ratios);
  (iii) genuinely external (DP, DI, Theil, distinct-taxi recount). "Improves metrics it
  never optimizes" claims ride ring (iii) ONLY.
- Fidelity: Fidelity-A 0.844 vs raw 0.848 (SZ; gate passed); Fidelity-B 0.187 disclosed
  as the by-design distributional cost, concentrated in tail-relocation components.

## Things FATE is NOT (guard against drift)
- NOT an imitation-learning or generative method: it edits real trajectories and
  generates nothing (data-augmentation family; downstream BC is the evaluation vehicle).
- NOT a causal method: F_demo is associational (partial R² over ~10 district profiles).
- NOT guaranteeing realism: identity preservation only; Fidelity-B cost disclosed.
- NOT a per-trajectory fairness fix: fairness is collective (corpus-level), measured
  over the aggregate service allocation.
- There is NO edit-budget (k) sweep in the current results. The dose-response sweeps
  that exist are: upweighting dose (w10–w50), oversampling dose (d2.5k–d10k), α weight
  sweep (6 editor runs). A k-sweep would be NEW compute — flag any text implying it
  exists.
