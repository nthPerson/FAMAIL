# Supply-Lift Editing — Meeting 42 Briefing

> Source document for slide assembly. Sections map roughly to slides; tables and bolded
> numbers are the load-bearing content. Items marked **[TBD — validation run]** get filled
> from tonight's Shenzhen PRIMARY run before presenting.

---

## 1. Where we left off: the finding that forced this

Meeting 41's P0 deliverable ("the big one") landed: **external fairness metrics — demographic
parity, disparate impact, supply/demand ratio, Theil — improve before→after editing**, on
metrics the editor never optimizes. Shenzhen improves unanimously across all three
demographic feature sets.

But the audit surfaced a serious caveat: **the improvement is leveling-down.** The
over-served group's service is reduced; the under-served group is (nearly) untouched. The
gap closes from the top. Philosophically and practically, that's the weak form of fairness
(Parfit's classic objection): nobody is better off, some are worse off.

## 2. Why the existing editor *cannot* lift up (mechanism analysis)

We traced the leveling-down to three compounding structural causes in the trim editor —
none of them bugs, all of them design consequences:

| Cause | Evidence |
|---|---|
| **Selection never sees the poor group** | All 2,455 trim edits originate *and terminate* inside the advantaged group's cells |
| **Leverage asymmetry** | Y = S/max(D, 0.5); ∂Y/∂D = −S/D² is ~32× stronger for removing demand from rich cells; **93.2% of disadvantaged units sit at/below the demand floor**, where demand edits are inert |
| **Supply is frozen** | The editor can only move demand; median supply: disadvantaged cells 1.8 taxis vs advantaged 17.6 (10×) |

Two follow-ups closed the escape routes:

- **Demand-only "lift" is provably perverse.** A greedy oracle showed demand editing could
  raise mean(Y | disadvantaged) by +1.54 — but only by *deleting service from poor areas*
  (removing recorded pickups). Optimizing the ratio by shrinking its numerator's demand is
  laundering, not fairness.
- **Option A (training-side reweighting) is negative.** Weighted-BC policies trained on
  edited data allocate *fewer* pickups to poor areas, dose-dependently (share 0.0500 →
  0.0452 at w30, 0/6 seeds favorable, p = .031). The published F_causal gains at the policy
  level are system-level leveling-down.

**Conclusion: the only non-perverse lever that can raise Y = S/D for the under-served group
is the numerator — supply.** At the demand floor, ΔY = 2·ΔS: adding taxi presence to
starved cells is the direct, honest mechanism. The editor needed a supply channel.

## 3. Gate G0: is there even enough headroom? (Stage-0 oracle)

Before building anything, we bounded the achievable effect. A greedy oracle reroutes
seeking tails (last 4 states + pickup, up to 2 cells) toward disadvantaged cells and
measures the ceiling on Δmean(Y | disadvantaged). Threshold to proceed: **≥ +0.3**.

| Ceiling (baseline mean(Y|D) = 7.07) | Value |
|---|---|
| Full accounting (supply + relocated demand) | **+0.882** |
| Distinct-seeking semantics | +0.827 |
| **Supply channel alone** (demand pinned) | **+0.786 — 2.6× the threshold** |

The decomposition row exists because we caught the oracle exploiting the demand floor
(moving pickup demand out of near-empty poor cells inflates Y artificially — the same
perverse pattern as before). **The supply channel alone clears the gate comfortably**, so
the go-decision rests on the honest number. 5,538 trajectories qualify as reroute
candidates; the binding constraint is candidate supply, not the edit budget.

## 4. What supply-lift editing is

Three additions to the trajectory editor; everything else is unchanged.

### 4.1 The supply channel (endogenous ΔS)
Taxi presence becomes part of the objective. One 5-minute seeking state = **1/12 of an
hourly presence unit**, spread over its 5×5 neighborhood (matching how the supply grid was
originally built from GPS). When a tail moves, its presence mass moves with it —
*differentiably* — and the objective sees supply as `clamp(S_base + ΔS, floor)`. Trim
edits also contribute their (small) ΔS to evaluation — honestly accounted, but never fed
back into their own optimization (published trim behavior is preserved bit-for-bit).

### 4.2 Supply-gradient attribution (how lift candidates are chosen)
The intuitive version, in three moves:

1. **Ask the map a question.** For every cell-hour of the city, ask the fairness
   objective: *"if one more taxi cruised here, how much fairer would service be?"* One
   backward pass answers all ~34,500 what-ifs at once → a **value-of-presence heat map**,
   glowing where neighborhoods are starved relative to demand and demographics. (The
   F_causal component of this gradient has a closed form; the autograd matches it exactly
   in tests.)
2. **Find drivers who almost pass through the glow.** For each trajectory's final approach
   (last ~4 seeking states + pickup), slide it up to 2 blocks in each of 24 directions and
   read the heat map. Rank all 95k trajectories by their best slide: *which drivers were
   already passing close enough to a taxi desert that a two-block detour puts them in it?*
3. **Let the real optimizer decide.** The ranking only nominates. Each selected trajectory
   runs the full iterative editor — exact detour derived from the complete objective
   (fairness + realism/fidelity), supply endogenous — so the final reroute is optimized,
   not the screen's linear guess.

Selection: trim keeps absolute precedence (its 2,455 edits are untouched); lift fills the
remaining budget (k = 10,000 → ~7,545 lift edits), skipping non-positive scores.

### 4.3 Tapered, physically-valid rerouting
The pickup moves the full offset; earlier tail states move progressively less
(taper 0.25/0.5/0.75/1.0); the anchor never moves. A repair step guarantees every
consecutive step of the edited trajectory still satisfies the **king-move rule**
(max(|dx|,|dy|) ≤ 1) — verified correct by exhaustive enumeration on 3,000 random cases.

**This also fixes a latent double standard we found during design:** preprocessing filters
*source* trajectories that violate king-move adjacency, but the legacy editor's ≥2-cell
pickup moves violated it on *output* with no filter. Now: lift edits are 100% compliant by
construction (unrepairable ⇒ edit skipped, counted); trim edits are repaired wherever
compatible with reproducing the legacy pickup cell exactly (~87–90% in smokes; the
remainder deliberately falls back to legacy behavior so published numbers reproduce, and is
counted). Production percentages: **[TBD — validation run]**.

## 5. Validation discipline (what makes this defensible)

Seven gates, two hard human checkpoints:

| Gate | What it proves | Status |
|---|---|---|
| G0 | Oracle headroom ≥ +0.3 | **PASSED** (+0.786 supply-only) |
| G1 | Legacy mode reproduces today's pipeline **bit-for-bit** | Test green (exact-equality end-to-end) |
| G3 | Trim pickups + demand grid identical to legacy in combined runs | **Proven at production scale**: combined run reproduces 2,455 trim edits, F_causal 0.7988 → 0.8132 (+0.0144) exactly |
| G2 | Soft ΔS matches hard recounts (two tiers) | Tier-1 exact; tier-2 see §6 · **[TBD]** |
| G4 | King-move compliance | Lift 100% by construction · trim **[TBD]** |
| G5 | Fidelity stable vs trim-only | **[TBD — validation run]** |
| G6 | Δmean(Y|D) > 0 with CI, F_causal not regressed, + external metrics | **[TBD — validation run]** |

Engineering war stories worth one slide (they show the pipeline is *audited*, not just
built): three production incidents, all the same root family — float32 residues from the
editor's per-edit demand accounting (~10⁻⁹-scale) — surfaced as a crash, a pathological
slowdown, and a metrics-path rejection. Each was reproduced, root-caused (one via live
stack sampling of a stalled 14-hour run), minimally fixed without touching the legacy
bit-reproduction path, and regression-tested.

Plus one optimization with teeth: profiling showed the fidelity discriminator's LSTM is
~70–75% of every optimization iteration, re-encoding inputs that don't change between
iterations. Caching the constant encodings — **verified bitwise-identical on CUDA, both
modes** — cut production lift edits from ~6.2 s to ~2.5 s (~2.5×). Edit runs that projected
15.5 h now finish in ~8.

## 6. The honesty ledger (say these before a reviewer does)

1. **Metric firewall, three rings.** (i) *Optimized*: F_spatial / F_causal / F_fidelity —
   what the gradient sees. (ii) *Design-targeted, not optimized*: mean(Y|D) and the
   SDR family — we aimed the supply mechanism at these; improvement there is confirmatory.
   (iii) *Genuinely external*: DP gap, disparate impact, Theil, per-group service levels,
   and the tier-2 recount. **"Improves metrics we never optimized" claims ride on ring
   (iii) only.**
2. **Two-tier supply accounting.** Tier-1 (the optimizer's convention: fractional presence
   mass) is generous. Tier-2 re-derives *distinct-taxi* counts from raw GPS with edited
   tails substituted — an audit the optimizer can't see (and it reproduces the production
   supply grid exactly, MAE = 0, before any edits). Early smoke evidence: tier-2 ΔS ≈
   **0.25–0.30×** tier-1 (a driver already cruising nearby adds no *distinct* presence),
   directionally consistent where nonzero. The paper reports both tiers; production-scale
   tier-2 numbers: **[TBD]**.
3. **Channel decomposition.** Any Δmean(Y|D) we report gets split into supply-channel vs
   demand-channel contributions, so the lift-up claim demonstrably rides supply provision,
   not the demand-floor artifact we ourselves flagged.
4. **Expectations on F_causal.** Lift edits are individually non-harmful to the internal
   objective by construction, but their F_causal yield per edit should be *smaller* than
   trim's (trim exhausted the sharpest 2,455 candidates; lift's supply masses are small and
   its pickups carry demand *into* poor cells, which partially offsets). The headline
   success criterion is lift-up on external metrics with F_causal not regressing — not a
   bigger F_causal.

## 7. Status and what's next

- **Now:** full Shenzhen PRIMARY validation run (k = 10,000: 2,455 trim + ~7,545 lift) is
  in flight on the optimized stack; completes tonight. Gate battery (G2/G4/G5/G6) runs on
  its output; hard checkpoint review follows.
- **On approval (Task 11, the headline plan):** re-run all 4 datasets (3 Shenzhen feature
  sets + SF) with supply-lift as the paper's main edited data; external metrics
  before→after on all; weighted-BC downstream sweeps; Option-A rollout re-evaluation on
  lift-edited data; curated `PAPER/supply-lift/` results.
- **Deadlines:** KDD abstract Jul 19, paper Jul 26. Pre-agreed fallback ladder: if
  validation gates fail → oracle number goes to limitations and the published trim results
  stand; if the build had slipped past ~Jul 16 → drop the BC re-run; if metrics come back
  weak → supply-lift becomes an additive subsection instead of the headline.

---

### Numbers to fill in from tonight's run (single source: the checkpoint gate package)

- [ ] Combined-run F_causal after trim+lift (trim-only reference: 0.8132)
- [ ] Δmean(Y|D) with bootstrap CI + supply/demand channel split
- [ ] External metrics table (DP, DI, Theil, per-group levels) before→after
- [ ] Tier-2 recount at production scale (ΔS ratio, metric survival)
- [ ] G4 adjacency percentages (lift / trim / fallback counts)
- [ ] G5 fidelity vs trim-only
- [ ] n_lift, n_taper_infeasible_{trim,lift}, run wall-time on the cached stack
