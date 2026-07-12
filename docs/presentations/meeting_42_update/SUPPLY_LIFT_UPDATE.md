# Supply-Lift Editing — Meeting 42 Update (2026-07-09)

> Presentation context for the supply-lift workstream (built and validated 2026-07-07 → 07-09).
> Complements `famail_temporal/baselines/meeting_prep/MEETING_42_PREP.md` (other P0 topics).
> All numbers below are reproducible from the results dirs listed at the end; statistical claims
> carry 95% bootstrap CIs (paired over spatial units, B=1000–2000).

---

## 1. One-paragraph summary (suggested opening slide)

We extended the trajectory editor with a **supply lever**: in addition to relocating pickups
("trim", the published mechanism), the editor can now reroute a trajectory's **seeking tail**
(last 4 states + pickup, tapered translation) toward under-served cells, with the supply grid
**endogenous** to the objective via a differentiable ΔS channel. Result on Shenzhen PRIMARY:
**F_causal 0.7988 → 0.8210 (+0.0222, vs +0.0144 trim-only)**, and — the new capability — the
**under-served (migrant) group's service ratio rises for the first time**, with the supply
component of that rise **statistically significant on both cities**. The improvement is no longer
pure leveling-down.

## 2. Why we did this (recap, 1 slide)

Two findings from last week motivated the change:
- **Leveling-down mechanism** (`PAPER/external-metrics/LEVELING_DOWN_MECHANISM.md`): all 2,455
  trim edits stay inside the advantaged group; demand-only editing provably cannot raise the
  under-served group's Y = supply/demand (supply was frozen; 93% of disadvantaged cells sit at
  the demand floor).
- **Rollout-allocation eval, negative** (formerly "Option A"): BC policies trained on trim-edited
  data allocated *fewer* pickups to poor areas, dose-dependently (0/6 seeds, p = .031).

A Stage-0 oracle gated the build: greedy upper bound on Δmean(Y | disadvantaged) from tail
rerouting = **+0.88** (supply-channel-only variant: **+0.79**) vs. a go/no-go threshold of +0.3.

## 3. What the new editor does (1–2 slides, mechanism)

- **Two edit modes.** `trim` (existing behavior, byte-identical optimization) and `lift`
  (new): candidate trajectories are scored by the objective's **supply gradient** ∂L/∂S, and the
  optimizer translates the whole seeking tail with linearly tapered offsets (0.25/0.5/0.75/1.0),
  so the trajectory *physically cruises through* the under-served area before pickup.
- **Endogenous supply.** Each tail state contributes presence mass (1 state = 1/12 hourly
  presence, 5×5 neighborhood) to a differentiable ΔS grid that feeds F_spatial and F_causal
  *during* optimization — the editor sees the supply consequences of its own moves.
- **Adjacency repair (new data-quality property).** The legacy editor silently broke the
  king-move rule (max(|dx|,|dy|) ≤ 1) on every ≥2-cell move. Tail translation includes a
  provably-exact repair (reviewer brute-forced 3,000 cases: returns "infeasible" exactly when no
  compliant assignment exists). **Edits that cannot be repaired are skipped** — the uniform rule
  adopted this week (see §6).
- **Budget split.** k = 10,000: trim keeps its published selection (2,455); lift fills the rest
  (7,545). Trim's pickups and demand grid reproduce legacy exactly (verified in production:
  the combined run reproduced the published trim numbers mid-run before lift began).

## 4. Headline results (the key slides)

### 4.1 Shenzhen PRIMARY (filtered dataset = paper headline)

| Metric | Before | After | Δ | vs trim-only |
|---|---|---|---|---|
| F_causal (optimization label) | 0.7988 | **0.8210** | **+0.0222** | +0.0144 |
| F_spatial | 0.1034 | 0.1098 | **+0.0064** | ~0 (trim-only: −0.0002) |
| Theil (unfairness, ↓ better) | 0.1550 | **0.1468** | −0.0082 (CI excl. 0) | — |
| Gini-DSR (↓ better) | 0.8981 | 0.8856 | −0.0125 | — |
| mean(Y | migrant-disadvantaged) | 7.073 | **7.120** | +0.047 (CI excl. 0) | ≈ 0 (untouched) |
| SDR gap (migrant, adv−dis) | 14.20 | **13.34** | narrows | narrowed by leveling-down only |

Fidelity: **stable** — Fidelity-A (identity) 0.846 edited vs 0.849 raw (published trim-only delta
was −0.002; lift edits are *less* identity-damaging than trim edits: −0.003 vs −0.006).
Fidelity-B shows lift's expected distributional cost (0.264 vs trim 0.160) — whole-tail
relocation vs pickup-only; disclose as a by-design trade-off.

### 4.2 The channel decomposition (THE framing slide — see §5)

Δmean(Y|D) is one metric evaluated at three points: Y(S,D) → Y(S,D′) → Y(S′,D′). The edit moves
both demand (pickups relocate) and supply (tails relocate), so the total splits exactly:

| Channel (Shenzhen PRIMARY, migrant axis) | Δ | 95% CI | Significant? |
|---|---|---|---|
| **Supply** (tails add presence to D cells) | **+0.0091** | [+0.0054, +0.0130] | **YES** |
| Supply, tier-2 distinct-taxi recount | **+0.0242** | [+0.0208, +0.0279] | **YES** |
| Demand (pickup relocation) | +0.0378 | [−0.0065, +0.0836] | no |
| Total | +0.0468 | [+0.0021, +0.0930] | yes |

**Tier-2** = we re-counted supply from raw GPS as *distinct taxis per neighborhood-hour* with the
edited tails substituted (the honest semantics; the editor's internal convention is a fractional
presence approximation). The recount machinery reproduces the production supply grid **exactly**
(MAE 0.0) and matched 100% of the 10,000 edit histories to raw pings. Under the honest count the
supply effect is ~2.7× larger than the internal convention credits.

### 4.3 SF / Cabspotting (second city, filtered)

| Metric | Value | Note |
|---|---|---|
| F_causal | 0.8752 → **0.9079 (+0.0328)** | trim-only published: +0.0139 → **2.4×** |
| Theil | −0.0081, CI [−0.0095, −0.0067] | significant |
| Migrant DP gap (district extremes) | −0.076, CI excl. 0 | **was n.s. under trim-only** |
| Migrant DI | +0.0061 toward parity, CI excl. 0 | new |
| Supply channel Δmean(Y|D) | **+0.0195, CI excl. 0** | **core claim replicates** |
| Demand channel | −0.0525, CI excl. 0 (negative) | see §5.2 |
| Total Δmean(Y|D) | −0.0330, significant negative | see §5.2 |
| Fidelity-A (sf_12 discriminator) | raw 0.9578 → edited 0.9581 | **stable**; matches published to ~1e-7; lift Fid-B cost 0.265 ≈ Shenzhen's 0.2645 (same by-design signature both cities) |

## 5. The two framing discussions for this meeting

### 5.1 Lead the lifting-up claim with the SUPPLY channel

The demand channel's contribution is statistically indistinguishable from zero on Shenzhen (huge
variance — near-floor cell shuffling) and *negative* on SF. The supply channel is significant,
consistent, and replicates across both cities (+0.009/+0.024 SZ tier-1/tier-2; +0.020 SF).
Recommended claim: *"rerouting seeking behavior adds real (distinct-taxi-verified) supply to
under-served areas, producing a small but statistically robust rise in their service ratio —
the first non-leveling-down improvement in this line of work."* A reviewer who decomposes our
numbers will find exactly this; we should present it decomposed ourselves.

### 5.2 The SF cross-metric tension (demand endogeneity, discussion item)

On SF, lift edits reroute pickups INTO under-served areas: demand there rises, so the S/D ratio
falls (total Δmean(Y|D) = −0.033) — **yet DP, DI, and Theil all improve significantly**, including
the migrant axis that trim-only editing could not move. These are not contradictory: more rides
*served* in poor areas is a demand-side fairness gain (DP measures pickup-share parity) even as
it lowers the ratio metric. This connects directly to the demand-endogeneity argument in the
objective-motivation write-up (recorded demand is suppressed by under-supply). **Decision needed:
which SF fairness story goes in the paper** — external metrics (uniformly better) with the ratio
metric caveated, or hold the SF mean(Y|D) claim entirely. Shenzhen needs no such caveat.

## 6. The skip-on-infeasible rule (disclosure slide)

- ~5% of trim edits (115/2,455 SZ; 47/1,371 SF) have *no* king-compliant tail repair. Originally
  they fell back to the legacy (adjacency-violating) move; we adopted the uniform rule **"an edit
  is applied only when a compliant repair exists"** and reverted them (post-process; identification
  provably exact — it replays the editor's own feasibility decision; verified the same 115 on
  re-derivation; ΔS/demand grids rebuilt from scratch match persisted artifacts bit-for-bit).
- **Disclosure:** the rule was adopted *before* its metric effects were computed, but it moved
  every metric favorably (SZ +0.0209 → +0.0222; SF +0.0223 → +0.0328 — the infeasible edits were
  actively counterproductive). Provenance documented in each `_filtered/PROVENANCE.md`, including
  that survivors were not re-optimized (negligible coupling, avoids multi-hour re-runs).
- Compliance after filtering: SZ **100%** absolute. SF: **100% edit-relative** (no edit introduces
  a violation); absolute is 87.4% because **14.9% of SF's raw source trajectories already violate
  adjacency** (Cabspotting GPS gaps ≤ 18.6 cells) — a data-quality fact worth one caveat sentence,
  and our edited corpus is *more* compliant than the raw corpus (87.4% vs 85.0%).

## 7. Validation rigor (1 slide, bullets)

- Two hard human checkpoints (oracle gate; full gate review) + fresh-agent review of every task.
- **G1**: legacy mode (TAIL_LEN=0) is bit-for-bit the published pipeline (end-to-end exact-equality
  test); published numbers remain reproducible.
- **G2**: tier-2 raw-GPS recount, exact reproduction + 100% history matching (§4.2).
- **G4**: adjacency (§6). **G5**: fidelity stable (§4.1). **G6**: external metrics + CIs (§4).
- Three float32-residual production incidents found and fixed with reviewed regression tests
  (root causes proven, e.g. −1.86e-9 drift demonstrated arithmetically).
- Performance: discriminator-encoding cache (bitwise-verified equivalence) roughly **halved**
  edit-run time (15.5h → 7.9h at k=10k); SF runs in ~40 min.

## 8. Weighted-BC sweep — LANDED this morning (Pillar 2 strengthens)

Full published protocol (10 arms × 6 seeds, paired edited−raw, Wilcoxon):

| Arm | ΔF_causal | p | vs published trim-only |
|---|---|---|---|
| edited, w=1 (vanilla BC) | +0.0023 | .16 (n.s.) | same null — vanilla BC averages it away |
| edited, w=10 | **+0.0232** | .031, 6/6 seeds | +0.0186 |
| edited, w=20 | **+0.0280** | .031 | +0.0242 |
| edited, w=30 | **+0.0310** | .031, 6/6 | +0.0274 |
| random placebo w10/w30 | −0.001 / −0.003 | n.s. | null — gain remains edit-specific |
| most-fair control w10–30 | +0.003 → +0.001 | n.s. | null — not a selection artifact |

**New vs trim-only:** ΔF_spatial now ALSO propagates — positive and significant at every weight
(+0.0042 w10 → +0.0057 w30, p=.031, 6/6), which the trim-only corpus never achieved. Fidelity-A
within noise at all weights (largest effect +0.0006 at w30 — technically p=.031 but trivial in
magnitude and in the favorable direction); Fidelity-B rises modestly with weight (paired diff
+0.016 at w30) — same trade-off family as the data-level Fidelity-B note in §4.1.

## 8b. Rollout-allocation eval — LANDED (attenuated, NOT reversed — disclose)

Same protocol as the prior negative result (4 arms × 6 seeds, corpus-matched rollouts, paired
vs raw). Pickup share allocated to migrant-disadvantaged areas by the trained policies:

| Arm | trim-only era | supply-lift era | Read |
|---|---|---|---|
| edited (w=1) | +0.0003 n.s. | −0.0008 (p=.031) | ~0 both |
| w=10 | −0.0033 (0/6, p=.031) | **−0.0023 (0/6, p=.031)** | drain attenuated ~30% |
| w=30 | −0.0048 (0/6, p=.031) | **−0.0029 (0/6, p=.031)** | drain attenuated ~40% |
| seeking-state share (all arms) | n.s. | n.s. | policies did NOT learn to cruise poor areas |

**Honest three-level story for the paper:** (1) data-level lift-up is real and
distinct-taxi-verified (supply channel, CI-significant both cities); (2) fairness metrics
propagate through weighted BC strongly (F_causal +0.031, F_spatial now too); (3) the *allocation
behavior* of trained policies still tilts away from poor areas — ~40% less than under trim-only
data, but systematically (0/6 seeds). We can claim (1)+(2); we must disclose (3) as
attenuated-but-persistent, and it motivates future work (training-side allocation constraints,
not just data-side editing). Recommend presenting this as the honest boundary of the method.

- Remaining today: curation slots filled + final whole-branch review.

## 9. Deferred (parked deliberately, not forgotten)

- 2 alternate Shenzhen feature-set runs (robustness parity) — idle-GPU work later.
- SF tier-2 recount plumbing (SF supply-channel number is internal-convention only for now).
- A further ~25% speedup that is *not* bitwise-safe (~7e-9 drift) — needs sign-off if wanted.

## 10. Where everything lives

- Headline datasets: `famail_temporal/results/2026-07-08T14-03-03_supply_lift_v1_shz_primary_filtered/`
  and `.../2026-07-08T22-43-06_supply_lift_v1_sf12_filtered/` (each with `PROVENANCE.md`).
- External metrics: `famail_temporal/baselines/external_fairness/results/shz-primary-supplylift-filtered/`
  and `.../sf12-supplylift-filtered/` (tables + figures + report.md).
- Design + plan: `docs/superpowers/specs/2026-07-08-supply-lift-editing-design.md`,
  `docs/superpowers/plans/2026-07-08-supply-lift-editing.md`.
- Full execution log (incidents, reviews, every number's origin): `.superpowers/sdd/progress.md`.
- Motivation docs: `PAPER/external-metrics/LEVELING_DOWN_MECHANISM.md` (+ §6.4 rollout-eval result).
