# Meeting 41 Prep — Progress Since Meeting 40

**Date:** 2026-07-02 (prep written 2026-07-01) · **Covers:** everything since Meeting 40 (2026-06-25).
**Purpose:** one briefing for Dr. Zhang covering the week's two workstreams: (1) the **second dataset**
(San Francisco) — search, selection, implementation, and the demonstrated dual claim; and (2) the
**demographic-feature re-runs + the curated `PAPER/` deliverable** (done in a separate session; recapped here).

**Bottom line.** The paper now has **external validity on a second city**: the FAMAIL editor makes San
Francisco taxi trajectories **fairer (ΔF_causal +0.0139, beating Shenzhen's own +0.0128) while they stay
realistic (F_fidelity 0.968)** — with **no change to the algorithm, fairness metric, or fidelity
architecture**. Separately, the entire Shenzhen result set was re-run under **three demographic feature sets**
and packaged into a committed, adversarially-reviewed `PAPER/` bundle; the two-pillar story reproduces under
all three. Two decisions are teed up for the meeting: **(a) the `F_causal → F_demo` rename**, and **(b) how
hard to lean on the second dataset given one honest caveat (the fidelity signal is profile-dominated — but
*equally so on Shenzhen*).**

---

## TL;DR — what changed since Meeting 40

1. **Second dataset chosen and implemented: SF Cabspotting.** A grounded compatibility analysis showed the
   realism claim *requires* dense per-driver traces, which **rules out every OD-only US source** (NYC TLC,
   Chicago, DC). SF Cabspotting is the only US dense-trace set with a native occupancy flag, persistent taxi
   IDs, and an ACS/Census join — it drops into the pipeline with **zero algorithm change**. **§A.**

2. **Dual claim demonstrated on SF.** Editor causal-emphasis run: **F_causal 0.8752 → 0.8891 (Δ +0.0139)**
   while **F_fidelity = 0.968** (edit-induced change ~1e-5). +0.0139 **beats Shenzhen's +0.0128** from a
   density-comparable baseline. A matched fidelity-OFF run confirms the fidelity term is inert (+0.01392 vs
   +0.01394). **§C.**

3. **The discriminator retrains cleanly on SF — val-AUC 0.998** (Shenzhen 0.982), identical architecture and
   training protocol. Getting there required fixing 3 latent SF-pipeline day-encoding bugs and a learning-rate
   problem (first run failed at chance; diagnosed and fixed). Fairness baseline **provably preserved**
   (byte-identical count artifacts). **§B.**

4. **One honest caveat — the fidelity signal is profile-dominated (and so is Shenzhen's).** The discriminator
   classifies driver identity on the profile and largely ignores the seeking trajectory (fidelity gradient
   ≈0). I verified the **Shenzhen primary discriminator behaves identically** — this matches the earlier
   `fidelity-grad ≈ 0` gradient-heatmap finding. So F_fidelity is a *driver-identity-preservation* metric, not
   an active constraint; **it's a property of the whole method, not an SF weakness.** Decision needed on
   framing. **§C.4.**

5. **Separate session: demographic-feature re-runs + the `PAPER/` bundle.** The full Shenzhen experiment set
   (editor, L1, L2, weighted-BC, placebo, variance) was re-run under **three feature sets** and curated into a
   committed `PAPER/` directory with per-set READMEs, figures, tables, and **two adversarial reviews (0 of 29
   findings refuted)**. Two-pillar story reproduces under all three; only the F_causal *scale* shifts. **§D.**

6. **Open decisions for the meeting:** the `F_causal → F_demo` rename (held from the reviews), and the
   second-dataset framing given the profile-dominance caveat. **§E.**

**Where the second dataset stands:** search **DONE**; implementation **DONE** (pipeline + discriminator);
dual claim **DEMONSTRATED**; results curated into `PAPER/second-dataset/` (kept deliberately separable in case
the second dataset is later swapped).

---

## §A. Second-dataset search & selection

**The constraint that drove everything.** The realism half of the paper's claim is enforced by
**F_fidelity — a pre-trained, driver-conditioned, 3-stream Siamese discriminator** over *dense per-driver
trajectory sequences* (seeking + driving + an 11-dim profile). It **cannot score origin–destination (OD)
pairs** and **must be retrained per city**. That single fact decides second-dataset compatibility:

- **OD-only US data is INCOMPATIBLE with the dual claim** — NYC TLC, Chicago, DC publish trip records (pickup
  → dropoff), no dense traces, weak/no persistent driver IDs. They can support the *fairness* half but not
  *realism*. (This contradicts the Meeting-40 "NYC + Census" idea, which is why the search was worth doing.)
- **Dense-trace + driver-ID data is compatible (with per-city discriminator retraining).**

**Ranking → SF Cabspotting (#1).** It is the only **US** dense-trace taxi set with (i) a **native occupancy
flag** (splits seeking vs. driving for free), (ii) **persistent per-taxi IDs**, and (iii) a **native US
Census/ACS demographic join** — so it fits the existing pipeline with **zero algorithm change**. Fallbacks:
#2 Porto ECML/PKDD (non-US, INE demographics), #3 Rome; DiDi excluded. Dataset: 536 SF Yellow-Cab taxis,
~11.2M GPS pings, 2008-05-17 → 06-10, format `[lat lon occupancy time]`.

Full analysis: `famail_temporal/second_dataset/docs/SECOND_DATASET_COMPATIBILITY.md`.

## §B. SF implementation

### B.1 Faithful pipeline (no algorithm/metric/fidelity change)
A city-switchable SF pipeline (`FAMAIL_CITY` env var; `shenzhen` default stays bit-identical) emits
`source_data` in the **existing loader schema**, so `preprocess → DataBundle → FAMAILObjective` all work
unchanged. Key decisions (`SF_PHASE2_DECISIONS.md`): **faithful constant 0.01° grid** (matches Shenzhen's
cell size → SF footprint = **32×30**, *not* forced to 48×90, which would distort the ε-ball edit scale);
**majority-overlap of ACS 2006–2010 tracts** onto cells (matches Shenzhen's district mapping); reuses the
Shenzhen feature *names* filled with ACS values (housing = median home value, comp = per-capita income,
migrant = foreign-born share).

### B.2 The regime discovery — why the sf12 subsample
A fairness-only smoke on the **full 536-taxi fleet** found **F_causal ≈ 0.982 and the editor a near-no-op.**
Root cause = **fleet density**: SF is a *near-complete* fleet (**0.56 drivers/cell**) vs Shenzhen's *50-driver
sample* (**0.012/cell, ~47× sparser**). The 5×5 distinct-taxi supply measure saturates on the dense fleet, so
the service residual becomes supply-noise orthogonal to demographics → F_causal → 1, nothing to edit. **Fix =
fleet subsampling.** A GPU comparison chose **sf12** (12 drivers, Shenzhen-density-matched): full-pool
causal-emphasis ΔF_causal **+0.0199** vs sf50's saturated +0.0041. Supply/demand heatmap:
`PAPER/second-dataset/figures/sf_supply_demand.png`.

### B.3 Discriminator retrain (val-AUC 0.998)
Retrained the 3-stream discriminator on sf12 with the **identical architecture + Ren-aligned training
protocol** as Shenzhen (concatenation combo, [200,100] BiLSTM, N=5, 1.556M params; 10k day-based pairs,
7500/1500/1000, 12 drivers). Three things had to be right, and Phase 4 was the *first* time the discriminator
training path ran on SF data, so three latent bugs surfaced — all fixed, with the **fairness baseline provably
preserved** (rebuild → **byte-identical** count/demographic artifacts; baseline F_causal 0.8752 unchanged):
1. trajectory day column was an absolute epoch-day serial → remapped to **day-of-week (1..7)** with
   `days_in_week=7`;
2. calendar-day sidecars were sorted-distinct → **parallel-per-trajectory** (the generation contract);
3. a missing `calendar_day_map.pkl` → now emitted.
Plus the FeatureNormalizer config (x_max=32, y_max=30, days_in_week=7) is **baked into the checkpoint** so
inference normalization matches training (backward-compatible; Shenzhen loads bit-identical). *First training
run failed at chance (val-AUC 0.495) — diagnosed to a too-low learning rate (6e-5) + premature early-stop in
the warmup; refit at lr 1e-3 → **0.998**.* Detail: `SF_PHASE4_DISCRIMINATOR.md`.

## §C. SF results — the dual claim

### C.1 Headline (`PAPER/second-dataset/`)
Editor causal-emphasis (α = 0.2 / 0.7 / 0.1), fidelity ON, `-k 2000` (1371 edits):

| metric | before | after | Δ |
|---|---:|---:|---:|
| **F_causal** (fairness) | 0.8752 | 0.8891 | **+0.0139 ↑** |
| F_spatial (secondary) | 0.1846 | 0.1817 | −0.0030 ↓ |
| **F_fidelity** (realism) | — | **0.968** | edit-induced Δ ≈ **−1.5e-5** |

**+0.0139 beats Shenzhen's own +0.0128** from a density-comparable baseline. Edited SF trajectories stay
realistic (still recognized as the same driver; the edit barely moves the score).

### C.2 Two ΔF_causal figures, both correct (avoid confusion at the meeting)
**+0.0199** = the subsample-*selection* metric (causal-emphasis over the *entire unfair pool*, ~762
trajectories, fidelity off). **+0.0139** = the *dual-claim headline* (`-k 2000` → 1371 edits, fidelity on).
Different edit subsets, not a regression.

### C.3 Fidelity is inert as a gradient (matched control)
Same run with `ALPHA_FIDELITY=0`: ΔF_causal **+0.01392** vs **+0.01394** with fidelity on — a 2e-5 difference.
Turning fidelity on costs zero fairness and only adds the per-iteration discriminator pass (33 min vs 4.4 min).

### C.4 The caveat to raise — F_fidelity is profile-dominated (shared with Shenzhen)
In the editing use case both branches share the **same driver's profile**, so the discriminator must read the
**seeking trajectory** to notice an edit. A direct probe shows it does not: swapping a *different driver's
entire seeking trajectory* into one branch changes the score by ~0 (sf12 −0.0001, gradient 2.6e-11). **I ran
the same probe on the Shenzhen primary discriminator — identical (−0.0012, grad 4.7e-6)**, consistent with the
earlier `fidelity-grad ≈ 0` gradient-heatmap result. So F_fidelity is a **driver-identity-preservation
metric**, not an active gradient — **a property of the whole mechanism, not an SF weakness.** *Interim
decision (pending your input): report as-is for parity with Shenzhen; a stronger seeking-sensitive
discriminator (drop the profile stream, or add same-driver-corrupted-seeking hard negatives) is deferred and
would require re-running Shenzhen the same way for a fair comparison.*

## §D. Separate session — demographic-feature re-runs + the `PAPER/` bundle

*(Recap of work completed in a separate session; packaged in the committed `PAPER/` directory.)*

**What was done.** The entire Shenzhen experiment set (editor, L1 data-quality, L2 vanilla transfer,
weighted-BC recovery, random-subset placebo, variance suite) was **re-run under three demographic feature
sets**, and the results curated into a self-contained, committed `PAPER/` deliverable (per-set READMEs,
figures, tables, provenance for every artifact).

**The three feature sets** (F_causal is *feature-set-specific*, so the axis choice is defended explicitly):
| set | before-edit F_causal | role |
|---|---|---|
| **{housing, comp, migrant}** | **0.799** | **PRIMARY** (equity-salient axes; *higher* baseline → not the unfairness-maximizing lens) |
| {housing, GDP, comp} | 0.807 | sensitivity (original SES set; shows conclusions predate the migrant choice) |
| {housing, comp, migrant, logpopdensity} | 0.725 | sensitivity (adds a demand-density control; scale drop is ~90% LogPopDensity, a geography var, not protected) |

**The two-pillar story reproduces under all three sets** (only the F_causal *scale* shifts):
- **Pillar 1 (L1, data quality):** the **edited** dataset is the *fairest faithful* source (higher F_causal
  than raw/BC-gen; GAN-gen disqualified by distributional collapse) while identity-faithful (Fidelity-A
  unchanged). PRIMARY: F_causal 0.7988 → **0.8132**.
- **L2 (vanilla transfer):** driver-conditioned BC on edited data does **not** transfer the fairness
  (edited−raw within the ±0.003 cross-seed band; null) — vanilla BC averages it away.
- **Pillar 2 (weighted BC):** **upweighting** the edited demonstrations **recovers** it — PRIMARY ΔF_causal
  **+0.0205 / +0.0278 / +0.0311** at weights 10/20/30 (monotone dose-response, 6/6 seeds, t-CIs exclude 0).
- **Edit ≫ select > random:** under the PRIMARY metric SELECT is **genuinely null** (edit beats select ~70×);
  the random placebo is null; **filtering** unfair trajectories *lowers* F_causal (Pareto). The gain is
  **edit-driven**, not reproducible by selecting or removing data.
- Data cleanup (6 stuck-GPS "sink" artifacts) handled in `PAPER/shared_cleanup/` — affects the secondary
  F_spatial by ~23%, leaves the headline F_causal essentially untouched.

**Adversarial review.** Two independent reviews of the bundle; **0 of 29 findings refuted** (all were
framing/labeling overreaches, now fixed across the bundle — figure honesty, statistical caveats, feature-choice
scoping). Statistical conventions to keep in mind when citing: **n=6 Wilcoxon floors at p=0.03125** (= sign
unanimity, not a magnitude — read effects from means + t-CIs); **F_causal is associational** (partial R² on 10
district-level profiles), with an ecological-fallacy caveat.

## §E. Decisions / open items for the meeting

1. **`F_causal → F_demo` rename** (held from the adversarial reviews). The metric measures *demand-adjusted
   demographic predictability of service* — associational, not causal. RA favors **`F_demo`**; interim decision
   was keep `F_causal` + the associational caveat now, **raise the rename with you before any paper-wide
   change**. Numbers are unaffected either way. *(Your call.)*
2. **Second-dataset framing vs the profile-dominance caveat (§C.4).** Report SF's F_fidelity as-is for parity
   with Shenzhen (recommended, deadline-friendly), or invest in a seeking-sensitive discriminator (stronger
   realism claim, but a method change requiring a Shenzhen re-run for parity). *(Your call.)*
3. **Optional next second-dataset work:** the SF result is deliberately isolated in `PAPER/second-dataset/`; if
   you'd prefer a different second dataset, it can be swapped without touching the Shenzhen bundle.

## Pointers
- Second dataset: `PAPER/second-dataset/README.md` (results) · `famail_temporal/second_dataset/docs/`
  (SF_SECOND_DATASET_STORY.md, SF_PHASE4_DISCRIMINATOR.md, SECOND_DATASET_COMPATIBILITY.md, SF_PHASE2_DECISIONS.md).
- Shenzhen `PAPER/` bundle: `PAPER/README.md` · `PAPER/by_feature_set/housing-comp-migrant/` (PRIMARY) ·
  `PAPER/feature_selection/` · `PAPER/reviews/`.
