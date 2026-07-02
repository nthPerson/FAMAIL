> **Superseded (2026-07-01):** the current, results-backed paper argument lives in `PAPER/argument/`.
> This document is retained for historical context; its numbers predate the PRIMARY re-run + the SF
> second dataset.

# FAMAIL — Paper Argument Plan (rough, for Dr. Zhang)

**Date:** 2026-06-25 · **Status:** argument skeleton + evidence map. Figures are placeholders (no image
files exist yet); each is tied to the data + script that produces it. Numbers are from the verified result
artifacts cited inline.

---

## Thesis (one sentence)

**FAMAIL turns real human trajectories into a *fairer but still human* dataset (data-level), and that injected
fairness can be *propagated into trained generative/imitation policies* by fairness-aware training
(model-level) — vanilla training alone does not inherit it.**

Two pillars, in the order a reader should meet them:

| | Pillar 1 — **the data** | Pillar 2 — **the model** |
|---|---|---|
| **Goal** | keep human-derived data **and** improve fairness | train BC/GAN models that are **more fair** |
| **Method** | edit trajectories with the FAMAIL attribution+editing approach | train those models **on FAMAIL-edited data** |
| **Claim** | editing yields the *fairest faithful* dataset | edited-data fairness *propagates to rolled-out trajectories* (with a fairness-aware trainer) |
| **Status** | **DONE — positive** | **BC: DONE (negative→recovered); GAN/WGAN: future** |

---

## Pillar 1 — Edit trajectories with FAMAIL → fairer, human-faithful data

**Claim.** Attribution-guided editing improves *both* fairness axes (F_spatial, F_causal) while preserving
human **identity** (Fidelity-A) and **distributional** realism (Fidelity-B), and it **dominates synthetic
generation** on the fairness×fidelity plane.

**Evidence / experiments.**
- **P1.1 — The editor.** Per-(cell,time) attribution → ST-iFGSM signed-gradient edit of the terminal/pickup
  cell within an ε=2 L∞ ball; objective = weighted sum of F_spatial, F_causal, fidelity (causal-emphasis
  config α=(0.2,0.7,0.1)). Data-level **F_causal +0.0128 vs raw**. *(code: `algorithm/`, `config.py`)*
- **P1.2 — Level-1 data-quality table (headline of Pillar 1).** Among *faithful* sources, **edited is the
  fairest**: F_causal **edited 0.8180 > raw 0.8052 >** BC-generated > GAN-generated; edited stays
  identity-faithful (Fidelity-A 0.838 ≈ raw; real-anchored identity gate **PASSED 0.840 vs 0.174**);
  **GAN-generated is disqualified by distributional collapse** (Fidelity-B ≈ 0.32). *(exp:
  `run_level1_table_v2.py` → `results/level1_table_v2/`; docs: `LEVEL1_V2_RESULTS.md`,
  `LEVEL1_V2_METHODOLOGY.md`)* → **[Fig 1]**
- **P1.3 — Data Pareto (support).** Sweeping α (and k) traces a fairness↔fidelity frontier; **edited data
  Pareto-dominates generated alternatives** — it buys fairness at far lower fidelity cost than generation.
  *(exp: `run_data_pareto.py` → `results/data_pareto/`; `pareto.py`)* → **[Fig 2]**
- **P1.4 — Interpretability / verification gate.** The gradient-heatmap explorer shows *where* edits act and
  confirms the causal gradient concentrates at **district boundaries** (the demographic-granularity wall) —
  used to validate edits and as a figure source. *(tool: `visualization/gradient_heatmap/`)* → **[Fig 3]**

**Takeaway.** Editing beats *generating* on exactly the two things a dataset must keep — fairness and human
fidelity — so the **edited dataset is the durable asset**.

---

## Pillar 2 — Train BC/GAN on FAMAIL-edited data → fairness propagates to rollouts

**Claim.** A policy trained on edited data produces **fairer rolled-out trajectories** than one trained on raw
— *provided the trainer is fairness-aware* (the ~3.6% edited minority must not be averaged away). Told as a
**negative → resolved** arc:

**Evidence / experiments.**
- **P2.1 — Vanilla BC does *not* transfer (the honest negative).** Driver-conditioned BC on edited vs raw →
  paired **ΔF_causal −0.0022 ± 0.0016, 5/5 seeds**; identity gate passed, not a fidelity trade-off. *(exp:
  `run_level2_table.py` → `results/level2_table/`; doc: `LEVEL2_RESULTS.md`)* → **[Fig 4]**
- **P2.2 — Diagnosis.** BC's teacher-forced MLE **averages over the unedited ~96.4%**; the bottleneck is
  *edited fraction + averaging operator*, **not** the 1/N metric wall and **not** edit magnitude. *(analysis:
  `MEETING_40_PREP.md §A`)*
- **P2.3 — Weighted BC recovers it (the resolution).** Upweighting the edited demonstrations' loss flips
  transfer positive and significant: paired **ΔF_causal +0.0186 (w10) / +0.0242 (w20) / +0.0274 (w30), 6/6
  seeds, Wilcoxon p=0.031**; identity fidelity unchanged; distributional realism a small **tunable** cost
  (a fairness↔realism knob, w10 most efficient). *(exp: `run_weighted_bc_smoke.py` →
  `results/weighted_bc_sweep/sig_6seed_w10_w20_w30/`; doc: `MEETING_41_PREP.md §1`)* → **[Fig 5, Fig 6]**
- **P2.4 — Placebo: the gain is edit-specific.** Upweighting a *random*, size-matched non-edited subset moves
  fairness essentially to zero (**ΔF_causal −0.0012 / −0.0015, non-significant**), an order of magnitude below
  the edited arms → **not an oversampling artifact** (adversarially verified). *(exp:
  `run_weighted_bc_smoke.py --placebo` → `results/weighted_bc_sweep/placebo_6seed_w10_w30/`; doc:
  `MEETING_41_PREP.md §6`)* → **[Fig 5 overlay]**
- **P2.5 — GAN / WGAN on edited data (future, completes the pillar).** Train adversarial generators on edited
  vs raw and measure rollout-fairness transfer; WGAN-GP for collapse resistance. Mirrors the BC arc for a
  *second* model class. *(code: `gan/train_adversarial.py`; prior model-level GAN baseline was null at n=5 —
  `MEETING_38_PREP.md` — so this is the open forward experiment)* → **[Fig 7 — PLACEHOLDER, to run]**

**Takeaway.** Data-level fairness is **realizable in models** by fairness-aware training (shown for weighted
BC, edit-specific; GAN/WGAN forthcoming). That vanilla training *fails* to inherit it is itself a contribution
— a clean **diagnosis-and-fix**.

---

## How the two pillars compose (the umbrella)

> The FAMAIL editor produces the **fairest faithful dataset** (Pillar 1); whether that fairness reaches a
> *policy* is a property of the **trainer**, and a fairness-aware trainer realizes it (Pillar 2). Editing is
> the reusable asset; fairness-aware training is what cashes it out.

**Framing lean (decision for Dr. Zhang):** *"data is the asset"* spine + *"negative-then-resolved"* arc — the
L1 win stands regardless of how far the model-level story is pushed, and the L2 negative→weighted-BC recovery
gives the model section its narrative. Four framings + trade-offs detailed in `MEETING_41_PREP.md §3`;
consolidated progress + the relocation/data-quality context in `MEETING_40_PREP.md`.

---

## Figure manifest (all placeholders — none rendered yet)

| Fig | Pillar | Shows | Data source | Generator / status |
|---|---|---|---|---|
| **Fig 1** | 1 | Level-1 fairness×fidelity — edited dominates raw / BC-gen / GAN-gen | `results/level1_table_v2/` | `figure.py` · **placeholder** |
| **Fig 2** | 1 | Data Pareto frontier (fairness vs fidelity across α/k) — edited above generation | `results/data_pareto/` | `pareto.py` · **placeholder** |
| **Fig 3** | 1 | Gradient heatmap — causal gradient at district boundaries (interpretability) | `visualization/gradient_heatmap/` | export from tool · **placeholder** |
| **Fig 4** | 2 | Level-2 vanilla-BC parity (edited ≈ raw) — the negative | `results/level2_table/` | new plot · **placeholder** |
| **Fig 5** | 2 | Weighted-BC **dose-response** — ΔF_causal vs w, **edited (rising) vs random/placebo (flat)** | `results/weighted_bc_sweep/` | new plot · **placeholder** |
| **Fig 6** | 2 | Fairness ↔ Fidelity-B **trade-off knob** across w | `results/weighted_bc_sweep/` | new plot · **placeholder** |
| **Fig 7** | 2 | GAN/WGAN-on-edited fairness transfer | *(experiment not yet run)* | `gan/train_adversarial.py` · **PLACEHOLDER + experiment** |
| **Fig 0** *(opt.)* | 1 | Before/after example edited trajectory (terminal-cell relocation) | `results/level1_table_v2/` histories | new plot · **placeholder** |

---

## Known limitations / caveats (carry into the paper, 1-liners)

- **Stuck-GPS "sinks":** one raw-data artifact cell *depresses* the *secondary* F_spatial by ~23% (we
  under-claim spatial fairness; true ≈0.10 vs reported ~0.082); **F_causal — the headline axis — is unaffected
  (Δ+0.0004)**, so the causal results are robust. Clean-up is a data-hygiene item.
  *(`MEETING_40_PREP.md §D`, `MEETING_41_PREP.md §7`)*
- **Model-level scope:** recovery shown for *importance-weighted BC*, not full IL/cGAIL (yet); GAN/WGAN is the
  next experiment (P2.5).
- **Statistics:** n=6 → Wilcoxon **p=0.031** is the unanimous-sign floor; extend to n≈8–10 for a stronger p.
- **Assumption:** weighted BC needs a *labeled* edited subset (we have it because we did the editing).
