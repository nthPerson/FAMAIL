# Design: How-To Document for the Fairness-Attribution Export

**Date:** 2026-05-07
**Status:** Approved for implementation
**Scope:** A standalone document that researchers receiving a FAMAIL
fairness-attribution export use to apply that data to GAN, GAIL, and
generic offline-RL training. Prescriptive ("here is what to do") rather
than descriptive ("here is what the columns mean") — the latter is
already covered by the auto-generated per-export `README.md`.

---

## Motivation

The export tool at
`famail_temporal/evaluation/export_fairness_attributions.py` produces a
timestamped directory of per-cell fairness attributions consumed by
downstream collaborators training GAN, GAIL, and (potentially) other
fairness-aware models. Each export ships with an auto-generated
`README.md` that defines the schema, sign convention, and column
semantics — but that README is a *reference card*, not a *how-to*.

A consumer reading only the per-export README is left to figure out:

- How to map the per-(cell, hour-block) attribution onto a
  per-state reward in their training loop.
- Which axes carry independent signal versus broadcast duplicates,
  and what that distinction means for sampling and variance.
- How to mask inactive cells without silently corrupting NaN-aware
  reductions.
- Which sanity checks tell them their load is correct before they
  spend GPU time training on a misread tensor.
- Which mistakes are common enough to warrant pre-emption.

Manuel (the GAN/GAIL collaborator) is the immediate consumer; this
project's own baseline GAN is the second; future researchers running
fairness-aware training on FAMAIL exports are the long tail. None of
them should need to reverse-engineer the answers from
`FAIRNESS_DECOMPOSITION_FORMULATION.md` and the export source code.

The how-to closes that gap. It is **not** a replacement for the
per-export README, the methodology notes, or the decomposition
formulation document — it cites all three at section breaks.

---

## Goals

1. **Actionable.** A FAMAIL-fluent researcher reading the document
   end-to-end should be able to: (a) load the export in any of the
   three formats, (b) compute a per-state fairness reward / loss
   signal usable inside a training loop, (c) detect when their load
   is corrupt before training runs, and (d) avoid the seven
   pre-catalogued pitfalls.
2. **Compact.** Target ~5–7 pages rendered (≈3,000–4,500 words).
   Short enough that a researcher reads the whole thing before
   touching their training code, long enough to cover three
   training-method recipes plus shared scaffolding.
3. **Pointer-rich at boundaries.** Every section that touches
   methodology cites the in-tree doc that owns the math, so a
   reader who wants depth has a single click to find it.
4. **Standalone for practical decisions.** A reader does not have
   to flip between the how-to and the per-export README to make a
   loading or masking decision; the load instructions and sign
   convention are restated inline.
5. **Neutral on consumer-side normative choices.** The doc does not
   prescribe when to use F_spatial vs F_causal vs both, what
   normalization to apply, or what reward weighting to use. It
   shows how to apply each option correctly, not which option to
   pick.

## Non-goals

1. **Not a re-derivation of the math.** The 1/N-shifted decomposition,
   the Gini formulation, and the demographic-projection R² all live
   in `FAIRNESS_DECOMPOSITION_FORMULATION.md` and
   `F_CAUSAL_METHODOLOGY_NOTES.md`. The how-to references results;
   it does not re-derive them.
2. **Not a tutorial for newcomers.** The audience is FAMAIL-fluent;
   `RESEARCHER_HANDOFF.md` is the orientation entry point. The
   how-to assumes the reader can answer "what does F_causal
   measure?" before opening it.
3. **Not framework-specific.** No PyTorch, JAX, or TensorFlow code.
   Pseudocode only. A FAMAIL-fluent reader can transcribe.
4. **Not a worked end-to-end example.** Each training-method recipe
   is self-contained and short; there is no consolidated
   tutorial-style worked example threading them together.
5. **Not opinionated about which metric or normalization to choose.**
   See goal 5 above.

---

## Audience

ML researchers who already have FAMAIL context — i.e., they have read
`docs/RESEARCHER_HANDOFF.md` and at least skimmed
`docs/FAIRNESS_DECOMPOSITION_FORMULATION.md`. The doc opens with a
line confirming this assumption and points newcomers back to the
handoff.

---

## Location

`famail_temporal/exports/USING_FAIRNESS_ATTRIBUTION_EXPORTS.md`.

Sits at the exports root next to all per-export subdirectories. The
auto-generated per-export `README.md` is updated to reference it via
relative path (`../USING_FAIRNESS_ATTRIBUTION_EXPORTS.md`) so a
recipient who has only the export directory still finds the how-to
in one click. Keeping it at `exports/` rather than in `docs/` puts
it physically closer to the data while avoiding per-export
duplication.

---

## Approach

**Hybrid: shared preamble + recipe library + appendices.**

The doc opens with a single shared preamble covering the load,
sign-convention, axis-semantics, NaN, and metric-overview content
that every recipe depends on. Three self-contained recipes follow
(GAIL, GAN, generic offline RL); each links back to the preamble
rather than re-stating shared content. Two appendices anchor the
bottom: a numbered pitfalls catalogue and a sanity-check checklist.

This structure was chosen over a linear-tutorial alternative
(stronger first-read flow, weaker for return visits) and over a
pure-recipe-book alternative (cleaner section independence,
duplicates the preamble three times). The hybrid keeps shared
material in exactly one place — minimizing drift — while leaving
each recipe individually findable on a return visit.

---

## Document structure

### Front matter

- **Title:** "Using FAMAIL Fairness-Attribution Exports".
- **Audience line.** One sentence: "Assumes you have read
  `RESEARCHER_HANDOFF.md`. If you have not, start there."
- **TL;DR (≤1 paragraph).** What the doc gives the reader: load
  patterns, three training-method recipes, gotchas, sanity checks.
  Explicit non-goals (no math re-derivation, no framework
  prescription, no metric-choice opinion).
- **What you have.** Three-bullet recap of the export directory's
  contents (three `.pkl` artifacts + `metadata.json` + per-export
  `README.md`) with a sentence each on what each artifact is good
  for.

### §1 — Shared preamble

Five short subsections. The reader is expected to read these once
before touching any recipe; recipes link back to subsection IDs
rather than re-stating content.

**§1.1 — Loading the export.** Pseudocode for each of the three
formats, plus the one-line decision rule:

- `*_dense.pkl` for tensor lookups inside a training loop.
- `*_long.pkl` for pandas filtering and analysis.
- `*_tuples.pkl` for dependency-free iteration.

**§1.2 — Sign convention and scale.** Per-cell α is signed and
unbounded; sum over active cells equals F ∈ [0, 1]; positive means
"contributes more than the 1/N baseline to fairness." Two
anti-pattern callouts: (a) clamping per-cell α to [0, 1] silently
discards the negative-fair / unfair distinction; (b) treating
per-cell magnitudes as probabilities is a category error.

**§1.3 — Axis semantics: the broadcast trap.** The export's 4D
shape `(x, y, time_bucket, day)` looks like four independent axes
but is really two computed axes (`(x, y, time_block)`) broadcast
across two synthetic axes (12 buckets per block, `n_days` days).
Three concrete consequences:

- IID sampling along broadcast axes inflates effective sample
  size and corrupts variance estimates.
- "Per-bucket" or "per-day" features built from these axes are
  duplicated copies of the per-block / pooled value.
- Aggregations (`mean`, `std`) along broadcast axes are
  pass-throughs of the per-block / pooled value, not means of
  independent samples.

**§1.4 — Active vs inactive cells.** What `is_active = False`
means (no supply per `ACTIVE_SUPPLY_THRESHOLD`, out of bounds, or
NaN demographics). NaN propagation rule: every numeric column at
inactive cells is NaN. Recommended masking pattern (mask before
reducing; use `nansum` / `nanmean` if masking is impractical;
never replace NaN with zero without intent — zero is a valid
attribution value).

**§1.5 — Two metrics, briefly.** One sentence each on what
F_spatial (Gini-based exposure equity across cells) and F_causal
(demographic explanatory power of the demand-adjusted residual)
capture. Pointer to `FAIRNESS_DECOMPOSITION_FORMULATION.md` for
the math. Explicit non-recommendation: "This document does not
prescribe which metric to use; both are exported because they
measure different things and the choice depends on your model's
objective."

### §2 — Recipes

Three self-contained sections. Each recipe is structured
identically: (1) where α enters the loss/reward, (2) pseudocode
for the signal-construction step, (3) one paragraph on
interaction with active masks and broadcast axes (linking back
to §1.3 and §1.4), (4) a one-sentence pointer to the
relevant pitfalls in §3.

**§2.1 — Recipe: GAIL / imitation learning reward shaping.**
α as a per-state bonus added to the discriminator-derived reward.
Pseudocode shows the per-step lookup `α_lookup(s) = α[x, y,
time_block]` with a fallback rule for the agent visiting an
inactive cell (recommend an explicit "off-support" penalty; do
not silently zero). Mentions the natural framing from
`FAIRNESS_DECOMPOSITION_FORMULATION.md` §3: "agents that
randomly visit any cell average a reward of F/N; only agents
that preferentially visit positive-α cells beat that."

**§2.2 — Recipe: GAN training.** Three sub-uses, in order of
increasing intrusiveness on the existing training pipeline:

- α as an evaluation diagnostic (compare generator output's
  α-weighted pickup mass against real data's). No change to
  training; an offline scoring step.
- α as an auxiliary loss (penalize generated trajectories whose
  pickup distribution under-weights positive-α cells). Adds a
  loss term; no architectural change.
- α as a conditioning input (broadcast a per-cell channel into
  the generator's spatial features). Architectural change.

Pseudocode for each. Notes the dataset's grid alignment with the
project's 48 × 90 cell grid is a precondition.

**§2.3 — Recipe: Generic offline RL.** α as a per-state reward
bonus in any value-based or actor-critic offline RL setup
(Q-learning, CQL, BCQ, IQL). Same per-state lookup as GAIL but
without the imitation-discriminator structure. Notes that
"state" alignment with the dataset's per-(cell, hour-block)
granularity is the key precondition; an RL state finer than the
hour-block (e.g. per-bucket) gets duplicated α values, which is
the §1.3 broadcast trap surfacing again.

### §3 — Pitfalls catalogue

Numbered list, one paragraph each. Initial seven items (final
content may add or merge during writing):

1. **Treating broadcast axes as independent.** Per-bucket and
   per-day axes carry duplicated values; sampling them as
   independent corrupts variance and effective-sample-size
   calculations.
2. **Clamping per-cell α to [0, 1].** Per-cell values are
   signed and unbounded; only the overall metric is in [0, 1].
   Clamping silently discards the negative-fair signal.
3. **Mistaking the F_spatial / F_causal magnitude imbalance for
   a bug.** A representative export shows F_spatial ≈ 0.08 and
   F_causal ≈ 0.80. The two metrics measure different things on
   different scales and are not expected to match. Pre-empts a
   common "F_spatial looks broken" reaction.
4. **Treating attribution as per-trajectory.** Attribution is
   per-(cell, hour-block); two trajectories whose pickups land
   in the same cell-block share α. Per-driver fairness is
   explicitly out of scope (cite
   `FAIRNESS_ATTRIBUTION_EXPORT_DESIGN.md` §8).
5. **Uniform sampling over the long format without
   stratifying on `is_active`.** ~85% of the dense grid is
   inactive in the manuel-handoff export; uniform sampling
   trains on a lot of NaN.
6. **Treating `demand_D` and `supply_S` as per-bucket counts.**
   They are mean-hourly rates at the block level; downstream
   features that assume bucket-level counts are off by a factor
   of 12.
7. **Combining attributions across exports without re-checking
   `n_days` and `famail_git_sha`.** Attribution magnitudes scale
   with the active-set size and the underlying source-data
   commit; cross-export aggregation requires explicit
   reconciliation.

### §4 — Sanity-check checklist

Concrete invariants the reader runs after loading. Each is one
line of pseudocode plus a one-sentence pass criterion:

- `nansum(spatial_attribution)` ≈ `metadata["overall_F_spatial"]`
  (within `1e-5`).
- `nansum(causal_attribution)` ≈ `metadata["overall_F_causal"]`.
- `isnan(spatial_attribution)` agrees with `~active_mask`
  element-wise.
- Broadcast equality: `spatial[x, y, b1] == spatial[x, y, b2]`
  for every pair of buckets `b1`, `b2` in the same hour-block;
  same across all `n_days` day indices.
- `n_active_cells_per_block` from metadata agrees with
  `active_mask.sum(axis=(0, 1))` per block.
- Cross-format equality: dense, long, and tuples carry the same
  values for any chosen `(x, y, time_bucket, day)` row.

### §5 — Pointers

Plain bullet list. Math, causal methodology, project orientation,
export design rationale — paths into `famail_temporal/docs/`.

---

## Side change: per-export README link

`_README_TEMPLATE` in
[`famail_temporal/evaluation/export_fairness_attributions.py:276`](../../famail_temporal/evaluation/export_fairness_attributions.py#L276)
gains a link to the how-to via relative path
`../USING_FAIRNESS_ATTRIBUTION_EXPORTS.md`. Suggested location:
new line at the top of the existing "Contact" section, or a new
top-level "How to use this export" pointer above "TL;DR". The
exact placement is an implementation detail for writing-plans.

This is the only mechanism by which a recipient with just an
export directory finds the how-to. Without it the doc is
effectively orphaned.

---

## Voice and style

- **Prescriptive, not exploratory.** Each recipe and pitfall has
  a recommended action; alternatives are mentioned only when the
  trade-off is real and consumer-dependent.
- **Pseudocode, not framework code.** Per the brainstorming
  decision: NumPy-style or torch-style indexing notation is fine;
  no framework imports.
- **Compact.** Each pitfall and sanity-check is one paragraph or
  one line, respectively. Recipes are 8–15 lines of pseudocode
  plus 2–4 short paragraphs of accompanying prose.
- **Sign-convention coherent with the rest of `famail_temporal/`.**
  Positive α = above-baseline fair. F = 1 = fairer. Sum-to-F
  decomposition. No (1 − F) detours.

---

## Acceptance criteria

The how-to is complete when:

1. The five preamble subsections (§1.1–§1.5) are written and
   each is short enough to read in under a minute.
2. The three recipes (§2.1, §2.2, §2.3) are written with
   identical four-part structure (entry point, pseudocode,
   active-mask / broadcast notes, pointer to relevant pitfalls).
3. The pitfalls catalogue (§3) has at least the seven
   pre-catalogued items, each one paragraph long.
4. The sanity-check checklist (§4) has the six pre-listed
   invariants, each as one line of pseudocode plus one sentence.
5. The pointers (§5) link to all four upstream docs
   (decomposition, causal methodology, researcher handoff, export
   design).
6. The per-export README template includes a link to the how-to
   (side change above).
7. The doc renders to ~5–7 pages and stays under 4,500 words.
8. All cross-references resolve correctly relative to the
   exports/ root.

---

## Out of scope (explicit)

- Worked end-to-end example threading the recipes together.
- Framework-specific code (PyTorch, JAX, TensorFlow).
- Guidance on which metric or weighting to choose.
- Consumer-side normalization recipes (e.g. min-max, z-score,
  clamp). The doc names normalization as a consumer choice and
  stops.
- Per-driver fairness recipes (excluded by the export design;
  see `FAIRNESS_ATTRIBUTION_EXPORT_DESIGN.md` §8).
- Per-day fairness recipes (excluded by the export design;
  see `FAIRNESS_ATTRIBUTION_EXPORT_DESIGN.md` §2 and the
  forward note in `F_CAUSAL_METHODOLOGY_NOTES.md` §9).

---

## Change log

- **2026-05-07** — Initial design. Approved structure: hybrid
  preamble + recipe library + appendices, located at
  `famail_temporal/exports/USING_FAIRNESS_ATTRIBUTION_EXPORTS.md`.
