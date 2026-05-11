# Design: Researcher Handoff Document for `famail_temporal/`

**Date:** 2026-05-07
**Status:** Approved for implementation
**Scope:** A single self-contained document that lets a same-lab researcher
with no prior `famail_temporal/` context ramp up well enough to give the
project a meaningful methodology sanity-check.

---

## Motivation

Multiple lab members will be reading and reviewing the FAMAIL Temporal
codebase: paper co-authors, the GAN/GAIL-baseline collaborator, future
contributors. The codebase already has thorough per-module READMEs and
two methodology notes (`FAIRNESS_DECOMPOSITION_FORMULATION.md`,
`F_CAUSAL_METHODOLOGY_NOTES.md`), but there is no single artifact that
threads these together into a top-down narrative a new reader can follow
end-to-end.

Without that, every collaborator who joins the project has to reconstruct
the picture by hopping between READMEs in an order they have to discover.
That is wasteful when several reviewers are converging on the project at
once, and it makes meaningful sanity-checking effectively gated on hours
of independent code reading.

The handoff document closes this gap. It is **not** a replication of the
in-tree material — it is a deliberately compact orientation that surfaces
load-bearing design choices and points back into the in-tree docs at every
section break, so a reader can drill into specifics on demand.

---

## Goals

1. **Sanity-checkable.** A competent same-lab researcher reading the
   document end-to-end should be able to: (a) state the research question;
   (b) describe the dataset and the active-unit construction; (c) write
   the three fairness/realism formulations in their own notation; (d) read
   the trajectory-modification pseudocode and explain the gradient flow;
   (e) identify at least three places where they would push back on a
   methodological choice if they wanted to.
2. **Compact.** Target ~6 pages rendered (≈3,500–4,500 words). Tight
   enough that a reviewer reads the whole thing in one sitting.
3. **Pointer-rich.** Every section ends with or contains pointers into
   the in-tree docs for the reader who wants more depth on that section.
4. **Self-contained for the algorithm proper.** Cover the full
   trajectory-modification algorithm (problem → metrics → attribution →
   pseudocode) without requiring a click-out for any of the core math.
5. **Sign-convention coherent.** Use the same sign conventions as the
   in-tree material (positive α = above-baseline fair contribution;
   `F = 1` = fairer; sum-to-F decomposition).

## Non-goals

1. **Not a per-cell-attribution-export reference.** That tool is
   downstream tooling, not part of the trajectory-modification algorithm,
   and will get its own standalone document later.
2. **Not a replication of the methodology notes.** The handoff cites
   `FAIRNESS_DECOMPOSITION_FORMULATION.md` and
   `F_CAUSAL_METHODOLOGY_NOTES.md` rather than reproducing their content.
3. **Not an operational manual.** The top-level `README.md` and
   `evaluation/EVALUATION_QUICKSTART.md` are the places researchers go
   to actually run the code; the handoff sends them there with one
   pointer rather than competing.
4. **Not a paper draft.** The handoff is structured for sanity-checking,
   not paper extraction. Paper-section-hook notes already exist on each
   in-tree README; those are the right inputs for the eventual paper.

---

## Output artifact

- **Path:** `famail_temporal/docs/RESEARCHER_HANDOFF.md`
- **Format:** GitHub-flavored Markdown.
- **Length:** ~6 pages rendered (≈3,500–4,500 words).
- **Header block:** title, date, audience-framing line ("intended as a
  sanity-check enabler for collaborating researchers in the lab"),
  one-line status (which `config.py` and source-data SHA the document was
  written against).
- **Linking convention:** all in-tree references use markdown relative
  links (`[text](path)`) so they resolve when the file is rendered on
  GitHub or in IDEs.

---

## Document structure (final)

The document follows the **Algorithm-First** structure agreed during
brainstorming. Section length targets sum to ~6 pages.

### §1. Project context (~½ page)

Covers:

- Research question: detect spatial-temporal unfairness in taxi service,
  and algorithmically remediate it by modifying pickup locations.
- Two-part contribution: (a) a fairness audit at `(cell, time-block)`
  granularity; (b) the trajectory-modification algorithm.
- One sentence on lineage: `famail_temporal/` is a ground-up rewrite that
  supersedes prior 2D and 4-block iterations; explicit temporal granularity
  at T = 24 hourly blocks is the headline methodological change.
- One sentence on data locale: 50-driver Shenzhen dataset, 3 months,
  weekdays only.
- Out-of-scope: per-driver fairness, supply-side modification, real-time
  deployment.

Pointer-out: top-level `famail_temporal/README.md` for the architectural
invariants list and the quickstart.

### §2. Dataset and active-unit construction (~1 page)

Covers:

- Grid geometry: 48 × 90 spatial grid × T = 24 hourly blocks → up to
  103,680 candidate `(cell, time-block)` units.
- Three primary tensors (`pickup_3d`, `dropoff_3d`, `active_taxis_3d`)
  in mean-hourly-rate units.
- Active-unit filter: a unit `(c, t)` is active iff supply ≥
  `ACTIVE_SUPPLY_THRESHOLD` AND cell is inside the Shenzhen boundary AND
  no NaN in selected demographics for cell `c`.
- **Load-bearing design choice — supply-based not demand-based mask.**
  Short paragraph: observed demand is endogenous to historical service
  patterns (a chronically under-served residential cell may have ~zero
  observed demand); a demand-based mask would conflate "no service
  territory" with "unfair service territory" and excise the very cells
  most relevant to the fairness question.
- Canonical active-unit ordering (cell-major, then block-within-cell)
  asserted at every load boundary.
- Demographics: three z-scored features (avg housing price per sqm,
  GDP/capita, companies/capita).

Pointer-out: `famail_temporal/data/README.md` for `DataBundle` API,
`source_data/README.md` for source-file inventory,
`F_CAUSAL_METHODOLOGY_NOTES.md` §5 for full active-mask rationale.

### §3. The objective at a glance (~½ page)

Covers:

- The combined objective:
  `L = α_s · F_spatial + α_c · F_causal + α_f · F_fidelity`
- All three terms in [0, 1], higher = better; weighted sum maximized
  during ST-iFGSM.
- Default weights: `α_s ≈ α_c ≈ 0.33`, `α_f ≈ 0.34`.
- `ALPHA_FIDELITY = 0` is a clean ablation (skips the discriminator
  pathway entirely; supports fairness-only experiments).
- One-line preview of each term, full formulations follow in §4–§6.

Pointer-out: `algorithm/README.md` (the orchestration module's design),
`config.py` for the actual default values.

### §4. F_spatial — pooled Gini fairness (~¾ page)

Covers:

- Definitions: DSR = pickup_N / active_taxis_N, ASR = dropoff_N /
  active_taxis_N.
- Formula:
  `F_spatial = 1 − ½ · (Gini(DSR) + Gini(ASR))`
- Pairwise Gini formula with the `Σᵢ Σⱼ |xᵢ − xⱼ| / (2 N² μ)` form;
  brief note on differentiability (everywhere except measure-zero ties).
- Sign convention: `F_spatial = 1` ⇔ perfect equality across all active
  units; `F_spatial = 0` ⇔ one unit absorbs all the service mass.

Two design choices, each with ~2-sentence rationale:

1. **Pooled, not block-averaged.** The Gini is computed over all N
   active units simultaneously rather than per-time-block then averaged.
   Time-blocks with more active units carry proportionally more weight,
   reflecting their larger contribution to total service exposure.
2. **DSR + ASR equal weighting.** Pickups and dropoffs are dual signals
   of service; weighting either alone biases the metric toward an origin
   or destination view.

Pointer-out: `fairness/README.md` (Option B selection rationale, API
surface).

### §5. F_causal — demographic-projection R² (~1¼ pages)

This is the metric that will attract the most reviewer attention. It gets
the most space.

Covers:

- The double-regression construction:
  - Stage 1: `g_0(D) = β₀ + β₁/(D+1) + β₂/√(D+1) + β₃√(D+1)` fitted via
    OLS on all N active units; produces a baseline service-rate
    prediction from demand.
  - Stage 2: residuals `R = Y − g_0(D)` projected onto demographics
    `H_demo = X̃(X̃ᵀX̃)⁻¹X̃ᵀ` (where `X̃` is z-scored demographics with
    intercept).
  - Final form:
    `F_causal = R'(I − H_demo)R / R'MR = 1 − r²_demo`
- Sign convention: `r²_demo` high ⇔ demographics explain the residuals
  well ⇔ unfair; `F_causal = 1 − r²_demo` orients high = fair, matching
  the `F_spatial` convention.

Four design choices, each with ~2-sentence rationale:

1. **Power basis for `g_0`.** Linear-in-parameters so OLS plugs cleanly
   into the hat-matrix algebra; the four basis terms together capture
   hyperbolic demand-saturation (`1/(D+1)`, `1/√(D+1)`) plus sub-linear
   growth (`√(D+1)`).
2. **`DEMAND_FLOOR = 0.5` as a clamp, not a filter.** Cells with
   `D_raw < 0.5` keep their identity in the active set; only their `D`
   value is replaced by 0.5 inside `Y = S/D`. This preserves residual-
   scale balance (without a clamp, low-D cells get `Y` orders of
   magnitude larger than signal-regime cells and dominate `R'MR`).
   Filtering would break the inclusive-audit property (cf. §2).
3. **Two-R² diagnostic.** Report both the all-cells R² (~0.04 on current
   data) and the signal-regime R² on cells with `D ≥ DEMAND_FLOOR`
   (~0.69). The first measures audit-set inclusivity; the second measures
   model-class adequacy. Reviewers can assess each independently.
4. **`g_0` evaluated under `torch.no_grad()` during the modifier loop.**
   Only the residual `R` carries gradient; the demand-response baseline
   is a fixed function of `D`. Otherwise the demand term would be
   double-counted in the gradient.

Pointer-out: `F_CAUSAL_METHODOLOGY_NOTES.md` (full empirical justification
of all four choices), `fairness/g0_power_basis.py` (basis implementation),
`fairness/hat_matrices.py` (compact form).

### §6. F_fidelity — discriminator-based realism (~½ page)

Covers:

- The Multi-Stream Siamese discriminator is pre-trained and treated as
  an opaque inference module in `famail_temporal/`. Four ported classes:
  `FeatureNormalizer`, `SiameseLSTMEncoder`, `ProfileEncoder`,
  `MultiStreamSiameseDiscriminator`.
- Inputs per call: anchor trajectory + modified trajectory, each rendered
  as multi-stream context (driving stream, seeking stream, profile
  features).
- Output: similarity score in [0, 1]; `F_fidelity = 1` ⇔ modified
  trajectory indistinguishable from authentic expert trajectories.

Two design choices, each with ~2-sentence rationale:

1. **Opaque inference-only port.** The fidelity term is inherited from
   prior work, not a methodological contribution of `famail_temporal/`.
   Only the four classes needed for inference are ported; training code
   and deprecated architectures are excluded.
2. **`ALPHA_FIDELITY = 0` is a supported ablation.** Cleanly skips the
   discriminator forward pass; the checkpoint does not need to exist and
   no GPU memory is consumed. Useful for fairness-only experiments and
   for isolating the fairness gradient signal from the realism gradient
   signal.

Brief note in-text: cuDNN's optimized RNN kernel does not support
backward in eval mode; the discriminator forward is wrapped in
`torch.backends.cudnn.flags(enabled=False)` to fall back to a pure-PyTorch
implementation that supports backward. This is an implementation detail,
not a methodological choice.

Pointer-out: `fidelity/README.md` (full architecture port rationale + the
four multi-stream context-builder decisions D1–D4),
`discriminator_checkpoints/README.md` (checkpoint provenance).

### §7. Per-cell fairness attribution (~¾ page)

Covers:

- The decomposition problem: both `F_spatial` and `F_causal` are scalars
  in [0, 1]. We want a signed per-cell value `αᵢ` such that
  `Σᵢ αᵢ = F` (matching the published metric) and `αᵢ > 0` aligns with
  "this cell helps fairness."
- The 1/N-shifted decomposition:
  ```
  αᵢ = (1 / N_active) − unfairness_contribᵢ
  Σᵢ αᵢ = F
  ```
- For F_spatial: `unfairness_contribᵢ = ½ · (gini_dsrᵢ + gini_asrᵢ)`
  where `giniᵢ(x) = Σⱼ |xᵢ − xⱼ| / (2 N² · mean(x))`.
- For F_causal:
  `unfairness_contribᵢ = ((MR)ᵢ² − ((I−H)R)ᵢ²) / R'MR`
  (FWL-derived per-cell decomposition of `r²_demo`).
- Sign convention table:

  | `αᵢ` value | Cell semantics |
  |---|---|
  | `> 1/N` | Above-baseline fair contribution |
  | `≈ 1/N` | Neutral; uniform-share contribution |
  | `0 < αᵢ < 1/N` | Mildly underperforming |
  | `< 0` | Drags fairness below baseline; priority for modification |

- Why uniform `1/N` as the baseline:
  - In the perfect-fair limit (`Gini = 0` or `r²_demo = 0`) every
    `αᵢ = 1/N` and `Σ αᵢ = 1 = F`.
  - In the perfect-unfair limit, outlier cells absorb the unfairness
    mass and `Σ αᵢ = 0 = F`.
  - Information-theoretically, the uniform baseline is the
    minimum-assumption prior — no auxiliary signal (demand, supply,
    demographics) is injected into the baseline itself.

Pointer-out: `FAIRNESS_DECOMPOSITION_FORMULATION.md` for full derivation,
worked perfect-fair / perfect-unfair examples, the function reference
table, and the audit trail comparing this to the prior `(1 − F)` form.

### §8. Trajectory-modification algorithm (~1¼ pages)

The longest single section. Two pseudocode blocks, then a list of design
choices.

**Outer pipeline pseudocode.** Approximately:

```text
bundle    ← DataBundle.load()
objective ← FAMAILObjective(bundle)
F_before  ← objective.forward(bundle.pickup_3d)
α[1..N]   ← compute_per_unit_attribution(bundle)        # wraps the causal form
ranking   ← rank_trajectories(bundle.trajectories, α, bundle.unit_map)
top_k     ← select_top_k(ranking, k)                    # only αᵢ < 0 selected
modifier  ← TrajectoryModifier(objective, bundle)
for τ in top_k:
    history[τ] ← modifier.modify_single(τ)              # mutates _base_pickup_3d
F_after   ← objective.forward(modifier.current_pickup_3d())
```

**Inner ST-iFGSM pseudocode.** Approximately:

```text
function modify_single(τ):
    (cx, cy), t* ← pickup cell and time block of τ
    if neighborhood at (cx, cy, t*) has no active units: skip
    pickup_mass ← 1 / (n_hours_per_block[t*] · n_days)
    base ← _base_pickup_3d.clone()
    base[cx, cy, t*] −= pickup_mass             # subtract original
    Δ ← (0, 0)
    for it in 1..MAX_ITERATIONS:
        τ_anneal ← anneal_temperature(it)
        p ← (cx, cy) + Δ                        # requires_grad=True
        probs ← SoftCellAssignment(p, τ_anneal) # (2k+1, 2k+1) Gaussian softmax
        soft_3d ← base + inject(probs · pickup_mass, t*)
        total, terms ← Objective(soft_3d, τ_features, ...)
        g ← ∂total / ∂p
        Δ ← clip(Δ + α_step · sign(g), −ε, ε)   # ST-iFGSM step
        Δ ← clip-to-grid(Δ)
        if |total − total_prev| < tol: break
    new_cell ← integer-rounded (cx, cy) + Δ
    _base_pickup_3d[cx, cy, t*]   −= pickup_mass
    _base_pickup_3d[new_cell, t*] += pickup_mass
    return modified_τ
```

Then seven compact design-choice notes (~1–2 sentences each):

1. **Soft-cell assignment via Gaussian softmax.** Makes the (continuous)
   `(pickup_x, pickup_y)` differentiable while still producing an
   integer cell assignment in expectation.
2. **Delta-tensor injection pattern.** `soft_3d = base_3d + delta`
   instead of in-place mutation. Keeps the autograd graph intact when
   the only perturbed slice is `delta[:, :, t*]`.
3. **Single grid-to-unit conversion point.** The
   `(48, 90, T) → (N,)` masking happens exactly once, at the top of
   `FAMAILObjective.forward()`. Every fairness module sees only
   N-vectors; eliminates a class of "function silently received a
   full-grid tensor" bugs.
4. **Sequential modification with shared `_base_pickup_3d`.**
   Trajectory `k+1`'s optimization sees the updated baseline after
   trajectory `k`'s modification. Order-dependence is intentional;
   attribution scores are computed once before any modification so the
   selection order is stable.
5. **Strictly-negative top-k filter.** A trajectory only enters the top-k
   if its pickup cell has `αᵢ < 0` (cell actively drags fairness below
   the 1/N baseline). Cells at or above baseline have no priority.
6. **Pickup-mass conservation.** Because `pickup_3d` is mean-hourly
   rates, a single trajectory contributes mass
   `1 / (n_hours_per_block[t*] · n_days)`. Subtract at original cell,
   add at new cell — total mass is preserved.
7. **ST-iFGSM signed-gradient step.** `Δ = clip(α · sign(grad), −ε, ε)`.
   Robust to gradient-magnitude variation across the three terms; the
   ε-ball constrains how far a pickup can move from its original
   location.

Pointer-out: `algorithm/README.md` (gradient-flow diagram, full API),
`algorithm/modifier.py` (canonical implementation),
`algorithm/attribution.py` (`compute_per_unit_attribution`,
`rank_trajectories`, `select_top_k`).

### §9. Known limitations and open questions (~½ page)

Six bullets, each ~2 sentences:

1. **Zero-supply cells are excluded entirely from the audit.** The active
   mask cannot distinguish "unfair zero supply" from "no service
   territory." Auditing this class would require coupling supply
   prediction with the current framework.
2. **Endogenous demand is controlled but not modeled.** F_causal treats
   observed `D` as-is, not as a noisy proxy for latent "demand under
   fair service." Modeling latent demand as an instrument would be a
   more sophisticated extension.
3. **`DEMAND_FLOOR = 0.5` is pragmatic, not derived.** A sensitivity
   analysis across `{0.1, 0.25, 0.5, 1.0}` would be a useful robustness
   check.
4. **Per-day fairness aggregation is pooled, not per-day.** The metrics
   currently average over all weekdays; weekday-to-weekday variation in
   fairness is observable in principle but not exposed. Future
   research direction.
5. **F_fidelity inherits any bias in the discriminator.** If the
   discriminator was trained predominantly on commercial-area expert
   trajectories, that bias propagates into the realism gradient signal.
   Discriminator provenance is documented at
   `discriminator_checkpoints/README.md`.
6. **Soft-cell-assignment kernel size + temperature schedule are
   unswept.** The `(2k+1) × (2k+1)` neighborhood and the τ-anneal
   schedule are configured but not yet ablated.

Pointer-out: `F_CAUSAL_METHODOLOGY_NOTES.md` §9 (which already enumerates
many of these for the F_causal-specific subset).

### §10. Where to read more (~¼ page)

Pointers organized by concern:

- **Math and methodology:**
  - `famail_temporal/docs/FAIRNESS_DECOMPOSITION_FORMULATION.md` — the
    1/N-shifted decomposition full derivation.
  - `famail_temporal/docs/F_CAUSAL_METHODOLOGY_NOTES.md` — power basis,
    DEMAND_FLOOR, two-R² diagnostic, paper-ready text.
- **Module-by-module designs:**
  - `data/README.md`, `fairness/README.md`, `fidelity/README.md`,
    `algorithm/README.md`, `evaluation/README.md`.
- **Operational:**
  - `famail_temporal/README.md` — top-level quickstart.
  - `evaluation/EVALUATION_QUICKSTART.md` — running the eval framework.
- **Tests as living spec:**
  - `tests/README.md` — math-invariant and bug-class guards.
- **Forthcoming:**
  - The per-cell attribution export tool will get its own standalone
    document; the design notes are at
    `docs/FAIRNESS_ATTRIBUTION_EXPORT_DESIGN.md`.

### §11. Diagnostics snapshot (~¼ page)

Single dated table, exactly:

| Quantity | Value |
|---|---|
| Active-unit count `N` | (filled at write time) |
| All-cells `g_0` R² | (filled) |
| Signal-regime `g_0` R² (D ≥ DEMAND_FLOOR) | (filled) |
| Baseline `F_spatial` | (filled) |
| Baseline `F_causal` | (filled) |
| `n_days` (weekdays) | (filled) |
| `DEMAND_FLOOR` | 0.5 |
| `ACTIVE_SUPPLY_THRESHOLD` | 0.5 |
| `T` | 24 |

Footer: "Snapshot dated YYYY-MM-DD against source-data git SHA `<sha>`;
rerun `python -m famail_temporal.preprocess --force` to refresh."

The numbers are filled at write time from the latest `preprocess` output
and the corresponding `processing_metadata.json`. The footer makes
clear that the snapshot may go stale; readers are pointed back to the
preprocess output for current values.

---

## Style and editorial conventions

1. **Voice.** Third-person, present tense, declarative. Avoid the
   first-person plural ("we") except where stating a deliberate research
   choice. Avoid hedging ("might," "perhaps") where the codebase has
   already settled the question.
2. **Notation.** Match the in-tree docs: `Y = S/D` (capital letters for
   per-unit observables), `R` for residual vector, `H` and `M` for the
   demographic and centering hat matrices, `αᵢ` for per-cell
   attribution. Use the project's `1/N` baseline notation, not `1/N_active`
   except where ambiguity would arise.
3. **Sign conventions.** "Higher is fairer" for both `F_spatial` and
   `F_causal` (matches the in-tree convention). Per-cell `αᵢ` is the
   1/N-shifted form; positive = fair.
4. **Formulas.** Display equations in fenced code blocks (matching the
   convention in `FAIRNESS_DECOMPOSITION_FORMULATION.md`). Inline math
   uses backticks, e.g., `F_causal = 1 − r²_demo`.
5. **Pseudocode.** `text` fences (not `python` fences) so reviewers
   read it as algorithmic shorthand, not runnable code. Variable names
   match the codebase where reasonable (`bundle`, `_base_pickup_3d`,
   `α`, `ε`).
6. **Cross-links.** All in-tree references use markdown relative paths
   (e.g., `[fairness/README.md](../fairness/README.md)`). The handoff
   lives at `famail_temporal/docs/RESEARCHER_HANDOFF.md`, so the
   relative-path basis is `famail_temporal/docs/`.
7. **Table preference.** Where five or more design choices, sign
   conventions, or limitations are listed, prefer a markdown table with
   one column for the item and one for the rationale.

---

## Test plan

There are no executable tests for a documentation artifact. The handoff
is reviewed as follows:

1. **Self-review checklist (run by author before handing to reviewers):**
   - [ ] Word count is in the 3,500–4,500 range (≈6 pages rendered).
   - [ ] Every `(N,)`, `(48, 90, T)`, sign-convention, and threshold
         claim cross-checks against the in-tree docs.
   - [ ] All cross-links resolve when the document is rendered on
         GitHub.
   - [ ] Diagnostics snapshot table values match the latest
         `preprocess` output and the `processing_metadata.json` SHA.
   - [ ] No "TBD" / "TODO" / placeholder text remains.
   - [ ] All seven design-choice rationales in §8 are 1–2 sentences,
         not paragraphs.
2. **Sanity-check probe (run by a fresh reader):** ask a same-lab
   researcher who has not read the codebase to read the document and
   answer the five sanity-checkable questions in **Goals §1**. If they
   cannot answer all five from the document alone, the section that
   gated them is too thin and gets a revision.
3. **Drift check (run later):** when `preprocess` is rerun against
   updated source data, the diagnostics snapshot table is refreshed
   and the footer date is bumped. No other section should require
   updating unless a methodological choice has changed.

---

## Decision audit trail

| Decision | Choice | Rationale |
|---|---|---|
| Audience framing | Same-lab researcher, full setup needed | Captured during brainstorming; collaborators have lab context but no `famail_temporal/` context. |
| Document length | ~6 pages (tight technical brief) | Sanity-check enablement, not paper extraction. Tight length forces editorial selection. |
| Document structure | Algorithm-first (Approach A) | Surfaces design choices next to the things they justify; best for sanity-checking. |
| Pseudocode scope | Outer pipeline + inner ST-iFGSM (two blocks) | Lets reviewers sanity-check orchestration and per-trajectory math separately. |
| Empirical numbers | Structural body + one dated diagnostics table | Easy to refresh; isolates point-in-time content. |
| Export-tool coverage | Out of scope for this document | Mechanical necessity, not part of the trajectory-modification algorithm; gets its own document later. |
| Limitations section | Dedicated short section | Sanity-checkers want a roadmap of where to push back. |
| Output path | `famail_temporal/docs/RESEARCHER_HANDOFF.md` | Sits alongside `FAIRNESS_DECOMPOSITION_FORMULATION.md` and `F_CAUSAL_METHODOLOGY_NOTES.md`. |
| Spec storage | `docs/superpowers/specs/2026-05-07-...` | Matches existing brainstorming-spec convention. |

---

## Implementation note

The handoff document itself will be drafted in the implementation phase
following this design. Drafting will be guided by:

- The section structure and length targets above.
- The pointer-out targets specified in each section (so the document
  cross-links into the right in-tree material from the first revision).
- The diagnostics snapshot values pulled from the latest `preprocess`
  output and `processing_metadata.json` available at draft time.

The implementation plan will be produced separately (via the
writing-plans skill) once this design is approved.
