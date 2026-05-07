# Researcher Handoff Document Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce `famail_temporal/docs/RESEARCHER_HANDOFF.md` — a ~6-page tight technical brief that lets a same-lab researcher with no `famail_temporal/` context sanity-check the project's methodology, fairness formulations, attribution decompositions, and trajectory-modification algorithm.

**Architecture:** Single markdown artifact authored section-by-section after a pre-flight that locks notation and verifies the narrative spine. Cohesion is enforced through a unified notation key, paired section openers/closers, per-section claim-density acceptance criteria, and a final cross-cutting consistency audit before commit. The tight length budget (~3,500–4,500 words) forces editorial selection; depth is offloaded to existing in-tree material via per-section pointer-out targets.

**Tech Stack:** GitHub-flavored Markdown; relative-path links rooted at `famail_temporal/docs/`; pseudocode rendered in `text` fences; equations rendered inline via backticks or in fenced code blocks (matching `FAIRNESS_DECOMPOSITION_FORMULATION.md` convention).

---

## Reference materials (read-only)

The drafter consults these throughout. None are modified by this plan.

| Path | Used for |
|---|---|
| `docs/superpowers/specs/2026-05-07-researcher-handoff-document-design.md` | The approved design spec — single source of truth for section structure, length budgets, and pointer-outs |
| `famail_temporal/README.md` | Top-level architecture invariants (§1, §2, §3) |
| `famail_temporal/docs/FAIRNESS_DECOMPOSITION_FORMULATION.md` | Per-cell attribution math (§7) |
| `famail_temporal/docs/F_CAUSAL_METHODOLOGY_NOTES.md` | F_causal formulation, DEMAND_FLOOR rationale, two-R² diagnostic (§5, §9) |
| `famail_temporal/data/README.md` | Active-unit construction, canonical ordering (§2) |
| `famail_temporal/fairness/README.md` | Pooled Gini, Option B causal (§4, §5) |
| `famail_temporal/fidelity/README.md` | Discriminator port, multi-stream context (§6) |
| `famail_temporal/algorithm/README.md` | Gradient flow, single grid-to-unit conversion (§3, §8) |
| `famail_temporal/algorithm/modifier.py` | Per-trajectory ST-iFGSM loop (§8 inner pseudocode) |
| `famail_temporal/algorithm/attribution.py` | `compute_per_unit_attribution`, `rank_trajectories`, `select_top_k` (§8 outer pseudocode) |
| `famail_temporal/config.py` | Default values for the diagnostics snapshot (§11) |
| `famail_temporal/source_data/processing_metadata.json` | Source-data git SHA, n_days, GPS bounds for §11 footer |

## File structure

| Path | Status | Responsibility |
|---|---|---|
| `famail_temporal/docs/RESEARCHER_HANDOFF.md` | **Create** | The handoff document itself — the only artifact this plan produces |

The plan also reads (does not write) the design spec at `docs/superpowers/specs/2026-05-07-researcher-handoff-document-design.md` to recover section-by-section requirements.

---

## Pre-flight: cohesion mechanisms locked before any section is drafted

These three tasks produce no commits — they are working notes that the drafter holds in mind (and can copy into a scratch buffer) while writing every subsequent section. The outputs of all three are inlined into this plan so a fresh executor can pick up without re-deriving them.

### Task 0a: Lock the unified notation key

**Files:** No file changes. Output is the table below; drafter consults it while writing every section.

- [ ] **Step 1: Read the spec's Style and editorial conventions section**

Path: `docs/superpowers/specs/2026-05-07-researcher-handoff-document-design.md`, the section titled "Style and editorial conventions." Confirms notation is locked from the in-tree convention, not invented.

- [ ] **Step 2: Verify the locked notation key against the in-tree docs**

The notation key below is the canonical form for the handoff. Every drafted section MUST use these symbols with these meanings, and ONLY these symbols for these concepts.

```text
Symbol            Meaning                                        First introduced in §
-------           -------                                        ---------------------
N, N_active       Count of active (cell, time-block) units        §2
(c, t), (cx,cy,t) An active unit / cell triple                    §2
T                 Number of time blocks (= 24, hourly)            §2
D, D_u            Per-unit demand (mean hourly pickups)           §5
S, S_u            Per-unit supply (mean hourly active taxis)      §5
Y, Y_u            Service rate = S / max(D, DEMAND_FLOOR)         §5
g_0(D)            Power-basis baseline service-rate function      §5
R, R_u            Residual = Y − g_0(D)                           §5
X̃                Z-scored demographics with intercept (N × p+1)  §5
H_demo            Demographic hat matrix X̃(X̃'X̃)⁻¹X̃'           §5
M                 Centering matrix I − 11'/N                      §5
I                 Identity matrix                                 §5
DSR_u             Demand-service ratio = pickup_u / S_u           §4
ASR_u             Arrival-service ratio = dropoff_u / S_u         §4
F_spatial         Spatial fairness scalar in [0, 1]               §3
F_causal          Causal fairness scalar in [0, 1]                §3
F_fidelity        Realism scalar in [0, 1]                        §3
α_s, α_c, α_f     Objective weights (summing to ≈1)               §3
α_i               Per-cell fairness attribution                   §7
1/N               Uniform-baseline term in attribution            §7
ε                 ST-iFGSM ε-ball radius (`EPSILON_BALL`)         §8
α_step            ST-iFGSM step size (`STEP_SIZE_ALPHA`)          §8
τ                 Soft-cell-assignment temperature                §8
Δ                 Cumulative pickup perturbation (2-vec, x and y) §8
t*                The trajectory's pickup time block              §8
k                 Soft-assignment neighborhood half-width         §8
```

**Discipline:** if a section needs a symbol not in this table, add it to the table BEFORE drafting the section, and verify no clash with an existing entry. The most likely clash candidates: `R` is the residual vector (do NOT use `R` for "ranking" or "reward"); `α` without subscript is ambiguous (always use `α_s`, `α_c`, `α_f`, `α_i`, or `α_step`).

### Task 0b: Lock the section spine — opener + forward-pointer for every section

**Files:** No file changes. Output is the table below; drafter writes the section opener and closer verbatim from this list.

- [ ] **Step 1: Verify the spine reads as a coherent narrative**

Read the openers below in order. They should compose into a one-paragraph narrative that walks a reader from "what's the question" to "what's the algorithm" to "what we don't yet know."

```text
§1  OPEN:  "FAMAIL Temporal asks whether a city's taxi-service supply is fair
            across both space and time, and whether trajectories can be
            algorithmically rerouted to make it fairer."
    CLOSE: "The remainder of the document defines what 'fair' means in this
            project (§3–§7) and how trajectories are rerouted (§8)."

§2  OPEN:  "Fairness is measured over a discrete set of active spatial-
            temporal units; this section defines that set."
    CLOSE: "All N-vectors and (48, 90, T) tensors in the rest of the document
            share the active-unit ordering established here."

§3  OPEN:  "Three terms compose the optimization objective: two fairness
            metrics and one realism check."
    CLOSE: "§4–§6 give each term in full; §7 decomposes the two fairness
            terms per cell; §8 puts everything inside the trajectory-
            modification loop."

§4  OPEN:  "F_spatial is a Gini-based measure of equity in service exposure
            across active units."
    CLOSE: "F_spatial enters the objective in §3 as the first term and is
            decomposed per cell in §7."

§5  OPEN:  "F_causal asks whether demographics — not demand — explain the
            service rate, via a double regression."
    CLOSE: "F_causal enters the objective in §3 as the second term and is
            decomposed per cell in §7. DEMAND_FLOOR's empirical
            justification is reprised in §9 as a sensitivity-study
            opportunity."

§6  OPEN:  "F_fidelity is a similarity score from a pre-trained discriminator
            that constrains modified trajectories to remain realistic."
    CLOSE: "F_fidelity enters the objective in §3 as the third term. Unlike
            the fairness terms, it is not decomposed per cell — it is a
            per-trajectory check, not a per-unit audit."

§7  OPEN:  "Both fairness metrics admit a per-cell decomposition that sums
            to F itself, signed so that positive = fair."
    CLOSE: "Per-cell α_i drives trajectory selection in §8 (cells with
            α_i < 0 are highest-priority modification targets) and is the
            primary export downstream tooling consumes."

§8  OPEN:  "The algorithm modifies a small set of high-priority trajectories
            using ST-iFGSM, with cohesion preserved by a single
            grid-to-unit conversion point and a delta-tensor injection
            pattern."
    CLOSE: "§9 lists the methodological gaps a reviewer should know about
            before assessing results."

§9  OPEN:  "Six known limitations bound the methodology's claims."
    CLOSE: "§10 points to the in-tree material that develops any of these
            in greater depth."

§10 OPEN:  "Pointers into the in-tree material, organized by concern."
    CLOSE: "§11 gives the dataset numbers a reviewer can use to anchor
            scale judgments."

§11 OPEN:  "Diagnostic snapshot dated against the source-data git SHA at
            the document's writing date."
    CLOSE: "(no closer — terminal section)"
```

The narrative spine, read end-to-end: the question is fairness across space and time → the audit is at (cell, time-block) granularity → three terms compose the objective → each term has a formulation and design choices → both fairness terms decompose per cell → the algorithm uses those decompositions to select and modify trajectories → here are the gaps → here are the pointers and the numbers.

### Task 0c: Gather the diagnostics snapshot values for §11

**Files:** No file changes. Output is a working note recorded inline below; drafter pastes it into §11 when writing that section.

The values are gathered through the project's own public APIs (`DataBundle.load()`, `FAMAILObjective`, the JSON metadata file) — never by reaching into the cache pickle files directly.

- [ ] **Step 1: Read the source-data JSON metadata for git SHA and n_days**

Run:

```bash
python -c "import json; m = json.load(open('famail_temporal/source_data/processing_metadata.json')); print('git_sha:', m['git_sha']); print('n_days:', m['n_days'])"
```

Expected output: `git_sha: a532ead` (or the current SHA) and `n_days: 66`.

- [ ] **Step 2: Run preprocess to obtain g_0 R² values from its console output**

Run:

```bash
python -m famail_temporal.preprocess
```

Watch the stdout for the lines that report:
- `g_0` all-cells R² on N active units
- `g_0` signal-regime R² on cells with D ≥ DEMAND_FLOOR
- N_active count, signal-regime n

If preprocess has already been run and the cache is current, it will print these from the existing cache; otherwise it computes them. Record the printed values.

- [ ] **Step 3: Compute baseline F_spatial and F_causal using the public API**

Run:

```bash
python -c "
from famail_temporal.data.loader import DataBundle
from famail_temporal.algorithm.objective import FAMAILObjective
import torch
b = DataBundle.load()
obj = FAMAILObjective(b)
fs, fc, ff, total = obj.forward(torch.tensor(b.pickup_3d, dtype=torch.float32))
print(f'N_active:  {b.unit_map.n_units}')
print(f'n_days:    {b.n_days}')
print(f'F_spatial: {float(fs):.4f}')
print(f'F_causal:  {float(fc):.4f}')
"
```

Expected: `N_active` between 5,800–8,000; `n_days` matches step 1; `F_spatial`, `F_causal` both in [0, 1].

If `ALPHA_FIDELITY > 0` and the discriminator checkpoint is present, this also reports `F_fidelity`. Either way, only `F_spatial` and `F_causal` baselines are needed for the snapshot table.

- [ ] **Step 4: Record the values inline below**

Replace the placeholders in the table below with the values gathered. Then this table becomes the canonical content for §11.

```text
| Quantity                                | Value      |
|-----------------------------------------|-----------:|
| Active-unit count N                     | <fill>     |
| All-cells g_0 R²                        | <fill>     |
| Signal-regime g_0 R² (D ≥ DEMAND_FLOOR) | <fill>     |
| Baseline F_spatial                      | <fill>     |
| Baseline F_causal                       | <fill>     |
| n_days (weekdays)                       | <fill>     |
| DEMAND_FLOOR                            | 0.5        |
| ACTIVE_SUPPLY_THRESHOLD                 | 0.5        |
| T (hourly time blocks)                  | 24         |
| Demographic features                    | housing, gdp, comp |
```

Footer: `Snapshot dated 2026-05-07 against source-data git SHA <sha-from-step-1>. Refresh by running python -m famail_temporal.preprocess --force.`

---

## Drafting tasks

Each drafting task targets one section, follows a uniform structure (read → draft → self-check → commit), and contains an explicit acceptance-criteria checklist that the drafter MUST verify before committing. The prose itself is written during execution; this plan does not duplicate the section text.

### Task 1: Front-matter + executive TL;DR

**Files:**
- Create: `famail_temporal/docs/RESEARCHER_HANDOFF.md`

The front-matter is the first thing every reader sees. It must orient a busy reviewer in 30 seconds.

The design spec specified a header block (title, date, audience-framing line, one-line status). This task adds a 3–5 sentence TL;DR before §1, since the design discussion did not explicitly raise this and a TL;DR is conventional for handoff documents. **If the user does not want the TL;DR, this task is reduced to just the header block.**

- [ ] **Step 1: Acceptance criteria for the front-matter**

```text
[ ] Title is "FAMAIL Temporal — Researcher Handoff: Trajectory-Modification
    Algorithm and Fairness Formulations".
[ ] Date line: "Document date: 2026-05-07".
[ ] Audience-framing line: one sentence beginning "This document is intended
    as a sanity-check enabler for collaborating researchers..."
[ ] Status line: "Written against config.T = 24, source-data git SHA
    <sha>." (filled from Task 0c step 1 output)
[ ] TL;DR: 3–5 sentences. MUST cover: (a) what the project optimizes,
    (b) the three-term objective at a level appropriate for a busy reader,
    (c) what the trajectory-modification algorithm does, (d) what's
    out-of-scope. Word count: 80–120 words.
[ ] No claims that aren't substantiated in later sections.
```

- [ ] **Step 2: Draft the front-matter**

Draft the header block + TL;DR following the criteria. The TL;DR must use the locked notation (Task 0a) for any technical terms.

- [ ] **Step 3: Self-check**

Re-read the criteria; verify each is satisfied. Verify no notation drift from the locked table.

- [ ] **Step 4: Commit**

```bash
git add famail_temporal/docs/RESEARCHER_HANDOFF.md
git commit -m "docs(handoff): scaffold researcher handoff with front-matter + TL;DR"
```

### Task 2: §1 Project context (~½ page, ~250–350 words)

**Files:**
- Modify: `famail_temporal/docs/RESEARCHER_HANDOFF.md` (append §1)

- [ ] **Step 1: Acceptance criteria**

```text
[ ] Section opener (verbatim from Task 0b §1 spine).
[ ] Section closer (verbatim from Task 0b §1 spine).
[ ] States the research question explicitly.
[ ] Names the two-part contribution: (a) (cell, time-block)-granularity
    fairness audit; (b) trajectory-modification algorithm.
[ ] One-sentence lineage claim (rewrite of prior 2D / 4-block iterations,
    T = 24 is the headline change).
[ ] One-sentence data locale claim (50 drivers × 3 months × weekdays only,
    Shenzhen).
[ ] Out-of-scope bullet list: per-driver fairness, supply-side
    modification, real-time deployment.
[ ] At least 3 load-bearing claims a reviewer could push back on.
[ ] Pointer-out: famail_temporal/README.md (relative path:
    "../README.md").
[ ] Word count: 250–350.
```

- [ ] **Step 2: Read source material**

Read `famail_temporal/README.md` (Quickstart through Key design commitments) and the design spec's §1 description.

- [ ] **Step 3: Draft §1**

- [ ] **Step 4: Self-check**

Verify each criterion. Especially: read the opener and closer aloud — do they match Task 0b verbatim? If not, fix.

- [ ] **Step 5: Commit**

```bash
git add famail_temporal/docs/RESEARCHER_HANDOFF.md
git commit -m "docs(handoff): §1 project context"
```

### Task 3: §2 Dataset and active-unit construction (~1 page, ~500–650 words)

**Files:**
- Modify: `famail_temporal/docs/RESEARCHER_HANDOFF.md` (append §2)

- [ ] **Step 1: Acceptance criteria**

```text
[ ] Section opener and closer (verbatim from Task 0b §2 spine).
[ ] Grid geometry stated: 48 × 90 spatial × T = 24 hourly blocks.
[ ] Three primary tensors named with their semantic: pickup_3d, dropoff_3d,
    active_taxis_3d — all in mean-hourly-rate units.
[ ] Active-unit filter stated as a conjunction of three conditions:
    supply ≥ ACTIVE_SUPPLY_THRESHOLD, inside Shenzhen boundary, finite
    demographics.
[ ] DEDICATED PARAGRAPH on the load-bearing design choice: supply-based
    not demand-based mask. MUST give the endogeneity argument
    (residential cell with ~zero demand because residents gave up).
[ ] Canonical active-unit ordering stated (cell-major, then block-within-
    cell), with one sentence on why it matters (every (N,) array shares
    the ordering; asserted at every load boundary).
[ ] Demographics: three z-scored features named (housing price per sqm,
    GDP/capita, companies/capita).
[ ] At least 4 load-bearing claims.
[ ] Pointer-outs: data/README.md, source_data/README.md,
    F_CAUSAL_METHODOLOGY_NOTES.md §5.
[ ] Word count: 500–650.
```

- [ ] **Step 2: Read source material**

Read `famail_temporal/data/README.md` (Consumer-side key design choices §1–§3), `famail_temporal/source_data/README.md`, `famail_temporal/docs/F_CAUSAL_METHODOLOGY_NOTES.md` §5.

- [ ] **Step 3: Draft §2**

- [ ] **Step 4: Self-check against criteria**

- [ ] **Step 5: Commit**

```bash
git add famail_temporal/docs/RESEARCHER_HANDOFF.md
git commit -m "docs(handoff): §2 dataset and active-unit construction"
```

### Task 4: §3 The objective at a glance (~½ page, ~250–350 words)

**Files:**
- Modify: `famail_temporal/docs/RESEARCHER_HANDOFF.md` (append §3)

- [ ] **Step 1: Acceptance criteria**

```text
[ ] Section opener and closer (verbatim from Task 0b §3 spine).
[ ] The combined objective stated in symbolic form:
    L = α_s · F_spatial + α_c · F_causal + α_f · F_fidelity.
[ ] All three terms in [0, 1], higher = better.
[ ] Default weights stated: α_s ≈ 0.33, α_c ≈ 0.33, α_f ≈ 0.34.
[ ] ALPHA_FIDELITY = 0 named as a clean ablation (one sentence).
[ ] One-line preview of each term — these are forward-pointers to §4, §5,
    §6 respectively.
[ ] Pointer-outs: algorithm/README.md, config.py.
[ ] At least 3 load-bearing claims.
[ ] Word count: 250–350.
```

- [ ] **Step 2: Read source material**

Read `famail_temporal/algorithm/README.md` (purpose + key design choice 1) and the locked notation table (Task 0a).

- [ ] **Step 3: Draft §3**

- [ ] **Step 4: Self-check**

- [ ] **Step 5: Commit**

```bash
git add famail_temporal/docs/RESEARCHER_HANDOFF.md
git commit -m "docs(handoff): §3 objective at a glance"
```

### Task 5: §4 F_spatial — pooled Gini fairness (~¾ page, ~400–500 words)

**Files:**
- Modify: `famail_temporal/docs/RESEARCHER_HANDOFF.md` (append §4)

- [ ] **Step 1: Acceptance criteria**

```text
[ ] Section opener and closer (verbatim from Task 0b §4 spine).
[ ] DSR and ASR defined explicitly: DSR_u = pickup_u / S_u; ASR_u =
    dropoff_u / S_u. Use locked notation.
[ ] F_spatial formula stated: F_spatial = 1 − ½(Gini(DSR) + Gini(ASR)).
[ ] Pairwise Gini formula stated:
    G(x) = Σ_i Σ_j |x_i − x_j| / (2 N² mean(x)).
[ ] One-sentence note on differentiability (everywhere except measure-
    zero ties).
[ ] Sign convention stated: F_spatial = 1 ⇔ perfect equality across all
    active units; F_spatial = 0 ⇔ one unit absorbs all service mass.
[ ] EXACTLY TWO design choices, each 1–2 sentences:
       1. Pooled, not block-averaged. Rationale.
       2. DSR + ASR equal weighting. Rationale.
[ ] Pointer-out: fairness/README.md.
[ ] At least 3 load-bearing claims.
[ ] Word count: 400–500.
```

- [ ] **Step 2: Read source material**

Read `famail_temporal/fairness/README.md` (Pooled Gini section) and `famail_temporal/fairness/spatial.py` (just the docstring + formula comment, not the implementation).

- [ ] **Step 3: Draft §4**

- [ ] **Step 4: Self-check**

- [ ] **Step 5: Commit**

```bash
git add famail_temporal/docs/RESEARCHER_HANDOFF.md
git commit -m "docs(handoff): §4 F_spatial pooled Gini"
```

### Task 6: §5 F_causal — demographic-projection R² (~1¼ pages, ~700–850 words)

**Files:**
- Modify: `famail_temporal/docs/RESEARCHER_HANDOFF.md` (append §5)

This is the longest section and the one most likely to attract reviewer scrutiny. Plan time carefully.

- [ ] **Step 1: Acceptance criteria**

```text
[ ] Section opener and closer (verbatim from Task 0b §5 spine).
[ ] Stage 1 stated: g_0(D) = β₀ + β₁/(D+1) + β₂/√(D+1) + β₃√(D+1),
    fitted via OLS on all N active units, produces baseline service-rate
    prediction from demand alone.
[ ] Stage 2 stated: residuals R = Y − g_0(D); demographic projection
    H_demo = X̃(X̃'X̃)⁻¹X̃'.
[ ] Final form stated: F_causal = R'(I − H_demo)R / R'MR = 1 − r²_demo.
[ ] Sign convention stated: r²_demo high ⇔ unfair; F_causal high ⇔ fair.
[ ] EXACTLY FOUR design choices, each 1–2 sentences with rationale:
       1. Power basis for g_0 (linear-in-params + hat-matrix algebra; four
          terms together capture hyperbolic + sub-linear).
       2. DEMAND_FLOOR = 0.5 as clamp not filter (residual-scale balance;
          inclusive-audit property).
       3. Two-R² diagnostic (separates model-class adequacy from set-
          composition; the all-cells / signal-regime split).
       4. g_0 evaluated under torch.no_grad() in the modifier loop
          (avoids double-counting the demand gradient).
[ ] Pointer-outs: F_CAUSAL_METHODOLOGY_NOTES.md (full justification),
    fairness/g0_power_basis.py, fairness/hat_matrices.py.
[ ] At least 6 load-bearing claims.
[ ] Word count: 700–850.
```

- [ ] **Step 2: Read source material**

Read `famail_temporal/docs/F_CAUSAL_METHODOLOGY_NOTES.md` §1 (formulation), §2 (power basis), §3 (two-R²), §4 (DEMAND_FLOOR rationale). Also `famail_temporal/fairness/README.md` (Option B section).

- [ ] **Step 3: Draft §5**

Draft sub-structure:
- Sub-paragraph 1: Stage 1 g_0 fit
- Sub-paragraph 2: Stage 2 demographic projection
- Sub-paragraph 3: Final form + sign convention
- Sub-paragraph 4 (or compact list): the four design choices
- Closer

- [ ] **Step 4: Self-check**

Especially: does the section actually defend the DEMAND_FLOOR choice, or just state it? A reviewer needs to be able to push back on the 0.5 value specifically.

- [ ] **Step 5: Commit**

```bash
git add famail_temporal/docs/RESEARCHER_HANDOFF.md
git commit -m "docs(handoff): §5 F_causal demographic-projection R²"
```

### Task 7: §6 F_fidelity — discriminator-based realism (~½ page, ~250–350 words)

**Files:**
- Modify: `famail_temporal/docs/RESEARCHER_HANDOFF.md` (append §6)

- [ ] **Step 1: Acceptance criteria**

```text
[ ] Section opener and closer (verbatim from Task 0b §6 spine).
[ ] Discriminator named: Multi-Stream Siamese; pre-trained; opaque
    inference port in famail_temporal/.
[ ] Four ported classes named: FeatureNormalizer, SiameseLSTMEncoder,
    ProfileEncoder, MultiStreamSiameseDiscriminator.
[ ] Inputs per call described: anchor trajectory + modified trajectory,
    each rendered as multi-stream context (driving stream, seeking stream,
    profile features).
[ ] Output: similarity score in [0, 1]; F_fidelity = 1 ⇔ indistinguishable
    from authentic expert.
[ ] EXACTLY TWO design choices, each 1–2 sentences:
       1. Opaque inference-only port (inherited not contributed).
       2. ALPHA_FIDELITY = 0 as clean ablation.
[ ] One-sentence in-text mention of the cuDNN-backward-in-eval workaround
    (NOT a section).
[ ] Pointer-outs: fidelity/README.md, discriminator_checkpoints/README.md.
[ ] At least 3 load-bearing claims.
[ ] Word count: 250–350.
```

- [ ] **Step 2: Read source material**

Read `famail_temporal/fidelity/README.md` (full file).

- [ ] **Step 3: Draft §6**

- [ ] **Step 4: Self-check**

- [ ] **Step 5: Commit**

```bash
git add famail_temporal/docs/RESEARCHER_HANDOFF.md
git commit -m "docs(handoff): §6 F_fidelity discriminator-based realism"
```

### Task 8: §7 Per-cell fairness attribution (~¾ page, ~400–500 words)

**Files:**
- Modify: `famail_temporal/docs/RESEARCHER_HANDOFF.md` (append §7)

- [ ] **Step 1: Acceptance criteria**

```text
[ ] Section opener and closer (verbatim from Task 0b §7 spine).
[ ] State the decomposition problem: F is a scalar, but we want a signed
    per-cell α_i with Σ α_i = F.
[ ] State the 1/N-shifted decomposition:
       α_i = (1/N) − unfairness_contrib_i.
[ ] For F_spatial: unfairness_contrib_i = ½(gini_dsr_i + gini_asr_i)
    where gini_i(x) = Σ_j |x_i − x_j| / (2 N² mean(x)).
[ ] For F_causal: unfairness_contrib_i =
       ((MR)_i² − ((I−H)R)_i²) / R'MR.
[ ] Sign-convention table (markdown) with at least four bands:
       α_i > 1/N         (above-baseline fair)
       α_i ≈ 1/N         (neutral / uniform-share)
       0 < α_i < 1/N     (mildly underperforming)
       α_i < 0           (drags fairness below baseline; priority)
[ ] Justification for uniform 1/N: minimum-assumption prior;
    perfect-fair-limit check (every α_i = 1/N, sum = F = 1);
    perfect-unfair-limit check (outliers absorb mass, sum = 0 = F).
[ ] Pointer-out: FAIRNESS_DECOMPOSITION_FORMULATION.md (full derivation
    + worked examples + audit trail).
[ ] At least 4 load-bearing claims.
[ ] Word count: 400–500.
```

- [ ] **Step 2: Read source material**

Read `famail_temporal/docs/FAIRNESS_DECOMPOSITION_FORMULATION.md` (full file). Also `famail_temporal/fairness/causal.py::per_cell_fairness_attribution_causal` and `famail_temporal/fairness/spatial.py::per_cell_fairness_attribution_spatial` (just the function signatures and docstrings).

- [ ] **Step 3: Draft §7**

Sub-structure:
- Decomposition problem statement (1–2 sentences)
- The 1/N-shifted form (with sum-to-F invariant called out)
- Per-metric forms (compact: spatial, then causal)
- Sign convention table
- Justification of 1/N baseline (with both limits)
- Closer

- [ ] **Step 4: Self-check**

Verify the math is consistent with FAIRNESS_DECOMPOSITION_FORMULATION.md TL;DR.

- [ ] **Step 5: Commit**

```bash
git add famail_temporal/docs/RESEARCHER_HANDOFF.md
git commit -m "docs(handoff): §7 per-cell fairness attribution"
```

### Task 9: §8 Trajectory-modification algorithm (~1¼ pages, ~700–850 words)

**Files:**
- Modify: `famail_temporal/docs/RESEARCHER_HANDOFF.md` (append §8)

This is the second-longest section. It contains TWO pseudocode blocks plus seven design-choice notes.

- [ ] **Step 1: Acceptance criteria**

```text
[ ] Section opener and closer (verbatim from Task 0b §8 spine).
[ ] Outer pipeline pseudocode (text fence) covering the full pipeline
    from DataBundle.load() through F_after measurement, INCLUDING:
       - F_before measurement
       - α computation via compute_per_unit_attribution(bundle)
       - rank_trajectories(...)
       - select_top_k(ranking, k) with the α_i < 0 filter
       - sequential modify_single calls
       - F_after measurement
[ ] Inner ST-iFGSM pseudocode (text fence) covering one trajectory's
    modification, INCLUDING:
       - Pickup-cell + time-block extraction
       - Pickup-mass = 1 / (n_hours_per_block[t*] · n_days)
       - The "subtract original" line
       - The for-loop over MAX_ITERATIONS:
            * temperature anneal
            * pickup-tensor with requires_grad
            * SoftCellAssignment(p, τ) → probs
            * delta-tensor injection: soft_3d = base + inject(probs, t*)
            * Forward through Objective
            * Gradient through ∂total/∂p
            * Δ ← clip(Δ + α_step · sign(grad), −ε, ε)
            * grid clip
            * convergence check
       - The "commit to shared base" mass-balance lines
[ ] Pseudocode style: text fences (NOT python); concept-named variables;
    error handling and bookkeeping omitted; only algorithmic essence.
[ ] EXACTLY SEVEN design-choice notes, each 1–2 sentences, in this order:
       1. Soft-cell assignment via Gaussian softmax (continuous→discrete
          differentiability).
       2. Delta-tensor injection pattern (autograd safety; no in-place).
       3. Single grid-to-unit conversion point (fairness modules see only
          N-vectors).
       4. Sequential modification with shared _base_pickup_3d (order-
          dependence is intentional; attribution computed once before
          any modification).
       5. Strictly-negative top-k filter (only modify trajectories whose
          pickup is actively dragging fairness down).
       6. Pickup-mass conservation (mean-hourly aggregation; subtract at
          original, add at new).
       7. ST-iFGSM signed-gradient step (robust to gradient-magnitude
          variation across the three terms; ε-ball constrains pickup
          movement).
[ ] Pointer-outs: algorithm/README.md (gradient-flow diagram),
    algorithm/modifier.py, algorithm/attribution.py.
[ ] At least 7 load-bearing claims (one per design choice, plus the
    pseudocode itself).
[ ] Word count: 700–850.
```

- [ ] **Step 2: Read source material**

Read `famail_temporal/algorithm/README.md` (full file), `famail_temporal/algorithm/modifier.py::modify_single`, `famail_temporal/algorithm/attribution.py` (full file).

- [ ] **Step 3: Draft §8**

Sub-structure:
- Section opener
- One paragraph on the high-level approach (rank by attribution, modify top-k)
- Outer pipeline pseudocode (text fence)
- One sentence on what the inner loop does
- Inner ST-iFGSM pseudocode (text fence)
- Seven design-choice notes (numbered list)
- Section closer

- [ ] **Step 4: Self-check**

Especially: read both pseudocode blocks aloud. Are they algorithm-level (a reviewer can follow without the codebase) or implementation-level (only makes sense if you've read modifier.py)? They should be the former.

- [ ] **Step 5: Commit**

```bash
git add famail_temporal/docs/RESEARCHER_HANDOFF.md
git commit -m "docs(handoff): §8 trajectory-modification algorithm"
```

### Task 10: §9 Known limitations and open questions (~½ page, ~250–350 words)

**Files:**
- Modify: `famail_temporal/docs/RESEARCHER_HANDOFF.md` (append §9)

- [ ] **Step 1: Acceptance criteria**

```text
[ ] Section opener and closer (verbatim from Task 0b §9 spine).
[ ] EXACTLY SIX limitations, in this order, each 2 sentences:
       1. Zero-supply cells excluded entirely (active-mask design;
          can't distinguish "unfair zero supply" from "no service
          territory"). Cross-reference §2.
       2. Endogenous demand controlled but not modeled. Cross-reference
          §5.
       3. DEMAND_FLOOR = 0.5 is pragmatic, not derived. Cross-reference
          §5 design choice 2; sensitivity study suggested.
       4. Per-day fairness aggregation is pooled, not per-day. Future
          research direction. Cross-reference §2.
       5. F_fidelity inherits any bias in the discriminator. Cross-
          reference §6.
       6. Soft-cell-assignment kernel size + temperature schedule are
          unswept. Cross-reference §8 design choice 1.
[ ] Each limitation MUST cross-reference the specific earlier-section
    design choice it limits — not a generic "see above."
[ ] Pointer-out: F_CAUSAL_METHODOLOGY_NOTES.md §9.
[ ] Word count: 250–350.
```

- [ ] **Step 2: Read source material**

Read `famail_temporal/docs/F_CAUSAL_METHODOLOGY_NOTES.md` §9.

- [ ] **Step 3: Draft §9**

- [ ] **Step 4: Self-check**

For each of the six bullets, confirm the cross-reference resolves to a section/design-choice that has actually been written.

- [ ] **Step 5: Commit**

```bash
git add famail_temporal/docs/RESEARCHER_HANDOFF.md
git commit -m "docs(handoff): §9 known limitations and open questions"
```

### Task 11: §10 Where to read more (~¼ page, ~150–200 words)

**Files:**
- Modify: `famail_temporal/docs/RESEARCHER_HANDOFF.md` (append §10)

- [ ] **Step 1: Acceptance criteria**

```text
[ ] Section opener and closer (verbatim from Task 0b §10 spine).
[ ] Pointers organized by concern, in this order:
       - Math and methodology: FAIRNESS_DECOMPOSITION_FORMULATION.md,
         F_CAUSAL_METHODOLOGY_NOTES.md.
       - Module-by-module designs: data/README.md, fairness/README.md,
         fidelity/README.md, algorithm/README.md, evaluation/README.md.
       - Operational: ../README.md (top-level), ../evaluation/
         EVALUATION_QUICKSTART.md.
       - Tests as living spec: ../tests/README.md.
       - Forthcoming: per-cell attribution export tool will get its own
         standalone document; current notes at FAIRNESS_ATTRIBUTION_
         EXPORT_DESIGN.md.
[ ] All paths use markdown relative links from famail_temporal/docs/.
[ ] Word count: 150–200.
```

- [ ] **Step 2: Draft §10**

The content is deterministic — fill from the design spec.

- [ ] **Step 3: Self-check**

Verify every relative path resolves: from `famail_temporal/docs/RESEARCHER_HANDOFF.md`, `../README.md` → `famail_temporal/README.md`; `FAIRNESS_DECOMPOSITION_FORMULATION.md` → `famail_temporal/docs/FAIRNESS_DECOMPOSITION_FORMULATION.md`; etc.

- [ ] **Step 4: Commit**

```bash
git add famail_temporal/docs/RESEARCHER_HANDOFF.md
git commit -m "docs(handoff): §10 where to read more"
```

### Task 12: §11 Diagnostics snapshot (~¼ page, ~100–150 words)

**Files:**
- Modify: `famail_temporal/docs/RESEARCHER_HANDOFF.md` (append §11)

- [ ] **Step 1: Acceptance criteria**

```text
[ ] Section opener (verbatim from Task 0b §11 spine).
[ ] Single markdown table with the 10 rows from Task 0c step 4.
[ ] Footer line stating the snapshot date and the source-data git SHA,
    plus the refresh command.
[ ] Numbers come from Task 0c step 4 — NO placeholders remain.
[ ] Word count (excluding the table): 100–150.
```

- [ ] **Step 2: Verify the diagnostics snapshot from Task 0c is current**

If Task 0c step 4 was run more than a few hours ago and source data has been regenerated since, re-run Task 0c.

- [ ] **Step 3: Draft §11**

- [ ] **Step 4: Self-check**

Verify zero placeholders in the table.

- [ ] **Step 5: Commit**

```bash
git add famail_temporal/docs/RESEARCHER_HANDOFF.md
git commit -m "docs(handoff): §11 diagnostics snapshot"
```

---

## Cohesion review

The drafting tasks each verify their own internal consistency. This task verifies cross-section consistency — the cohesion property the document overall must have.

### Task 13: Cross-cutting cohesion audit

**Files:**
- Modify: `famail_temporal/docs/RESEARCHER_HANDOFF.md` (any fixes uncovered)

This is the single most important task for the user's stated goal ("cohesive handoff document that researchers can use to fully understand the famail_temporal approach"). Each step is a separate audit pass.

- [ ] **Step 1: Notation audit**

Open the document and search for every symbol in the locked notation table (Task 0a). For each symbol:

- Is it defined the first time it appears (or referenced back to its definition)?
- Is the meaning consistent across all sections it appears in?
- Are subscripts (`_u`, `_i`, `_s`, `_c`, `_f`) used consistently?

Common drift cases to check explicitly:
- `R` only ever means the residual vector (never "ranking" or "reward")
- `α` always has a subscript (`_s`, `_c`, `_f`, `_i`, `_step`); `α` alone is forbidden
- `N` and `N_active` are interchangeable but only one form appears in any single sentence
- `F_spatial`, `F_causal`, `F_fidelity` always carry subscripts (never bare `F` except in generic references like "Σ α_i = F")

If drift found, fix in place.

- [ ] **Step 2: Cross-reference integrity audit**

For every "see §X" or "as in §X" or "developed in §X" reference:
- Does §X exist?
- Does §X actually contain the thing being referenced?
- For each section's CLOSER (Task 0b): is the forward-pointer paid off in the section it points to?
- For each §9 limitation: does the cross-reference resolve to a specific design choice?

If broken cross-references found, fix in place.

- [ ] **Step 3: Length budget audit**

Run: `wc -w famail_temporal/docs/RESEARCHER_HANDOFF.md`

Expected: total word count between 3,500 and 4,500.

If under 3,500: identify the section that is most underdeveloped relative to its design-spec budget; review whether its claim-density target was met; expand if needed.

If over 4,500: identify the section most over budget; trim. The most likely culprit is §5 or §8 (longest sections). Trim to design-choice rationales (each should be 1–2 sentences, not paragraphs).

- [ ] **Step 4: Claim-density audit**

For each section, count the number of distinct claims a reviewer could individually push back on. (A claim is something stated as fact that is not self-evident — e.g., "DEMAND_FLOOR = 0.5 is chosen because it balances residual scale" is one claim; "the power basis is fitted via OLS" is not a claim, it's a definition.)

Section minimums (matching the per-section acceptance criteria):
- §1: ≥ 3 claims
- §2: ≥ 4 claims
- §3: ≥ 3 claims
- §4: ≥ 3 claims
- §5: ≥ 6 claims
- §6: ≥ 3 claims
- §7: ≥ 4 claims
- §8: ≥ 7 claims (one per design choice)
- §9: 6 claims (one per limitation)

If any section is under its minimum, expand its design-choice rationales or limitations.

- [ ] **Step 5: Sanity-check probe**

Read the handoff document straight through. Self-test against the five sanity-checkable questions from the design spec's Goals §1:

1. Can I state the research question? (§1)
2. Can I describe the dataset and active-unit construction? (§2)
3. Can I write the three fairness/realism formulations in my own notation? (§4, §5, §6)
4. Can I read the trajectory-modification pseudocode and explain the gradient flow? (§3, §7, §8)
5. Can I identify at least three places where I would push back on a methodological choice? (§5 design choices, §9 limitations)

If any of the five questions cannot be answered confidently from the document alone, the gating section is too thin. Mark which one(s) and revise.

- [ ] **Step 6: Final commit (only if any fixes were made)**

If any of steps 1–5 surfaced fixes:

```bash
git add famail_temporal/docs/RESEARCHER_HANDOFF.md
git commit -m "docs(handoff): cohesion audit fixes (notation, cross-refs, length, claim-density)"
```

If no fixes were needed, no commit is necessary.

---

## Self-Review

This plan was reviewed inline against the spec and the cohesion-risk register before publication. Findings:

**Spec coverage:**
- Every section §1–§11 in the design spec maps to exactly one drafting task (Tasks 1–12).
- Front-matter (header block) maps to Task 1.
- TL;DR adds to the design spec; called out explicitly so the user can override.
- Style and editorial conventions are encoded into the locked notation table (Task 0a) and the per-task acceptance criteria.
- Test plan from the spec maps to Task 13 (cohesion audit), which implements the self-review checklist + the sanity-check probe + the drift-check rule (the last as Task 0c step 2's re-run condition).

**Placeholder scan:** This plan contains explicit placeholders only inside Task 0c step 4 (the diagnostics-snapshot table) and inside the "fill from the gathered values" instruction in Task 12 step 3. These are intentional template placeholders, not plan failures: the values are gathered as part of Task 0c and pasted in during Task 12. Every other "<fill>" or similar marker has been eliminated.

**Type consistency:** Notation symbols are locked once in Task 0a and referenced by every drafting task. Pointer-out paths and section numbers are consistent across the plan.

**Cohesion-risk coverage:**
- Notation drift → Task 0a + Task 13 step 1.
- Sections-as-checklist → Task 0b (section spine) + per-task verbatim opener/closer requirement.
- Insufficient claim-density → per-section claim-count minimum + Task 13 step 4.
- Math without setup → spine sequencing in Task 0b ensures §3 stages §5.
- Codebase-specific pseudocode → Task 9 step 1 explicitly forbids implementation-level pseudocode.
- Limitations as throat-clearing → Task 10 acceptance criteria require each bullet to cross-reference a specific earlier design choice.
- Decorative diagnostics table → Task 12 footer requirement; numbers pulled from a verifiable source via Task 0c.
- No TL;DR → Task 1 adds a TL;DR with explicit user-override path.

---

## Assessment of cohesion potential

The user asked specifically whether this plan will produce a cohesive handoff document that lets researchers fully understand the project and scrutinize it. Honest assessment:

**What the plan does well:**

- The notation key (Task 0a) is locked before any prose is written. This eliminates the most common cohesion failure in technical documents — drift in symbol meaning between sections.
- The section spine (Task 0b) is a narrative spine: openers and closers are written verbatim, and they compose into a coherent paragraph that walks the reader from problem to algorithm to gaps. This is the most important cohesion mechanism in the plan; it's what differentiates a document that reads as one argument from a document that reads as a checklist.
- Per-section acceptance criteria specify *minimum claim density*. A reviewer's ability to scrutinize is bottlenecked by the number of substantive, defensible claims a section makes. Specifying minimums forces the drafter to write rationale, not just statement.
- Task 13's five-step cohesion audit is a stand-alone pass that re-reads the document against five orthogonal axes (notation, cross-refs, length, claim-density, sanity-check probe). This catches drift introduced during drafting.
- §9 limitations are tied by cross-reference to specific earlier design choices. A reviewer scrutinizing the project doesn't just want "what's wrong" — they want "what's wrong about this specific choice." The plan enforces this.

**Residual risks the plan does NOT fully eliminate:**

- The pseudocode in §8 is a synthesis point. If it leans too implementation-level (variable names from `modifier.py`, error handling, bookkeeping), reviewers will read it as code review, not algorithm review. Task 9 step 1 forbids this in the criteria, but the criterion is a discipline rather than a checkable invariant. The drafter must self-police.
- The §5 (F_causal) design-choice rationales must be 1–2 sentences each, but DEMAND_FLOOR specifically has a substantial empirical defense in `F_CAUSAL_METHODOLOGY_NOTES.md`. The 1–2-sentence budget may force an oversimplification that loses the residual-scale-balance argument. If this happens, the criterion is "rationale is present and pointer-out resolves to the full empirical defense" — the reader can drill down. Task 6 step 4's self-check is the gate.
- The TL;DR is added by Task 1 without explicit user approval (the design discussion did not raise it). If the user prefers no TL;DR, Task 1's content is reduced to just the header block. The plan should be transparent about this addition.
- The diagnostics snapshot (§11) freezes a specific point-in-time view. If source data is regenerated between Task 0c and final commit, the snapshot drifts. Task 12 step 2 mitigates this with a re-run check, but the rule is "re-run if more than a few hours have passed" — a reasonable but not-rigorous heuristic.

**Bottom line.** The plan should produce a document that meets the user's stated goal — a researcher can read it end-to-end, understand the project, and identify methodologically scrutable points. The two specific risks worth watching during execution are (a) pseudocode drift toward implementation-level, and (b) the §5 design-choice 1–2-sentence budget squeezing out too much of the empirical defense. Both are flagged in the per-task self-checks; neither is a blocker.

---

## Execution handoff

Plan complete and saved to `docs/superpowers/plans/2026-05-07-researcher-handoff-document.md`. Two execution options:

1. **Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration. Each section gets a focused subagent that reads the relevant in-tree material and drafts against the acceptance criteria.

2. **Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints. I draft the document in this conversation; we review at section boundaries.

Which approach?
