# §4 Restructure Plan v2 — Dr. Zhang's 2026-07-26 revision demands

**Deadline:** 2026-07-26 14:00 (hard; withdrawal is the stated alternative).
**Status:** v1 was verified by 6 adversarial agents; **all six returned FAIL**.
v2 is rebuilt against their findings. Every number below is *measured*, not
estimated. Do not re-introduce v1's assumptions.

---

## 0. Measured geometry (the currency)

acmart sigconf, this paper: **58 line-slots per column, 116 per page.**
Page 8 spans slots 813–928. The last body line is slot **927**
("with it is unmeasured.", §6 close). **Slack today = 1 slot.**

Budget in slots. Source lines and `%` comments are NOT slots — comments are free.

**Invariant:** no number, CI, p-value, or seed count may change. Content may be
*moved* or *cut*, never altered. Every `% src:` and `% lint-allow:` comment
travels on the same line as its number.

---

## 1. Demands

| # | Demand | Addressed |
|---|---|---|
| D1 | Reorganize §4 around problem-driven questions; drop defensive framing | §4 |
| D2 | Simplify language; remove defensive explanation from main text | §4, §7 |
| D3 | Every major claim supported by a **visible main-text** float | §5 |
| D4 | Dose-response **figure** for downstream transfer | §6 |
| D5 | **All** baseline results in a main-text table | §5.2 |
| D6 | Baselines cited + explained in the **experiment setting** section | §4.1 |
| D7 | Move implementation detail + secondary analyses to appendix | §3 |
| D8 | Adopt her 4.1–4.5 structure | §4 |
| D9 | Follow ST-SiameseNet / cGAIL organization and style | §2 |
| R1 | (Robert) State diplomatically WHY these baselines | §4.1 |
| R2 | (Robert) Enforce simplicity HARD; cutting detail/results authorised | §3 |

---

## 2. Style target (from her two papers)

**ST-SiameseNet §4:** 4.1 Evaluation Metrics · 4.2 Baseline Algorithms
(numbered list, citation each) · 4.3 Results (4.3.1 → **main-text Table 1**,
prose walks the table saying *why* each baseline loses) · 4.4 Case Studies.

**cGAIL §6:** 6.3 Experiment Settings (baselines named+cited here) ·
6.4 Baseline Methods Comparison (**Table + Figure**) · 6.5, 6.6 focused studies.

**Adopt:** declarative sentences; baselines as a numbered list in setup; one
comparison table per claim with ours first and bolded; percentage improvements;
figures carry sensitivity analyses.

**Delete from body (pure register, no information loss):** "class (i)/(ii)/(iii)"
labels at point of use, "claim discipline", "reported for completeness rather
than as evidence", "a by-design trade-off we disclose rather than net out",
"we decompose it rather than take it at face value", and every sentence whose
job is to pre-empt an objection rather than report a result.

---

## 3. THE SLOT LEDGER (binding)

### Removals

| Item | Now | After | Freed |
|---|---|---|---|
| **§4.6 San Francisco** (04:491–581) | **56** | 20 | **36** |
| RQ paragraph (04:17–37) | 21 | 9 | 12 |
| Metric-class block (04:112–131) | 18 | 6 | 12 |
| §4.4 opener duplication (04:333–341) — replaced by the 4.1 list | 8 | 0 | 8 |
| Per-dose / control numeric prose → figure | 14 | 0 | 14 |
| Protocol (04:81–93) | 14 | 7 | 7 |
| Fidelity paragraph (04:210–217) | 8 | 2 | 6 |
| Provenance disclosures (04:220–224) | 6 | 0 | 6 |
| §4.5 robustness compression | — | — | 10 |
| **Total freed** | | | **111** |

### Additions

| Item | Slots |
|---|---|
| `tab:baselines` → main, panel-split (§5.2) | 17 |
| Dose-response figure, `figsize=(3.35, 2.0)`, caption ≤5 lines | 20 |
| Baseline list in 4.1, **one clause per item**, 7 items + `\topsep` | 16 |
| `tab:l1` four-source → main §4.3 | 12 |
| **Total added** | **65** |

**Projected net: 46 slots freed.** Required margin ≥15 (float rounding can cost
10+ discontinuously). Margin holds.

### Contingency, in drop order
1. `tab:l1` out of main, back to appendix (+12)
2. SF compression 20 → 12 (+8)
3. Baseline list to bare names, no descriptions (+6)
4. **Never drop:** `tab:baselines` in main (D5), dose figure (D4).

### The SF decision (dominates everything)
§4.6 is 56 slots, 18% of §4, and v1 never budgeted it. **Ruling: compress to
~20 slots.** Her own 4.5 title ("Robustness Across Cities…") wants cross-city
evidence as *robustness*, not as a deep dive. Retained in body: the two-city
reproduction statement, ΔF_demo +0.0316, the n=12 SF flagship, the supply-channel
replication, and the F_spatial non-replication (a negative result — must stay
visible). Routed to a new Appendix C block: the tier-1/tier-2 recount narrative,
the Reading-B resolution, per-arm SF baseline values, and the SF caveats.
Nothing is deleted; the numbers move.

---

## 4. Target structure

### 4.1 Experimental Setup
- **Datasets** — unchanged.
- **Editor configuration** — unchanged.
- **Evaluation metrics** — rewrite. Plain list of instruments with direction
  arrows. **Keep a one-clause naming of the three classes** ("optimized,
  design-targeted, and external") — this costs ~2 slots and preserves
  `three-class criterion` for 03:209, 04:66, 04:451–457, App C:189–196 and the
  `fig:alpha-pareto` caption. Do NOT delete the term.
- **Baseline algorithms** — NEW `\textbf{}` run-in (**not** a `\subsection`;
  a sixth subsection would renumber everything). Opens with R1:

  > No existing method edits trajectories for fairness under a fidelity
  > constraint, so we draw baselines from the two nearest families: bounded
  > perturbation methods, which edit trajectories but not for fairness, and
  > fairness interventions, which target fairness but act on the model rather
  > than on the demonstrations. Each is adapted to our setting and run at the
  > same edit budget as FATE.

  Then one clause per arm: (1) ST-iFGSM \cite{hu2023stifgsm}; (2) FGSM
  \cite{goodfellow2015fgsm}; (3) random jitter (direction placebo);
  (4) demographic oversampling + untargeted placebo; (5) Kamiran–Calders
  reweighing \cite{kamirancalders2012} — **"by demographic group, not by edit"**
  (keep, `fbe876f`); (6) parity penalty in the spirit of \cite{zheng2023};
  (7) BC- and GAN-generated corpora as alternative data sources.
  **Delete 04:333–341** when this lands, but **carry its "expected not to
  improve" clause into §4.4 prose** — it is what makes the comparison honest.
- **Protocol** — 3 sentences. The bootstrap-CI understatement disclosure
  ("units are spatially correlated and demographics district-constant") exists
  in exactly ONE place in the paper: it must land in `app:protocol`, not vanish.

### 4.2 Data-Level Fairness and Resource Lift-Up
Opens on: *does FATE reduce service disparity, and does it raise under-served
service rather than only cutting service elsewhere?*
- `tab:external-sz` stays. **Move `\label{sec:exp-ablation}` to sit immediately
  after the `\subsection` line** — today it anchors to `table.caption.6`
  (Table 1), a live defect (`main.aux:59`).
- **Keep 04:130–131 and 04:168–169 verbatim.** Both are read-aloud repairs from
  the last 24h (`54b0ff1`, `2689524`) and together are worth ~1 slot. Do not
  touch the accounting-convention parenthetical, and do **not** point at
  "Appendix D" — that block defines *grouping*, not *accounting*, conventions.
- Trim-only ablation: keep, compressed. It answers her question 2 directly.
- Fidelity → one clause. Provenance disclosures → new Appendix B block beside
  "King-move repair" (07:121), and **repair the 03_methodology.tex:429 pointer**.

### 4.3 Downstream Transfer
- **Table 2** = `tab:l1` moved from App C. Its lead-in prose (07:303–306) must
  travel with it or be rewritten; fix the `tab:featsets` cell at 07:257 that
  says "see Tab.~\ref{tab:l1}".
- **Figure 3** = new dose-response (§6).
- Prose keeps: vanilla null (+0.0016, n.s.), flagship (+0.0297, 12/12,
  p=.00049), both controls null. Everything else → appendix.

### 4.4 Comparison with Baselines
- **Table 3** = `tab:baselines`, **two panels in one float** (§5.2).
- Prose walks the table, ST-SiameseNet style. Keep the random-jitter surprise
  and the "expected not to improve" clause. Keep "by demographic group rather
  than by edit" at point of claim.

### 4.5 Robustness Across Cities and Demographic Features
- **Both labels on the heading:**
  `\label{sec:exp-robustness}\label{sec:exp-sf}` — four rendered refs to
  `sec:exp-sf` (04:35, 04:278, 04:303, 07:465) die otherwise.
- Fix RQ5 so it does not print "(§4.5) … (§4.5)" twice.
- Fix 04:527, which becomes a self-reference after the merge.
- SF first (~20 slots), then feature sets, then weight sensitivity.

---

## 5. Floats

| Float | Now | After |
|---|---|---|
| `tab:external-sz` | main §4.2 | main §4.2 |
| `tab:l1` | App C | **main §4.3** |
| `tab:baselines` | App C | **main §4.4** |
| dose-response fig | — | **main §4.3** |
| `fig:alpha-pareto`, `tab:featsets` | App C | App C |

**D3 honesty:** §4.1 and §4.5 carry no float. §4.5 hosts the San Francisco
claim, so D3 is met for the three result subsections only. This is stated, not
hidden. If the measured margin exceeds 25 slots after the first build, promote
a compact 4-row SF before→after table into §4.5.

**Placement rule:** declare every float AFTER its first `\ref` in source order
(the fix already recorded at 04:177–186), then verify in the *rendered PDF* that
each lands in its citing subsection's column.

### 5.2 The D5 table, specified
One float, two panels, so the two measurement levels are never read as one scale:

- **Panel A — data-level (corpus edits, matched budget):** FATE trim+lift
  **+0.0226**, ST-iFGSM, FGSM, random jitter, demographic oversampling,
  untargeted placebo. Columns: ΔF_demo, ΔF_spatial, inflation.
- **Panel B — training-side (downstream paired BC, n=6, w30):** FATE **+0.0302**
  as the reference row, Kamiran–Calders reweighing −0.0227, parity penalty
  (inert; state the λ range in the caption, it has no single value).

Panel B's header must name the quantity. Putting −0.0227 in the same column as
+0.0226 is a category error the invariant forbids.

---

## 6. The dose-response figure, specified

v1's spec was **not buildable**. Corrected:

- **No `random_w20` exists.** Keys present in
  `alpha_sweep_s10_c80_f10_filtered_6seed/paired_stats.json`: edited,
  edited_w10/20/30, most_fair_w10/20/30, random_w10, random_w30. Draw random as
  two markers joined by a dashed segment; **state the w20 omission in the caption.**
- **Knee/saturation** (w40 +0.0323, w50 +0.0339) lives in
  `famail_temporal/results/weighted_bc_sweep/alpha_sweep_s10_dose_ext_6seed`,
  NOT in the directory v1 named. Include it or the curve shows no knee.
- **All series are n=6.** `dose_response.json` w30 = +0.0302; the n=12 flagship
  +0.0297 is a *different* quantity — keep it in prose, out of the figure.
- **Vanilla** is +0.0016 (n=12, n.s.), not zero. Plot with an n.s. marker or
  annotate; do not draw it at the origin.
- `make_figures.py` has **no** dose-response generator — one must be written.
  `figsize=(3.35, 2.0)`, caption ≤5 lines.
- Verify every plotted value against source JSON before committing.

---

## 7. §3 edits are required (v1 wrongly forbade them)

Four are forced by content moves, and a `\ref` repair cannot fix a missing
definition:
- `03:209` "all three metric classes (\S\ref{sec:exp-setup})" — survives only
  because 4.1 keeps the one-clause naming. Verify after the rewrite.
- `03:213` "both accounting conventions (\S\ref{sec:exp-data-level})" — survives
  because we are NOT moving that definition.
- `03:429` provenance pointer — repoint to the new Appendix B block.
- `03:376` `\ref{sec:exp-ablation}` — verify the re-anchored label still prints 4.2.

---

## 8. Gates (v1's were insufficient)

1. `latexmk -pdf -g` exit 0, 0 LaTeX errors.
2. **New lint gate:** assert "REFERENCES" begins on page 9 — a body spill must
   fail loudly, not be caught by eye.
3. `lint.sh` green incl. the 5pt Overfull gate (an extended table can trip it).
4. **Label superset check:** `grep -o 'newlabel{[^}]*}' main.aux | sort` before
   and after; the after-set must contain every prior key.
5. `grep -c 'multiply defined' main.log` == 0; 0 undefined.
6. Anchor check: `sec:exp-ablation`, `sec:exp-sf`, `sec:exp-robustness` must
   anchor to `subsection.*`, not `table.caption.*`.
7. Manual read of the ~20 refs enumerated by the ref-integrity verifier.
8. Rendered-float check: each of Tables 1–3 and the figure lands in its citing
   subsection.
9. Stale-stamp sweep: `grep -n '§4\.[0-9]' sections/*.tex` and re-stamp.

---

## 9. Residual, to disclose to Robert on wake

Her papers benchmark against competing methods for the same task. Ours are
adapted perturbation and fairness methods, because no prior method edits
trajectories for fairness — now stated in 4.1 per R1. Restructuring cannot
close that; only the cGAIL-as-downstream-learner experiment can.
