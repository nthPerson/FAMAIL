# FAMAIL paper-argument documentation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create the `PAPER/argument/` documentation set (README + 8 docs) that encapsulates the FAMAIL paper argument and doubles as self-contained context for a generic slide-building agent.

**Architecture:** Additive documentation only — no code, no experiments, no figure regeneration. Each doc is a focused, self-bounded markdown file that embeds verified headline numbers + key tables, references existing `PAPER/` figures by path, and ends with a provenance footer. Number-heavy results docs are written first (they anchor the values); the overview and README are written last (they must agree with the results docs).

**Tech Stack:** Markdown. Verification via `grep`, `test -f`, and small `python3` snippets that read the source JSONs. Design spec: `docs/superpowers/specs/2026-07-01-paper-argument-docs-design.md` (read it before starting — it has the full per-doc spec).

## Global Constraints

Every task implicitly includes these (from the spec):

- **Product-agnostic:** no document names any specific tool/product. Agent-facing guidance is written for a generic "presentation agent" / "an agent building slides". (Verify: `grep -rniE 'cowork|co-work' <file>` returns nothing.)
- **Numbers are seed MEANS, never seed-0.** Every cited fairness number must be traceable to a listed source JSON. When a task gives a value, confirm it against the cited JSON before writing; if it differs, use the JSON value.
- **`p = 0.03125` is the n=6 Wilcoxon floor** (sign-unanimity, not a magnitude). Always pair it with the mean Δ + t-CI + monotone dose-response + null controls. `n=5` two-sided Wilcoxon cannot reach p<0.05 (floor 0.0625) — report those nulls vs the cross-seed noise band.
- **Figures are referenced by `PAPER/`-relative path, never regenerated or copied.** Every referenced path must resolve (`test -f`).
- **Metric naming:** use `F_causal` + the associational caveat; note the `F_demo` rename is a pending PI decision. Do not rename.
- **Thesis (PI-approved):** FAMAIL is a fairness-oriented data-augmentation method — edit a small unfair slice of real trajectories, then upweight it in training so fairness propagates (two pillars: edited data is fairest-faithful; vanilla BC averages it away, weighted BC recovers it edit-specifically). Validated on two cities. Shenzhen is primary; SF is external validity.
- **Audience:** technical peers (ML / fairness / mobility-literate).
- **Each content doc ends with a "Sources / provenance" footer** listing the JSONs/tables/methodology docs it draws from.
- **Commit after each task.** Branch is `paper-argument-docs` (already checked out).

## File Structure

```
PAPER/argument/
  README.md                 index + generic-agent entry point + suggested slide outline      (Task 9)
  00_overview.md            elevator argument + headline-numbers table + money-figure refs    (Task 3)
  01_motivation_goals.md    motivation + edit-vs-generate + contributions                     (Task 4)
  02_datasets.md            Shenzhen + SF + compatibility rationale + cleanup                 (Task 5)
  03_fairness_theory.md     F_causal/F_spatial/Fidelity-A/B + editor + Resources              (Task 6)
  04_evaluation.md          two-pillar experimental design + gate + statistical conventions   (Task 7)
  05_results_shenzhen.md    Shenzhen primary results                                          (Task 1)
  06_results_sf.md          SF external-validity results                                      (Task 2)
  07_limitations.md         candid limitations + open questions + review-credibility note     (Task 8)
```
Also modified (Task 9): `famail_temporal/baselines/PAPER_ARGUMENT_PLAN.md` and `docs/two_level_argument.md` (add a one-line "superseded" pointer at the top).

---

### Task 1: `05_results_shenzhen.md` (Shenzhen primary results — anchors the Shenzhen numbers)

**Files:**
- Create: `PAPER/argument/05_results_shenzhen.md`
- Read (sources): `PAPER/by_feature_set/housing-comp-migrant/data/{editor_hcm_metrics,L1v2_hcm_multiseed,L2_hcm_metrics,weighted_bc_hcm_sweep,weighted_bc_hcm_paired_stats,variance_hcm_aggregate}.json`; `PAPER/by_feature_set/housing-comp-migrant/tables/pareto_points_hcm.csv`; `PAPER/feature_selection/tables/comparison_across_sets.md`; `PAPER/by_feature_set/housing-comp-migrant/README.md`.

**Interfaces:**
- Produces: the canonical Shenzhen numbers that Task 3 (`00_overview`) headline table must match exactly.

**Verified values to use (confirm each against the JSON before writing; means, not seed-0):**
- Editor (causal-emphasis α=0.2/0.7/0.1): F_causal 0.7988 → 0.8132 (Δ **+0.0144**); F_spatial 0.1034 → 0.1025.
- L1 per-source F_causal means: edited **0.8132** > raw 0.7988 ≈ bc 0.7980; gan 0.8089 (distributionally disqualified). Fidelity-A: raw 0.848, edited 0.843, bc 0.848, gan 0.848. Fidelity-B: raw 0.000, edited 0.149, bc 0.011, gan 0.173.
- L2 vanilla edited−raw ΔF_causal **−0.0012** (n=5, p=0.44, n.s., mixed signs).
- Weighted-BC edited ΔF_causal **+0.0205 / +0.0278 / +0.0311** at w10/20/30 (t-CIs [+0.019,+0.022] / [+0.025,+0.031] / [+0.027,+0.035]; 6/6 seeds; p=0.03125 = n=6 floor).
- Controls @ w30: most_fair **+0.0004** (p=1.0, null, mixed signs); random **−0.0009** (p=0.56, null). Edit÷select ratio @ w30 ≈ **70×**.
- Pareto: raw F_causal 0.7988; filter@K *lowers* it (0.7935 at K=2455); edit 0.8132. F_spatial small movements (raw 0.1034, edit 0.1025, filter 0.1046) — causal-emphasis run, F_spatial not the objective (do NOT claim "edit improves both").
- Model-level variance (b0 vs FAMAIL) ΔF_causal **−0.0011 ± 0.0032** (null).
- Three feature sets before-edit F_causal: 0.799 (PRIMARY {housing,comp,migrant}) / 0.807 ({housing,gdp,comp}) / 0.725 ({housing,comp,migrant,logpop}); the two-pillar story reproduces under all three (3-way `comparison_across_sets.md`).

**Section outline:** (1) Editor dual-metric result; (2) Pillar 1 — L1 data quality (edited fairest faithful; GAN disqualified; deterministic-gap caveat — raw/edited std=0, editor's own objective); (3) L2 vanilla null; (4) Pillar 2 — weighted-BC recovery + edit≫select>random + Pareto (filtering lowers F_causal); (5) model-level variance null; (6) 3-feature-set robustness (scale shifts, conclusions hold). Sources/provenance footer.

**Figures to reference:** `by_feature_set/housing-comp-migrant/figures/{fig_dose_response.png, fig_l1_data_quality.png, fig_l2_negative_transfer.png, fig_fidb_components.png, pareto_causal_hcm.png, pareto_spatial_hcm.png}`; `feature_selection/figures/fig_feature_robustness.png`.

- [ ] **Step 1: Verify the source numbers.** Run:
```bash
cd /home/robert/FAMAIL && python3 - <<'PY'
import json, numpy as np
d="PAPER/by_feature_set/housing-comp-migrant/data/"
m=lambda v: round(float(np.mean(v)),4)
ed=json.load(open(d+"editor_hcm_metrics.json"))
print("editor F_causal", round(ed["metrics_before"]["f_causal"],4), "->", round(ed["metrics_after"]["f_causal"],4))
l1=json.load(open(d+"L1v2_hcm_multiseed.json"))["per_source"]
for s in ["raw","edited","bc","gan"]: print("L1",s,"F_causal",m(l1[s]["f_causal"]["values"]),"FidA",m(l1[s]["fidelity_a"]["values"]),"FidB",m(l1[s]["fidelity_b"]["values"]))
ps=json.load(open(d+"weighted_bc_hcm_paired_stats.json"))["f_causal"]
for a in ["edited_w10","edited_w20","edited_w30","most_fair_w30","random_w30"]: print("WBC",a,round(ps[a]["mean"],4),"p",ps[a]["wilcoxon_p"])
l2=json.load(open(d+"L2_hcm_metrics.json"))["paired"]["f_causal"]["raw"]; print("L2 edited-raw",round(l2["mean"],4),"p",l2.get("wilcoxon_p"))
var=json.load(open(d+"variance_hcm_aggregate.json"))["paired_delta"]["f_causal"]; print("variance",round(var["mean"],4),"+/-",round(var["std"],4))
PY
```
Expected: values match the "Verified values" block above (edited 0.8132 > raw 0.7988 ≈ bc 0.7980; WBC +0.0205/+0.0278/+0.0311; L2 −0.0012; variance −0.0011).

- [ ] **Step 2: Confirm figure paths exist.** Run:
```bash
cd /home/robert/FAMAIL && for f in by_feature_set/housing-comp-migrant/figures/fig_dose_response.png by_feature_set/housing-comp-migrant/figures/fig_l1_data_quality.png by_feature_set/housing-comp-migrant/figures/fig_l2_negative_transfer.png by_feature_set/housing-comp-migrant/figures/fig_fidb_components.png by_feature_set/housing-comp-migrant/figures/pareto_causal_hcm.png by_feature_set/housing-comp-migrant/figures/pareto_spatial_hcm.png feature_selection/figures/fig_feature_robustness.png; do test -f "PAPER/$f" && echo "ok $f" || echo "MISSING $f"; done
```
Expected: all `ok`.

- [ ] **Step 3: Write `PAPER/argument/05_results_shenzhen.md`** to the section outline, embedding the verified numbers and a small markdown table per pillar, referencing the figures by path, honoring every Global Constraint (deterministic-gap caveat; n=6-floor framing; do-not-claim-improves-both on Pareto F_spatial). End with a Sources/provenance footer listing the JSONs above.

- [ ] **Step 4: Verify the doc.** Run:
```bash
cd /home/robert/FAMAIL && grep -niE 'cowork|co-work' PAPER/argument/05_results_shenzhen.md || echo "no-product-names OK"; grep -c "0.8132\|+0.0311\|0.7988" PAPER/argument/05_results_shenzhen.md
```
Expected: "no-product-names OK"; the grep count ≥ 3 (headline numbers present).

- [ ] **Step 5: Commit.**
```bash
cd /home/robert/FAMAIL && git add PAPER/argument/05_results_shenzhen.md && git commit -m "docs(argument): Shenzhen primary results (05)"
```

---

### Task 2: `06_results_sf.md` (SF external-validity results — anchors the SF numbers)

**Files:**
- Create: `PAPER/argument/06_results_sf.md`
- Read (sources): `PAPER/second-dataset/data/{sf12_dual_metrics,sf12_fairoff_k2000_metrics,eval_l1v2_sf12_metrics,eval_l2_sf12_metrics,eval_l2_sf12_paired_stats,eval_weighted_bc_sf12_paired_stats,eval_variance_sf12_aggregate,sf12_discriminator_training}.json`; `PAPER/second-dataset/tables/*.csv`; `PAPER/second-dataset/FINDINGS.md` §5–7.

**Interfaces:**
- Produces: the canonical SF numbers that Task 3 (`00_overview`) headline table must match exactly.

**Verified values to use (confirm against the JSONs/FINDINGS; means, not seed-0):**
- Dual claim: F_causal 0.8752 → 0.8891 (Δ **+0.0139**); F_spatial 0.1846 → 0.1817; **F_fidelity 0.968** (edit-induced Δ ≈ −1.5e-5). Fidelity inert as a gradient: fair-off +0.01392 vs fair-on +0.01394. Full-unfair-pool selection metric +0.0199 (§C.2 — different edit subset, not a regression). Discriminator val-AUC **0.998**.
- Pillar 1 (L1): edited **0.8891** > raw 0.8752 ≈ bc 0.8789 ≈ gan 0.8794; Fidelity-A **0.958** (= raw); Fidelity-B: raw 0.000, edited 0.106, bc 0.010, gan 0.027.
- L2 vanilla edited−raw ΔF_causal **+0.0004 ± 0.0033** (n=5, p=0.81, null).
- Pillar 2 weighted-BC edited **+0.0296 / +0.0348 / +0.0387** (w10/20/30, 6/6 seeds). **Both controls negative:** random −0.0071 / −0.0095; most-fair −0.0117 / −0.0068 / −0.0027 → sharper than Shenzhen.
- Model-level variance ΔF_causal **−0.0005 ± 0.0043** (null).
- GAN did **not** collapse on SF (Fidelity-B 0.027 vs Shenzhen ~0.32) — honest divergence, not load-bearing. Edited fraction ~12.6%.

**Section outline:** (1) the dual claim (fairer + realistic, no algorithm change; the two ΔF_causal figures; fidelity inert; val-AUC); (2) Pillar 1 reproduction; (3) L2 null; (4) Pillar 2 recovery with both controls negative (sharper); (5) variance null; (6) head-to-head table (Shenzhen vs SF); (7) the GAN-did-not-collapse divergence + why not load-bearing. Sources/provenance footer.

**Figures to reference:** `second-dataset/figures/sf_supply_demand.png`.

- [ ] **Step 1: Verify the source numbers.** Run:
```bash
cd /home/robert/FAMAIL && python3 - <<'PY'
import json,numpy as np
d="PAPER/second-dataset/data/"
du=json.load(open(d+"sf12_dual_metrics.json")); print("dual keys", list(du.keys())[:8])
ps=json.load(open(d+"eval_weighted_bc_sf12_paired_stats.json"))["f_causal"]
for a in ["edited_w10","edited_w20","edited_w30","random_w10","random_w30","most_fair_w10","most_fair_w20","most_fair_w30"]:
    if a in ps: print("WBC",a,round(ps[a]["mean"],4),"p",ps[a].get("wilcoxon_p"))
l2=json.load(open(d+"eval_l2_sf12_paired_stats.json")); print("L2 keys", list(l2.keys())[:6])
var=json.load(open(d+"eval_variance_sf12_aggregate.json"))["paired_delta"]["f_causal"]; print("variance",round(var["mean"],4),"+/-",round(var["std"],4))
PY
```
Expected: WBC edited +0.0296/+0.0348/+0.0387; random & most-fair arms negative; variance −0.0005. (If a key name differs, inspect the JSON and use the actual structure; cross-check against `PAPER/second-dataset/tables/eval_weighted_bc_recovery.csv` and FINDINGS §6.3.)

- [ ] **Step 2: Confirm the figure path exists.** Run:
```bash
cd /home/robert/FAMAIL && test -f PAPER/second-dataset/figures/sf_supply_demand.png && echo ok || echo MISSING
```
Expected: `ok`.

- [ ] **Step 3: Write `PAPER/argument/06_results_sf.md`** to the outline, embedding the verified SF numbers + the head-to-head table (Shenzhen editor +0.0144 vs SF +0.0139; WBC +0.0311 vs +0.0387; controls ~null vs negative; both variance null; GAN collapsed vs not), referencing the figure, honoring the Global Constraints (SF F_causal is associational + ACS-proxy → not cross-city-absolute-comparable; frame SF as "on par with / reproduces" Shenzhen, not "beats"; n=6-floor framing). Sources/provenance footer.

- [ ] **Step 4: Verify the doc.** Run:
```bash
cd /home/robert/FAMAIL && grep -niE 'cowork|co-work' PAPER/argument/06_results_sf.md || echo "no-product-names OK"; grep -c "0.8891\|+0.0387\|0.968" PAPER/argument/06_results_sf.md
```
Expected: "no-product-names OK"; count ≥ 3.

- [ ] **Step 5: Commit.**
```bash
cd /home/robert/FAMAIL && git add PAPER/argument/06_results_sf.md && git commit -m "docs(argument): SF external-validity results (06)"
```

---

### Task 3: `00_overview.md` (elevator argument — must agree with Tasks 1 & 2)

**Files:**
- Create: `PAPER/argument/00_overview.md`
- Read: `PAPER/argument/05_results_shenzhen.md`, `PAPER/argument/06_results_sf.md` (numbers MUST match these).

**Interfaces:**
- Consumes: the Shenzhen numbers from Task 1 and SF numbers from Task 2. The headline-numbers table must be identical to the values in those docs.

**Section outline:** (1) the problem (demographic-driven service inequity in real mobility data); (2) the approach (edit the unfair slice, don't generate); (3) the two-pillar result in two sentences; (4) external validity (SF reproduces it, no algorithm change); (5) the data-augmentation framing; (6) a **headline-numbers table** with Shenzhen PRIMARY + SF side by side (editor ΔF_causal +0.0144 / +0.0139; L1 edited-fairest 0.8132 / 0.8891; L2 null −0.0012 / +0.0004; weighted-BC w30 +0.0311 / +0.0387; controls — Shenzhen ~null, SF negative); (7) "money figures" pointer (`fig_dose_response.png` for each city's recovery; `fig_l1_data_quality.png`). Sources/provenance footer (points to 05/06).

- [ ] **Step 1: Re-read the two results docs** to lift the exact numbers. Run:
```bash
cd /home/robert/FAMAIL && grep -hoE '\+?-?0\.0[0-9]{3}' PAPER/argument/05_results_shenzhen.md PAPER/argument/06_results_sf.md | sort -u | head -40
```
Expected: the set of numbers used; use only values present in 05/06.

- [ ] **Step 2: Write `PAPER/argument/00_overview.md`** to the outline (≤ 2 pages), with the side-by-side headline table, honoring Global Constraints.

- [ ] **Step 3: Verify consistency + no product names.** Run:
```bash
cd /home/robert/FAMAIL && grep -niE 'cowork|co-work' PAPER/argument/00_overview.md || echo "no-product-names OK"; for n in 0.8132 0.8891 +0.0311 +0.0387 +0.0144 +0.0139; do grep -q -- "$n" PAPER/argument/00_overview.md && echo "has $n" || echo "MISSING $n"; done
```
Expected: "no-product-names OK"; every `has <n>` (the headline numbers appear and match 05/06).

- [ ] **Step 4: Commit.**
```bash
cd /home/robert/FAMAIL && git add PAPER/argument/00_overview.md && git commit -m "docs(argument): overview / elevator argument (00)"
```

---

### Task 4: `01_motivation_goals.md`

**Files:**
- Create: `PAPER/argument/01_motivation_goals.md`
- Read: `docs/two_level_argument.md` (umbrella-claim framing); `PAPER/second-dataset/FINDINGS.md` §8 (data-aug framing).

**Section outline:** (1) why taxi/mobility service inequity matters (real-world equity stakes); (2) how imitation-learned demand models inherit/amplify demographic service bias present in the data; (3) why *editing real data* beats *generating* synthetic data — keep human fidelity, target only the unfair slice, avoid the distributional collapse that plagues generation; (4) the data-augmentation positioning (edit a slice → upweight in training); (5) explicit contributions: a demand-adjusted demographic-fairness metric + per-cell attribution, an attribution-guided trajectory editor, the two-pillar training recipe (weighted BC), and two-city validation. Sources/provenance footer.

- [ ] **Step 1: Write `PAPER/argument/01_motivation_goals.md`** to the outline (no numbers required beyond framing; if any number appears, it must match 05/06). Honor Global Constraints (data-augmentation thesis; audience).

- [ ] **Step 2: Verify.** Run:
```bash
cd /home/robert/FAMAIL && grep -niE 'cowork|co-work' PAPER/argument/01_motivation_goals.md || echo "no-product-names OK"; test -s PAPER/argument/01_motivation_goals.md && echo "non-empty OK"
```
Expected: "no-product-names OK"; "non-empty OK".

- [ ] **Step 3: Commit.**
```bash
cd /home/robert/FAMAIL && git add PAPER/argument/01_motivation_goals.md && git commit -m "docs(argument): motivation & goals (01)"
```

---

### Task 5: `02_datasets.md`

**Files:**
- Create: `PAPER/argument/02_datasets.md`
- Read: `PAPER/shared_cleanup/README.md` + `PAPER/shared_cleanup/tables/dataset_summary.md`; `PAPER/second-dataset/FINDINGS.md` §1–3; the three editor metrics JSONs for the before-edit baselines (`PAPER/by_feature_set/*/data/editor_*_metrics.json`).

**Verified values to use:**
- Shenzhen: 50-driver sample; grid 48×90 @ 0.01°; T=24 hourly; demographics resolve to **10 district profiles**; three feature sets before-edit F_causal **0.799 / 0.807 / 0.725**; `{housing,comp,migrant}` is PRIMARY (higher baseline → not unfairness-maximizing). Cleanup: **10 calibrated stuck-GPS sink cells across 9 driver plates; 106,677 phantom pickups removed**; headline sink grid **(29,53)** (local F_spatial +0.089 / net global +0.021).
- SF Cabspotting: 536 taxis / ~11.2M pings (2008-05-17→06-10) → fleet-density regime discovery (0.56 vs Shenzhen 0.012 drivers/cell) → **sf12** density-matched (12 drivers, baseline F_causal 0.8752); grid 32×30; ACS 2006–2010 (housing = median home value, comp = per-capita income, migrant = foreign-born share).
- Compatibility: the dual claim needs dense per-driver traces + persistent driver IDs (F_fidelity can't score OD pairs) → OD-only US data (NYC TLC / Chicago / DC) is incompatible; SF is the only compatible US set (fallbacks Porto, Rome).

**Section outline:** (1) Shenzhen (primary) — sample, grid, time, demographics + the three feature sets + why hcm is PRIMARY, the stuck-GPS cleanup; (2) SF Cabspotting (external validity) — dataset, the fleet-density regime discovery, sf12 subsample, ACS demographics; (3) compatibility rationale (why dense-trace + driver-ID required); (4) a small Shenzhen-vs-SF comparison table. Sources/provenance footer.

**Figures to reference:** `shared_cleanup/figures/sink_spatial_attr_before_after.png`; `second-dataset/figures/sf_supply_demand.png`.

- [ ] **Step 1: Confirm cleanup counts + figure paths.** Run:
```bash
cd /home/robert/FAMAIL && python3 - <<'PY'
import json
m=json.load(open("famail_temporal/source_data/processing_metadata.json"))["stuck_gps_sinks"]
print("cells",len(m["flagged_cells"]),"plates",len({s["plate_id"] for s in m["sinks"]}),"removed",m["n_pickups_removed"])
PY
for f in shared_cleanup/figures/sink_spatial_attr_before_after.png second-dataset/figures/sf_supply_demand.png; do test -f "PAPER/$f" && echo "ok $f" || echo "MISSING $f"; done
```
Expected: `cells 10 plates 9 removed 106677`; both figures `ok`.

- [ ] **Step 2: Write `PAPER/argument/02_datasets.md`** to the outline with the verified values + comparison table + figure refs. Honor Global Constraints.

- [ ] **Step 3: Verify.** Run:
```bash
cd /home/robert/FAMAIL && grep -niE 'cowork|co-work' PAPER/argument/02_datasets.md || echo "no-product-names OK"; for n in "10 " "9 " 106,677 sf12 48×90 32×30; do grep -q -- "$n" PAPER/argument/02_datasets.md && echo "has $n" || echo "check $n"; done
```
Expected: "no-product-names OK"; the cleanup counts (10 / 9 / 106,677), sf12, and both grid sizes present.

- [ ] **Step 4: Commit.**
```bash
cd /home/robert/FAMAIL && git add PAPER/argument/02_datasets.md && git commit -m "docs(argument): datasets (02)"
```

---

### Task 6: `03_fairness_theory.md`

**Files:**
- Create: `PAPER/argument/03_fairness_theory.md`
- Read: `famail_temporal/docs/F_CAUSAL_METHODOLOGY_NOTES.md`, `famail_temporal/docs/FAIRNESS_DECOMPOSITION_FORMULATION.md`, `docs/mathematical_foundations.md`, `docs/site/methodology/{objective-function,discriminator,algorithm,soft-cell-assignment}.md`, `famail_temporal/docs/TRAJECTORY_EDITING_METHODOLOGY.md`, `famail_temporal/fairness/README.md`.

**Section outline (core formulas + intuition, NOT full derivations):**
- **F_causal** `= R'(I−H_demo)R / R'MR`, `R = Y − g₀(D)` (1 = fairest). Intuition: after removing demand `D` (power-basis `g₀`), how much of the leftover service variation is explained by demographics — 1 means none (fair). Caveats: **associational, not causal** (partial R² of a cross-sectional OLS, no identification); **10 district-level DOF** → ecological-fallacy exposure. One sentence on the per-cell attribution αᵢ (with a pointer to the decomposition doc).
- **F_spatial** (spatial-attribution / Gini on channel 0), 1 = fairest; secondary metric.
- **Fidelity-A** — frozen driver-conditioned 3-stream (seeking BiLSTM + driving LSTM + 11-dim profile) HuMID-style Siamese discriminator; same-driver probability. Note the profile-dominance property (identity-preservation, not shape realism).
- **Fidelity-B** — discriminator-free Jensen–Shannon divergence of trajectory-statistic distributions vs raw.
- **The editor** — per-(cell,time) attribution → ST-iFGSM signed-gradient edit of the pickup cell within an ε=2 L∞ ball; weighted objective (causal-emphasis α=0.2/0.7/0.1).
- **Resources** — internal: `F_CAUSAL_METHODOLOGY_NOTES.md`, `FAIRNESS_DECOMPOSITION_FORMULATION.md`, `docs/mathematical_foundations.md`, `docs/site/methodology/*`, `TRAJECTORY_EDITING_METHODOLOGY.md`. External lineage (as grounded in the repo's methodology docs; exact bibrefs to be finalized by the authors): cGAIL (imitation-learning base), HuMID/Ren (identity discriminator), FGSM/iFGSM (the editing step), Frisch–Waugh–Lovell (the residualization).
Sources/provenance footer.

- [ ] **Step 1: Confirm the F_causal formula against source.** Run:
```bash
cd /home/robert/FAMAIL && grep -niE "I-H|I_minus_H|R'MR|1 - r|partial R|residual" famail_temporal/docs/F_CAUSAL_METHODOLOGY_NOTES.md | head -8
```
Expected: lines confirming `F_causal = R'(I−H)R/R'MR = 1 − r²_demo` and the residual definition. Use the doc's exact formulation.

- [ ] **Step 2: Write `PAPER/argument/03_fairness_theory.md`** to the outline. Formulas in inline/blocks; intuition in prose; the associational + 10-DOF caveat prominent; Resources section with internal pointers + external lineage. Do NOT reproduce full derivations (link them). Honor Global Constraints (F_causal naming + F_demo-pending note).

- [ ] **Step 3: Verify.** Run:
```bash
cd /home/robert/FAMAIL && grep -niE 'cowork|co-work' PAPER/argument/03_fairness_theory.md || echo "no-product-names OK"; for t in F_causal F_spatial Fidelity-A Fidelity-B associational cGAIL Frisch; do grep -q -- "$t" PAPER/argument/03_fairness_theory.md && echo "has $t" || echo "MISSING $t"; done
```
Expected: "no-product-names OK"; all four metrics + "associational" + the external-lineage names present.

- [ ] **Step 4: Commit.**
```bash
cd /home/robert/FAMAIL && git add PAPER/argument/03_fairness_theory.md && git commit -m "docs(argument): fairness-metric theory + resources (03)"
```

---

### Task 7: `04_evaluation.md`

**Files:**
- Create: `PAPER/argument/04_evaluation.md`
- Read: `PAPER/by_feature_set/housing-comp-migrant/README.md`, `PAPER/second-dataset/FINDINGS.md` §6, `famail_temporal/baselines/LEVEL1_V2_METHODOLOGY.md`.

**Section outline:** (1) the two-pillar experimental design — **L1 data-quality** (four sources raw/edited/BC-gen/GAN-gen scored on F_causal, F_spatial, Fidelity-A, Fidelity-B); **L2 vanilla transfer** (driver-conditioned BC per source, paired edited−raw); **weighted-BC recovery** (upweight edited demos; dose-response w10/20/30; **random-placebo** and **most-fair-select** controls); **model-level variance** (b0 vs FAMAIL). (2) the **real-anchored Fidelity-A validation gate** (matched vs mismatched real-driver pairs — trusts Fidelity-A). (3) **Statistical conventions:** paired seeds; the n=6 Wilcoxon floor (0.03125 = sign-unanimity) → lead with mean Δ + t-CI + monotone dose-response + null controls; n=5 nulls reported vs the cross-seed noise band; the deterministic (std=0) L1 data-level gap has no sampling CI. Sources/provenance footer.

- [ ] **Step 1: Write `PAPER/argument/04_evaluation.md`** to the outline (procedures + conventions; no per-city result numbers — those live in 05/06; may reference the four-source/arm structure). Honor Global Constraints (the statistical-conventions block is load-bearing).

- [ ] **Step 2: Verify.** Run:
```bash
cd /home/robert/FAMAIL && grep -niE 'cowork|co-work' PAPER/argument/04_evaluation.md || echo "no-product-names OK"; for t in "L1" "L2" weighted 0.03125 placebo most-fair gate; do grep -q -- "$t" PAPER/argument/04_evaluation.md && echo "has $t" || echo "MISSING $t"; done
```
Expected: "no-product-names OK"; all listed tokens present (esp. the n=6 floor 0.03125 and the controls).

- [ ] **Step 3: Commit.**
```bash
cd /home/robert/FAMAIL && git add PAPER/argument/04_evaluation.md && git commit -m "docs(argument): evaluation procedures + statistical conventions (04)"
```

---

### Task 8: `07_limitations.md`

**Files:**
- Create: `PAPER/argument/07_limitations.md`
- Read: `PAPER/reviews/REVIEW_C_primary.md`, `PAPER/reviews/README.md`, `PAPER/second-dataset/FINDINGS.md` §9.

**Section outline (candid):** (1) F_causal is associational (partial R² on 10 district profiles) + ecological-fallacy exposure; (2) n=6/n=5 Wilcoxon floors + no multiple-comparison survival → evidence rests on CIs + dose-response + controls, not uncorrected p; (3) the deterministic L1 data-level gap (no sampling CI; also the editor's own objective); (4) profile-dominated fidelity — F_fidelity certifies identity preservation, not trajectory-shape realism (true on *both* cities); (5) GAN did not collapse on SF → the Shenzhen "GAN disqualified" sub-claim does not transfer; (6) small-n (SF 12 drivers, 5–6 seeds); (7) SF demographics are ACS proxies (migrant = foreign-born share, not hukou) → cross-city absolute F_causal not comparable; (8) the pending `F_causal → F_demo` rename; (9) **open questions** (what training procedure best realizes the data-level fairness; other model classes e.g. GAN/WGAN on edited data). Close with the **credibility note**: three adversarial-review rounds (REVIEW_A/B produced 29 findings, 0 refuted; REVIEW_C on the PRIMARY + branch code: 0 critical / 1 substantive-fixed / ~8 minor). Sources/provenance footer.

- [ ] **Step 1: Write `PAPER/argument/07_limitations.md`** to the outline. Candid, specific, each limitation one short paragraph. Honor Global Constraints.

- [ ] **Step 2: Verify.** Run:
```bash
cd /home/robert/FAMAIL && grep -niE 'cowork|co-work' PAPER/argument/07_limitations.md || echo "no-product-names OK"; for t in associational 0.03125 profile GAN F_demo review; do grep -qi -- "$t" PAPER/argument/07_limitations.md && echo "has $t" || echo "MISSING $t"; done
```
Expected: "no-product-names OK"; all tokens present.

- [ ] **Step 3: Commit.**
```bash
cd /home/robert/FAMAIL && git add PAPER/argument/07_limitations.md && git commit -m "docs(argument): limitations + open questions (07)"
```

---

### Task 9: `README.md` (index + generic-agent entry point) + stale-doc pointers + whole-set consistency

**Files:**
- Create: `PAPER/argument/README.md`
- Modify: `famail_temporal/baselines/PAPER_ARGUMENT_PLAN.md` (add pointer at top), `docs/two_level_argument.md` (add pointer at top)
- Read: all of `PAPER/argument/*.md` (to index them).

**Section outline (README):** (1) one-paragraph thesis; (2) reading order + one-line description of each of the 8 docs; (3) **"Suggested slide outline"** mapping docs → deck sections (Title/Thesis ← 00; Motivation ← 01; Data ← 02; Metrics ← 03; How we evaluated ← 04; Shenzhen results ← 05; A second city ← 06; Limitations & next ← 07); (4) a **"For a presentation agent"** note — GENERIC directives only (lead with 00's thesis; use the headline-numbers table; place each doc's referenced figures on the corresponding slides; keep the associational-`F_causal` caveat on any fairness-metric slide; the numbers in 05/06 are authoritative); (5) pointers to the deeper `PAPER/` artifacts (`by_feature_set/`, `feature_selection/`, `second-dataset/`, `reviews/`, `shared_cleanup/`).

**Stale-doc pointer (prepend to both files, adjusted per file):**
```
> **Superseded (2026-07-01):** the current, results-backed paper argument lives in `PAPER/argument/`. This document is retained for historical context; its numbers predate the PRIMARY re-run + the SF second dataset.
```

- [ ] **Step 1: Write `PAPER/argument/README.md`** to the outline (product-agnostic; the "presentation agent" note is generic).

- [ ] **Step 2: Prepend the superseded pointer** to `famail_temporal/baselines/PAPER_ARGUMENT_PLAN.md` and `docs/two_level_argument.md` (use each file's relative path context; keep the existing content below the pointer).

- [ ] **Step 3: Whole-set consistency + no-product-name sweep.** Run:
```bash
cd /home/robert/FAMAIL && echo "=== files ==="; ls PAPER/argument/*.md; echo "=== product-name sweep (expect nothing) ==="; grep -rniE 'cowork|co-work' PAPER/argument/ || echo "CLEAN"; echo "=== figure refs resolve ==="; grep -rhoE '(by_feature_set|feature_selection|shared_cleanup|second-dataset)/[A-Za-z0-9_/-]+\.png' PAPER/argument/ | sort -u | while read p; do test -f "PAPER/$p" && echo "ok $p" || echo "MISSING $p"; done; echo "=== headline numbers agree (00 vs 05/06) ==="; for n in 0.8132 0.8891 +0.0311 +0.0387; do a=$(grep -lc -- "$n" PAPER/argument/00_overview.md); echo "$n in 00: $a"; done
```
Expected: 9 files listed; "CLEAN"; every figure ref `ok`; the headline numbers present in `00_overview`.

- [ ] **Step 4: Commit.**
```bash
cd /home/robert/FAMAIL && git add PAPER/argument/README.md famail_temporal/baselines/PAPER_ARGUMENT_PLAN.md docs/two_level_argument.md && git commit -m "docs(argument): README/index + slide outline + supersede stale argument docs (09)"
```

---

## Self-Review

**Spec coverage:** motivation/goals → Task 4; datasets → Task 5; fairness theory + resources → Task 6; evaluation → Task 7; results (Shenzhen + SF split) → Tasks 1 & 2; limitations → Task 8; index + slide outline + supersede stale docs → Task 9; self-containment (numbers/tables inline + figure refs + provenance) → in every task; product-agnostic + numbers-discipline + F_causal-naming → Global Constraints + per-task verify steps. All spec sections covered.

**Placeholder scan:** no "TBD/TODO"; every task gives exact files, verified values, section outlines, real verification commands with expected output, and a commit. The external-citation "authors finalize exact bibrefs" is an intentional deferral (avoid fabricating references), not a placeholder in the deliverable.

**Type/name consistency:** figure paths are consistent across tasks (e.g. `pareto_causal_hcm.png` used identically in Tasks 1 and 9); the headline numbers used in Task 3 (00) are exactly those anchored in Tasks 1/2 and re-verified in Task 9; doc filenames match the File Structure and the spec throughout.

**Build-order dependency:** Tasks 1→2 (anchor numbers) precede Task 3 (overview must match) and Task 9 (indexes all + final consistency), per the spec's implementation notes. Tasks 4–8 are independent content docs and may run in any order after the constraints are known.
