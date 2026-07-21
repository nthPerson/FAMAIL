# Re-costed cut inventory vs current text (2026-07-20, READ-ONLY)

Basis: current section files as of this build (§1 =164 ln, §2 =91, §3 =486, §4 =654,
§5 =47). Line yields are **rendered single-column lines**, counted from source prose at
≈68 chars/rendered line and **rounded down / deliberately conservative** (better to
under-promise). Deficit to hit: ≈290 rendered lines (2.7pp), **+2 lines slack reserved**
for a possible Figure-1 (b)+(c) swap → effective target ≈292.

Naming note (was ambiguous in the 07-18 docs): **fig:teaser** (§1, `\begin{figure}[t]`
l.32, inputs `figures/figure-1/figure-1`) is the WHY figure — this is the "Figure 1" the
2026-07-20 addendum reduced to the (c)-only single panel. **fig:overview** (§3,
`\begin{figure*}[t]` l.317, inputs `figures/figure-2/figure-2`) is the full-width HOW
explainer — this is what the 07-18 candidate `meth-figure1-resize` targets. They are
different floats.

---

## (A) Summary line-yield accounting

**Grand conservative total of all LIVE candidates ≈ 248 rendered lines** (old survivors +
new + compression), split by mechanism:

| Mechanism | Lines | Share |
|---|---:|---:|
| **Appendix-relocation** (body keeps stub + headline) | **≈169** | 68% |
| **Deletion / table-merge / figure-resize / body-cut** | **≈60** | 24% |
| **Compression (reword, no content loss)** | **≈19** | 8% |
| **TOTAL (live, conservative)** | **≈248** | |

Sub-totals feeding the grand total:
- Surviving old candidates (LIVE, primary variants): **≈220**
- New post-07-18 candidates: **≈9**
- Verbose-prose compression (Section D, distinct from above): **≈10** (+ the ≈9 of new
  candidates that are compression-type). Compression mechanism row above = 19 total.

**Headline finding — the target has almost no margin.** Taking essentially *every* live
candidate at conservative yields sums to ≈248, which is **≈44 lines short** of the 290-line
deficit (≈292 with slack). Closing that gap depends on (i) the two big relocations landing
at full value (`exp-figure2-relocate` ~42 + `exp-table6-relocate` ~32 = ~74 lines, **both
appendix-only**), and (ii) rewrite-compaction reclaiming widow/orphan lines beyond these
estimates — the ranked-list doctrine ("rewrite-compaction + float re-packing usually beat
estimates"). On strict conservative counting the candidate set alone does **not**
guarantee 8.0pp; the appendix is not optional, it is load-bearing.

MOOT (already banked, do not re-count): `teaser-resize` / all figure-1(teaser) items —
the (c)-only single-panel reduction is already in the current 10.6pp geometry (~11 lines
already realized; this is why the deficit fell 330→290).
STRUCK (Zhang reversal, off-plan): `rw-contrast-tighten`, `rw-leveling-compress`.
OUT OF SCOPE: `abstract-tighten` (abstract 00 excluded by the brief); `teaser-remove`
(HIGH, Robert/Zhang decision only).

---

## (B) Re-costed table — surviving 07-18 candidates

Yield = conservative rendered lines. Tag = APPENDIX-SAFE / BODY-CUT-ONLY / NOT-SAFE / MOOT
/ STRUCK. "Δ vs 07-18" flags where the number moved.

### §1 Introduction (`sections/01_introduction.tex`)

**`intro-tier-breakdown`** — STILL EXISTS, moved to **l.100–103**.
Quote: *"and add statistically robust taxi presence to the…"* (l.100).
Drop the tier clause "raising its mean service ratio by +0.0176 … (tier-1) and by +0.0411
… (tier-2)" (both restated at §4.2 l.138–139 / tab:channels). **Yield ≈3** (was 6;
optimistic). Risk LOW. **BODY-CUT-ONLY** (pure duplication; +0.0226 headline stays).

**`intro-contributions-compress`** — STILL EXISTS, **l.145–164** (four-item itemize).
Quote: *"\textbf{A fairness-aware trajectory editor.} Two bounded…"* (l.146).
Tighten each bullet's 2nd clause ~1 line. **Yield ≈4** (was 6). Risk **MEDIUM** (itemize is
the reviewer's claim map). **BODY-CUT-ONLY**. Reserve.

**`teaser-resize` / `teaser-remove`** — **l.32–67** (fig:teaser).
`teaser-resize` = **MOOT** (single-panel already banked). `teaser-remove` = OUT OF SCOPE
(HIGH, Robert/Zhang). Note the ≥2-line slack requirement is tied to this float (b)+(c)
swap = `figures/figure-1/figure-1-bc`, commented at l.38.

### §2 Related Work (`sections/02_related_work.tex`)

**`rw-recourse-compress`** — STILL EXISTS, **l.58–74**.
Quote: *"\textbf{Adversarial perturbation and recourse.} The editing…"* (l.58).
The FGSM→ST-iFGSM→Gumbel/STE chain (l.58–64) is re-explained in §3.5 "Shared machinery"
(l.385–407). Compress to a compact cite-carrying clause, **keep every `\cite`**. **Yield
≈3** (was 4). Risk LOW. **BODY-CUT-ONLY** (mechanism survives in §3.5). *Caution:* Zhang
wants §2 more explicit — this theme is not the per-group-limitation theme she flagged, so
it is admissible, but treat as the single §2 cut.

**`rw-contrast-tighten`** — **STRUCK** (Zhang reversal; §2 to grow, not shrink).
**`rw-leveling-compress`** — **STRUCK**. Text still present at **l.86–91** ("Both critiques
are load-bearing…") but off-plan.

### §3 Methodology (`sections/03_methodology.tex`) — the appendix vein

**`meth-fcausal-derivation-relocate`** — STILL EXISTS, **l.69–95** (F_causal→**F_demo**
renamed; content unchanged).
Quote: *"Despite its appearance, Eq.~\eqref{eq:fdemo} is nothing…"* (l.69).
Relocate the constant-H/implicit-refit elaboration, the N×N normal-equations identity
(l.81–84), and the FWL paragraph (l.87–95). **Keep in body:** Eq.(1); a 2-sentence gloss;
the idempotence half-sentence (l.79–80, dependency for §3.3); the interpretation clause
("r²_demo = share of demand-adjusted inequity attributable to demographics"); the boundary
sentence (l.96–98). **Yield ≈18** (was 24; lowered because the gloss *adds* ~3 lines and
the interpretation clause stays). Risk LOW. **APPENDIX-SAFE.** Stub: *"Eq.(1) is the
stage-two regression in closed form (RSS/TSS); H is constant so it re-fits exactly each
step; FWL exactness and the O(N) evaluation identity are in App. X."*
Dependency: §3.3 l.199–200 ("Because M and I−H are idempotent…") needs the idempotence
half-sentence to remain.

**`meth-fspatial-gini-relocate`** — STILL EXISTS, Eq.(2) **l.125–132**, remark **l.133–136**.
Quote: *"As with $F_{\mathrm{demo}}$, the quadratic pairwise sum…"* (l.133).
(a) SHORTEN: drop only the "never materialized / O(N log N) prefix-sum" aside (l.133–136).
**Yield ≈2** (was 3). Risk LOW. **BODY-CUT-ONLY**.
(b) RELOCATE all of Eq.(2), keep one-line def. **Yield ≈7** (was 8). Risk MEDIUM.
**APPENDIX-SAFE** (stub: "F_spatial = 1 − mean Gini of DSR/ASR, App. X"). Variants are
mutually exclusive; (a) is the safe default.

**`meth-attribution-eq-relocate`** — STILL EXISTS. Eq.(3) unit-attr **l.202–208**; Eq.(4)
supply-grad closed form **l.232–236** + verification l.237.
Quote (Eq.4): *"The map's $F_{\mathrm{demo}}$ component has the closed form"* (l.230–231).
Eq.(4)+verification: **Yield ≈7** (was 8). Risk LOW. **APPENDIX-SAFE**.
Eq.(3): additional **≈6** (was 7). Risk MEDIUM (backs the "two exact attribution
mechanisms" contribution). **APPENDIX-SAFE**, stub "an exact per-unit partition (App. X)".
Combined ≈13. Dependency: §3.4 **l.285** cross-refs `\eqref{eq:unit-attr}` — retarget to
appendix.

**`meth-screen-detail-shorten`** — STILL EXISTS, **l.240–260**.
Quote: *"To turn the map into candidates, the screen asks…"* (l.240).
Compress the per-offset translation + linearized-gain scoring (l.242–253) to ~2 sentences;
**keep** 80k/95k eligibility and "screen nominates, editor derives the move" (l.255–259).
**Yield ≈6** (unchanged). Risk MEDIUM. **APPENDIX-SAFE** (scoring recipe → App.; stub keeps
"a linearized-offset screen ranks trajectories whose tails could bend into high-value cells
(~80k of ~95k); it only nominates").

**`meth-editor-impl-relocate`** — STILL EXISTS; content shifted down by the commented-out
figure block (l.348–378). Shared machinery **l.385–407**; taper constants
w_j=0.25/0.5/0.75/1.0 & 1/12 presence mass & s0=0.1 **l.416–426**; king-move repair
mechanics **l.434–445**.
Quote: *"\textbf{Shared machinery.} Both editing phases repurpose…"* (l.385).
Relocate Gumbel/STE bridging detail + editor constants + repair procedure. **Keep in
body:** ε=2 identity-budget reinterpretation (l.401–403), "supply is thus endogenous"
(l.426–427), and the **~5% (118/2,455) revert disclosure** (l.443–444). **Yield ≈13** (was
17; conservative). Risk LOW. **APPENDIX-SAFE.** **Do NOT** touch the two-phase-as-control
paragraph (l.456–463, DO-NOT-CUT). Dependency: ~5% number echoed in §4.2 provenance (l.157)
— keep consistent.

**`meth-weight-dup-compress`** — STILL EXISTS, **l.167–186** (vs §4.6 l.462–471).
Quote: *"The weights are selected \emph{empirically}: we sweep…"* (l.167).
Shorten §3.2 to a forward-pointer; §4.6 carries the sweep detail. **Yield ≈8** (unchanged).
Risk LOW. **BODY-CUT-ONLY** (dup with §4.6). **Couples with `exp-figure2-relocate`**: if
Fig.2 relocates, the pointer must aim at the appendix, and the three-class criterion must
still be stated in the body.

**`meth-figure1-resize`** *(= the §3 fig:overview HOW explainer, not the teaser)* — STILL
EXISTS, **l.317–347**, still `figure*` full-width, inputs `figures/figure-2/figure-2`.
FIGURE-OP: `figure*`→single-column or width 0.85. **Yield ≈12** (was 15; TikZ height may
not shrink proportionally). Risk MEDIUM (best method-teacher). **BODY-CUT-ONLY** (in-place
resize). Reserve. *Flag:* single-column reflow of a 3-panel TikZ risks legibility.

### §4 Experiments (`sections/04_experiments.tex`)

**`exp-setup-stats-shorten`** — STILL EXISTS, **l.36–49**.
Quote: *"Every model-based comparison uses paired seeds…"* (l.36).
Relocate floor arithmetic (0.03125 / 0.0625 / .00049 derivation) to footnote/appendix;
keep sign-unanimity + n=12 flagship + bootstrap-first-order. **Yield ≈3** (was 4). Risk LOW.
**APPENDIX-SAFE**. Keep ".00049 survives correction" (echoed in §5).

**`exp-setup-instruments-shorten`** — STILL EXISTS, **l.63–72**.
Quote: *"\textbf{External fairness instruments.} On the service ratio…"* (l.63).
Compress DP/DI/Theil/levels definitions; **keep DP≡gap disclosure** (l.69–70). **Yield ≈3**
(was 4). Risk LOW. **APPENDIX-SAFE**. Stub keeps DP≡gap + the strict-count sentence.

**`exp-tables-merge-A`** — STILL EXISTS: tab:ablation **l.178–195** + tab:baselines
**l.425–443**. Fold FATE trim-only into the cross-arm table; SF ablation rows as a
sub-panel. **Yield ≈12** (was 18; lowered — the two tables now have divergent column sets:
ablation is a 2-col trim-only/trim+lift comparison with SF DP-gap rows, baselines is an
arm-list with an `inflation` column, so the merge is a genuine restructure, not a clean
fold). Risk LOW-MEDIUM. **BODY-CUT-ONLY** (layout op; one caption saved).

**`exp-tables-merge-B`** — STILL EXISTS: tab:external-sz **l.91–108** + tab:channels
**l.128–144**. Shared `mean(Y|disadv) +0.0529` row; stack under one caption. **Yield ≈10**
(was 13). Risk LOW. **BODY-CUT-ONLY**.

**`exp-table6-relocate`** — STILL EXISTS: prose **l.494–510** + tab:featsets **l.512–536**
(largest table, footnotesize).
Quote: *"\textbf{Feature-set robustness.} $F_{\mathrm{demo}}$ is specific…"* (l.494).
Relocate grid + prose; **keep in body 2–3 sentences**: directional reproduction; supply
channel tier-2-sig on all three (+0.0411/+0.0211/+0.0771); most-fair leak +0.0054/+0.0072
(already body-resident at l.452–453); edited arm ≥3×. **Yield ≈32** (was 35). Risk MEDIUM.
**APPENDIX-SAFE.** Dependency: table cells cross-ref `Tab.~\ref{tab:featsets}` / §4.x
(l.528–533) — retarget to appendix; keep the leak disclosure inline.

**`exp-figure2-relocate`** — STILL EXISTS: prose **l.462–471** + fig:alpha-pareto
**l.474–492** (single-column `figure`, `includegraphics`, ~0.40pg). **Largest single
candidate.**
Quote: *"\textbf{Weight sensitivity.} Fig.~\ref{fig:alpha-pareto} reports…"* (l.462).
Relocate figure; keep 2 sentences (flatness within 0.001; monotone lift-up decline,
tier-1-sig only α_sp≤0.2; adopted (0.1,0.8,0.1) = criterion's best point). **Yield ≈42**
(was 50; conservative on a single-column float). Risk MEDIUM. **APPENDIX-SAFE.** Couples
with `meth-weight-dup-compress`.

**`exp-fourseource-gan-shorten`** — STILL EXISTS, **l.246–252**.
Quote: *"One instability we surface rather than smooth: the Shenzhen GAN's…"* (l.246).
Relocate per-seed spread (±0.129; 0.197–0.295 / 0.03–0.04; length-comp 0.247); keep
1-sentence honesty beat. **Yield ≈4** (was 5). Risk LOW–MED. **APPENDIX-SAFE**.

**`exp-dose-saturation-shorten`** — STILL EXISTS, in the upweighting paragraph **l.271–274**.
Quote: *"Extending the dose shows saturation, not unbounded growth…"* (l.271).
Relocate w40/w50 values + increment list (+0.0050/+0.0035/+0.0021/+0.0016); keep
"saturation, w30 at the knee". **Yield ≈3** (was 5; the removable span is small). Risk
LOW–MED. **APPENDIX-SAFE**. Twin = `exp-sf-downstream-shorten`.

**`exp-variance-shorten`** — STILL EXISTS, **l.293–301**.
Quote: *"\textbf{Model-level variance.} A paired baseline-vs-FATE…"* (l.293).
Keep +0.0030±0.0022, n=10, p=.0039, magnitude comparison; relocate the n=5-vs-n=10 aside.
**Yield ≈3** (was 4). Risk LOW–MED. **APPENDIX-SAFE**. Twin = SF variance (l.624–629).

**`exp-provenance-shorten`** — STILL EXISTS, **l.157–165**.
Quote: *"\textbf{Provenance disclosures.} About 5\% of trim edits…"* (l.157).
**Keep ~5% disclosure**; relocate oracle-ceiling arithmetic (l.161–164: +0.786, 2.6×,
+0.882, realized +0.053). **Yield ≈4** (unchanged). Risk LOW–MED. **APPENDIX-SAFE**, stub
"realized lift far below the realism-free oracle ceiling (App. X)".

**`exp-baselines-perturbation-note-shorten`** — STILL EXISTS, **l.333–343**.
Quote: *"A naming note: the gradient arms are \emph{iFGSM/FGSM…"* (l.333).
Compress the δ=0 / concatenation-head stationarity beat to 2 sentences. **Yield ≈7** (was
8). Risk MEDIUM. **APPENDIX-SAFE** (stationarity detail → App.; keep "gradient arms are
iFGSM/FGSM with random restart; the δ=0 no-op did not stall because the deployed head
compares by concatenation").

**`exp-fairness-penalty-shorten`** — STILL EXISTS, **l.388–419**; λ-grid **l.401–412**.
Quote: *"\textbf{Fairness-method baselines (training-side).} The remaining…"* (l.388).
**Keep** Kamiran–Calders −0.0227 (6/6, wrong way); "inert at every trainable dose,
destructive only where it dominates"; "neither reproduces the recovery". Relocate the full
λ-grid (λ∈{1,3.16,10,100,1000}, signed vs absolute, −0.2053/−0.1293 collapse, 10⁻⁵
derivation). **Yield ≈8** (was 9). Risk MEDIUM. **APPENDIX-SAFE.** *Note:* this now
absorbs the new penalty-formulation-independence passage (see C).

**`exp-filtering-shorten`** — STILL EXISTS, **l.539–546**.
Quote: *"\textbf{Filtering is not a substitute.} Removing the $K$…"* (l.539).
Compress to 1 sentence (0.7935 at K=2,455 vs 0.8214 edited). **Yield ≈4** (was 5). Risk
MEDIUM. **APPENDIX-SAFE / BODY-CUT-ONLY** (1-sentence rebuttal stays in body).

**`exp-sf-downstream-shorten`** — STILL EXISTS, **l.599–642**.
Quote: *"\textbf{Downstream.} The recovery reproduces on San Francisco."* (l.599).
Keep n=12 flagship (+0.0333, 12/12, p=.00049) and "F_spatial does not propagate on SF"
city-difference; compress SF saturation (l.612–618), variance (l.624–629), four-source
(l.636–642) to one pointer sentence each. **Yield ≈6** (was 8). Risk LOW. **BODY-CUT-ONLY /
APPENDIX-SAFE**. **Must not** touch the tier-2/Reading-B block (l.567–597, DO-NOT-CUT). The
n=6-per-dose parenthetical (l.609–610) is a further ~1-line relocate-safe extra.

### §5 Conclusion (`sections/05_conclusion.tex`)

**`concl-restatement-shorten`** — STILL EXISTS, **l.5–18**.
Quote: *"FATE makes fairness a property of the demonstrations."* (l.5).
Trim restatement to ~4 sentences; **keep bounds paragraph l.20–37 intact** (DO-NOT-CUT).
**Yield ≈4** (was 6). Risk LOW. **BODY-CUT-ONLY**.

**`concl-future-shorten`** — STILL EXISTS, **l.39–47**.
Quote: *"Two directions follow naturally. The editor's two-phase…"* (l.39).
Trim to ~3 sentences. **Yield ≈2** (was 3). Risk LOW. **BODY-CUT-ONLY**.

---

## (C) New candidates from post-07-18 content

**`new-intro-extmetric-numbers`** — §1 **l.121–133** (the un-commented external-metrics
paragraph; a shorter commented variant sits at l.110–120).
Quote: *"Because a fairness method scored on its own objective proves little…"* (l.127),
then l.129–133: "…on the migrant axis, disparate impact +0.0162 and the demographic-parity
gap −0.890 (14.199 → 13.309); between regions, the Theil index −0.0087". These three
numbers are **verbatim duplicates of tab:external-sz (l.101–103)** — fresh duplication that
did not exist at 07-18. Drop the specific figures, keep "all three improve on the primary
city, every interval excluding zero (§4.x)". **Yield ≈3.** Risk LOW. **BODY-CUT-ONLY**.

**`new-intro-categories-compress`** — §1 **l.19–30** (the intervention-categories paragraph,
added post-07-18).
Quote: *"Interventions typically target one end of the pipeline or the other."* (l.19).
Overlaps §2's pre/in/post-processing framing (l.6–14). Compress the in-processing and
data-generation sub-clauses (esp. "replacing or extensively altering the original data can
reduce realism, introduce distribution shift, and make it difficult to identify the
source…", l.26–28). **Yield ≈3.** Risk MEDIUM (motivation; sets up FATE's "third
position"). **BODY-CUT-ONLY** (compression).

**`new-baselines-sf-perturbation-compress`** — §4.5 **l.359–365**.
Quote: *"The three perturbation arms replicate on San Francisco: every arm lands well…"*
(l.359). SF mirror of the Shenzhen perturbation-arm result; compress the per-arm
percentages (72.0/87.0/97.5%; 0.198–0.418 vs 0.098) to one sentence. **Yield ≈3.** Risk
MEDIUM (robustness beat). **APPENDIX-SAFE** (per-arm numbers → App.; keep "the arms
replicate on SF — all well below the editor, all violating adjacency").

**`new-penalty-formulation-independence`** — §4.5 **l.401–412** + the RESOLVED comment
l.413–417. The signed-vs-absolute "both formulations" testing is **already inside the span
of `exp-fairness-penalty-shorten`**; no separate yield claimed. Incremental relocate-safe
text ≈2 lines (the "shallower absolute collapse is the gap pushing back… neither
formulation offers a constructive operating range" sentence) is counted within that
candidate. Flagged here only so it is not double-counted.

New-candidate net yield ≈ **9 lines** (3+3+3).

---

## (D) Verbose-prose compression candidates (reword only, no content loss)

Author preference: explicit and to the point, no clever language. Each below yields ≥3
rendered lines by rewording alone. None touches protected content.

**`comp-intro-hook`** — §1 **l.7–17** (opening). Current ≈11 rendered lines → target ≈8
(**≈3**). The hook carries rhetorical flourish the author wants plainer.
Example: *"Models that imitate such data faithfully learn the inequity as competence, and
are doomed to re-enact it when deployed, sending supply where supply already went ---
exhibiting the feedback loop documented in other data-driven allocation systems."*
→ *"Models that imitate such data reproduce the inequity and re-enact it when deployed --- a
feedback loop documented in other data-driven allocation systems \cite{ensign2018,
lumisaac2016}."* (drops "as competence / doomed / sending supply where supply already went";
same claim + same cites.)

**`comp-fidelity-guardrail`** — §3.2 **l.141–156** (Realism guardrail paragraph). Current
≈16 rendered lines → target ≈13 (**≈3**).
Example: *"We state its role honestly: under the small, bounded edits FATE makes, its
gradient with respect to the edit is near zero, so it serves as a realism \emph{guardrail}
rather than a driver of edits --- verified with respect to the edited pickup cell, with
identity-axis stability under both editing phases confirmed empirically
(\S\ref{sec:experiments})…"*
→ *"Under FATE's small bounded edits the discriminator's gradient is near zero, so it acts
as a realism \emph{guardrail}, not a driver of edits; identity-axis stability under both
phases is confirmed empirically (\S\ref{sec:experiments})."* (removes "We state its role
honestly" and the "verified with respect to the edited pickup cell" aside; the guardrail
claim and the empirical pointer survive.)

**`comp-supply-grad-exposition`** — §3.3 **l.211–230** (supply-gradient attribution
exposition). Current ≈20 rendered lines → target ≈16 (**≈4**). The ΔS auxiliary-input
mechanism is explained twice over (setup + "the input exists so that…"); collapse to one
statement. Value-of-presence-map concept and Eq.(4) stub survive.
Example: *"To obtain it, we give the objective an auxiliary \emph{supply-perturbation}
input $\Delta S \in \mathbb{R}^{N}$ --- one entry of hypothetical added taxi presence per
active unit --- and have it evaluate supply as $S + \Delta S$ in every term of the
objective where supply appears… At $\Delta S = 0$ the objective's value is untouched; the
input exists so that automatic differentiation with respect to it reads out $\partial
\mathcal{L}/\partial S_i$ at the city's current supply…"*
→ *"We add an auxiliary input $\Delta S \in \mathbb{R}^{N}$ of hypothetical added presence
per unit, evaluate every supply-dependent term at $S + \Delta S$, and read $\partial
\mathcal{L}/\partial S_i$ by automatic differentiation at $\Delta S = 0$ --- the marginal
fairness effect of added presence at each unit, in one backward pass."*

Compression subtotal ≈ **10 lines** (3+3+4).

---

## (E) Appendix-safe subset — stubs + appendix size

Items whose evidence can move to a KDD-appropriate appendix while the 8-page body stays
self-contained (keeps claim + headline number, appendix keeps derivation/detail).

| id | ≈lines | body stub that MUST remain |
|---|---:|---|
| `meth-fcausal-derivation-relocate` | 18 | Eq.(1); RSS/TSS gloss; idempotence half-sentence; r²_demo interpretation; boundary sentence; "FWL exactness + O(N) identity, App. X" |
| `meth-attribution-eq` (Eq.4) | 7 | "closed form and autograd check in App. X" |
| `meth-attribution-eq` (Eq.3) | 6 | "an exact per-unit partition (App. X)" |
| `meth-screen-detail-shorten` | 6 | 80k/95k eligibility + "screen nominates, editor derives the move" |
| `meth-editor-impl-relocate` | 13 | ε=2 identity budget; "supply is endogenous"; ~5% (118/2,455) revert disclosure |
| `meth-fspatial-gini` (b, alt) | 7 | "F_spatial = 1 − mean Gini of DSR/ASR (App. X)" |
| `exp-setup-stats-shorten` | 3 | sign-unanimity certificate; n=12 flagship survives correction; bootstrap first-order |
| `exp-setup-instruments-shorten` | 3 | DP≡gap disclosure + strict-count sentence |
| `exp-table6-relocate` | 32 | tier-2-sig-all-three (+0.0411/+0.0211/+0.0771); most-fair leak +0.0054/+0.0072; edited ≥3× |
| `exp-figure2-relocate` | 42 | flatness within 0.001; monotone lift-up decline (tier-1-sig α_sp≤0.2); adopted (0.1,0.8,0.1) = criterion best point |
| `exp-fourseource-gan-shorten` | 4 | 1-sentence GAN-collapse honesty beat (SF stays healthy) |
| `exp-dose-saturation-shorten` | 3 | "saturation, not unbounded growth; w30 at the knee" |
| `exp-variance-shorten` | 3 | +0.0030±0.0022, n=10, p=.0039; order-of-magnitude-below-upweighting |
| `exp-provenance-shorten` | 4 | ~5% revert disclosure; "far below the realism-free oracle ceiling (App. X)" |
| `exp-baselines-perturbation-note-shorten` | 7 | "iFGSM/FGSM w/ random restart; δ=0 no-op did not stall (concatenation head)" |
| `exp-fairness-penalty-shorten` | 8 | reweigh −0.0227 wrong-way; inert/destructive verdict; "neither reproduces the recovery" |
| `exp-filtering-shorten` | 4 | 1-sentence: 0.7935 at K=2,455 vs 0.8214 edited |
| `exp-sf-downstream-shorten` | 6 | n=12 flagship (+0.0333, p=.00049) + "F_spatial does not propagate on SF" |
| `new-baselines-sf-perturbation-compress` | 3 | "arms replicate on SF, all below the editor, all violating adjacency" |
| **Appendix-safe subtotal** | **≈179** | (fspatial(b) and attribution-Eq3 are the MEDIUM ones) |

**Appendix size estimate.** If the whole appendix-safe subset relocates:
- Relocated **prose** ≈179 − fig2(42) − table6(32) = ~105 single-col lines ≈ **~1.0 pg**
  (≈2 columns).
- Relocated **floats**: fig:alpha-pareto ~0.40pg + tab:featsets ~0.28pg ≈ **~0.68pg**.
- `\appendix` headers / stubs / equation displays overhead ≈ **~0.15pg**.
- **Total appendix ≈ 1.7–1.9 pages** — comfortably under the ≈2.5pg camera-ready ceiling.

In practice only enough of this subset fires to reach 8.0pp body (wave-based, re-measure
after each). Because the conservative full total is only ≈248 and the deficit is ≈290, the
realistic plan is to fire **most** of the appendix-safe subset plus the two body merges,
and rely on rewrite-compaction for the last ~40 lines — leaving the ≥2-line Figure-1 swap
slack intact.

---

## DO-NOT-CUT register (flagged, never proposed for cutting) — confirmed present

- Both pillars w/ numbers in body: +0.0226 F_demo & +0.0529 mean(Y|disadv) & DP/DI/Theil
  (tab:external-sz l.91–108); n=12 flagship +0.0297/+0.0333, 12/12, p=.00049 (l.262–270,
  l.599–611); vanilla-BC null; random + most-fair controls.
- Disclosures (main-text, shorten-only where noted): allocation drain both cities
  (l.303–319, l.630–635); most-fair leak +0.0054/+0.0072 (l.449–460); leveling-down scoping
  (§1 l.81–89, §3.4 l.262–299); SF caveats (l.644–654); §5 bounds (l.20–37); DP≡gap
  (l.69–70); tier-2/Reading-B (l.567–597); ~5% revert (l.157, l.443–444); oversampling
  fabrication + placebo-degrades (l.368–386).
- §3.4 structural-diagnosis numbers (2,455/2,455; 32×; 93% at floor; 1.8 vs 17.6; oracle
  perversity) l.264–299; two-phase-as-control paragraph l.456–463; associational caveat
  l.101–111. **≥2 lines slack reserved** for the Figure-1 (b)+(c) swap.
