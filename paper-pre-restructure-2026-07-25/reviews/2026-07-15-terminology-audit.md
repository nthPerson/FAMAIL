# Terminology self-documentation audit (2026-07-15)

**Auditor:** parallel audit session. **Scope:** all of `paper/sections/01–05` +
`main.tex` (abstract). Report-only; recommendations, not decisions — Robert picks.
**House style applied as the yardstick:** explicit over clever; a name should carry its
meaning to a fairness/ML reader who knows nothing about this project; defined-at-first-use
is the acceptable fallback for genuinely load-bearing terms of art.
Counts are of *rendered* text (comments stripped), by grep with word boundaries;
`file:line` references are to the sources at HEAD (`8367bf8`).

---

## 1. Primary target: "leveling down"

**Usage map (12 rendered occurrences).** Intro first use `01:40` (italicized, then the
objection at `01:42`); contribution bullet `01:92`; §2 run-in head "Leveling down and
feedback loops" + genealogy (`02:70,80`); §3.4 title "Why Demand-Only Editing Levels
Down" + body (`03:261`) + "level *up*" contrast (`03:269`); §4.2 (`04:165`); §4.3 "pure
leveling-down" (`04:197`); conclusion "pure leveling-down" (`05:14`); related-work
"level up" (`02:81`).

**Defined at first use? Yes, three times over, and well:**
- `01:40-44`: "*it improves measured fairness — by leveling down.* Every one of the
  2,455 pickups it edits originates in an advantaged cell, and the under-served group's
  service never moves; the classic leveling-down objection to equalization
  [parfit1997, mittelstadt2024] applies literally." The mechanism sentence *is* the
  definition, in place.
- `02:70-74`: the formal version — "equality achieved by making the better-off worse
  rather than the worse-off better — originates in the ethics of equality [26]".
- `04:195-197` re-derives it numerically ("service level unchanged to four decimals …
  while the advantaged level falls; the improvement is pure leveling-down").

**Citation alignment (checked in the rendered bibliography):** this is not our coinage —
it is the literature's own term, and *two of our cited titles contain it*:
- [26] Parfit 1997, *Equality and Priority* — the origin of the "leveling-down
  objection" (the paper's phrase "originates in the ethics of equality" is anchored here).
- [24] Mittelstadt, Wachter & Russell 2024 — title: "The Unfairness of Fair Machine
  Learning: **Levelling Down** and Strict Egalitarianism by Default" (UK spelling).
- [36] Zietlow et al., CVPR 2022 — title: "**Leveling Down** in Computer Vision: Pareto
  Inefficiencies in Fair Deep Classifiers" (US spelling — ours matches the ML venue).

Replacing the term would orphan those two titles and blunt §2's genealogy sentence,
which currently does real work connecting our diagnosis to the fairness literature.

**Options, ranked:**
1. **Keep, as defined (recommended).** The term passes the house test *because* every
   first-contact site pairs it with its mechanism, and the citation alignment is worth
   more than any paraphrase. Two hygiene items if kept:
   (a) **noun-form consistency** — noun uses are open in five places ("by leveling
   down", "as leveling down") but hyphenated in two ("pure leveling-down", `04:197`,
   `05:14`). Pick one; house convention would be open noun / hyphenated attributive
   ("leveling-down objection").
   (b) The abstract avoids the term entirely (says "purely by reducing over-service") —
   good; keep it that way so the defined-at-first-use guarantee holds.
2. **"Improving the ratio only by reducing the advantaged group's service"** (and
   variants: "closes the gap only from the top", which the contribution bullet already
   uses as a gloss). Fully explicit; zero reader risk. Cost: ~10 words per site × 12
   sites, a clumsy §3.4 title ("Why Demand-Only Editing Closes Gaps Only from the
   Top"), and the [24]/[26]/[36] connection has to be re-made parenthetically anyway.
   Best used as what it already is: the recurring gloss, not the name.
3. **"Downward equalization" / "equalizing downward".** Explicit-ish, compact, and
   close to the egalitarianism literature's paraphrase. Cost: not the cited term either,
   so it buys self-documentation but still needs the definition sentence — weakly
   dominated by option 1.

---

## 2. Sweep — coinage-by-coinage

Verdict key: **fine** (self-documenting or properly defined, leave it),
**define-better** (keep the word, strengthen/relocate the definition or fix drift),
**rename** (candidate for replacement). Replacements ranked, explicit first.

### 2.1 "metric firewall" — 1 use, `04:45` (§4.1 run-in head)
Defined at first use: yes — the paragraph it heads *is* the definition, and it is a good
one (one clause per ring + the claim-discipline rule "'Improves metrics it never
optimizes' claims ride ring (iii) only").
**Verdict: define-better (rename the head only if the ring decision below changes).**
The metaphor appears exactly once and does no further work; as a paragraph label it is
harmless, but it is the one place a reader may pause ("firewall against what?" —
answer: against claim leakage between metric classes, which the head doesn't say).
Options: 1) "**Metric rings and claim discipline.**" (says what the paragraph does);
2) "**Three classes of metrics.**" (pairs with a class-rename, below); 3) keep
"Metric firewall." (cost ≈ 0; flavor).

### 2.2 ring-(i)/(ii)/(iii) scheme — 11 exact uses + 2 "three-ring"
Where: forward mention `03:166` ("the design-targeted and external **rings** as well
(§4.6)"); §4.1 `04:24` ("three-ring selection criterion") **before** the definition at
`04:45-54`; then `04:54,73,77,105,393,394,397,398,450`.
Defined at first use: **no** — defined at `04:45-54` (and well), but *used* twice before
that (`03:166`, `04:24`), and `03:166`'s pointer sends the reader to §4.6 (the sweep),
not to the §4.1 definition.
Does the definition carry the weight? The definition itself, yes. The *deployment* has
three problems: (a) the forward uses above; (b) orthography drift — "ring (i)" (spaced),
"ring-(ii)" (hyphenated), "(ring iii)" (bare numeral, in the `04:77` heading) all
coexist; (c) a mild but real hazard specific to this paper: in a manuscript about city
grids and districts, "ring" invites a literal spatial misreading ("ring (iii)
instruments" ≈ instruments in some geographic ring) precisely at first contact.
**Verdict: define-better at minimum; rename is defensible.** Options:
1) **Keep "ring", fix deployment** — one orthography everywhere (suggest "ring (i)"
   spaced, matching the definition), and give `03:166` a five-word gloss + pointer to
   §4.1 ("the design-targeted and external rings — metric classes defined in §4.1 —").
   Smallest diff.
2) **Rename to "class (i)/(ii)/(iii)"** ("three-class criterion", "class-(iii)
   instruments"): kills the spatial collision, stays compact, definition paragraph
   survives verbatim with one word swapped. Cost: loses the concentric flavor
   (optimized at the core, external outermost) that "ring" was chosen for.
3) **Drop the labels; always use the adjectives** ("optimized / design-targeted /
   genuinely external"): maximally explicit, no notation to remember. Cost: wordy at 11
   sites, and the compact criterion name ("three-ring criterion") has no obvious form.

### 2.3 "three-ring criterion" vs "three-part criterion" — 2 vs 2, same object
`04:24,398` say "three-ring"; `03:173` and the Figure 2 caption (`04:411`) say
"three-part". A reader can legitimately wonder if these are two criteria.
**Verdict: rename one — unify.** Follow the §2.2 decision: "three-ring criterion" if
rings stay, "three-class criterion" if renamed; "three-part" should survive nowhere
(it under-specifies *which* three parts).

### 2.4 "lift-up" (7) vs "lifting-up" (2) — plus "level up" (2)
"lift-up" (`03:169,175`, `04:104,291,395,400,408`) is the property/claim name; the two
"lifting-up" strays are "the naive **lifting-up** baseline" (`04:336`) and "the SF
**lifting-up** claim" (`04:487`). "level up" (`02:81`, `03:269`) is the *literature's*
verb (Mittelstadt's prescription) and is correctly kept distinct, citation-adjacent.
Defined at first use: functionally yes — §3.2 (`03:169`) uses it in a sentence whose
subject is the under-served group's service rising.
**Verdict: define-better (standardize).** 1) Normalize both "lifting-up" → "lift-up";
2) keep "level up" only in the two cited-prescription spots; 3) no rename needed —
"lift-up" is anchored to the *lift* mode's name, which is the paper's own mechanism
vocabulary and self-documenting one hop away. (Fully explicit alternative if ever
wanted: "the added-presence claim".)

### 2.5 "trim+lift" (13) and the stray "trim-then-lift" (1)
"trim+lift" is the canonical editor name (13 uses, README-canonical). The conclusion
(`05:36`) introduces "the two-phase **trim-then-lift** structure" — a third compound, and
a reader may ask whether trim-then-lift differs from trim+lift (it doesn't; it stresses
phase order, which §3.5 "Budget and phase order" already established).
**Verdict: define-better (reword the one instance).** Suggest "The editor's two-phase
structure — trim first, then lift — is a scientific control…", avoiding the new coinage.

### 2.6 "tier-1 / tier-2 accounting" — 11 uses
First use `01:57-59` **with inline definitions** ("under the optimizer's
fractional-presence accounting (tier-1) and … recounted as distinct taxis from raw GPS
(tier-2)"); re-defined at `04:128-129`. Deployment is consistent (always tier-1/tier-2,
hyphenated).
**Verdict: fine.** The numbers are arbitrary but the two definitions are attached at
both load-bearing sites. If ever renamed: 1) lead with the semantic names
("fractional-presence" / "distinct-taxi recount") and demote tier-N to parenthetical;
2) "optimizer-units vs. taxi-units accounting". Neither seems worth 11 edits.

### 2.7 the fairest-selection control — FIVE surface forms for one arm
Observed: "select-the-fairest control" (abstract `main.tex:65`, `01:98`, `04:379`),
"most-fair selection" (`04:263`), "most-fair select (w30)" (Table 6 rows), "most-fair
control" (`04:455,501`), "the corpus's already-fairest trajectories" (`01:71`,
`03:425`, `04:412`).
Defined at first use: the *abstract* form is self-defining; the problem is not
definition but **aliasing** — a careful reader must verify that five names denote one
control.
**Verdict: rename (unify).** Options: 1) **"most-fair selection control"** everywhere
in prose (self-documenting; matches the existing table shorthand "most-fair select
(w30)", which can stay as the declared table abbreviation); keep "the corpus's
already-fairest trajectories" once, as the defining clause at first §4.4 use. 2)
"fairest-subset control". 3) Keep "select-the-fairest control" as the single prose form
(it is the most explicit of the five) and align the others to it. Any one of these; the
point is one name + one declared table shorthand.

### 2.8 "vanilla BC" — 7 uses
Forms: "vanilla behavior-cloning objective" (abstract, `01:67`), "vanilla BC"
(`04:274,496`), "vanilla training" (`04:244` head), "vanilla transfer" (`04:386`),
"vanilla-transfer null" (`03:415`), Table 6 row "vanilla-BC transfer (w1)".
**Verdict: fine.** "Vanilla" is ML-idiomatic and every fairness/ML reader parses it;
the contrastive content ("weighting every demonstration equally") is stated at `03:414`
and `04:244-246`. Explicit alternatives if Robert prefers a colder register:
1) "uniform-weight BC" (the contrast *is* the weighting — most precise);
2) "unweighted BC" (slightly less accurate: weights exist, they're equal).
Cost of renaming: none technical; 7 sites + a Table 6 row label.

### 2.9 "king-compliant" / "king-move" — 8 uses
First use `03:379-380` defines it **with the formula**: "a king-move rule: consecutive
states move at most one cell per axis, max(|dx|,|dy|) ≤ 1" (and ties it to the
preprocessing pipeline of [33]). Later: "king-compliant assignment/repair"
(`03:384`, `04:148`), "king-move adjacency" (`04:283,326,521`), "king-compliant by
construction" (`04:288,329`).
**Verdict: fine (kept honest by its definition).** The chess metaphor is
standard-adjacent (Chebyshev/chessboard distance) and the formula removes all ambiguity.
Optional hygiene: at the first §4 reuse (`04:148` or `04:283`), a two-word reminder
"(one cell per axis)". Explicit alternatives if renamed: 1) "one-cell-per-axis rule";
2) "unit Chebyshev step rule" (house style would dislike the naked eponym).

### 2.10 "oracle gate" family — 3 uses, 2 names
"greedy oracle" ×2 (`03:285`, `03:302` — "bounded its headroom with a greedy oracle,
which cleared a pre-registered go/no-go threshold"); "the pre-build oracle gate"
(`04:150`).
Defined at first use: partially — `03:285`'s context ("closes the remaining escape
route", "bounded its headroom") implies upper-bound-search, but never says it.
**Verdict: define-better.** Two small moves: 1) gloss at first use — "a greedy oracle
(an unconstrained search that upper-bounds what any editor could achieve)"; 2) link the
§4.3 name back — "the pre-build oracle gate (§3.4's greedy-oracle headroom check)" — or
unify both sites on one name ("oracle headroom gate"). Explicit alternatives:
"upper-bound (oracle) analysis"; "greedy headroom bound".

### 2.11 "dose-response" family — 11 uses
"dose-response" (`01:69`, `04:37,385,498`, `05:29`), "dose" (`04:258,264,339-341,350`),
"dose-monotone" (`04:339`).
**Verdict: fine, with one flag.** The clinical-trial register (dose + placebo +
pre-registered) is coherent, standard in empirical ML writing, and each use is next to
its quantitative meaning. The flag: **two different knobs are both "dose"** — the
upweight w (§4.4/§4.6) and the oversampling duplication budget d = 10,000 (§4.5,
"matched dose"). §4.5 does define its dose explicitly, so confusion risk is modest;
if Robert wants zero risk, say "matched duplication dose" once at `04:339` and leave
the weight-dose language alone.

### 2.12 "placebo" — 10 uses
Three distinct placebos exist and every use is qualified: "random-subset placebo"
(weighting control), "seeded random-jitter placebo" (perturbation arm), "untargeted
placebo (same fabrication, no targeting)" (oversampling control). The abstract avoids
the word (uses "a random-upweighting … control") — good.
**Verdict: fine.** Keep the qualifiers universal (no bare "the placebo" where two are
in scope — currently true). No rename proposed; "placebo" is the one clinical borrow
that is strictly more precise than its explicit paraphrase ("control constructed to be
identical except for the treated ingredient").

### 2.13 Additional coinages found in the sweep (added per instructions)

| Term | Where / count | Defined? | Verdict + options |
|------|---------------|----------|-------------------|
| "honest / honestly / the honest X" | 6 sites: `01:37`, `03:146`, `04:65` ("the honest count"), `04:116` ("tier-2 is the honest semantics"), `04:278`, `04:350` | n/a (tone, not a term) | **Style flag.** Six uses read as a tic and carry an edge (implies others are dishonest). Suggest keeping 1–2 (e.g. `01:37`, `03:146`) and cooling the rest: "the honest count" → "the strict count"; "the honest semantics" → "the stricter semantics"; "cannot supply the matched dose honestly" → "…without re-duplication". |
| "SF caveat block" | `04:519` (run-in head) | — | **Rename.** "Block" is document furniture, not content. → "**San Francisco caveats.**" (or "SF caveats."). |
| "the screen" / "nominates" | §3.3, ~3 uses around `03:236-252` | Yes — operational definition in place ("the screen asks which drivers…", "The screen only nominates: every selected trajectory is subsequently re-optimized") | **Fine.** "Screening/nomination" is standard selection vocabulary and the paragraph defines the mechanics. |
| "value-of-presence map" | `03:223,241` | Yes — defined in its first sentence ("large where added presence would most repair the fairness terms…") | **Fine** — a model explicit coinage; no action. |
| "sign-unanimity certificate" | `04:37` | Yes — the sentence defines it (p = .031 attained exactly when all six seeds share a sign) | **Fine**; if "certificate" feels clever, "sign-unanimity reading" is the flattest swap. |
| "effect-versus-noise" | `04:39` | Yes, in-sentence | **Fine.** |
| "skip-on-infeasible rule" | `04:26` | Pointer to §3.5 sits in the same sentence; the rule itself is described at `03:381-386` | **Fine / NIT** — consider naming it at §3.5 too, so §4.1's mention has a landing site with the same words. |
| "two-cell ball" | abstract, `01:29`, §3.1 | §3.1 formalizes (L∞, ε = 2) | **Fine** — abstract-level shorthand with a §3 formal anchor. |
| "demand deficit attribution" / "supply-gradient attribution" | §1, §3.3 onward | Yes — §3.3 defines both in their first paragraph | **Fine** — self-documenting names, consistently deployed. |

---

## 3. Summary of recommended actions (Robert picks; none applied)

Highest leverage first:
1. **Keep "leveling down"** (citation-aligned via [24]/[26]/[36], defined three ways);
   normalize the two hyphenated noun uses ("pure leveling-down" → "pure leveling down")
   — or the reverse, but one form.
2. **Unify the selection-control name** (currently five aliases): one prose name +
   the declared Table-6 shorthand.
3. **Ring scheme:** fix orthography to one form + gloss the forward use at `03:166`;
   decide keep-"ring" vs "class" (spatial-misreading argument favors "class");
   whichever wins, kill "three-part criterion" in favor of the matching
   "three-ring/three-class criterion".
4. **Standardize "lift-up"** (2 "lifting-up" strays); keep "level up" only as the
   cited prescription.
5. **Reword the single "trim-then-lift"** (conclusion) to avoid a third editor compound.
6. Small ones: "SF caveat block" → "San Francisco caveats"; cool the "honest" tic;
   optional glosses for "greedy oracle" and the §4.5 "matched (duplication) dose".
7. No action recommended: tier-1/tier-2, vanilla, king-move (all defined and
   consistent), placebo/dose family, screen/nominates, value-of-presence,
   sign-unanimity certificate, two-cell ball, the attribution names.

Cross-reference: the render-QA report (same date) flags the *typographic* side effects
of several of these compounds (they cause most of the margin overflows), so rewording
decisions here can be batched with those fixes.
