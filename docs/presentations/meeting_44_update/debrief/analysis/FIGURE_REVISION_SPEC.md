# Figure + Formatting Revision Spec — Meeting 44 (2026-07-23)

> ⚠️ **POST-DEBRIEF CORRECTIONS (Robert, 2026-07-23) — parts of this document are superseded.**
> 1. Dr. Zhang WAS reviewing the CURRENT paper content (Robert transferred it to Overleaf pre-meeting; she views and will edit there). Retract every "stale copy / older version / discount-as-render-artifact" inference in this file — ALL her feedback binds against the current text.
> 2. Template: per Robert, conform to her direction to NOT use the `\keywords{...}` block (the "corrected template adds keywords" reading is superseded); verify against KDD template standards.
> 3. Raw data: releasable IF 100% anonymous (not a flat no); in-paper data references must not leak identifying information.
> 4. Hat-matrix citation: stays in the main body (derivation content still moves to the appendix).
> Authoritative record: ../MEETING_44_DEBRIEF.md (§2, §3, §6, §7).

**Lens:** every visual / formatting critique from the meeting, turned into an
element-level revision spec.
**Scope:** Figure 1 (`fig:teaser`, intro) and Figure 2 (`fig:overview`,
methodology), their captions, and cross-cutting formatting rules.
**Sources:** `meeting_44_transcript.txt` (345 lines), `plaud_marked_photo.png`,
`plaud_summary.md`, `plaud_discussion_summary_raw.txt`, `plaud_highlights_raw.txt`,
`paper/figures/figure-1/figure-1.tex`, `paper/figures/figure-2/figure-2.tex`,
captions in `sections/01_introduction.tex` + `03_methodology.tex`,
rendered `paper/main.pdf` (Fig 1 = p.1 top-right; Fig 2 = p.5 full-width top).

> **Headline verdict (Dr. Zhang):** Figure 1 is *not self-explanatory*. She could
> not understand it on sight, had to read the caption, and could not see where the
> "3× service" claim lives. **Fix: rebuild Figure 1's two rollout panels in
> Figure 2's style** (map background, explicit district labels, duplicated legend),
> **put the fairness numbers inside the figure**, and **conserve taxi/passenger
> totals** between before/after so the story reads as *relocation*, not *removal*.
> She physically demonstrated the district labels she wants by writing **"A."** and
> **"D."** on a printout (see §Pen-Markup).

---

## PART 1 — EXTRACTED CRITIQUES & AGREEMENTS (quote-anchored)

### 1.1 Figure 1 is not self-explanatory (the core failure)

| # | Failure | Quote (Dr. Zhang unless noted) | Time |
|---|---------|-------------------------------|------|
| A | Reader can't parse it at a glance | "we want the figure itself to be self-explanatory instead of having to read a figure" | [0:54:49] |
| B | Grid cells unexplained | "I know it's a grid cell. But I don't know what is a grid cell. Is it a map or is it something? I have no idea." | [1:15:22] |
| C | Color coding unexplained | "there are different two districts, yellow part and blue part. Yellow part only has taxis, blue part only has people. So. What does that mean? I have no idea." | [1:15:22]–[1:16:03] |
| D | Districts unlabeled | "So what are the advantage areas? … the yellow part or the blue part?" | [1:04:25]–[1:04:29] |
| E | Fairness itself not visible | "I cannot see what is fairness. I can only see certain vehicles towards the left, certain passengers towards the right, and I cannot see why this creates a fairness." | [1:11:34] |
| F | Glyph identity ambiguous | "I need to guess what are these vehicles. Are these vehicles taxis or other things? Are these like people icons, passengers or anything?" | [1:11:34] |
| G | 3× gap not visually evident | "explain to me what does a three times means?" (after trying to read a ratio off the glyphs and failing) | [1:08:01]–[1:08:29] |
| H | Reliance on caption | Robert: "if we read the figure one caption, it kind of implies the advantaged district receives three times the taxi service." Zhang: "we want the figure itself to be self-explanatory instead of having to read a figure." | [1:04:32]–[1:04:58] |
| I | Two figures too far apart to co-explain | "figure one, figure two, they are so far away from each other. And we want to make sure that each figure is self explanatory. It's by itself." | [1:16:59] |
| J | Misread the tint as parking | "it looks to me like this orange, this yellow area is where people park their cars instead of where the taxis are." | [1:09:37] |

Robert's defense that Fig 1 + Fig 2 explain each other "in combination" [0:55:55],
[1:14:38] was **explicitly overruled** — each figure must stand alone [1:16:59].

### 1.2 Mandated fixes for Figure 1

| # | Fix | Quote | Time |
|---|-----|-------|------|
| 1 | Adopt Figure 2's style for the two rollout panels | "the style of the Figure two would work better to replace these two smaller figures in Figure one. Because in this Figure two you have advantaged districts and disadvantaged districts." | [1:17:25] |
| 2 | Add explicit district text labels A/D | "if it's a yellow color, we just add a … text saying that this part is the advantage. And … the blue part is disadvantage." | [1:04:57] |
| 2b| (restated) put the label text ON the region | "if we want to convey that this advantaged area, this yellow part is advantage area, you put a text advantage area over there." | [1:09:55] |
| 3 | Duplicate the legend(s) onto Fig 1 | Robert: "I will make sure that the legends and everything are duplicated on both figure two or figure one, figure two." Zhang assents: "It's going to be way more obvious." | [1:18:23]–[1:18:40] |
| 4 | Add map background behind the grids | Robert: "putting the … city map behind the other grid." Zhang: "I see. I think the Figure two would work better…" | [1:17:14]–[1:17:25] |
| 5 | Clear icons for advantaged/disadvantaged + vehicle/taxi counts | "having all those … tasks in terms of advantage, disadvantages regions. And also the number of vehicles and the number of taxis." | [1:18:40] |
| 6 | Legend/key icon defining the districts | Robert: "you want a … legend icon that defines these?" (Zhang: yes, or a text label) | [1:05:24]–[1:05:32] |
| 7 | Larger, more realistic rendering | "I can show you more realistic kind of drawing, so people can instantly know what you are trying to convey the moment they look at the paper." | [1:10:49] |

### 1.3 Show the fairness numbers INSIDE the figure

| # | Requirement | Quote | Time |
|---|-------------|-------|------|
| 1 | Pre-editing fairness score is missing | "one thing is not reflected is that what is the fairness score before editing?" | [0:52:46] |
| 2 | Put a fairness value on each side | "we can directly show one value over here. Like fairness score for this data and fairness score for this data." | [0:53:04] |
| 3 | The 3× must be shown in-figure, not just caption | Robert: "the caption gives a three x service gap." Zhang: "if that is the case we'd better show that information in the figure." | [0:53:37]–[0:53:52] |
| 4 | Worked example of how to make 3× visible via glyph counts | "if the current claim is this part is having three times more service than this part, then you are assigning six taxis over here and you're having two passengers over here. And over here you just have one taxi and one passenger. … people would know what … three times service difference you're talking about." | [1:12:04]–[1:12:45] |

### 1.4 Conservation rule (totals constant between before/after) — the sharpest mandate

| # | Rule | Quote | Time |
|---|------|-------|------|
| 1 | Totals must stay the same, else it reads as elimination | "make sure that within these two figures, the total number of passengers and the total number of vehicles should stay the same. Otherwise, it's kind of trying to assume that you are … essentially eliminating taxis or services or eliminating demands." | [1:05:32] |
| 2 | The story is relocation, not removal | "because we are not removing taxis or removing passengers, what we are doing is we are relocating them." | [1:18:40] |
| 3 | Final crisp statement of the rule | "we want to make sure in the teasing figure, at least that the number of taxis stays the same as the number of passengers stays the same." | [1:19:22] |
| 4 | Teasing figure need not be rigorous (so equal totals + stylized distribution is fine) | "The teasing figure doesn't need to be rigorous. … It doesn't have to be rigorous." | [1:13:42]–[1:14:03] |

**Context / tension inside this thread:** Robert initially *defended* the current
unequal counts — "if that's meaningful, that is intentional that there are less
taxis right there" [1:06:06] — arguing that equal taxi counts would break the
service-per-unit-demand math and "destroy our actual results" [1:06:25]–[1:07:24].
Dr. Zhang's reply: the teaser conveys the *idea* via distribution across districts,
not literal counts [1:13:42], and later Robert conceded the figure "is not literal;
it is very much stylized" [1:14:01]–[1:14:38]. **Net decision = conservation holds.**
(One earlier nuance at [1:06:15]: "in the city level, we want less service in
general across all the regions" — i.e., if anything drops it must drop city-wide,
not asymmetrically; superseded by the equal-totals rule at [1:18:40]/[1:19:22].)

### 1.5 Metric definition consistency (text ↔ figure)

| # | Issue | Quote | Time |
|---|-------|-------|------|
| 1 | No service ratio is visible in Fig 1 | "currently I do not see a service ratio." | [1:07:24] |
| 2 | Ratio DIRECTION is ambiguous (demand/supply vs supply/demand) | "I'm only observing if the service is demand versus supply. Over here I'm looking at infinity supply … if the service ratio is demand over supply, over here I'm looking at a value which is zero … one over four … five over zero, this is infinity … three over two. So explain to me what does a three times means?" | [1:07:24]–[1:08:29] |
| 3 | Metric name must be pinned | Robert (blanking): "we have disparate impact. It's not disparate impact. It's the other one I can't remember … this word impact." | [1:08:29]–[1:09:54] |
| 4 | Caption phrasing of the metric | Caption: "$3.0\times$ the taxi service per unit demand"; Zhang reads it back verbatim | [1:12:54]–[1:13:39] |
| 5 | Don't over-formalize the ratio in the drawing | "you don't need to include all the important details in terms of how we are calculating those service ratios. What most important is you want to convey the idea." | [1:13:42] |

Standardize: one direction (paper §3.2 defines **both** `Y_i = S_i/max(D_i,d_0)` =
supply/demand **and** `DSR_i = D_i/S_i` = demand/supply — this is the ambiguity),
one name (the 3.0× is sourced from **disparate impact**, `disparate_impact.before
0.33252 → 1/0.33252 = 3.0×`, per the caption src comment), used identically in
abstract/intro text, Fig 1, and the external-metrics table.

### 1.6 Cross-figure consistency

| # | Point | Quote | Time |
|---|-------|-------|------|
| 1 | Colors already consistent Fig1↔Fig2; keep it | Robert: "the coloring is consistent and Figure Two … Figure One and Figure Two in combination provide the motivation and then the solution." | [0:55:55] |
| 2 | Fig 1 = motivation (WHY), Fig 2 = solution (HOW), jointly tell the story | same turn; and figure-1.tex header: "This figure is the WHY; fig:overview is the HOW." | [0:55:55] |
| 3 | Color conveys fair/less-fair (cautionary) regions | Robert: "a more cautionary color … there are more fair regions and less fair regions" | [0:55:55]–[0:56:55] |

### 1.7 In-figure text size, spacing, compactness, blank space

| # | Rule | Quote | Time |
|---|------|-------|------|
| 1 | Too much text, too small, enlarge the figure | "there are too many texts in the figure. And they're so small and difficult to read, I think we also need to enlarge this figure." | [0:13:46] |
| 2 | THE FONT-SIZE RULE (exact) | "the text in the figure are like slightly smaller than this caption size or than the text in the paper. So that way people are having less difficulty reading." | [0:15:10] |
| 3 | Bigger text, smaller gaps | Robert (confirming): "Bigger text, smaller gaps is what I'm hearing." Zhang: "Uh-huh." | [0:14:08]–[0:14:13] |
| 4 | Compact the figure / kill blank space | "there are a lot of blank spaces over here … make figures more compact." | [0:13:18]–[0:13:30] |
| 5 | Don't waste space — lots of content to fit | "we have a lot of content to put in the figure, so we don't want to waste space." | [0:14:13] |
| 6 | Robert's cause + fix | "I reduced the text size when the figure was larger … So I'll just increase the text size there." | [0:15:44] |

### 1.8 Caption length policy

| # | Rule | Quote | Time |
|---|------|-------|------|
| 1 | Captions need not be long; blank space after caption | "the caption doesn't have to be kind of a long caption. As you can see, there are a lot of space after the caption." | [0:16:47] |
| 2 | Essential info only (8-page pressure) | "because we have limited space is just eight pages … we just explain the most important information in the caption." | [0:17:24] |
| 3 | Applies to BOTH figures | Robert: "this goes for both figures, both Figure one and Figure two." (current format = "explain everything") | [0:16:55] |
| 4 | ⚠ Possible counter-signal | "we don't modify the caption of the figures or anything, and we just leave it as is, like how the paper looks" — ambiguous; likely means *follow the template's caption formatting*, not "don't shorten." | [0:19:16] |

### 1.9 References-in-text, bold refs, all-figures-referenced

| # | Rule | Quote | Time |
|---|------|-------|------|
| 1 | Main-text figure refs must be clear | "in the main text where we are referencing the figures, we want to make sure that they are clear." | [0:18:22] |
| 2 | Bold-reference formatting challenge | Robert: "without bolding the references. Which is tricky because of how text works, it will only bold one part of it. So it's either you choose to not bold it, or you choose to bold the figure and not the number." | [0:18:31] |
| 3 | Verify EVERYTHING is referenced (tables flagged) | Robert: "I didn't check that figure one was referenced. They did figure two and all the other, and tables actually was the big one." | [0:19:02] |

### 1.10 Template (CCS vs keywords) — layout impact only

| # | Point | Quote | Time |
|---|-------|-------|------|
| 1 | Replace CCS-concepts block with keywords | "you are removing the CCS concepts part and replace it by keywords." | [0:10:09] |
| 2 | Already fixed by Zhang | "the template is wrong but it's okay, I have modified it to the correct version." | [0:11:22] |
| 3 | Package conflict when SHE inserted a figure | "for the figures, I'm trying to add it in, by using the package that you use, but it doesn't seem to be showing correctly … I think that is a package conflict issue." (her workaround: screenshots [0:12:54]) | [0:12:15] |

**Layout note:** the front matter already renders **KEYWORDS** (p.1), so the swap is
in. The package-conflict + screenshot-workflow is a *risk* (see Conflicts §C7), not a
figure change.

### 1.11 Figure 2 and other visuals — remarks

| Target | Remark | Quote | Time |
|--------|--------|-------|------|
| **Figure 2** | Praised as the model to emulate; has the district split Fig 1 lacks | "in this Figure two you have advantaged districts and disadvantaged districts. And we know disadvantaged districts are having more people less vehicle." | [1:17:25]–[1:18:40] |
| **Figure 2** | No standalone critique; inherits the global rules (enlarge text, shorten caption, compact spacing, conserve nothing — Fig 2 is not a before/after) | — | — |
| **Figure 2** | Miscount while reading it live ("decreased number of taxis … one two three") shows even Fig 2's counts can mislead — keep counts legible | "I know that there are a decreased number of taxis in total, right? One two three." | [1:16:09] |
| **Tables** | Flagged as the reference-coverage risk ("the big one") | see §1.9 #3 | [0:19:02] |
| **Frontier / external-metrics tables** | *Not discussed by name this meeting.* No specific critique captured. | — | — |

---

## PART 2 — CURRENT STATE (what is actually there now)

### 2.1 Figure 1 (`fig:teaser`) — `figures/figure-1/figure-1.tex`, column-width, on p.1

**Structure (top → bottom):**
1. Bold headline: *"Bias is learned from data — fairness can be, too"* (`\footnotesize\bfseries`).
2. Two **corpus chips** over GPS-trace PNGs (`corpus_background_raw.png`,
   `corpus_background_edited.png`), joined by a cobalt **"FATE"** provenance arrow
   (raw → edited). Edited chip has an accent slice + baked-in "+".
3. Two **model boxes**: "standard imitation model" / "upweighted imitation model"
   (latter has pale-blue fill).
4. Two **rollout city grids** (7 cols × 4 rows, `0.45cm` cells) with amber/blue
   regional tints + a dashed district boundary. **No street-map background. No
   district text labels.**
5. Two **outcome captions** under the grids ("the fairness gap, learned and
   re-enacted — bias propagates to rolled out corpus" / "fairness gains survive
   training — less human bias in rolled out corpus").
6. One **boxed legend footer** (dotted border): *taxi presence · service pickup ·
   changed by the edit* (the "+").

**Text sizes:** headline `\footnotesize\bfseries`; every other label `\tiny`
(≈5pt in the 9pt body → this is the "too small" Zhang flagged).

**Palette (shared, CVD/grayscale-verified — MUST preserve):** `figink #4B5563`,
`figaccent #2563EB` (added/edited/FATE), `figtrim #D97706` (excess/trim),
`figtintamber #FFF3D6` (over-served), `figtintblue #E8F1FC` (under-served).
Verification on record: deut/prot separation >200 linear-RGB; charcoal text 7.6:1;
glyphs/dashes/plus marks carry semantics without color (figure-1.tex lines 38-40).

**GLYPH COUNTS (counted from TikZ source; conservation check):**

| Panel | Taxis | Pickups | Total | Source lines |
|-------|-------|---------|-------|--------------|
| LEFT (unfair / raw rollout) | **7** | **5** | **12** | taxis 183–186, picks 187–189 |
| RIGHT (fair / edited rollout) | **6** (4 plain + 2 `taxinew`+) | **4** (3 plain + 1 `picknew`+) | **10** | taxis 201–203, taxinew 208–209, picks 206–207, picknew 205 |
| **Δ** | **−1** ❌ | **−1** ❌ | **−2** ❌ | — |

➡ **The current figure VIOLATES the conservation rule on both axes** (7→6 taxis,
5→4 pickups). As drawn it reads as *eliminating* one taxi and one passenger —
precisely the "eliminating taxis/demands" failure Zhang named at [1:05:32]. The 2
`taxinew` + 1 `picknew` "+" glyphs sit in under-served cells (good relocation
*intent*) but because totals drop, they read as *net additions*, not moves.

**Caption:** 5 sentences, renders as ~8 lines on p.1 (`sections/01_introduction.tex`
lines 37–51, plus a separate `\Description`). Leads with the 3.0× claim.

### 2.2 Figure 2 (`fig:overview`) — `figures/figure-2/figure-2.tex`, full-width `figure*`, on p.5

Three panels over a **real Shenzhen street-map PNG** (`SZ_street_background_5x4_rotated`),
10×8 cells (`4.9mm`), same palette:
- **(1) Attribute:** taxis dense in advantaged (amber) district, unmet pickups in
  disadvantaged (blue); **explicit `\lbl` text: "advantaged district (over-served)"
  / "disadvantaged district (under-served)"**; lower-left legend (taxi presence /
  service pickup).
- **(2) Trim:** amber arrows relocate 2 pickups within the advantaged district;
  vacated origins as pale-amber glyphs; per-panel legend.
- **(3) Lift:** map zoomed 2×; quantized cobalt "value-of-presence" heat cells;
  solid original tail vs dashed cobalt rerouted tail; upper-right legend (original
  tail / rerouted tail / value gradient).

Text: panel titles `\footnotesize`, body labels `\tiny` (same too-small issue).
**Caption:** 4 sentences, ~9 rendered lines (`03_methodology.tex` lines 193–205).
This figure already embodies everything Fig 1 must adopt (map, district labels,
duplicated legends) — it is the reference standard.

### 2.3 Pen-markup photo (`plaud_marked_photo.png`, 911×236 strip)

Shows **only the two rollout grids of Figure 1** (bottom band) + their outcome
captions. Dr. Zhang's magenta-pen marks are all on the **LEFT (unfair) panel**:

- **"A."** written over the **amber / over-served region** (bottom-center-left,
  among the clustered taxis) — she is labeling it the **Advantaged** district.
- **"D."** written large over the **blue / under-served region** (bottom-right,
  among the passenger stick-figures) — labeling it the **Disadvantaged** district.
- A **diagonal slash/arrow** descending from the district boundary down toward the
  caption line "bias propagates to rolled out corpus" — points at the region
  boundary / at the text, emphasizing where the label + boundary read is meant to
  land.

The photo is a literal demonstration of §1.2 fix #2: she wants the words
**Advantaged** and **Disadvantaged** printed on the district regions. The
Plaud highlight for this moment (`mark_type 2`, ts 3942000 ms = **[1:05:42]**):
*"why districts are not clearly labeled as advantaged or disadvantaged … adding
explicit text labels and legends."*

---

## PART 3 — ELEMENT-LEVEL REVISION SPEC

Actions: **KEEP** (leave as-is, do not regress) · **ADD** · **CHANGE** · **REMOVE**.
Palette rule for every ADD/CHANGE: reuse the existing `figink/figaccent/figtrim/
figtint*` colors — do **not** introduce new hues, so the CVD/grayscale verification
(figure-1.tex 38-40) survives.

### 3.1 FIGURE 1 (`fig:teaser`)

| Element | Action | Exact requirement | Mandate | Notes / risk |
|---------|--------|-------------------|---------|--------------|
| District labels on rollout panels | **ADD** | Text "Advantaged" (amber region) and "Disadvantaged" (blue region) on **each** rollout grid, matching Fig 2's `\lbl` style ("advantaged district (over-served)" / "…(under-served)") | [1:04:57], [1:09:55], pen-markup A./D. | Use white-backed `lbl` node (Fig 2 lines 96-99). TikZ real estate: cells are 0.45cm — labels may need to sit in an empty cell or just outside the frame. |
| Street-map background under grids | **ADD / CHANGE** | Put the city map behind the two rollout grids (Fig 2 style) so a cell reads as "a place on a map" | [1:17:14]–[1:17:25], [1:15:22] (B) | **Conflict C3**: figure-1.tex header (lines 12-15) *deliberately* keeps Fig 1 map-free ("plain schematic grids — no street map"). This reverses that decision. Reuse `SZ_street_background` but at column width the map may be busy behind 0.45cm cells → may need a lighter/blurred crop. |
| Legend | **CHANGE** | Keep the footer key but ensure it is **duplicated / self-contained on Fig 1** (taxi presence, service pickup, changed-by-edit) AND expand to gloss the district tints (amber = advantaged/over-served, blue = disadvantaged/under-served) | [1:18:23], [1:18:40], C (color) | Current legend (lines 222-230) omits the tint meaning — add 2 tint swatches. |
| In-figure fairness value (before) | **ADD** | Print the pre-edit fairness/service number on the LEFT panel and the after number on the RIGHT (e.g., "service ratio 3.0×" left → "≈1.2×" / "fairer" right) | [0:52:46], [0:53:04], [0:53:52] | **Conflict C2** with compactness [0:13:30]. Pull the `3.0×` out of the caption into the panel. Must match the standardized metric name/direction (§3.3). |
| 3× gap visibility | **CHANGE** | Make the advantaged/disadvantaged service gap *visible from glyph distribution*, per her worked example (e.g., 6 taxis / 2 pickups in A vs 1 taxi / 1 pickup in D on the "before" side) | [1:12:04]–[1:12:45], G | **Conflict C1** with conservation (below) — the per-district counts must still sum to equal totals across before/after. Stylized, not literal [1:14:01]. |
| Rollout glyph COUNTS | **CHANGE** | Make **total taxis LEFT = total taxis RIGHT** and **total pickups LEFT = total pickups RIGHT**. Current 7→6 / 5→4 must become e.g. 7→7 / 5→5, with the RIGHT panel showing the *same* glyphs **relocated** into under-served cells | [1:05:32], [1:18:40], [1:19:22] | Concretely: add 1 taxi + 1 pickup to the RIGHT panel (or drop 1 each from LEFT) and re-place, don't net-add. Re-count after editing. |
| "+" / `taxinew` / `picknew` motif | **CHANGE** | Redefine the "+" to mean **relocated** (this glyph moved here), not **added**. Legend text "changed by the edit" is already relocation-neutral — keep that wording; ensure the marked glyphs have a same-type counterpart removed from its old cell so totals conserve | [1:18:40] (relocate not remove) | **Conflict C1**: code comments call these "added presence" (lines 24, 69, 80) — semantics must flip to "moved," or the +marks + unequal counts keep telling the "added service" story Zhang rejected. |
| Grid-cell identity | **CHANGE** | A cell must read as a map location (via the map background + a one-word label like "city grid" or the map itself) | [1:15:22] (B), [1:11:34] (E/F) | Satisfied largely by the map-background ADD. |
| Glyph size / legibility | **CHANGE** | Enlarge taxi/passenger glyphs + all label text so figure text is *slightly smaller than caption/body*, not `\tiny` | [0:13:46], [0:15:10] | Bump `\tiny`→`\scriptsize`/`\small`; enlarge the whole figure. Preserve glyph proportions (car/stick-figure vocabulary is settled). |
| Outcome sub-captions under grids | **CHANGE (optional)** | Consider plain wording "fairer corpus / less-fair corpus" | Robert [0:59:52] | Minor; Robert's own suggestion, not a Zhang mandate. |
| Headline "Bias is learned from data — fairness can be, too" | **KEEP** | Retain (it does self-explanatory work) | — | Watch the em-dash (project style near-bans em-dashes; a colon/comma may be preferred). |
| Corpus chips + FATE provenance arrow | **KEEP** | Raw→edited chips + cobalt FATE arrow stay | not critiqued | The provenance arrow is doing real "edited comes from raw" work. |
| Model-box layer (standard vs upweighted imitation model) | **KEEP** | The intermediate imitation-model layer stays | Robert defended [0:58:29], [1:10:59]; Zhang conceded [1:03:10] | **Conflict C4**: Zhang floated *removing* this layer ("we don't have this intermediate layer" [0:57:12]); resolved in favor of keeping because upweighted imitation is core to the claim. Do not delete. |
| Blue-amber palette + CVD/grayscale verification | **KEEP** | Do not alter hues; re-verify if any fill opacity changes | figure-1.tex 30-40 | Any new tint swatch/label must stay within the verified set. |

### 3.2 FIGURE 2 (`fig:overview`)

| Element | Action | Exact requirement | Mandate | Notes / risk |
|---------|--------|-------------------|---------|--------------|
| Overall style, map, district labels, per-panel legends | **KEEP** | This is the reference standard Fig 1 must copy — do not regress | [1:17:25] praise | — |
| Label / title font sizes | **CHANGE** | Same font-size rule: bump `\tiny` body labels so figure text is *slightly smaller than* body, not much smaller | [0:15:10], [0:13:46] | Applies globally to all figures. |
| Internal spacing / compactness | **CHANGE** | Tighten gaps; remove wasted space around the 3 panels | [0:13:30], [0:14:13] | Full-width `figure*` — check the inter-panel gutters (shifts at 11.5, 23). |
| Glyph counts legibility | **KEEP/CHECK** | Keep taxi/pickup counts easy to count at a glance (she miscounted live) | [1:16:09] | Not a conservation figure (no before/after pairing), so no equal-totals rule — but keep counts readable. |
| Palette + CVD verification | **KEEP** | Shared verified palette; do not alter | figure-2.tex 18-25 | — |
| District labels "advantaged (over-served)/disadvantaged (under-served)" | **KEEP** | Retain — these are exactly what Fig 1 is being told to add | [1:17:25] | Wording should match Fig 1's new labels for consistency. |

### 3.3 METRIC STANDARDIZATION (text ↔ both figures)

| Item | Action | Requirement | Mandate |
|------|--------|-------------|---------|
| Ratio direction | **CHANGE** | Pick ONE direction and use it everywhere. Paper §3.2 currently defines both `Y_i = S_i/max(D_i,d_0)` (supply/demand) and `DSR_i = D_i/S_i` (demand/supply) — the source of Zhang's confusion. The 3.0× claim = supply/demand ("service per unit demand"). State the direction explicitly wherever 3.0× appears. | [1:07:24]–[1:08:29] |
| Metric name | **CHANGE** | Name the metric backing 3.0× consistently (it is **disparate impact**: `disparate_impact.before 0.33252 → 1/0.33252 = 3.0×`, caption src). Don't leave it unnamed/blanked. | [1:08:29]–[1:09:54] |
| Phrase "taxi service per unit demand" | **KEEP** | Fine phrasing; use identically in Fig 1 in-figure value, caption, abstract, and external-metrics table. | [1:12:54] |

### 3.4 CAPTIONS

| Caption | Action | Requirement | Mandate | Current |
|---------|--------|-------------|---------|---------|
| Fig 1 (`fig:teaser`) | **CHANGE / REMOVE** | Cut to essential info only; move the explanatory load into the (now self-explanatory) figure. Since the 3.0× and district labels move *into* the figure, the caption can shed those sentences. | [0:16:47], [0:17:24] | 5 sentences / ~8 rendered lines |
| Fig 2 (`fig:overview`) | **CHANGE** | Shorten; keep only the most important info (the phase-by-phase walk can lean on the in-panel titles/labels). | [0:16:47], [0:17:24], [0:16:55] | 4 sentences / ~9 rendered lines |
| `\Description` alt-text (both) | **KEEP** | Retain for accessibility (not the visible caption). | — | — |

---

## FORMATTING CHECKLIST (with source quotes)

| Item | Requirement | Quote | Time |
|------|-------------|-------|------|
| ☐ Template | Use the corrected KDD template: **keywords**, not CCS concepts (already applied — KEYWORDS renders on p.1) | "you are removing the CCS concepts part and replace it by keywords" | [0:10:09], done [0:11:22] |
| ☐ In-figure font size | Figure text **slightly smaller than caption/body**; kill `\tiny`-at-5pt | "the text in the figure are like slightly smaller than this caption size or than the text in the paper" | [0:15:10] |
| ☐ Bigger text + smaller gaps | Enlarge glyphs/labels; compact internal spacing | "Bigger text, smaller gaps" | [0:14:08] |
| ☐ Blank space | Remove excess blank space around/after figures; make figures compact | "there are a lot of blank spaces … make figures more compact" | [0:13:18]–[0:13:30] |
| ☐ Caption length | Essential info only, both figures | "we just explain the most important information in the caption" | [0:17:24] |
| ☐ All figures referenced in text | Verify `fig:teaser` **and** `fig:overview` **and every table** are `\ref`'d (tables flagged as the risk) | "tables actually was the big one" | [0:19:02] |
| ☐ Bold figure refs | Decide bold vs not; note the LaTeX limitation (can't cleanly bold "Figure~N" as one unit) | "either you choose to not bold it, or you choose to bold the figure and not the number" | [0:18:31] |
| ☐ Clear main-text refs | Ensure prose around each figure ref reads clearly | "we want to make sure that they are clear" | [0:18:22] |
| ☐ Package conflict | Resolve the TikZ insertion conflict Zhang hit; keep figures as vector TikZ (not screenshots) so CVD/grayscale verification holds | "it doesn't seem to be showing correctly … a package conflict issue" | [0:12:15] |
| ☐ Tighten bulleted lists / template real estate | Pull bullets in to reclaim space | Robert: "if you're familiar with it being okay to pull bullets in and tighten that up, I'm going to do it" | [0:40:30] |

---

## CONFLICTS & RISKS (exposed, not resolved)

- **C1 — Conservation vs the "3× gap made visible" ask.** [1:05:32]/[1:19:22] demand
  equal taxi & passenger totals across before/after; [1:12:04] asks the 3× gap to be
  visible via glyph counts per district. Both are satisfiable *only* by encoding the
  gap in the **distribution across districts** while keeping **panel totals equal** —
  the current figure does neither (unequal totals, no legible ratio). Needs a
  deliberate glyph re-layout.

- **C2 — In-figure metric text vs compactness.** Adding pre/post fairness values and
  district labels [0:53:04] pushes text INTO a figure Zhang simultaneously wants
  smaller and less blank [0:13:30]. Real-estate tradeoff on a column-width figure.

- **C3 — Map background in Fig 1 vs the TikZ grid-abstraction decision.** Zhang wants
  Fig 2's map style in Fig 1 [1:17:25]; figure-1.tex (lines 12-15) *deliberately*
  keeps Fig 1 map-free as "the WHY" abstraction. Adopting the map reverses a
  logged design decision and adds visual load behind 0.45cm cells.

- **C4 — "+ added presence" motif vs relocation/conservation.** The `taxinew`/`picknew`
  "+"-glyphs are coded as "added presence" (figure-1.tex 24, 69, 80). Conservation
  says nothing is added — only moved [1:18:40]. The "+" must be re-cast as
  "relocated," or removed, else it re-tells the rejected "adding service" story.

- **C5 — Removing the imitation-model layer (rejected but on record).** Zhang
  suggested dropping the intermediate model layer [0:57:12]; Robert's defense (it is
  core to the transfer claim) prevailed [0:58:29], [1:03:10]. Keep the layer — flag
  so a later editor doesn't "simplify" it away citing the transcript.

- **C6 — Fig 1 must stand alone vs Robert's "read them in combination."** Robert's
  repeated defense [0:55:55], [1:14:38] is overruled — [1:16:59] each figure must be
  self-explanatory alone. Any fix that still relies on Fig 2 to define the grid fails.

- **C7 — Screenshot workflow vs vector TikZ / CVD verification.** Zhang edits figures
  by screenshot [0:12:54] after hitting a TikZ package conflict [0:12:15]. If figures
  get replaced by screenshots, the vector palette + CVD/grayscale verification
  (figure-1.tex 38-40) is lost and text-size/greyscale guarantees break. Resolve the
  package conflict so figures stay TikZ.

- **C8 — Caption-shorten vs "leave caption as is."** [0:17:24] says shorten to
  essentials; [0:19:16] says "we don't modify the caption … leave it as is." Read
  the latter as *follow template formatting*, but the two statements are in tension —
  confirm intent before deep-cutting captions.

- **C9 — Metric direction ambiguity is a real inconsistency, not just figure polish.**
  §3.2 defines the ratio in *both* directions (`Y_i` supply/demand and `DSR_i`
  demand/supply). Until one is chosen, the figure value, caption, and text can't be
  made consistent [1:07:24]–[1:08:29].

---

## APPENDIX — file/line index

- Fig 1 source: `paper/figures/figure-1/figure-1.tex` (palette 30-48; glyph pics
  53-88; rollout LEFT taxis 183-186 / picks 187-189; RIGHT taxis 201-203, picknew
  205, picks 206-207, taxinew 208-209; legend 222-230).
- Fig 1 caption: `paper/sections/01_introduction.tex` lines 37-62 (3.0× src comment
  41-43); Fig 1 referenced in prose at line 25.
- Fig 2 source: `paper/figures/figure-2/figure-2.tex` (palette 31-35; district
  labels 96-99; panel legends 100-106, 147-155, 215-230).
- Fig 2 caption: `paper/sections/03_methodology.tex` lines 186-214.
- Rendered: `paper/main.pdf` p.1 (Fig 1, top-right col), p.5 (Fig 2, full-width top).
- Metric §: `03_methodology.tex` §3.2 (`Y_i`, `DSR_i`, `ASR_i` definitions).
