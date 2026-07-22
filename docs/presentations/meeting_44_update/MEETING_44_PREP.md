# Meeting 44 prep — status update + the Figure-1 decision

*Prepared 2026-07-20 for Robert's meeting with Dr. Zhang (week of 2026-07-20).
New location note: meeting prep now lives in `docs/presentations/<meeting>/`
(earlier meetings used `famail_temporal/baselines/meeting_prep/`).*

**Deadlines:** abstract **submitted 2026-07-19** (KDD 2027 Research Track
Cycle 1, OpenReview; "we can always modify the submission"). Full paper due
**Monday 2026-07-27 23:59**.

---

## 1. What has happened since Meeting 43 (2026-07-16)

**Submission + naming**
- **Abstract re-written and submitted.** Rebuilt to Dr. Zhang's four-beat
  structure, then polished by Robert; single-sourced in
  `paper/sections/00_abstract.tex` (shared by the manuscript and the
  standalone `kdd27-abstract-only.tex`, so the two can never drift).
- **Title:** "Mitigating Demonstration Bias via Fairness-Aware Trajectory
  Editing" (per Dr. Zhang's suggestion to lead with the problem).
- **Method named FATE** (Fairness-Aware Trajectory Editing; replaces the
  FAMAIL working name). **F_causal renamed F_demo** per Dr. Zhang's approval
  (2026-07-20): all prose, equations, tables, and the regenerated frontier
  figure; the associational caveat stays. Code and artifact keys keep
  `f_causal` (mapping recorded in `paper/README.md`).

**Writing**
- **Introduction re-written** to Dr. Zhang's six-beat outline: gap →
  why-existing-methods-fall-short (new intervention-categories paragraph) →
  FATE's position → mechanism summary → results with intervals (DI +0.0162,
  DP gap −0.890 (14.199 → 13.309), Theil −0.0087) → contributions.
  ST-SiameseNet is now cited at first mention.
- **Introduction reference check:** referenced resources verified to exist.
  The two citations added for the categories paragraph (FairGAN, DECAF) are
  the only refs still awaiting full manual verification — flagged P0 in
  `paper/CITATION_PRIORITY_CHECKLIST.md`, promised before submission per
  Dr. Kash's mandate.
- **Related work:** each of the four themes now closes with the concrete
  limitation FATE addresses (Dr. Zhang's "state the contrast" note).

**Figures**
- **Figure 2 (method overview):** glyph vocabulary per Dr. Zhang — passenger
  stick figure = service pickup, car glyph = taxi presence — plus legend
  updates.
- **Figure 1 redesigned** — see §2, the decision item for this meeting.
- **Color refresh applied to both figures** (the "more engaging colors"
  promised in Robert's email): a two-hue system — muted cobalt = added/
  edited (the FATE intervention), muted amber = excess/trimmed, charcoal
  neutrals, and pale amber/blue *regional* tints marking over-/under-served
  areas in both figures' maps. Figure 1's corpus boxes now show real
  GPS-trace renderings (the edited one with its blue slice and "+").
  Verified grayscale-safe and CVD-safe (blue–amber is the colorblind-safe
  axis; simulated deuteranopia/protanopia separation is large; shapes,
  dashes, and "+" marks still carry all semantics without color).

**Experiments closed since Meeting 43** (all landed in §4; no runs pending)
- **SF two-tier supply recount (D1):** counting taxis as *distinct vehicles*
  from raw GPS, the lift-up supply channel is +0.1027 (CI-significant) and
  the tier-2 total is +0.0493 (significantly positive) — the earlier tier-1
  net-negative was a fractional-presence accounting artifact. §4.7 now makes
  the two-tier statement. *Walk-through owed to Dr. Zhang — agenda item.*
- **Flagship n=12 in both cities:** the w30 recovery replicated on twelve
  paired seeds per city (p = .00049, 12/12 positive both cities; SZ
  +0.0297 ± 0.0029, SF +0.0333 ± 0.0050).
- **SF n=12 controls:** random-slice upweighting degrades fairness;
  most-fair-slice selection is positive but ~6× smaller than the edited
  slice — the effect is edit-specific.
- **Penalty-formulation probe:** the fairness-penalty baseline's failure is
  formulation-independent (absolute-value variant tracks the signed one);
  §4.5 records it.

**Manuscript logistics**
- Renders 12pp; ranked cut plan to the 8-page limit is ready
  (`paper/reviews/2026-07-18-ranked-cut-list.md`, wave-by-wave with
  re-measurement; appendix skeleton lands with the first relocation).
- Anonymous sigconf build with real venue metadata (KDD '27, San Jose).

---

## 2. Figure 1: options, tradeoffs, sizes (the decision for this meeting)

Dr. Zhang's request ("only showing c would be sufficient") is **already
executed** — the live manuscript figure is the (c)-only build. The question
for the meeting is whether to keep it or adopt the (b)+(c) variant.

| Option | Content | Size (figure+caption) | Saved vs 3-strip |
|---|---|---|---|
| A — 3-strip (archived) | problem → asset → stakes | 15.4 cm | — |
| B — (b)+(c) counter-proposal | asset → stakes | 12.0 cm | ~9 lines |
| C — (c)-only (**live**) | stakes only | 11.4 cm | ~11 lines |

**Tradeoff in one sentence: B costs only ~0.6 cm ≈ 1–2 text lines more than
C, and it is the only variant in which the paper's defining beat — the data
is valuable human expertise, so FATE *edits* rather than regenerates — has a
visual home.** Both B and C get nearly all their savings from dropping the
city panel (a); C then has to re-introduce labeled corpus chips as arrow
sources, which claws back most of panel (b)'s removal. The page budget does
not hinge on the difference: the cut plan reaches 8.0 pages via appendix
relocations and keeps ≥2 lines of slack for this swap.

**RESOLVED (Robert, 2026-07-21): C stays.** With the 8-page limit binding
hard after the cut campaign, Robert has settled on the live (c)-only figure
("it does the job, and the simplicity is a strength"); the (b)+(c)
counter-proposal is retired from the meeting agenda. The B variant and the
comparison material remain in the repo as design history.

Refinements applied to the live figure since the redesign (also on B where
applicable): an explicit **FATE provenance arrow** from the raw-corpus chip
to the edited-corpus chip (the edited corpus visibly *comes from* the raw
corpus), and a **boxed, centered legend** with a third entry glossing the
accent "+" marks ("changed by the edit").

**Meeting materials** (in `paper/figures/figure-1/design-archive/`):
`comparison-2026-07-20.png` (all three variants side by side, measured sizes
in the headers) · `preview-3strip-final.png` (the archived original) ·
`zhang-meeting-talking-points.md` (the full argument + anticipated
objection).

---

## 3. Other agenda items

1. **Reading-B / D1 acknowledgment** — the SF two-tier framing in §4.7 has
   not yet been walked through with Dr. Zhang (decided and executed
   post-Meeting-43); this meeting is the slot.
2. **Citation verification status** — the two new intro refs are the only
   unverified ones (P0); verification before submission per Dr. Kash's
   Meeting-43 mandate.
3. **Anonymity / artifact repo** — PII scrub + anonymous repo status check
   before the full-paper submission (Meeting-43 workstream).
4. **Colors** — the refreshed palette is already applied to both figures
   (and to both Figure-1 variants, so the A/B/C comparison also previews
   it); confirm Dr. Zhang is happy with the direction.
