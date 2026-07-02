# FAMAIL — Group Update Slides (content spec)

> **For the presentation agent:** build one slide per `## Slide` section below, in order.
> Text under **ON SLIDE** is written to be used **verbatim** (or lightly trimmed) — it is the slide's
> actual content. **NOTES** are speaker/context notes (do NOT put on the slide unless a "presenter
> notes" field exists). **VISUAL** names an existing image to place on the slide (use the path as-is;
> do not regenerate). Keep slides visually light — a headline, a few bullets, and the key numbers.
> Target: a ~15-minute talk, 5 content slides between a title and a closing overview. Audience is a
> technical research group that does **not** know this project's internals — keep jargon minimal.
>
> **One term to define once (put a one-line gloss the first time it appears):**
> *F_causal = our demographic-fairness score; 1 = perfectly fair (demographics explain none of the
> service gap). Higher is fairer.*

---

## Title slide

**ON SLIDE**
- Title: **FAMAIL — Making Fair Taxi-Service Data, and Making It Stick**
- Subtitle: Progress update since our last review
- Footer: Group research update · 2026-07-02

**NOTES**
FAMAIL = a fairness-oriented **data-augmentation** method for taxi-demand data. One-sentence hook to
open with: *"Last time we had a promising result on one dataset — since then we've made it robust:
cleaner data, three fairness lenses, a control experiment, and a second city."*

---

## Slide 1 — Roadmap: what we set out to do → what we delivered

**ON SLIDE**
Headline: **Since the last review: four commitments, four checkmarks**

| We said we'd… | Status |
|---|---|
| Clean a data-quality artifact and re-run everything | ✅ Done |
| Check the result holds across different fairness lenses | ✅ Done — 3 lenses |
| Prove the gain comes from *editing*, not cherry-picking | ✅ Done — control experiment |
| Validate on a second city | ✅ Done — San Francisco |

Bottom line: **no new claims — the same result, made robust.**

**NOTES**
This is the jumping-off point / roadmap; each row is one of the next four slides. Say: *"The headline
hasn't changed; what changed is how hard we've stress-tested it."* (F_causal gloss goes here, first
mention.)

---

## Slide 2 — #1: We cleaned the data and re-based the numbers

**ON SLIDE**
Headline: **A GPS artifact was inflating nothing important — but we removed it anyway**

- Found **10 "stuck-GPS" cells** across **9 drivers** — parked meters emitting phantom pickups
  (**106,677** fake pickup events), not real demand.
- Filtered them out and **re-ran the entire pipeline** on the cleaned data.
- Numbers shift slightly; **conclusions are identical.** Fairest lens, cleaned data:

| Fairness score (F_causal, 1 = fairest) | before edit | after edit |
|---|---|---|
| Shenzhen (primary) | 0.7988 | **0.8132**  (**+0.0144**) |

Takeaway: **cleaner inputs, same story — editing makes the data measurably fairer.**

**VISUAL** `PAPER/shared_cleanup/figures/sink_spatial_attr_before_after.png`
(spatial map, sinks circled, before vs after)

**NOTES**
"Re-based" = these numbers replace the ones from the last deck (old raw ~0.808 / edited ~0.818 on the
pre-cleaning data and an earlier fairness lens). The point for the audience: *we improved data hygiene
and re-picked the fairness lens for construct validity; the result didn't depend on either.* Don't dwell
on the old numbers.

---

## Slide 3 — #2: The result holds across three fairness lenses

**ON SLIDE**
Headline: **Not an artifact of one definition of "fair"**

- "Fairness" depends on *which* demographics you measure against. We tested **three lenses**:
  - housing + income + migrant share  ← **primary**
  - housing + GDP + income
  - housing + income + migrant + population density
- **Every conclusion reproduces in all three.** Only the absolute scale shifts:

| Lens | fairness gain from editing (Δ F_causal) |
|---|---|
| primary (housing, income, migrant) | **+0.0144** |
| housing, GDP, income | +0.0124 |
| + population density | +0.0156 |

Takeaway: **the fairness gain is robust — it's not cherry-picked to a favorable definition.**

**VISUAL** `PAPER/feature_selection/figures/fig_feature_robustness.png`
(3-way comparison / dumbbell)

**NOTES**
Reassure a skeptic: the primary lens is **not** the one with the lowest baseline, so we're not picking
the definition that maximizes apparent unfairness. Keep it qualitative — "same direction everywhere."

---

## Slide 4 — #3: It's the *editing* that helps — not selection, not oversampling

**ON SLIDE**
Headline: **A control experiment: edit ≫ select > random**

- The method upweights the edited trips during training. Skeptic's question: *would upweighting
  **any** small subset do this?*
- Equal-size control arms, same training:

| What we upweight | Fairness gain |
|---|---|
| the **edited** trips | **+0.0311** (6 / 6 runs positive) |
| the already-fairest existing trips (*select*) | +0.0004 — **no effect** |
| a random subset (*placebo*) | ≈ 0 — **no effect** |

- Editing beats merely selecting fair trips by **~70×**.

Takeaway: **the fairness comes specifically from the edits — you can't get it by picking or padding.**

**VISUAL** `PAPER/by_feature_set/housing-comp-migrant/figures/fig_dose_response.png`
(edited arm rising with weight; control arms flat)

**NOTES**
This is the most persuasive single slide — it rules out the obvious "you just oversampled" objection.
"6/6 runs positive" is the honest strength signal; skip p-values for this audience.

---

## Slide 5 — #5: A second city reproduces everything

**ON SLIDE**
Headline: **San Francisco: same method, no changes — same story**

- Independent city (SF taxi data + US Census), **zero algorithm changes.**
- Editing makes it **fairer *and* keeps it realistic**:
  - fairness F_causal **0.8752 → 0.8891** (**+0.0139**)
  - realism score **0.968** — edited trips still read as the same driver
- The full two-pillar result reproduces, and the control experiment is **even sharper** here
  (both control arms actually *hurt* fairness; upweighting the edits recovers **+0.0387**).
- Honest note: one supporting baseline (a GAN) behaved differently on SF — reported openly; it doesn't
  affect the main claim.

Takeaway: **external validity — the result isn't a quirk of one city.**

**VISUAL** `PAPER/second-dataset/figures/sf_supply_demand.png`
(SF supply/demand regime)

**NOTES**
"Realism score" = the identity-discriminator's same-driver probability; the edit barely moves it. If
asked why SF isn't directly comparable in absolute terms: the fairness score is city-specific and uses
Census proxies, so we report SF as *reproducing* the result, not beating it.

---

## Closing slide — The paper argument at 30,000 feet

**ON SLIDE**
Headline: **FAMAIL in one picture: edit a little, weight it, and fairness propagates**

Two pillars:
1. **Edit, don't generate.** Editing a small slice of real trips yields the **fairest data that is
   still realistic** — better than generating synthetic trips (which drift from real behavior).
2. **Make it stick.** Plain training averages the edit away; **upweighting the edited trips recovers
   the fairness** — and we proved that gain is edit-specific.

**A fairness-oriented data-augmentation method, validated on two cities.**

Takeaway: **the contribution is a simple, general recipe — edit the unfair slice, upweight it in
training — with the evidence to back it.**

**NOTES**
This is the one-slide summary to leave on screen for Q&A. If a 2nd overview slide is wanted, split into
"the problem" (imitation models inherit demographic service bias) + "our recipe" (these two pillars).
Optional closing line: *"Next: theory + the paper draft for the KDD deadline."*

**VISUAL (optional)** `PAPER/by_feature_set/housing-comp-migrant/figures/fig_l1_data_quality.png`
(edited = fairest faithful source, four-way comparison) — only if a supporting visual is wanted.
