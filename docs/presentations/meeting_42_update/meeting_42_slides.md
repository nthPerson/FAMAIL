# Meeting 42 — Slide Plan (synthesis)

> **This doc is the slide spine.** It sequences the two source briefings into **5 slides**
> (with an optional 6th) telling one story: *we delivered the P0 → found a catch → proved the
> catch is structural → built the lever that fixes it → here's the framing + what I need.*
> Deep numbers/detail live in the two source docs — pull from them per the cross-refs:
> [`external_metrics_results.md`](external_metrics_results.md) (the **result**) and
> [`supply_lift_briefing.md`](supply_lift_briefing.md) (the **fix**).
>
> Each slide below gives: a **title**, the **on-slide content** (terse bullets + the one
> load-bearing table/number), and a **"the point"** line for the presenter (speaker note, not
> on the slide). Arc = **problem → diagnosis → fix**.

---

## Slide 1 — The P0 landed: fairness improves on metrics we never optimized

*(Source: `external_metrics_results.md` §1–3)*

**On the slide**
- Meeting 41's "big one": prove the edit is fairer on **established metrics NOT in the
  objective** — you can't just gradient-ascend these. Measured **before-edit → after-edit**.
- 4 metrics (demographic parity, disparate impact, supply/demand ratio, Theil) on
  `Y = supply/demand` per active `(cell, hour)`; 4 datasets (Shenzhen ×3 feature sets + SF).
- **Shenzhen improves unanimously** — every equity axis, both groupings, Theil — and **every
  Δ 95% CI excludes zero.** Headline cell (migrant, district-extremes, PRIMARY):

| Metric | Before | After | Δ (95% CI) |
|---|---:|---:|---:|
| Disparate impact | 0.3325 | 0.3422 | **+0.0097** [0.0086, 0.0108] |
| Demographic-parity gap | 14.20 | 13.60 | **−0.60** [−0.67, −0.54] |
| Theil (between-region) | 0.155 | 0.149 | **−0.0059** [−0.0065, −0.0052] |

- **Robust across all 3 feature sets** (DI Δ = +0.0097 / +0.0092 / +0.0086): the gain does
  **not** depend on what the editor optimized.

**The point:** the P0 deliverable landed — improvement here can't be an artifact of the
objective. *(SF reproduces the direction but weakly — see Slide 5 honesty box.)*

---

## Slide 2 — The catch: the improvement is *leveling-down*

*(Source: `external_metrics_results.md` §4)*

**On the slide**
- Reporting the group **levels** (not just the gap) exposes **how** the gap closes.
  Shenzhen PRIMARY, district-extremes, mean `Y` before → after:

| Group | Before | After | move |
|---|---:|---:|---|
| **Disadvantaged** (poor, high-migrant) | 7.0734 | 7.0734 | **flat (+0.000)** |
| **Advantaged** | 21.27 | 20.67 | **−0.60** |

- The editor equalizes by **reducing whichever group is over-served — never by raising the
  under-served.** The disadvantaged group's absolute service is **unchanged**.
- The gap closes **from the top**. This is the weak form of fairness — **Parfit's
  leveling-down objection** — and a reviewer *will* raise it.

**The point:** we found this ourselves, and we raise it first. The rest of the talk is what we
did about it.

---

## Slide 3 — Why it levels down: **structural, not a bug** (this is a contribution)

*(Source: `external_metrics_results.md` §5 + `supply_lift_briefing.md` §2)*

**On the slide** — three compounding causes, all design consequences (not bugs):

| Cause | Evidence |
|---|---|
| **Selection never sees the poor group** | **2,455 / 2,455** edits originate *and* land in advantaged cells — zero touch a poor cell (attribution is residual-*variance*-based) |
| **Demand lever is ~inert on the poor side** | Adding demand to rich cells is **~32×** more effective on `Y`; **93%** of poor units sit at/below the demand floor |
| **The real inequity is supply-side — and supply is frozen** | Median taxi presence: poor **1.8** vs rich **17.6** (~10×); the editor moves only demand |

- **Oracle bound:** even a *perfect* demand-only editor could raise the poor group only by
  **deleting ~3k of its recorded pickups** — perverse.
- **Option A** (24 trained policies): the downstream stage doesn't lift up either — upweighted
  policies serve poor areas **~7–10% less** (0/6 seeds, p=.031).

**The point:** leveling-down is the **constrained optimum** of a demand-only editor over frozen
supply — a **proven property of the problem**, not a failure of our optimizer. *That* is the
answer to the Parfit objection, and it directly motivates the fix.

---

## Slide 4 — The fix we built this week: **supply-lift editing**

*(Source: `supply_lift_briefing.md` §3–5)*

**On the slide**
- The only non-perverse lever that raises `Y = S/D` for the under-served is the **numerator —
  supply.** At the demand floor, `ΔY = 2·ΔS`: adding taxi presence to starved cells is the
  honest mechanism.
- **Give the editor a supply channel:** reroute a trajectory's last few *seeking* states with
  its pickup toward starved cells via a **differentiable ΔS**; every step stays king-move-valid;
  the published **trim behavior is preserved bit-for-bit** (lift only fills the leftover budget).
- **Gate G0 — is there even headroom?** A greedy oracle bounds the achievable lift:

| Ceiling on Δmean(Y \| disadvantaged), baseline 7.07 | Value |
|---|---|
| **Supply channel alone** (the honest number) | **+0.786 — 2.6× the +0.3 go-threshold** |

- **G0 PASSED.** Full Shenzhen validation run is **in flight, completes tonight**; a 7-gate
  battery (bit-reproduction, king-move compliance, fidelity stability, lift-up with CI) +
  hard checkpoint review follows.

**The point:** the door the leveling-down result opened, we already walked through — the uplift
lever is **built and validating**, not just proposed.

---

## Slide 5 — The framing this earns + what I need from you

*(Source: both docs' framing sections + `../meeting_prep/MEETING_42_PREP.md` §3–4)*

**On the slide — the paper argument (reframed):**
- FAMAIL = a principled **over-service-reduction ("slack-trimming") fairness editor.** On
  Shenzhen it improves **metrics it never optimized**, unanimously and robustly, and **no
  group's absolute recorded service falls.**
- The leveling-down limit is a **demonstrated property of the demand-only / frozen-supply
  problem** — the contribution *and* the answer to the objection.
- **Supply-lift is the uplift lever** — now being built. Slogan: *"edit to trim over-service;
  reroute to lift under-service."*

**Decisions for Dr. Zhang:**
1. Approve presenting the current result as **over-service reduction**, with supply-lift as the
   uplift lever?
2. State the Pillar-2 downstream gain as **over-service trimming** (Option A), not uplift?
3. Leveling-down / constrained-optimality result → **main body as a contribution**, or limitations?
4. KDD scope — **supply-lift as the headline** (if tonight's gates pass) vs. an additive subsection?
   *(Deadlines: KDD abstract **Jul 19**, paper **Jul 26**.)*

**The point:** we turned a reviewer's attack into a proven problem-property and built the fix in
the same week — I need your calls on framing and scope.

---

### Optional Slide 6 — Split "Decisions for Dr. Zhang" onto its own slide

If a dedicated decision slide is wanted, move the 4 asks above to their own closing slide and
leave Slide 5 as the framing/summary. Recommended only if the meeting is decision-oriented and
you want to dwell on the asks; otherwise keep the 5-slide flow.

---

**Provenance (for Cowork — pull deep numbers/detail from these):**
result + all metric numbers → [`external_metrics_results.md`](external_metrics_results.md);
the fix, G0 oracle, validation gates → [`supply_lift_briefing.md`](supply_lift_briefing.md);
PI briefing + the 4 asks → [`../meeting_prep/MEETING_42_PREP.md`](../meeting_prep/MEETING_42_PREP.md).
Full underlying record: `PAPER/external-metrics/` (`FINDINGS.md`, `LEVELING_DOWN_MECHANISM.md`).
