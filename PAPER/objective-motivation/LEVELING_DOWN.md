# Leveling-down and demand endogeneity — framing the constraint as a contribution

> **⚠️ TERMINOLOGY NOTE (Meeting 43, 2026-07-16, Dr. Kash):** the MANUSCRIPT no longer uses
> "leveling down" as the *description* of trim-only behavior — pickups are relocated under
> conservation, not removed, so it is "not leveling down in the classic sense." The paper now
> says the mechanism ("redistributes among better-served areas; the under-served group gains
> nothing; the gap closes only from the top") and keeps leveling-down strictly as a cited
> ANALOGY (Parfit/Mittelstadt/Zietlow). This source doc predates that ruling and keeps its
> original vocabulary; read it as analysis provenance, not as manuscript phrasing.

A structural analysis proved that the current editor improves demographic fairness **only by reducing
over-service to advantaged areas, never by lifting service to under-served ones** ("leveling-down"). This doc
supplies the *ethical and fair-ML framing* that turns that property from a liability into a principled,
well-positioned contribution, and connects it to the deeper cause — demand endogeneity. The **empirical proof
and structural mechanism** (attribution selecting over-served cells; the near-inert demand lever on the
under-served side; the frozen-supply constraint; the oracle bound) live in
[`../external-metrics/LEVELING_DOWN_MECHANISM.md`](../external-metrics/LEVELING_DOWN_MECHANISM.md) — this doc
cross-references that analysis rather than restating its numbers. Citations resolve in
[`REFERENCES.md`](REFERENCES.md).

---

## 1. The ethical objection

The **leveling-down objection** to egalitarianism holds that if equality is achieved only by making the
better-off worse off with no gain to anyone, it is hard to see it as an improvement in any respect. It
originates with **Parfit (1997, "Equality and Priority," *Ratio*)** — note the 1991 Lindley Lecture version
carries the distinct title "Equality or Priority?", so the title must be cited to match the edition used.
**Temkin (1993, *Inequality*; 2000, "Equality, Priority, and the Levelling Down Objection")** defends
non-instrumental egalitarianism against it, rejecting the person-affecting "Slogan." Citing both signals that
FAMAIL understands *why* leveling down is normatively troubling — and does not rest its contribution on the
claim that reduction alone is a social good.

## 2. Algorithmic-fairness grounding

The fair-ML literature has formalized the same tension. **Mittelstadt, Wachter & Russell (2024)** show that
many fairness measures cause "levelling down," where fairness is reached by making every group worse off or by
bringing better-performing groups down to the worst-off level, and prescribe "levelling up" by design — for
example minimum-rate constraints / minimum acceptable-harm thresholds. **Zietlow et al. (2022)** show
empirically that common fairness heuristics degrade the worst-off group too, and that a single strategy escaped
it: an adaptive **augmentation** strategy that uniquely improved performance for the disadvantaged group — a
direct parallel to FAMAIL's data-augmentation stance and a pointer to where a lifting-up mechanism should come
from. **Pinzón et al. (2022)** prove that for some distributions a fairness constraint forces accuracy down to
trivial levels — i.e., leveling down can be *constraint-forced*, not an optimizer failure. We use Pinzón et al.
as an **analogy** only; FAMAIL's own oracle/structural bound (in the mechanism doc) is the load-bearing formal
claim about the demand-only editor.

## 3. FAMAIL's position — over-service reduction under a frozen-supply constraint

We present the editor honestly as a principled **over-service-reduction** operator. The leveling-down behavior
is the *constrained optimum* of a demand-only, frozen-supply problem — not an optimizer bug — for the reasons
established structurally in
[`../external-metrics/LEVELING_DOWN_MECHANISM.md`](../external-metrics/LEVELING_DOWN_MECHANISM.md): residual-
variance attribution selects over-served cells; the demand lever is near-inert on the under-served side; and
the real inequity is supply-side while supply is frozen, so the editor can only move pickups (demand). An
oracle bound there shows a demand-only editor could raise the under-served group's mean only by *deleting* its
recorded pickups — a perverse move — so leveling down is the boundary solution the problem geometry permits. We
make this explicit and, following the "levelling up" prescription (Mittelstadt et al. 2024) and Zietlow et
al.'s (2022) augmentation finding, name a **supply-side lever** — editing/augmenting active-taxi supply, not
just demand — as the only mechanism that can raise service to under-served areas, and hence our stated future
direction.

## 4. Demand endogeneity — the unifying thread

The metric-level blind spot and the editor-level leveling-down are the **same phenomenon**, and its root is
**demand endogeneity**. `F_causal` conditions out demand as a *legitimate* factor; but recorded demand
(pickups) in under-served areas is itself suppressed by historical under-supply — latent demand is censored by
the very inequity we aim to measure. This is the feedback-loop pathology documented for other public-allocation
systems: discovered-incident data understates true rates precisely where service/enforcement was historically
concentrated, producing self-reinforcing loops (**Ensign et al. 2018**; **Lum & Isaac 2016**). Conditioning on
an endogenous demand signal can therefore *launder away* real inequity.

The evidence is internal and decisive: the mechanism analysis finds the large majority — about 93% — of
under-served (high-migrant) units sit at or below the demand floor, where the editor sees essentially no
residual and never selects them (see
[`../external-metrics/LEVELING_DOWN_MECHANISM.md`](../external-metrics/LEVELING_DOWN_MECHANISM.md)). So the same
suppressed demand that makes `F_causal` under-detect inequity in poor areas is what makes the demand-only editor
unable to lift them. Recognizing this unifies the story: the honest limitation of the metric, the leveling-down
property of the editor, and the supply-side future work are three faces of demand endogeneity — and confronting
it directly is stronger than framing it away.
