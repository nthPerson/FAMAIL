# What Does "Fairness" Mean in FAMAIL?

FAMAIL operationalizes fairness through two complementary lenses, each capturing a different aspect of equitable service distribution.

---

## Spatial Fairness: Equal Access to Service

Spatial fairness asks: **Is taxi service distributed equally across the city?**

We measure this using the **Gini coefficient** — the same measure economists use to quantify income inequality — applied to taxi service rates. For each grid cell, we compute a service rate by normalizing the number of pickups (or dropoffs) by the number of active taxis available in that area. This controls for the fact that some areas simply have more taxis present.

<div class="equation-box">

$$G(\mathbf{x}) = \frac{\sum_{i}\sum_{j} |x_i - x_j|}{2n^2 \bar{x}}$$

</div>

A Gini coefficient of 0 represents perfect equality (every cell has the same service rate), while a coefficient of 1 represents maximum inequality (all service is concentrated in a single cell).

**Intuition:** If two neighborhoods have the same number of available taxis, they should have roughly similar pickup rates. Large deviations indicate that some areas are systematically favored over others.

---

## Causal Fairness: Service Driven by Demand, Not Demographics

Causal fairness asks a deeper question: **Is the service distribution explained by legitimate factors (demand) rather than sensitive factors (demographics)?**

It is acceptable — even expected — that areas with higher passenger demand receive more service. What is *not* acceptable is when two areas with the same demand receive different levels of service because one is wealthier than the other.

FAMAIL measures causal fairness by:

1. **Modeling the demand-service relationship:** What service level should each area receive, given its level of demand?
2. **Computing residuals:** How much does the actual service deviate from the demand-predicted level?
3. **Checking for demographic bias:** Do those residuals correlate with neighborhood demographics (income, housing prices, etc.)?

If service deviations are independent of demographics, the system achieves causal fairness — service allocation is driven by demand alone, not by where wealthy or poor residents live.

---

## Why Both?

These two fairness measures are complementary, and neither alone is sufficient:

| Scenario | Spatial Fairness | Causal Fairness | Assessment |
|----------|:----------------:|:---------------:|------------|
| Equal service everywhere, but driven by demographics | High | Low | Not truly fair — equality is coincidental |
| Unequal service, but fully explained by demand | Low | High | Not truly fair — some areas are still underserved |
| Equal service driven by demand | High | High | **Genuinely fair** |

!!! note "The goal"
    True fairness requires both: **equitable distribution** (spatial) that is **driven by legitimate factors** (causal). The [objective function](methodology/objective-function.md) captures both through separate, weighted terms.

---
