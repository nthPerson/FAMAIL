# ⚠️ DEPRECATED: Quality Term

**Status**: DEPRECATED as of January 12, 2026
**Reason**: Functionality overlap with Trajectory Fidelity Term ($F_{\text{fidelity}}$)

---

## Decision Summary

During the FAMAIL Meeting 18 (January 2026), the team decided to **eliminate the Quality Term ($F_{\text{quality}}$)** from the objective function.

### Rationale

The Quality Term was originally intended to measure trajectory quality metrics such as:
- Route efficiency
- Temporal consistency
- Spatial coherence
- Realistic speed profiles

However, upon review, it was determined that **these aspects overlap significantly with the Trajectory Fidelity Term**:

1. **The Fidelity Term's discriminator already validates trajectory realism**
   - The discriminator is trained to distinguish real from synthetic/edited trajectories
   - It implicitly captures route efficiency, temporal patterns, and spatial coherence
   
2. **Redundant validation**
   - Having both terms would effectively double-weight trajectory realism
   - This could skew optimization away from fairness objectives

3. **Simpler objective function**
   - Reducing from 4 terms to 3 terms simplifies the optimization landscape
   - Fewer hyperparameters to tune ($\alpha_4$ eliminated)

---

## Updated Objective Function

The FAMAIL objective function is now:

$$
\mathcal{L} = \alpha_1 F_{\text{causal}} + \alpha_2 F_{\text{spatial}} + \alpha_3 F_{\text{fidelity}}
$$

Where:
- $F_{\text{causal}}$: Causal Fairness (demand-service alignment)
- $F_{\text{spatial}}$: Spatial Fairness (Gini-based distribution equality)
- $F_{\text{fidelity}}$: Trajectory Fidelity (discriminator-based realism)

---

## Migration Notes

- Do not import from `objective_function.quality`
- All quality-related validation is now handled by the Fidelity Term
- See `fidelity/DEVELOPMENT_PLAN.md` for the trajectory modification algorithm where fidelity serves as a validation gate

---

## File Status

The `DEVELOPMENT_PLAN.md` in this directory is retained for historical reference only. It will not be implemented.

---

*Deprecated: January 12, 2026*
*Decision made in: FAMAIL Meeting 18*
