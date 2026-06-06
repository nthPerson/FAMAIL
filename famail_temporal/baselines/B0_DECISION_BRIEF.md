# B0 generator: decision brief

**For:** Dr. Xin Zhang **From:** (RA) **Date:** 2026-05-29
**Subject:** Switching the model-level baseline generator from the adversarial GAN to its maximum-likelihood (MLE) stage, and a metric caveat that affects the model-level headline.

---

## TL;DR (the decision)

The model-level baselines train **one generator architecture on different data** (raw → B0, FAMAIL-edited → FAMAIL, filtered → B2); the headline is *"a model trained on edited data produces fairer rollouts than one trained on raw data."* The planned generator recipe was **MLE pretrain → adversarial (GAN) fine-tune**.

The **adversarial fine-tune reproducibly collapses** and degrades the generator. The **MLE-pretrained generator alone is stable, faithful, and is all the model-level claim actually needs.** We are therefore:

1. Making **B0/FAMAIL/B2 use the MLE generator** (adversarial fine-tune off), identically across variants.
2. **Demoting the adversarial / pure-Gumbel GAN to the secondary "amplification/instability" ablation** the design spec already reserved for it — where the collapse is a *result to report*, not a bug to fix.
3. **Reframing the suite** from "GAN baselines" to "learned generative trajectory baselines" (the GAN remains a named ablation).

This is consistent with the spec, which already calls MLE-pretrain the load-bearing first stage ("keep the generator faithful so the small editing signal isn't drowned by mode-collapse noise") and the pure-GAN a *secondary* ablation.

---

## What we observed (5 GPU runs)

- **MLE pretrain works:** next-token loss 1.95 → 0.78; the MLE generator free-runs to ~18 tokens, matching the real mean (18.2), and reproduces corpus fairness.
- **The adversarial fine-tune always collapses:** the discriminator dominates — generator loss `g` explodes (5 → 9+), discriminator loss `d` pins at the label-smoothing floor (~0.326), and the generator's free-running rollouts **blow up to ~49–55 tokens** (vs ~18) even though teacher-forced loss stays low (~0.8).
- **Five stabilizers, none fixed it:** one-sided label smoothing, gradient clipping, slowing the discriminator (update every 3rd batch + lower LR), MLE-regularization of the generator loss, and capping the *training* rollout length at 24 (which only masked the symptom — generation still ran to 49).

## Root cause (diagnosed, not guessed)

Two compounding mechanisms:
1. **A length "leak."** The discriminator reads its real/fake decision off the *last step* of each sequence. Real sequences end at ~18; the generator's free-running sequences run much longer, so the discriminator separates real from fake on **length alone** — a trivial, unbeatable shortcut. (This is why slowing/regularizing the discriminator didn't help: the leak is in *what it looks at*, not how fast it learns.)
2. **Exposure bias.** The teacher-forced losses (the MLE loss and the MLE-regularizer) only constrain the model on *real* prefixes; they don't constrain free-running generation, so the adversarial gradient is free to push the generator into never-stopping, over-long sequences. The reproducible collapse on discrete sequences is itself the kind of "GANs are unstable / have no fairness mechanism" evidence the B1 baseline is meant to make.

## Why MLE-only is sound, not capitulation

- The headline is a **data contrast**, identified by swapping the training corpus. The generator only needs to be (a) faithful enough to transmit the data's fairness signal and (b) **identical across B0/FAMAIL/B2**. MLE-only satisfies both; the collapsing adversarial satisfies neither.
- An MLE-trained autoregressive model **is a legitimate generative baseline** (it is the dominant generative paradigm for sequences); reviewers accept it routinely.
- We lose nothing for the B0/FAMAIL/B2 headline; the GAN's distinctive contribution (mode-collapse *amplification*) is the spec's explicitly-secondary "bonus" claim and is kept as an ablation.

---

## Important caveat (needs your read): the fairness metric barely tests the generator

Our fairness metrics (F_causal, F_spatial) are computed from the **pickup demand grid, which is built only from each rollout's terminal cell**. The rest of the trajectory is discarded. Two consequences:

1. **"F_causal stays ~0.805 regardless of the collapse" is not reassurance — it shows the metric is insensitive to generator quality.** A model that emitted one cell and stopped would score the same.
2. For the **model-level headline (B0 vs FAMAIL)**, the test reduces to: *can the LSTM reproduce a ~1% shift in the marginal distribution of a single token?* The data-level edit moves F_causal by only **+0.0087** (or **+0.0128** for the no-dedup k=10000 edit), and it relocates the terminal pickup of only ~1,186 / ~105k trajectories (~1.1%). After passing through MLE smoothing + sampling, the model-level delta **may be at or below noise.** This is a weakness of the model-level design itself — **independent of the GAN-vs-MLE choice.**

**Proposed mitigations before we trust the model-level headline:**
- A **transmission-fidelity check** — JS divergence between the generated and target terminal-cell distributions (raw vs edited, paired) — to *prove* the ~1% edit survives the generator.
- Report a **disparate-impact (DI) ratio + a localized within-edited-units metric** (spec §3.3): larger dynamic range than F_causal's r², so a real effect is more detectable.
- Keep the **data-level Pareto as the documented fallback**: if the model-level transfer is null, that is itself a reportable finding (and the data-level result — editing dominates filtering — stands on its own).

---

## Results — MLE-only B0 (real corpus, 2026-05-29)

_Run: `--mle-epochs 5 --adv-epochs 0` on the full corpus (104,638 trajectories after the length filter). Total wall-clock ≈ 2m14s; MLE loss 1.95 → 0.78._

| Metric | Generated (MLE-only B0) | Corpus (reference) | Δ |
|---|---|---|---|
| **F_causal** | 0.8080 | 0.8052 | +0.0028 |
| **F_spatial** | 0.0837 | 0.0822 | +0.0015 |
| gini_dsr | 0.9353 | 0.9384 | −0.0031 |
| gini_asr | 0.8973 | 0.8973 | 0 (supply-side) |
| **mean generated rollout length** | **18.7** | ~18.2 (real) | ≈ match |

**Read:** the MLE-only generator reproduces the corpus's (un)fairness — **B0's "bias propagates" claim holds** — and, critically, it **free-runs to a realistic length (18.7 ≈ real 18.2)**, confirming a faithful generator. Contrast the collapsed adversarial run, which scored a near-identical `F_causal = 0.8086` *but free-ran to length ~49* — the same fairness number from a degraded generator, which is exactly the metric-blindness caveat above. Generation of all 104,638 rollouts took **~17 seconds** (batched), vs the ~1h47m the earlier per-trajectory loop projected.

**Bottom line:** MLE-only B0 gives the same headline number as the adversarial run, from a *faithful, stable, fast, reproducible* generator — vindicating the pivot.

---

## Recommendation & next steps

1. **Adopt MLE-only B0** across all model-level variants (already supported; no algorithm change).
2. **Add the transmission-fidelity (JS) + DI-ratio/localized metrics** before reporting the model-level headline.
3. **Demote the adversarial/pure-GAN to the amplification+instability ablation** and **reframe the suite** in the spec/paper ("learned generative trajectory baselines"); keep **B1** (the differentiable-fairness-loss model) as the home for differentiable generation.
4. **Keep the data-level Pareto as the fallback headline.**

**What I'd like your input on:** (a) the reframing of the primary training paradigm (MLE primary, GAN as ablation), and (b) the metric caveat — whether the JS + DI-ratio additions are sufficient to defend a model-level claim, or whether we lead with the data-level Pareto and present model-level as supporting evidence.

*Note: none of this touches the trajectory-editing algorithm itself — only the baseline generator's training recipe and the reporting metrics.*
