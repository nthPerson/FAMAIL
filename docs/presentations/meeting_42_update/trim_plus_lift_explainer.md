# How Trim+Lift Editing Works — PI Explainer

> Companion to `supply_lift_briefing.md`. This one is scoped to a single question: *how does
> the edited algorithm work now, and why is it structured the way it is?* Ends with a
> one-slide recommendation.

---

## 1. The algorithm at a high level: trim the over-served, then lift the under-served

The editor now improves fairness from **both ends of the service gap**, with a fixed edit
budget (k = 10,000 trajectories) split between two mechanisms:

**Phase 1 — Trim (unchanged from the published editor).** Attribution identifies the
trajectories whose pickups most concentrate service in already-over-served areas — in
practice ~2,455 trajectories, all in advantaged-group cells. Each one's pickup is nudged
(up to 2 cells) by the gradient of the fairness objective, which pulls demand *out of* the
hotspots. This is the mechanism behind all published results. It works — but on its own it
closes the gap by **leveling down**: the over-served lose a little service, the
under-served gain nothing, because trim can only move *demand*, and the under-served
group's problem is missing *supply*.

**Phase 2 — Lift (new).** The remaining budget (~7,545 edits) goes to trajectories whose
final cruising minutes pass *near* under-served areas. For each, the editor reroutes the
**tail** — the last ~4 seeking states plus the pickup — toward the cells where additional
taxi presence most improves fairness. The pickup moves the full offset; earlier tail states
move progressively less (25/50/75/100% taper), blending the detour back into the original
route; a repair step guarantees the edited path is still physically drivable (every step
moves at most one cell — which also fixes a latent inconsistency where the legacy editor
could output trajectories that our own preprocessing would have filtered out). Crucially,
the moved cruising time carries its **supply presence** with it, and that supply is inside
the objective — so the optimizer is directly rewarded for placing taxi presence where the
under-served need it.

**Shared machinery.** Both phases use the same per-trajectory optimizer, the same fairness
objective (F_spatial + F_causal + fidelity, same weights), the same ε = 2 edit ball, and the
same realism (fidelity) check. The difference is what each phase is allowed to move: trim
moves one pickup's demand; lift moves a tail's demand *and* supply. Every edit — trim or
lift — updates the shared running state, so later edits see earlier edits' effects.

One-sentence summary for the room: *"Trim quiets the over-served hotspots by relocating
excess pickups; lift answers the under-served deserts by rerouting drivers' final cruising
minutes into them — one objective, two levers, and for the first time the second lever
raises the group the fairness metrics say is being failed."*

## 2. How supply-gradient attribution works (human version)

Three moves:

**First, ask the map a question.** Take the fairness objective and ask it, at every cell of
the city at every hour: *"if one more taxi were cruising here, right now, how much fairer
would the city's service be?"* One backward pass answers all ~34,500 of those what-ifs
simultaneously. The result is a heat map of the **value of taxi presence** — it glows in
neighborhoods starved for taxis relative to their demand and demographics, and it's dark
(or negative) where taxis are already abundant. Because a cruising taxi is "visible" to the
whole 5×5 neighborhood around it, we blur the map accordingly — standing on a corner counts
for every block within two cells.

**Second, find drivers who almost pass through the glow.** Look at each trajectory's final
approach — the last few minutes of cruising before the pickup. Slide that approach rigidly
by up to two blocks in every direction and read the heat map: does the driver now cruise
through brighter territory, and how much brighter? Ranking all 95,000 trajectories by their
best slide answers: *which drivers were already passing so close to a taxi desert that a
two-block detour would put them inside it?* We're not inventing trips — just bending ones
that nearly went there anyway.

**Third, let the real optimizer make the call.** The ranking only *nominates* candidates to
fill the edit budget. Each nominee then goes through the full editor: the exact detour is
re-derived from the complete objective (including realism, via the fidelity discriminator),
and the final path is discretized with the taper and the drivability repair. No teleporting.

One-liner: **"Compute where an extra taxi-hour helps fairness most; find the drivers whose
pre-pickup cruising already passes nearby; nudge their last few minutes toward it — checking
every nudge is realistic and drivable."**

## 3. "Couldn't you do trim and lift in one pass?" — yes, and here's why we deliberately don't (yet)

This question decomposes into two different "one pass" ideas, and they have different
answers.

**First, a framing correction: execution already *is* one pass.** The runner assembles a
single edit plan (trim entries first, then lift entries filling the remaining budget) and
runs it through one sequential loop over one shared state. What's actually two-fold is
(a) the **selection criterion** and (b) the **optimization mode** per trajectory.

**Could selection be unified?** In principle, yes: one ranking of all 95k trajectories by
predicted objective gain, whatever the mechanism. Today the two scores live in different
currencies — trim uses the published per-unit *attribution of existing unfairness*; lift
uses a *linearized predicted gain* from the supply gradient — so merging them needs a
common calibration that doesn't exist yet. There is also a real ordering subtlety: the lift
heat map is deliberately computed **after** trim runs, so it reflects the post-trim city
(the two mechanisms interact through the shared demand grid). A fully interleaved pass
would need to periodically recompute the supply gradient as the state evolves — machinery,
not physics; doable, just not free.

**Could the optimization mode be unified?** Also yes — and this is the more interesting
answer for a PI. The "trim mode" restriction (only the pickup moves; supply stays frozen in
its objective) is **not an algorithmic necessity**. We could run every selected trajectory
in lift mode and let the gradient decide each edit's character — some edits would push
demand out of hotspots, some would pull supply into deserts, some would do both at once.
That is arguably the more elegant algorithm.

**Why we deliberately don't (this cycle):** the current two-phase structure is a
*scientific control*, not a technical limitation.

1. **Bit-level reproducibility of published results.** A hard invariant of this build is
   that trim's optimization is byte-identical to the published editor — the combined run
   provably reproduces the exact 2,455 published trim edits and their F_causal trajectory.
   Making supply endogenous for those trajectories would change their gradients and break
   that reproduction.
2. **A clean ablation.** Because trim is frozen, "trim-only" vs "trim+lift" is a pure
   ablation: every delta between the rows is attributable to the new mechanism. Unify the
   modes and the paper loses its cleanest causal claim about what lift contributes.
3. **Risk containment on a deadline.** The new mechanism is quarantined from the one that
   produced a year of results; every validation gate can bisect cleanly between them.

**Bottom line / recommended phrasing for the PI:** *"One-pass unified editing is possible
and is the natural v2 — but this cycle, freezing trim is what lets us (a) reproduce the
published numbers exactly inside the combined run and (b) attribute every improvement
cleanly to the new mechanism. Unification is future work with a real upside (mixed-motive
edits), not a gap in the current results."* This is **not** an "address asap" issue — the
two-phase design is load-bearing for the paper's validation story — but the unified version
is worth one line in future work.

## 4. Single-slide recommendation

**Slide title:** *Trim + Lift: closing the service gap from both ends*

**Layout:** visualization strip across the top two-thirds, 3–4 bullets underneath, one
footnote line. Keep §3 entirely in your back pocket as a speaker note — don't spend slide
real estate on it.

**Visualization (top):** a three-panel horizontal strip on a shared stylized city grid —
simple enough to sketch in PowerPoint shapes, or hand the spec below to Cowork:

- **Panel 1 — "The gap":** city grid with a dark **over-served** cluster (many taxi dots,
  few pickup marks) and a bright/hatched **under-served** area (many pickup marks, almost
  no taxi dots). Label the two regions.
- **Panel 2 — "Trim (published)":** same grid; 2–3 pickup marks in the over-served cluster
  with short arrows nudging them outward; caption *"~2.5k pickups moved out of hotspots —
  levels down."*
- **Panel 3 — "Lift (new)":** same grid with a soft heat-map glow over the under-served
  area; one trajectory drawn as a dotted polyline whose **last 4 segments bend** into the
  glow (arrowheads shrinking along the taper), pickup endpoint landing inside; caption
  *"~7.5k final approaches rerouted into taxi deserts — lifts up."*

If only one panel fits, keep Panel 3 and put Panel 2's caption in a bullet.

**Bullets (suggested language, pick 3–4):**

- **Trim (published mechanism, unchanged):** relocates ~2,455 excess pickups out of
  over-served hotspots — improves fairness, but by *leveling down*.
- **Lift (new):** reroutes the last ~4 cruising minutes of ~7,545 trajectories into
  under-served cells — adds real taxi *presence* where a one-shot gradient says it's most
  valuable, *lifting up* the group the metrics say is underserved.
- **One objective, two levers:** same optimizer and realism check for every edit; supply is
  now differentiable, so the editor is rewarded for *providing* service, not just
  redistributing demand.
- **Every edit stays physically drivable:** tapered reroute + repair guarantees no
  teleporting (max one cell per step) — closing a consistency gap the old editor had.

**Footnote line (small, bottom):** *Trim is kept bit-identical to the published editor
inside the combined run — so "trim-only vs trim+lift" is a clean ablation and published
results reproduce exactly.*

**Anticipated-question speaker note (for §3):** "Could this be one pass? Yes — and it's the
natural v2. We froze trim this cycle so the published numbers reproduce bit-for-bit inside
the combined run and every improvement is cleanly attributable to the new mechanism.
Unifying the modes is future work with a real upside, not a limitation of these results."
