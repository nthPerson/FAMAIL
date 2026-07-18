# D1 — SF tier-2 distinct-taxi supply recount: design spec (2026-07-17)

**Goal.** Extend the tier-2 recount to San Francisco so §4.7's supply channel stands on the
same distinct-taxi accounting as Shenzhen's — replacing the "tier-1 only, a lower bound"
disclosure and backstopping the Reading-B framing (Robert 2026-07-16: provisional on D1;
fallback to Reading A if D1 disconfirms).

**What exists.** `famail_temporal/analysis/supply_recount.py` implements tier-2 for
Shenzhen: rebuild distinct-taxi presence per (cell-neighborhood, hour) from raw GPS with
the edited seeking tails substituted for their originals, validated by (i) exact
reproduction of the production supply grid on the unedited corpus (MAE 0.0) and (ii) 100%
history→raw sequence matching. `--city sf12` currently writes a deferred stub
("sf ping-path needs new plumbing"). `channel_decomposition.py` already accepts
`--tier2-grid` and is city-agnostic given the bundle.

## Design

1. **Mirror, don't reinterpret.** The SF path reuses the same counting semantics as
   Shenzhen: a taxi is present in a cell-hour if any of its raw pings (with edited tails
   substituted) falls in the cell's 5×5 neighborhood that hour; distinct per taxi. The ONLY
   SF-specific code is the raw-data adapter: Cabspotting ping format (per-taxi files,
   native occupancy flag, lat/lon → the sf12 32×30 grid transform already used by the SF
   pipeline) instead of Shenzhen's plate-keyed format. Grid transform and seeking/occupied
   semantics are imported from the existing SF preprocessing (`second_dataset/` pipeline),
   never re-derived.
2. **Same validation gates, SF-instantiated (both REQUIRED before any result is read):**
   - G-repro: recount of the UNEDITED sf12 corpus reproduces the production
     `active_taxis_3d` grid exactly (MAE 0.0 target; any nonzero MAE = stop and diagnose,
     do not proceed to the edited recount).
   - G-match: 100% of the edited corpus's histories match their raw source sequences
     (same replay identification used on SZ).
3. **Outputs:** `supply_recount.json` + `S_tier2_{before,after}.npz` in the SF edit dir;
   then `channel_decomposition --edit-dir <sf12_filtered> --bootstrap 2000 --seed 0
   --tier2-grid <...>/S_tier2_after.npz` yields the SF tier-2 supply channel with CI.
4. **Decision rule (pre-committed, mirrors Reading-B's provisional status):**
   - Tier-2 supply channel positive with CI excluding 0 → §4.7 upgrades: lower-bound
     disclosure replaced by matched two-tier evidence; Reading B stands.
   - CI includes 0 or negative → REPORT AS MEASURED, surface to Robert immediately,
     and trigger his pre-committed reassessment toward Reading A. No smoothing.
5. **Non-goals:** no editor changes; no change to the SZ path (regression: SZ recount of
   the s10 corpus must reproduce its committed `supply_recount.json` byte-comparably);
   no SF re-editing — the corpus is the committed `2026-07-11T11-31-55_supply_lift_a10_
   sf12_filtered`.

## Risks
- Cabspotting ping density/gaps differ from SZ (the 14.9% raw king-violation finding
  showed SF GPS gaps up to ~18.6 cells) — the adapter must count presence from pings as
  they are, not from interpolated paths, exactly as the production SF supply grid was
  built (G-repro enforces this by construction).
- If G-repro cannot reach MAE 0.0 because the production SF grid was built with a
  convention the adapter can't reproduce, STOP and surface — a recount that doesn't
  reproduce the production grid proves nothing (the SZ gate's whole point).

## Estimate
~1–2 days SDD engineering (adapter + gates + tests), ~1h compute. Runs after the h-chain
drains (GPU) though the recount itself is CPU-heavy and may pair safely.
