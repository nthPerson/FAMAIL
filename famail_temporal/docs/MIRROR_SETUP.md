# Mirror setup: nthPerson/FAMAIL ↔ mkelly-sdsu/famail

This document covers the one-time manual setup needed for the two GitHub
Actions workflows that mirror `famail_temporal/` between Robert's personal
monorepo (`nthPerson/FAMAIL`) and the SDSU lab repo
(`mkelly-sdsu/famail`).

## Architecture at a glance

```
nthPerson/FAMAIL                          mkelly-sdsu/famail
─────────────────                         ──────────────────
famail_temporal/      ── forward sync ──> famail_temporal/   (auto on push)
                      <── backward sync ── famail_temporal/   (manual, opens PR)

(other monorepo files,                   (README.md, famail_proposal.pdf,
 not mirrored)                            langground/, etc. — untouched)
```

- **Forward** (personal → lab): runs automatically on every push to
  `main` that touches `famail_temporal/**`. Force-replaces the lab
  repo's `famail_temporal/` subdirectory using `rsync --delete`. Other
  paths in the lab repo are not modified.
- **Backward** (lab → personal): runs only when you click "Run workflow"
  in the Actions tab. Rsyncs the lab's `famail_temporal/` back into the
  monorepo on a new branch and opens a PR against `main` for review.

The workflow files live in
[`.github/workflows/mirror-famail-temporal-to-lab.yml`](../../.github/workflows/mirror-famail-temporal-to-lab.yml)
and
[`.github/workflows/pull-famail-temporal-from-lab.yml`](../../.github/workflows/pull-famail-temporal-from-lab.yml)
of the monorepo.

---

## One-time setup

### 1. Generate a Personal Access Token (PAT)

The workflows need a token that can write to `mkelly-sdsu/famail`. The
built-in `GITHUB_TOKEN` cannot reach other repos, so a PAT is required.

1. Go to <https://github.com/settings/personal-access-tokens/new>
   (fine-grained tokens — the modern, narrow-scoped variety).
2. **Token name:** `famail-temporal-mirror`
3. **Expiration:** 1 year is a reasonable default. (You can rotate it.)
4. **Resource owner:** select `mkelly-sdsu` (you need org membership, or
   the org owner needs to allow your token to act on their repos).
   - If you're not yet a member of `mkelly-sdsu`, ask the org owner to
     invite you with write access.
   - As an alternative if org-scoped fine-grained tokens are not
     permitted, use a "classic" PAT with `repo` scope, but expect
     broader permissions than necessary.
5. **Repository access:** "Only select repositories" → choose
   `mkelly-sdsu/famail`.
6. **Permissions** (Repository permissions section):
   - **Contents: Read and write** ✓
   - **Metadata: Read-only** ✓ (auto-selected)
   - Everything else can stay disabled.
7. Click **Generate token** and copy the value (it's shown once).

### 2. Store the PAT as a repo secret

1. In the monorepo (`nthPerson/FAMAIL`), navigate to:
   **Settings → Secrets and variables → Actions → New repository secret**
2. **Name:** `LAB_REPO_TOKEN` (exact spelling — the workflows look for
   this name).
3. **Secret:** paste the PAT from step 1.
4. Save.

### 3. (If applicable) Adjust lab-repo branch protection

If `mkelly-sdsu/famail`'s `main` branch has branch protection rules
(required reviews, required status checks, signed commits, etc.), the
forward workflow's automated push will fail. Two options:

- Add a bypass exception for the PAT's owner in the branch-protection
  rule.
- Temporarily relax the rule until the first sync succeeds, then adjust.

To check: in the lab repo, go to **Settings → Rules → Rulesets** (new
GitHub UI) or **Settings → Branches** (older UI). If you see no rules
on `main`, you're fine.

---

## End-to-end test plan

After completing the setup above:

### Test 1 — forward sync (the common case)

1. In the monorepo, make a trivial change to a tracked file under
   `famail_temporal/` (e.g., add a blank line at the end of
   [`famail_temporal/README.md`](../../famail_temporal/README.md)).
2. Commit and push to `main`:

   ```bash
   git add famail_temporal/README.md
   git commit -m "Test: trigger forward mirror"
   git push origin main
   ```

3. Open <https://github.com/nthPerson/FAMAIL/actions>. Within ~30
   seconds you should see a run of **"Mirror famail_temporal/ to lab
   repo"** in flight. Expand it; all four steps (Check out monorepo,
   Clone lab repo, Rsync, Commit and push if changed) should turn
   green.
4. Open <https://github.com/mkelly-sdsu/famail/tree/main/famail_temporal>.
   The trivial change should be visible there, in a commit authored by
   `github-actions[bot]` with the message
   `Sync famail_temporal/ from nthPerson/FAMAIL@<short-sha>`.
5. Confirm that the lab repo's `README.md`, `famail_proposal.pdf`, and
   `langground/` are still present at the lab repo's root (the
   workflow should never touch them).

### Test 2 — forward sync no-op on unrelated change

1. In the monorepo, modify a file OUTSIDE `famail_temporal/`
   (e.g., the root `README.md`). Commit and push.
2. The mirror workflow should NOT fire (no run appears in the Actions
   tab). The `paths:` filter is doing its job.

### Test 3 — backward pull

1. Make a small change directly in the lab repo's `famail_temporal/`
   subdirectory (e.g., via the GitHub web UI: edit a README, commit).
2. In the monorepo, go to
   <https://github.com/nthPerson/FAMAIL/actions/workflows/pull-famail-temporal-from-lab.yml>
   and click **"Run workflow"** → **"Run workflow"** (default branch).
3. The run should complete in ~30 seconds. The "Create branch, commit,
   push, open PR" step should report a created PR URL.
4. Open the Pull Requests tab in the monorepo. You should see a new PR
   titled `Sync famail_temporal/ from lab repo @ <short-sha>` with the
   lab-side change applied.
5. Review the diff and merge (or close, if it was a test).

### Test 4 — backward pull no-op

1. With the lab repo in sync with the monorepo, click **"Run workflow"**
   again on the backward workflow.
2. The run should complete cleanly with the log
   `Lab repo is in sync with monorepo; nothing to do.` and no PR should
   be created.

---

## Operational notes

- **Conflict semantics.** This is a snapshot-only mirror, not a merge.
  If the PI commits to the lab repo and you push to monorepo's `main`
  before pulling those commits back, the next forward sync will
  overwrite the PI's changes. **Run the backward workflow first**
  whenever the PI may have committed.
- **Rotating the PAT.** Fine-grained PATs default to a 1-year
  expiration. When the token expires, both workflows will fail with an
  authentication error. Regenerate the PAT and update the
  `LAB_REPO_TOKEN` secret; no workflow changes needed.
- **Where the workflow files live.** Both workflow YAMLs are at the
  monorepo's `.github/workflows/`. They are NOT inside
  `famail_temporal/`, so they don't get mirrored to the lab repo
  (which is the right thing — the lab repo doesn't need the workflow
  source). This setup doc IS inside `famail_temporal/docs/` and so
  does get mirrored, which is intentional: the PI can read it.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| Workflow fails at "Clone lab repo" with HTTP 401/403 | Token expired, revoked, or wrong repo selected | Regenerate the PAT (step 1) and update the `LAB_REPO_TOKEN` secret (step 2). |
| Workflow fails at "Commit and push" with `protected branch` | Branch protection on lab repo's `main` blocks bot pushes | Adjust the rule (see step 3 above). |
| Forward workflow doesn't fire on push | The push didn't touch `famail_temporal/**` | Confirm the changed paths with `git diff --stat HEAD~1` — only changes under `famail_temporal/` trigger the mirror. |
| Backward workflow runs but no PR appears | `gh pr create` failed silently, or no diff vs. monorepo | Check the step log for `gh` errors; if "no diff," lab is already in sync. |
| Want to dry-run forward without pushing | None — the workflow has no dry-run flag | Trigger via `workflow_dispatch` against a throwaway branch on the lab repo by temporarily editing `LAB_BRANCH` in the workflow file. |
