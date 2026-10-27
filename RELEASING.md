# Releasing

Three crates version independently (the kernel and the tool forward to the
workspace version, the wire crate has its own), so there is no per-tag version
to read. A release is: bump what changed, push a tag, let
[`.github/workflows/release.yml`](.github/workflows/release.yml) publish
whatever the index does not already have.

```
# bump billion-context + billion-context-tool (workspace version) and/or
# billion-context-wire (its own literal version), update CHANGELOG.md
git tag v0.4.0 && git push origin main --tags
```

The tag is a label for the release, not a version selector. Pushing `v0.4.0`
when only the kernel moved publishes the kernel and reports the other two as
skipped.

## One-time crates.io setup

Trusted Publishing is configured per crate, so this is done three times:
**Crate → Settings → Trusted Publishing → Add → GitHub**

| Field | Value |
|---|---|
| Repository owner | `MoRanYue` |
| Repository name | `billion-context-rs` |
| Workflow filename | `release.yml` |
| Environment | `release` (optional) |

`release` matches the `environment:` in the workflow. Leaving the field empty
also works — crates.io keeps a config with no environment for any exchange —
but if you fill it in it must match exactly, case-insensitively.

After this works, the long-lived `CARGO_REGISTRY_TOKEN` secret is no longer
needed for releases and can be revoked. Two things still need a personal token:
**the first-ever version of a new crate** (crates.io only offers the OIDC
exchange for crates that already exist) and **yanking**.

## What the workflow will not do

Four constraints come from crates.io's own validation
(`src/controllers/trustpub/tokens/exchange/mod.rs`), not from the docs page:

- **Triggers.** `pull_request_target` and `workflow_run` are rejected outright.
  The workflow uses `push` on `v*` tags and `workflow_dispatch`.
- **Thirty minutes.** The exchanged access token lives for thirty minutes, so
  the auth step runs after the test step, immediately before publishing.
- **One exchange per job.** The OIDC JWT carries a `jti` that crates.io records;
  replaying it is an error.
- **Owner ID, not just owner name.** The numeric `repository_owner_id` is
  compared too, captured when the config was created. Renaming the GitHub
  account means recreating every Trusted Publishing config.

## Why the workflow asks the index instead of trusting the exit code

It does not believe `cargo publish`. After each upload it polls
`https://crates.io/api/v1/crates/<crate>/<version>` until that returns 200.

This is not defensive habit. Twice while this workspace was released by hand,
`cargo publish` uploaded successfully and then failed its own
"waiting for availability" phase; twice more, the next crate in dependency order
failed because the index had not caught up. One of those outages lasted about
two and a half hours. The index is the only statement of fact, and a run that is
red in the log but green on the index should be green in the summary.

The same reasoning makes re-runs safe: publishing is idempotent here because
already-present versions are skipped, so a run that died halfway resumes by
finishing the rest.

## Verifying without publishing

```
gh workflow run release.yml -f dry_run=true
```

Packages and verifies all three crates, uploads nothing. The dry run skips the
crates.io authentication step, so it works *before* any Trusted Publisher is
configured — that order is the point: validate the packaging first, then turn on
the credential.
