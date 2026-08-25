# Release runbook — models to the site, numbers to the manuscript

One pipeline run produces artifacts with three different jobs. They are handled
differently, and the difference is about **role**, not size or regenerability
(regenerating anything invalidates everything downstream, so nothing here is
actually cheap to redo):

| tier | what | where it lives | why |
|---|---|---|---|
| 1 | site payload — `web/assets/models/*/{*.onnx,manifest.json}` | **git**, ~1 MB total | ships to S3; must be integrity-checked |
| 2 | manuscript evidence — `results/validation/*.json`, `results/*.json`, `results/mle_fits/*.csv`, `results/manuscript/**` | **git**, small + diffable | backs a number in the paper; `git log` must be able to answer "where does this come from" |
| 3 | bulk — checkpoints, `training_history/`, `data/simulated/`, exploratory `results/figures/` | **never git**; one archive to OSF | backs neither; recorded by hash so it can be dropped |

Git LFS is deliberately not used. The repo is public, so LFS bandwidth is metered
and billable, a plain `git clone` breaks for anyone without the extension, and the
tier-1 payload is about 1 MB — it solves a problem this project does not have.

The tool that makes tier 3 genuinely droppable is `results/run_manifest.json`:
one file per run recording the git commit, the machine, the config, and the
sha256 of **every** artifact including the bulk. Without it, "these figures are
regenerable" is a claim; with it, it is a fact you can check.

---

## Two invariants the tooling now enforces

**1. `.onnx` filenames carry a version.** `.github/workflows/deploy.yaml` ships
`.onnx` with `max-age=31536000, immutable`. CloudFront invalidation clears the CDN
edge but never a visitor's own browser cache. Overwriting `npe_model.onnx` in
place therefore leaves returning visitors silently running the previous release
for up to a year, with no error anywhere. `scripts/export_onnx.py --install`
writes `npe_model.v<version>.onnx` and points the manifest at it;
`grin-model.js` loads `${dir}/${mf.file}`, so nothing else needs to know the name.
CI rejects an unversioned filename.

**2. `artifact_sha256` is computed, never typed.** A hand-copied hash that
silently stops matching its file is worse than no hash, because it looks like
verification. `--install` computes it; the deploy workflow recomputes it and
refuses to upload on a mismatch, and also cross-checks the shipped weights
against `results/run_manifest.json`.

---

## The run

Do the compute-side steps **on the lab GPU machine**, driven from the laptop
through the VS Code tunnel terminal — the artifacts are there, and there is no
reason to move gigabytes to move a manifest. The laptop stays the control plane.

Work on a branch. Pushing the CI gate to `main` while the placeholder manifests
are still in place makes every deploy fail until the real weights land; a branch
lets the tooling and the weights arrive in `main` together, as one green deploy.

### 0. Laptop — publish the tooling to a branch

```bash
git checkout -b release/v1.0.0
git add -A && git commit -m "release: provenance tooling and deploy integrity gate"
git push -u origin release/v1.0.0
```

### 1. Lab machine — pick up the branch

```bash
cd <repo>
git fetch origin && git checkout release/v1.0.0
git status                      # must be clean before exporting; see step 3
```

### 2. Lab machine — export and install the weights

Choose the version once and use it for both models. `--status preview` is
available if the RT model is shipping as explicitly experimental rather than as a
release (see "Deciding the RT model's status" below).

```bash
python scripts/export_onnx.py       --install --version 1.0.0
python scripts/export_onnx.py --rt  --install --version 1.0.0
```

Each prints the installed path, the sha256, the manifest transition, and any
superseded `.onnx` it pruned. The pruning matters: without it every release
doubles the repo's model payload and the S3 `--delete` sync keeps shipping dead
files.

### 3. Lab machine — promote the manuscript figures

`results/figures/` is bulk and stays untracked. The figures that actually appear
in the paper, **and the CSVs they are drawn from**, go in `results/manuscript/`,
which is tracked.

```bash
mkdir -p results/manuscript
cp results/figures/<the ones in the paper>.png results/manuscript/
cp <the tables behind them>.csv               results/manuscript/
```

This is the step that makes the laptop self-sufficient for writing: with a CSV
per figure in git, figures can be redrawn locally with no GPU and no bulk
transfer, and a reviewer's question about a number has a `git log` answer.

### 4. Lab machine — build the run manifest and the archives

```bash
git add -A && git commit -m "release: v1.0.0 weights, validation, manuscript figures"

python scripts/release_bundle.py --version 1.0.0 --label manuscript-final \
    --notes "full manuscript-ready pipeline" --bulk
```

This writes `results/run_manifest.json`, a small tier-1+2 bundle, and (with
`--bulk`) the tier-3 archive. It **refuses to run on a dirty tree** — a release
built from uncommitted code records a commit that does not describe the code that
produced it, which makes every later provenance claim false in a way nothing can
detect afterwards. Commit first; `--allow-dirty` exists only for emergencies and
records the dirty file list in the manifest.

```bash
git add results/run_manifest.json
git commit -m "release: record run manifest" && git push
```

### 5. Lab machine — archive the bulk

Upload `grin-bulk-<run_id>.tar.gz` to OSF from the lab machine (it is going there
at preprint time anyway, so the archive has a home already). Then record the
DOI/URL in `results/run_manifest.json` under a `"bulk_archive"` key and commit
that one-line change. That link is what turns "we deleted 5 GB" into "the 5 GB is
at this DOI and here are its hashes".

### 6. Laptop — pull

Git is the transfer mechanism for tiers 1 and 2; that is the whole point of
keeping them small.

```bash
git checkout release/v1.0.0 && git pull
python scripts/provenance.py --verify results/run_manifest.json
```

That verify re-hashes every tier-1 and tier-2 file and must print `OK`. If the
lab machine could not push (no credentials, restricted network), use the bundle
instead — it carries the same files plus the manifest, and re-verifies every hash
before writing anything:

```bash
python scripts/release_unpack.py ~/Downloads/grin-release-<run_id>.tar.gz --dry-run
python scripts/release_unpack.py ~/Downloads/grin-release-<run_id>.tar.gz
```

---

## Shipping the website

```bash
# laptop
python scripts/provenance.py --verify results/run_manifest.json    # must be OK
git checkout main && git merge --no-ff release/v1.0.0 && git push
```

The push triggers `.github/workflows/deploy.yaml`. Its smoke check runs **before**
any upload and will refuse to deploy if: a manifest names a file not in the
commit, a sha256 disagrees with its file, `provenance_status` is still
`unverified_legacy_checkpoint`, the version is still `0.0.0-preview-*`, the
`.onnx` filename is unversioned, or a tier-1 artifact disagrees with
`results/run_manifest.json`.

Then confirm what the CDN actually serves — a green workflow proves the upload
succeeded, not that a visitor gets the new model:

```bash
python scripts/verify_deploy.py
```

It fetches the live manifest and the live `.onnx`, cache-busted, and hashes the
served bytes against this working tree. Exit 0 means the site is genuinely
current. If it fails right after a green run, wait a minute and retry — the
CloudFront invalidation is asynchronous.

**Sanity check in a real browser afterwards:** open the analyse page in a private
window, run one example, and confirm the numbers are plausible. The hash checks
prove the right bytes shipped; they do not prove the model is good.

---

## Is the manuscript safe to finalise?

Work through this before submitting. Each line is checkable from the laptop.

- [ ] `python scripts/provenance.py --verify results/run_manifest.json` prints OK.
- [ ] `results/run_manifest.json` records a **clean** tree (`git.dirty` is `false`).
      A dirty release means the recorded commit is not the code that ran.
- [ ] `results/validation/SUMMARY.md` has **no FAIL rows**. As of the last commit
      it had two: `v05` (speed) and `v11` (amortized comparison) were both in
      `ERROR`. Confirm the new run cleared them, or that the manuscript does not
      rest on them.
- [ ] Every figure in the paper has its backing CSV in `results/manuscript/`, and
      the figure can be redrawn on the laptop from that CSV alone.
- [ ] Every `\pending{}` in the manuscript is resolved against a number that
      exists in a tracked tier-2 file.
- [ ] `git.commit` in the run manifest is an ancestor of what you tag. Tag it:
      `git tag -a v1.0.0 -m "manuscript submission" && git push --tags`
- [ ] The bulk archive is uploaded and its DOI recorded in the run manifest.
- [ ] `CITATION.cff` version and `setup.py` version match the release (both were
      `0.1.0`).

### Deciding the RT model's status

`docs/dynamic_grt_rt_design.md` gates the RT work behind validation gates 4–8, and
`lab_computer_handoff.md` records that the manuscript's RT framing was explicitly
deferred until those passed — including that the legacy five-recipe LBA generator
and its 84.6% five-way architecture-recovery result were being retired. The
tracked `SUMMARY.md` still describes the **pre-pivot** RT checks (`v14` "5-way
SFT", `v15`, `v16`).

Before finalising, settle which is true of this run:

- **Gates 4–8 passed and the RT model was retrained on the dynamic-GRT
  generator** → export `cmrt` with `--status release`, and update the manuscript's
  RT section off the new numbers. The `v14`–`v16` rows must describe the dynamic
  model, not the retired one.
- **Gates 4–8 did not all pass, or the RT retrain did not happen** → export `cmrt`
  with `--status preview`, keep the site's RT model labelled experimental (CI
  allows `preview` and prints a note), and keep the RT section out of the
  submission as the handoff instructed.

Shipping a `cmrt` marked `release` whose numbers still come from the retired
generator is the one failure mode here that a hash check cannot catch.
