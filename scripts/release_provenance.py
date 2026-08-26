"""
release_provenance.py — RELEASE-time artifact provenance.

Not to be confused with src/provenance.py, which is TRAIN-time checkpoint
provenance: that module records what produced one .pt file (dataset hash, prior,
architecture, optimiser settings) and embeds it inside the checkpoint. This one
records what a whole pipeline RUN produced across the output tree, and is what
scripts/export_onnx.py, release_bundle.py and the deploy gate use. The two are
complementary and chain: export_onnx.py reads the checkpoint manifest written by
src/provenance.py and carries it into the site manifest, so a shipped .onnx can be
traced back to the dataset that trained it.

The problem this exists to solve: GRIN's pipeline produces artifacts at wildly
different sizes and lifetimes (a 300 KB .onnx that ships to the website, a 12 KB
JSON that backs a manuscript claim, a 4 GB checkpoint directory that backs
neither). Only the small ones can live in git. That is fine ONLY IF the big ones
can be proven to belong to the same run as the small ones -- otherwise "these
figures are regenerable" is a hope, not a fact, and a reviewer's "where does this
number come from" has no answer.

So: every release run writes ONE run manifest recording the git commit, the
machine, the config, and the sha256 of every artifact the run produced -- tracked
and untracked alike. The untracked bulk can then be dropped or archived off-repo
without losing the ability to say what it was.

No heavy imports at module scope (no torch): the deploy CI and the laptop both
import this without a training environment.
"""
from __future__ import annotations

import fnmatch
import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
from datetime import datetime, timezone

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

SCHEMA = "grin/run-manifest@1"
RUN_MANIFEST_PATH = os.path.join(PROJECT_ROOT, "results", "run_manifest.json")

# --------------------------------------------------------------------------- #
# Tiers.
#
# Tier is a statement about ROLE, not about size or regenerability. Regenerability
# is not a useful axis here -- regenerating anything invalidates everything
# downstream of it, so nothing in this pipeline is actually cheap to redo.
#
#   1  site payload     ships to S3, must be in git, must be integrity-checked
#   2  manuscript       backs a number in the paper, must be in git, small+diffable
#   3  bulk             backs neither; never in git, archived once with a checksum
#
# Order matters: first matching pattern wins, so put the narrow rules first.
# --------------------------------------------------------------------------- #
TIER_RULES: list[tuple[str, int]] = [
    # -- tier 1: the site payload -------------------------------------------
    ("web/assets/models/*/*.onnx", 1),
    ("web/assets/models/*/manifest.json", 1),
    # -- tier 2: manuscript evidence ----------------------------------------
    ("results/validation/*.json", 2),
    ("results/validation/sweeps/*.json", 2),
    ("results/validation/SUMMARY.md", 2),
    ("results/manuscript/*", 2),
    ("results/manuscript/*/*", 2),
    ("results/mle_fits/*.csv", 2),
    ("results/*.json", 2),
    # -- tier 3: bulk --------------------------------------------------------
    ("results/models/*", 3),
    ("results/training_history/*", 3),
    ("results/training_history/*/*", 3),
    ("results/mle_fits/*", 3),
    ("results/figures/*", 3),
    ("data/simulated/*", 3),
    ("data/simulated_grt/*", 3),
    ("data/processed/*", 3),
]

# Scanned for artifacts. Anything under these that matches no rule is reported as
# tier 3 with an "unclassified" flag rather than silently dropped -- a new output
# directory appearing in the pipeline should be visible, not invisible.
SCAN_ROOTS = ["web/assets/models", "results", "data/simulated", "data/simulated_grt"]

SKIP_DIRS = {"_archive", "__pycache__", ".ipynb_checkpoints"}
SKIP_NAMES = {".gitkeep", ".DS_Store", "Thumbs.db"}


def classify(relpath: str) -> tuple[int, bool]:
    """Return (tier, matched). Unmatched files are tier 3 but flagged."""
    rel = relpath.replace(os.sep, "/")
    for pattern, tier in TIER_RULES:
        if fnmatch.fnmatch(rel, pattern):
            return tier, True
    return 3, False


# --------------------------------------------------------------------------- #
# Hashing
#
# Text artifacts are hashed on EOL-NORMALISED content; binaries are hashed raw.
#
# This is not fastidiousness. The lab machine writes CRLF and a Linux checkout
# with core.autocrlf=input has LF, so hashing raw working-tree bytes makes every
# tracked .json and .csv "mismatch" across the two machines while being byte-for-
# byte identical in git. That turns the integrity check into noise, which is worse
# than not having one -- a check that cries wolf gets ignored on the day it is
# right. Normalising makes the hash a statement about content, which is what
# anyone reading it assumes it already was.
#
# Binaries (.onnx, .pt, .npz) are never normalised: a stray CRLF substitution in
# a weights file would be corruption, and must show up as one.
# --------------------------------------------------------------------------- #
TEXT_EXTS = {".json", ".md", ".csv", ".txt", ".tex", ".yaml", ".yml", ".py", ".js", ".r"}


def is_text_artifact(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in TEXT_EXTS


def sha256_file(path: str, _chunk: int = 1 << 20, raw: bool = False) -> str:
    """sha256, lowercase hex. Streams, so multi-GB checkpoints are fine.

    raw=True forces byte-exact hashing even for text -- used to diagnose whether a
    mismatch is a line-ending difference or a real content change.
    """
    if not raw and is_text_artifact(path):
        with open(path, "rb") as fh:
            data = fh.read()
        return hashlib.sha256(data.replace(b"\r\n", b"\n")).hexdigest()
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(_chunk), b""):
            h.update(block)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# --------------------------------------------------------------------------- #
# Environment capture
# --------------------------------------------------------------------------- #
def _run(cmd: list[str], cwd: str = PROJECT_ROOT, strip: bool = True) -> str | None:
    """strip=False matters for `git status --porcelain`: its first two columns are
    the status code and the first is a SPACE for unstaged changes, so stripping the
    output silently shifts every path one character left."""
    try:
        out = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=30)
        if out.returncode != 0:
            return None
        return out.stdout.strip() if strip else out.stdout.rstrip("\n")
    except (OSError, subprocess.SubprocessError):
        return None


def git_state() -> dict:
    """Git provenance. `dirty` is the one that matters: a release built from a
    dirty tree cannot be reproduced from its recorded commit, so the bundler
    refuses it unless explicitly overridden."""
    status = _run(["git", "status", "--porcelain"], strip=False)
    return {
        "commit": _run(["git", "rev-parse", "HEAD"]),
        "short": _run(["git", "rev-parse", "--short", "HEAD"]),
        "branch": _run(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "describe": _run(["git", "describe", "--tags", "--always", "--dirty"]),
        "dirty": bool(status),
        "dirty_files": [ln[3:] for ln in status.splitlines()] if status else [],
        "remote": _run(["git", "remote", "get-url", "origin"]),
    }


def machine_info() -> dict:
    """Records WHERE a run happened. The lab GPU box and the laptop produce
    different numbers from the same seed on some ops; knowing which one made an
    artifact is part of being able to defend it."""
    info: dict[str, object] = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "python": sys.version.split()[0],
    }
    # torch is optional -- the laptop unpack path must work without it
    try:
        import torch  # noqa: PLC0415

        info["torch"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["cuda_version"] = torch.version.cuda
            info["gpu"] = torch.cuda.get_device_name(0)
            info["gpu_count"] = torch.cuda.device_count()
    except Exception as exc:  # pragma: no cover - environment dependent
        info["torch"] = f"unavailable ({type(exc).__name__})"
    return info


def config_snapshot() -> dict:
    """The scalar knobs that define what a run MEANS. If TRIAL_RANGE changed
    between the run that made the checkpoint and the run that made the figures,
    the figures are describing a different model -- and without this snapshot
    there is no way to notice after the fact."""
    try:
        sys.path.insert(0, PROJECT_ROOT)
        from src import config as C  # noqa: PLC0415
    except Exception as exc:
        return {"error": f"could not import src.config ({type(exc).__name__}: {exc})"}

    keys = [
        "N_PER_CLASS", "TRIAL_RANGE", "TRIAL_IMBALANCE", "Z_MAX", "R_MAX",
        "DATA_SEED", "N_PARAMS", "RT_DRIFT_SD", "RT_HIDDEN_LAYERS", "RT_DROPOUT",
        "HIDDEN_LAYERS", "DROPOUT", "LR", "BATCH_SIZE", "EPOCHS", "N_ENSEMBLE",
        "WEIGHT_DECAY", "PATIENCE", "VAL_FRAC", "DEVICE",
    ]
    snap = {}
    for k in keys:
        if hasattr(C, k):
            v = getattr(C, k)
            snap[k] = list(v) if isinstance(v, tuple) else v
    return snap


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# --------------------------------------------------------------------------- #
# Artifact scanning
# --------------------------------------------------------------------------- #
def scan_artifacts(roots: list[str] | None = None, hash_tier3: bool = True) -> list[dict]:
    """Walk the output directories and describe every artifact found.

    hash_tier3=False skips hashing the bulk tier, which is the difference between
    a 5-second scan and a several-minute one on a full run. Use it for a dry run;
    never for the real bundle, since the whole point of the tier-3 entries is that
    the hash outlives the file.
    """
    roots = roots or SCAN_ROOTS
    found: list[dict] = []
    for root in roots:
        abs_root = os.path.join(PROJECT_ROOT, root)
        if not os.path.isdir(abs_root):
            continue
        for dirpath, dirnames, filenames in os.walk(abs_root):
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
            for name in filenames:
                if name in SKIP_NAMES:
                    continue
                # Never record the manifest inside itself. A fresh build never hits
                # this because the manifest is written AFTER the scan, but an amend
                # runs against a tree where it already exists -- and a self-entry can
                # never verify, since writing the file changes the bytes just hashed.
                if os.path.abspath(os.path.join(dirpath, name)) == RUN_MANIFEST_PATH:
                    continue
                abspath = os.path.join(dirpath, name)
                if os.path.islink(abspath) and not os.path.exists(abspath):
                    continue
                rel = os.path.relpath(abspath, PROJECT_ROOT).replace(os.sep, "/")
                tier, matched = classify(rel)
                st = os.stat(abspath)
                entry = {
                    "path": rel,
                    "tier": tier,
                    "bytes": st.st_size,
                    "mtime_utc": datetime.fromtimestamp(
                        st.st_mtime, timezone.utc
                    ).strftime("%Y-%m-%dT%H:%M:%SZ"),
                }
                if not matched:
                    entry["unclassified"] = True
                if tier < 3 or hash_tier3:
                    entry["sha256"] = sha256_file(abspath)
                found.append(entry)
    found.sort(key=lambda e: (e["tier"], e["path"]))
    return found


def build_run_manifest(
    label: str,
    notes: str = "",
    roots: list[str] | None = None,
    hash_tier3: bool = True,
) -> dict:
    git = git_state()
    artifacts = scan_artifacts(roots, hash_tier3=hash_tier3)
    host = socket.gethostname()
    stamp = utc_now()
    by_tier: dict[str, dict] = {}
    for t in (1, 2, 3):
        items = [a for a in artifacts if a["tier"] == t]
        by_tier[str(t)] = {
            "count": len(items),
            "bytes": sum(a["bytes"] for a in items),
        }
    return {
        "schema": SCHEMA,
        "run_id": f"{stamp[:10]}-{host}-{git['short'] or 'nogit'}-{label}",
        "label": label,
        "created_utc": stamp,
        "notes": notes,
        "git": git,
        "machine": machine_info(),
        "config": config_snapshot(),
        "totals": by_tier,
        "artifacts": artifacts,
    }


def write_json(path: str, obj: dict) -> None:
    # newline="\n": Python's text mode writes CRLF on Windows, which would make
    # this file's own bytes differ by platform. Pin it at the source.
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(obj, fh, indent=2, sort_keys=False)
        fh.write("\n")


def load_json(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _matches_any_eol(path: str, expected: str) -> bool:
    """True if the file's content matches `expected` under SOME line-ending
    convention. Old manifests recorded raw working-tree bytes, so the recorded
    hash may be of CRLF content (written on Windows) while this checkout is LF, or
    the reverse. Both directions have to be tried -- checking only one is how the
    first version of this reported 28 identical files as corrupt.
    """
    with open(path, "rb") as fh:
        raw = fh.read()
    lf = raw.replace(b"\r\n", b"\n")
    for candidate in (raw, lf, lf.replace(b"\n", b"\r\n")):
        if hashlib.sha256(candidate).hexdigest() == expected:
            return True
    return False


def verify_against_manifest(
    manifest: dict, tiers: tuple[int, ...] = (1, 2), root: str = PROJECT_ROOT
) -> tuple[list[str], list[str]]:
    """Re-hash the named tiers. Returns (problems, eol_only).

    `problems` empty == every artifact in those tiers has the content the run
    recorded. `eol_only` lists text files whose content matches but whose bytes
    differ by line-ending convention; those are reported, not failed, because
    treating them as corruption is what makes an integrity check get ignored.
    """
    problems: list[str] = []
    eol_only: list[str] = []
    for entry in manifest.get("artifacts", []):
        if entry["tier"] not in tiers:
            continue
        abspath = os.path.join(root, entry["path"])
        if not os.path.isfile(abspath):
            problems.append(f"MISSING   {entry['path']}")
            continue
        if "sha256" not in entry:
            continue
        actual = sha256_file(abspath)
        if actual == entry["sha256"]:
            continue
        # Distinguish "different content" from "same content, different newlines".
        # Manifests written before hashing was normalised recorded raw bytes, so a
        # cross-platform verify of an old manifest lands here for every text file.
        if is_text_artifact(entry["path"]) and _matches_any_eol(abspath, entry["sha256"]):
            eol_only.append(entry["path"])
            continue
        problems.append(
            f"MISMATCH  {entry['path']}\n"
            f"            expected {entry['sha256']}\n"
            f"            actual   {actual}"
        )
    return problems, eol_only


def amend_manifest(path: str, tiers: tuple[int, ...], reason: str,
                   hash_tier3: bool | None = None) -> dict:
    """Re-scan the named tiers of an EXISTING run manifest and record the change.

    Why this exists rather than "just rebuild the manifest": rebuilding requires
    the bulk artifacts to be present, and they live only on the machine that did
    the compute. Rebuilding on a laptop would silently replace a record of 1.5 GB
    of tier-3 output with a record of nothing -- destroying exactly the thing the
    manifest is for.

    So amendments are surgical and, crucially, LOGGED. An `amendments` list records
    what changed, when, on which machine and why. A provenance record that can be
    edited without trace is not a provenance record; one that cannot be edited at
    all just gets abandoned the first time reality moves.
    """
    manifest = load_json(path)
    before = {a["path"]: a.get("sha256") for a in manifest.get("artifacts", []) if a["tier"] in tiers}

    # Hash tier 3 whenever tier 3 is actually being amended. The old unconditional
    # hash_tier3=False meant `--amend --tier 3` REPLACED every tier-3 sha256 with
    # nothing and then reported all 34 of them as "changed" -- destroying the bulk
    # checksums this module exists to preserve, which is the very failure the
    # docstring above warns about. Skipping the hash is only safe when tier 3 is
    # being carried through untouched, and in that case it is not rescanned at all.
    if hash_tier3 is None:
        hash_tier3 = 3 in tiers
    kept = [a for a in manifest.get("artifacts", []) if a["tier"] not in tiers]
    rescanned = [a for a in scan_artifacts(hash_tier3=hash_tier3) if a["tier"] in tiers]

    missing_hash = [a["path"] for a in rescanned if not a.get("sha256")]
    if missing_hash:
        raise SystemExit(
            f"refusing to amend: {len(missing_hash)} rescanned artifact(s) have no "
            f"sha256, so writing them would erase the recorded checksum "
            f"(first: {missing_hash[0]}). Re-run with hashing enabled.")

    after = {a["path"]: a.get("sha256") for a in rescanned}
    added = sorted(set(after) - set(before))
    removed = sorted(set(before) - set(after))
    changed = sorted(k for k in set(before) & set(after) if before[k] != after[k])

    manifest["artifacts"] = sorted(kept + rescanned, key=lambda e: (e["tier"], e["path"]))
    for t in (1, 2, 3):
        items = [a for a in manifest["artifacts"] if a["tier"] == t]
        manifest.setdefault("totals", {})[str(t)] = {
            "count": len(items), "bytes": sum(a["bytes"] for a in items)
        }
    manifest.setdefault("amendments", []).append({
        "utc": utc_now(),
        "reason": reason,
        "tiers": list(tiers),
        "machine": socket.gethostname(),
        "git_commit": git_state()["commit"],
        "added": added,
        "removed": removed,
        "changed": changed,
    })
    write_json(path, manifest)
    return {"added": added, "removed": removed, "changed": changed}


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Inspect GRIN run provenance.")
    ap.add_argument("--scan", action="store_true", help="list artifacts by tier")
    ap.add_argument("--fast", action="store_true", help="skip hashing tier 3")
    ap.add_argument("--verify", metavar="MANIFEST", help="verify tiers 1+2 against a run manifest")
    ap.add_argument("--amend", metavar="MANIFEST",
                    help="re-scan tiers of an existing run manifest and log the amendment")
    ap.add_argument("--tier", type=int, action="append", default=None,
                    help="tier(s) to amend; repeatable (default: 1)")
    ap.add_argument("--reason", default="", help="why the amendment was needed (required)")
    args = ap.parse_args()

    if args.amend:
        if not args.reason:
            raise SystemExit("--amend requires --reason: an unexplained edit to a "
                             "provenance record is worse than no record")
        delta = amend_manifest(args.amend, tuple(args.tier or [1]), args.reason,
                               hash_tier3=None if not args.fast else False)
        for kind in ("added", "removed", "changed"):
            for item in delta[kind]:
                print(f"  {kind:8} {item}")
        if not any(delta.values()):
            print("  nothing changed")
        print(f"\namended {args.amend} (logged under \"amendments\")")
        raise SystemExit(0)

    if args.verify:
        probs, eol = verify_against_manifest(load_json(args.verify))
        if probs:
            print(f"FAIL — {len(probs)} problem(s):")
            for p in probs:
                print("  " + p)
            raise SystemExit(1)
        if eol:
            print(f"OK (with a note) — every tier 1+2 artifact has the recorded CONTENT.")
            print(f"\n  {len(eol)} text file(s) differ only in line endings from what the")
            print("  manifest recorded. That manifest was written on a machine with a")
            print("  different newline convention, before hashing was EOL-normalised.")
            print("  Nothing is corrupt. To clear the note, re-run")
            print("  scripts/release_bundle.py on the machine that holds the run.")
            print(f"\n  e.g. {', '.join(eol[:3])}")
        else:
            print("OK — every tier 1+2 artifact matches the run manifest.")
    else:
        arts = scan_artifacts(hash_tier3=not args.fast)
        for t in (1, 2, 3):
            items = [a for a in arts if a["tier"] == t]
            total = sum(a["bytes"] for a in items)
            print(f"\n--- tier {t}: {len(items)} files, {total/1e6:.1f} MB ---")
            for a in items:
                flag = "  [unclassified]" if a.get("unclassified") else ""
                print(f"  {a['bytes']:>12,}  {a['path']}{flag}")
