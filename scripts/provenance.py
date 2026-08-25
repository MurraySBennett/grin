"""
provenance.py — shared run-provenance helpers.

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
# --------------------------------------------------------------------------- #
def sha256_file(path: str, _chunk: int = 1 << 20) -> str:
    """Streaming sha256, lowercase hex. Handles multi-GB checkpoints."""
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
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, sort_keys=False)
        fh.write("\n")


def load_json(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def verify_against_manifest(
    manifest: dict, tiers: tuple[int, ...] = (1, 2), root: str = PROJECT_ROOT
) -> list[str]:
    """Re-hash the named tiers and return a list of human-readable problems.

    Empty list == every artifact in those tiers is byte-identical to what the run
    recorded. This is the check that makes the whole scheme worth anything.
    """
    problems: list[str] = []
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
        if actual != entry["sha256"]:
            problems.append(
                f"MISMATCH  {entry['path']}\n"
                f"            expected {entry['sha256']}\n"
                f"            actual   {actual}"
            )
    return problems


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Inspect GRIN run provenance.")
    ap.add_argument("--scan", action="store_true", help="list artifacts by tier")
    ap.add_argument("--fast", action="store_true", help="skip hashing tier 3")
    ap.add_argument("--verify", metavar="MANIFEST", help="verify tiers 1+2 against a run manifest")
    args = ap.parse_args()

    if args.verify:
        probs = verify_against_manifest(load_json(args.verify))
        if probs:
            print(f"FAIL — {len(probs)} problem(s):")
            for p in probs:
                print("  " + p)
            raise SystemExit(1)
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
