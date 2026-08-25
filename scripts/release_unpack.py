"""
release_unpack.py — install a release bundle into the working tree. RUN THIS ON
THE LAPTOP.

    python scripts/release_unpack.py ~/Downloads/grin-release-<run_id>.tar.gz
    python scripts/release_unpack.py <bundle> --dry-run     # show, change nothing

Every file is re-hashed after extraction and checked against the sha256 the lab
machine recorded in results/run_manifest.json inside the bundle. A mismatch aborts
before anything touches the working tree -- a half-installed release, where the
.onnx is new but the manifest is old, is exactly the failure the versioned
filenames and CI hash gate exist to prevent, and it should not be reachable by
accident here either.

Nothing is deleted. Files present in the tree but absent from the bundle are
reported, not removed; that decision stays yours, since it is how a stale figure
from a previous run gets noticed rather than silently overwritten.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import tarfile
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from provenance import PROJECT_ROOT, load_json, sha256_file  # noqa: E402

MANIFEST_ARCNAME = "results/run_manifest.json"


def _fmt(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{int(n)} B"
        n /= 1024.0
    return f"{n:.1f} GB"


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("bundle", help="path to grin-release-<run_id>.tar.gz")
    ap.add_argument("--dry-run", action="store_true", help="report only, write nothing")
    ap.add_argument("--tiers", default="1,2", help="which tiers to install (default 1,2)")
    args = ap.parse_args()

    if not os.path.isfile(args.bundle):
        print(f"no such bundle: {args.bundle}")
        return 1
    tiers = {int(t) for t in args.tiers.split(",") if t.strip()}

    print(f"bundle   {args.bundle}")
    print(f"         {_fmt(os.path.getsize(args.bundle))}   sha256 {sha256_file(args.bundle)}")

    tmp = tempfile.mkdtemp(prefix="grin-unpack-")
    try:
        with tarfile.open(args.bundle, "r:gz") as tar:
            # filter="data" rejects absolute paths, .. traversal, links, and
            # device nodes. Without it a tarball can write anywhere on disk.
            tar.extractall(tmp, filter="data")

        mf_path = os.path.join(tmp, MANIFEST_ARCNAME)
        if not os.path.isfile(mf_path):
            print(f"bundle has no {MANIFEST_ARCNAME} — not a GRIN release bundle")
            return 1
        mf = load_json(mf_path)

        git, machine = mf.get("git", {}), mf.get("machine", {})
        print(f"\nrun_id   {mf.get('run_id')}")
        print(f"version  {mf.get('release_version', '?')}   label {mf.get('label', '?')}")
        print(f"built    {mf.get('created_utc')} on {machine.get('hostname')}"
              f"  gpu={machine.get('gpu', 'n/a')}")
        print(f"commit   {git.get('short')} ({git.get('branch')})"
              f"{'  DIRTY' if git.get('dirty') else ''}")
        if mf.get("notes"):
            print(f"notes    {mf['notes']}")
        if mf.get("WARNING"):
            print(f"\n  ! {mf['WARNING']}")

        # -- verify every payload file against the manifest ------------------
        wanted = [a for a in mf.get("artifacts", []) if a["tier"] in tiers]
        problems, plan = [], []
        for entry in wanted:
            src = os.path.join(tmp, entry["path"])
            if not os.path.isfile(src):
                problems.append(f"MISSING FROM BUNDLE  {entry['path']}")
                continue
            if "sha256" in entry:
                actual = sha256_file(src)
                if actual != entry["sha256"]:
                    problems.append(
                        f"CORRUPT  {entry['path']}\n"
                        f"           manifest {entry['sha256']}\n"
                        f"           bundle   {actual}"
                    )
                    continue
            dest = os.path.join(PROJECT_ROOT, entry["path"])
            if not os.path.exists(dest):
                state = "new"
            elif sha256_file(dest) == entry.get("sha256"):
                state = "same"
            else:
                state = "changed"
            plan.append((state, entry))

        if problems:
            print(f"\nFAIL — {len(problems)} integrity problem(s), nothing installed:")
            for p in problems:
                print("  " + p)
            return 1
        print(f"\nverified {len(plan)} file(s) against the run manifest — all match")

        by_state = {s: [e for st, e in plan if st == s] for s in ("new", "changed", "same")}
        for state in ("new", "changed", "same"):
            items = by_state[state]
            if not items:
                continue
            print(f"\n  {state} ({len(items)}):")
            for e in items[:25] if state != "same" else items[:5]:
                print(f"      tier {e['tier']}  {_fmt(e['bytes']):>9}  {e['path']}")
            shown = 25 if state != "same" else 5
            if len(items) > shown:
                print(f"      ... and {len(items) - shown} more")

        if args.dry_run:
            print("\n--dry-run: nothing written.")
            return 0

        installed = 0
        for state, entry in plan:
            if state == "same":
                continue
            dest = os.path.join(PROJECT_ROOT, entry["path"])
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            shutil.copy2(os.path.join(tmp, entry["path"]), dest)
            installed += 1
        shutil.copy2(mf_path, os.path.join(PROJECT_ROOT, MANIFEST_ARCNAME))
        print(f"\ninstalled {installed} file(s) + {MANIFEST_ARCNAME}")

        # Superseded weights: the bundle carries exactly one .onnx per model dir,
        # so anything else there is from a previous release.
        bundled = {e["path"] for e in wanted}
        for model_dir in ("cm", "cmrt"):
            d = os.path.join(PROJECT_ROOT, "web", "assets", "models", model_dir)
            if not os.path.isdir(d):
                continue
            for name in sorted(os.listdir(d)):
                rel = f"web/assets/models/{model_dir}/{name}"
                if name.endswith(".onnx") and rel not in bundled:
                    print(f"  ! superseded, still on disk: {rel}")
                    print(f"      git rm {rel}")

        print("\nNext:")
        print("    git status")
        print("    python scripts/provenance.py --verify results/run_manifest.json")
        print("    git add -A && git commit -m 'release: install run <run_id>' && git push")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
