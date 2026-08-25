"""
release_bundle.py — package one pipeline run for release. RUN THIS ON THE MACHINE
THAT DID THE COMPUTE (the lab GPU box), not the laptop.

What it produces:

  results/run_manifest.json        tracked. Git commit + machine + config + the
                                   sha256 of EVERY artifact the run made, bulk
                                   included. This is the record that lets the bulk
                                   be deleted without the run becoming unprovable.

  grin-release-<run_id>.tar.gz     tiers 1+2 only. Small (single-digit MB). This is
                                   what crosses to the laptop.

  grin-bulk-<run_id>.tar.gz        tier 3, with --bulk. Large. Goes to OSF/cold
                                   storage once, never to git.

Usage on the lab machine:

    python scripts/release_bundle.py --version 1.0.0 --label manuscript-final \\
        --notes "full pipeline, count-only + RT" --bulk

Then transfer grin-release-<run_id>.tar.gz to the laptop and run
scripts/release_unpack.py there.

The dirty-tree check is not bureaucracy. A release built from uncommitted code
records a commit hash that does not describe the code that produced it, which
makes every downstream provenance claim false in a way nothing later can detect.
--allow-dirty exists for genuine emergencies and records the dirty file list in
the manifest so the compromise is at least visible.
"""
from __future__ import annotations

import argparse
import os
import sys
import tarfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from release_provenance import (  # noqa: E402
    PROJECT_ROOT,
    RUN_MANIFEST_PATH,
    build_run_manifest,
    sha256_file,
    write_json,
)


def _fmt(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} B"
        n /= 1024.0
    return f"{n:.1f} GB"


def _add(tar: tarfile.TarFile, entries: list[dict], label: str) -> int:
    total = 0
    for e in entries:
        abspath = os.path.join(PROJECT_ROOT, e["path"])
        if not os.path.isfile(abspath):
            print(f"  ! skipped (vanished mid-run): {e['path']}")
            continue
        tar.add(abspath, arcname=e["path"])
        total += e["bytes"]
    print(f"  {label}: {len(entries)} files, {_fmt(total)}")
    return total


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--version", required=True, help="release version, e.g. 1.0.0")
    ap.add_argument("--label", default="release", help="short run label, e.g. manuscript-final")
    ap.add_argument("--notes", default="", help="free text: what this run was for")
    ap.add_argument("--out", default=None, help="output directory (default: repo parent)")
    ap.add_argument("--bulk", action="store_true", help="also build the tier-3 archive")
    ap.add_argument("--allow-dirty", action="store_true", help="permit a dirty git tree")
    ap.add_argument("--fast", action="store_true",
                    help="skip hashing tier 3 (dry runs only -- never for a real release)")
    args = ap.parse_args()

    print("scanning artifacts and hashing (this walks every output directory)...")
    mf = build_run_manifest(
        label=args.label, notes=args.notes, hash_tier3=not args.fast
    )
    mf["release_version"] = args.version

    git = mf["git"]
    print(f"\n  commit   {git['short']} ({git['branch']})")
    print(f"  machine  {mf['machine']['hostname']}  gpu={mf['machine'].get('gpu', 'n/a')}")
    for t, name in ((1, "site payload"), (2, "manuscript"), (3, "bulk")):
        tot = mf["totals"][str(t)]
        print(f"  tier {t}   {tot['count']:>4} files  {_fmt(tot['bytes']):>10}   {name}")

    unclassified = [a["path"] for a in mf["artifacts"] if a.get("unclassified")]
    if unclassified:
        print(f"\n  ! {len(unclassified)} unclassified artifact(s) — treated as tier 3:")
        for p in unclassified[:10]:
            print(f"      {p}")
        if len(unclassified) > 10:
            print(f"      ... and {len(unclassified) - 10} more")
        print("    Add a rule to TIER_RULES in scripts/release_provenance.py if any of these")
        print("    back a manuscript number or ship to the site.")

    if git["dirty"]:
        print(f"\n  ! working tree is DIRTY ({len(git['dirty_files'])} file(s))")
        for f in git["dirty_files"][:10]:
            print(f"      {f}")
        if not args.allow_dirty:
            print("\nRefusing to bundle: the recorded commit would not describe the code")
            print("that produced these artifacts. Commit first, or pass --allow-dirty.")
            return 1
        print("    --allow-dirty given; the dirty file list is recorded in the manifest.")

    if args.fast:
        mf["WARNING"] = "built with --fast: tier-3 artifacts are NOT hashed, not a valid release"

    write_json(RUN_MANIFEST_PATH, mf)
    print(f"\nwrote {os.path.relpath(RUN_MANIFEST_PATH, PROJECT_ROOT)}")

    out_dir = os.path.abspath(args.out or os.path.dirname(PROJECT_ROOT))
    os.makedirs(out_dir, exist_ok=True)
    run_id = mf["run_id"]

    tier12 = [a for a in mf["artifacts"] if a["tier"] in (1, 2)]
    rel_path = os.path.join(out_dir, f"grin-release-{run_id}.tar.gz")
    print(f"\nbuilding {os.path.basename(rel_path)}")
    with tarfile.open(rel_path, "w:gz") as tar:
        _add(tar, tier12, "tiers 1+2")
        tar.add(RUN_MANIFEST_PATH, arcname="results/run_manifest.json")
    print(f"  -> {rel_path}")
    print(f"     {_fmt(os.path.getsize(rel_path))}   sha256 {sha256_file(rel_path)}")

    if args.bulk:
        tier3 = [a for a in mf["artifacts"] if a["tier"] == 3]
        bulk_path = os.path.join(out_dir, f"grin-bulk-{run_id}.tar.gz")
        print(f"\nbuilding {os.path.basename(bulk_path)} (this is the slow one)")
        with tarfile.open(bulk_path, "w:gz") as tar:
            _add(tar, tier3, "tier 3")
            tar.add(RUN_MANIFEST_PATH, arcname="results/run_manifest.json")
        print(f"  -> {bulk_path}")
        print(f"     {_fmt(os.path.getsize(bulk_path))}   sha256 {sha256_file(bulk_path)}")
        print("\n  Upload this to OSF, then record the DOI/URL in results/run_manifest.json")
        print("  under \"bulk_archive\" and commit that one-line change.")

    print("\nNext, on the LAB machine:")
    print("    git add -A && git commit && git push")
    print("Then, on the LAPTOP:")
    print(f"    python scripts/release_unpack.py <path to grin-release-{run_id}.tar.gz>")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
