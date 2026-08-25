"""
verify_deploy.py — confirm the LIVE site is serving the weights in this commit.

    python scripts/verify_deploy.py
    python scripts/verify_deploy.py --site https://grin.murraysbennett.com

A green GitHub Actions run proves the upload succeeded. It does not prove a
visitor gets the new model, because .onnx ships with `max-age=31536000,
immutable` and CloudFront invalidation never reaches a browser cache that already
holds the old file. This script fetches what the CDN actually serves and hashes
it, cache-busting the request so it reports the origin's bytes rather than
whatever an intermediate proxy is holding.

Exit 0 means: for every model, the live manifest's version and sha256 match this
working tree, and the live .onnx hashes to that value.

Requires no dependencies beyond the standard library.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from provenance import PROJECT_ROOT, sha256_file  # noqa: E402

DEFAULT_SITE = "https://grin.murraysbennett.com"
MODELS = ("cm", "cmrt")


def fetch(url: str, timeout: int = 60) -> bytes:
    # The cache-buster and no-cache header together are what make this a test of
    # the deploy rather than a test of some cache along the way.
    sep = "&" if "?" in url else "?"
    req = urllib.request.Request(
        f"{url}{sep}_cb={int(time.time() * 1000)}",
        headers={"Cache-Control": "no-cache", "Pragma": "no-cache",
                 "User-Agent": "grin-verify-deploy"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--site", default=DEFAULT_SITE, help=f"site root (default {DEFAULT_SITE})")
    ap.add_argument("--skip-weights", action="store_true",
                    help="compare manifests only; do not download the .onnx files")
    args = ap.parse_args()
    site = args.site.rstrip("/")

    failures: list[str] = []
    print(f"checking {site}\n")

    for model_id in MODELS:
        local_dir = os.path.join(PROJECT_ROOT, "web", "assets", "models", model_id)
        local_mf_path = os.path.join(local_dir, "manifest.json")
        if not os.path.isfile(local_mf_path):
            failures.append(f"{model_id}: no local manifest at {local_mf_path}")
            continue
        with open(local_mf_path, encoding="utf-8") as fh:
            local = json.load(fh)

        url = f"{site}/assets/models/{model_id}/manifest.json"
        try:
            live = json.loads(fetch(url))
        except (urllib.error.URLError, json.JSONDecodeError, TimeoutError) as exc:
            failures.append(f"{model_id}: could not fetch {url} ({exc})")
            continue

        print(f"  {model_id}")
        print(f"    local  version {local.get('version')}   file {local.get('file')}")
        print(f"    live   version {live.get('version')}   file {live.get('file')}")

        ok = True
        for key in ("version", "file", "artifact_sha256"):
            lv, rv = str(local.get(key, "")).lower(), str(live.get(key, "")).lower()
            if lv != rv:
                failures.append(
                    f"{model_id}: live manifest {key} is {live.get(key)!r}, "
                    f"this tree has {local.get(key)!r} — the deploy has not landed "
                    f"(or a CDN edge is still holding the old manifest)"
                )
                ok = False
        if not ok:
            continue

        local_onnx = os.path.join(local_dir, local["file"])
        if os.path.isfile(local_onnx):
            digest = sha256_file(local_onnx)
            if digest != str(local.get("artifact_sha256", "")).lower():
                failures.append(
                    f"{model_id}: the LOCAL .onnx does not match its own manifest — "
                    f"re-run scripts/export_onnx.py --install before trusting anything here"
                )
                continue
        else:
            print(f"    (no local copy of {local['file']}; comparing against the manifest hash)")

        if args.skip_weights:
            print("    manifest matches; --skip-weights, not downloading weights\n")
            continue

        onnx_url = f"{site}/assets/models/{model_id}/{live['file']}"
        try:
            blob = fetch(onnx_url, timeout=180)
        except (urllib.error.URLError, TimeoutError) as exc:
            failures.append(f"{model_id}: could not fetch {onnx_url} ({exc})")
            continue
        served = hashlib.sha256(blob).hexdigest()
        declared = str(live["artifact_sha256"]).lower()
        if served != declared:
            failures.append(
                f"{model_id}: the SERVED weights do not match the served manifest\n"
                f"      manifest {declared}\n"
                f"      served   {served}\n"
                f"      ({len(blob):,} bytes from {onnx_url})"
            )
            continue
        print(f"    served {len(blob):,} bytes, sha256 matches\n")

    if failures:
        print("DEPLOY NOT VERIFIED:")
        for f in failures:
            print("  - " + f)
        print("\nIf the workflow run was green, give CloudFront a minute and retry;")
        print("the invalidation is asynchronous. If it persists, check the Actions log.")
        return 1

    print("VERIFIED — the live site serves exactly the models in this working tree.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
