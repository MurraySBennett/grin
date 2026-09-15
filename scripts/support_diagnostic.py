"""Evaluate the production checkpoint's training-support deviance.

This is an input-support diagnostic, not a test of Gaussian GRT adequacy. It
calibrates the 95th-percentile warning threshold on 6,000 matrices drawn from the
training prior (500 from each of 12 classes) and evaluates partial B-response
mapping reversals at four severities, 150 matrices per severity. All matrices use
300 trials per stimulus. See ``src/inference/ood.py`` for the statistic.

    python scripts/support_diagnostic.py

Writes results/validation/support_diagnostic.json.
"""
import json
import os

from src.api import load_model
from validation.checks import v09_ood

OUT = os.path.join("results", "validation", "support_diagnostic.json")


def main():
    result = v09_ood(model=load_model())
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    print(json.dumps(result["result"], indent=2))
    print(f"wrote {OUT}")
    return result


if __name__ == "__main__":
    main()
