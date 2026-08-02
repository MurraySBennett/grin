"""
compare_frameworks.py — run BOTH inference frameworks on ONE frozen test set and
tabulate them with the shared ruler. This is the artefact the whole parallel-
directory exercise exists to produce: an honest, like-for-like pros/cons table.

This is deliberately still scored with the hand-rolled shared_eval.py rather
than workflow.compute_default_diagnostics — the bespoke model has no
BasicWorkflow object to call that on, so a framework-agnostic ruler is the
only way to score both sides fairly. For a BayesFlow-only run with the full
standard diagnostic report (recovery/calibration/coverage/contraction figures
+ report.md), use train_and_diagnose.py instead.

What it does:
  1. Generate a single frozen test set from the SHARED generative model.
  2. Train the BayesFlow workflow (with validation_data, for a real val_loss);
     sample posteriors; time it.
  3. If the bespoke model is available, sample its posteriors on the SAME matrices;
     time it. (Hook documented below — needs your trained bespoke model + src pkg.)
  4. Score both with shared_eval and print one table.

Run from the port directory:
    KERAS_BACKEND=torch python compare_frameworks.py --epochs 60 --train 200000

The defaults here are small so it runs anywhere; scale --train/--epochs up (and use
a GPU) to reach the accuracy numbers you would actually report.
"""
import argparse, os, time, contextlib
import numpy as np

os.environ.setdefault("KERAS_BACKEND", "torch")
import sys
sys.path.insert(0, os.path.dirname(__file__))
import bf_simulator as sim
import bf_workflow as wf
import shared_eval as ev


@contextlib.contextmanager
def _quiet():
    with open(os.devnull, "w") as dn, contextlib.redirect_stderr(dn):
        yield


def time_single_call(sample_fn, x, n_calls=40):
    """Median seconds for a single-matrix posterior — the loop-relevant latency.
    Uses perf_counter and several warmups; most of this cost is fixed per-call
    dispatch overhead, so it is sensitive to warmup and must not use time.time()."""
    for _ in range(3):
        sample_fn(x[:1])  # warmups (build/compile/caches)
    ts = []
    for _ in range(n_calls):
        t0 = time.perf_counter()
        sample_fn(x[:1])
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


# --------------------------------------------------------------------------- #
def run_bayesflow(train_data, val_data, test, kind, epochs, batch_size, num_samples,
                   results_dir):
    workflow = wf.build_workflow(results_dir, kind=kind, learning_rate=5e-4)
    with _quiet():
        workflow.fit_offline(data=train_data, epochs=epochs, batch_size=batch_size,
                              validation_data=val_data, verbose=0)
        t0 = time.time()
        draws = wf.sample_posterior(workflow, test["x"], num_samples=num_samples)
        batched = (time.time() - t0) / len(test["x"])
        single = time_single_call(
            lambda x: wf.sample_posterior(workflow, x, num_samples=num_samples),
            test["x"])
    return draws, single, batched


def run_bespoke(test, num_samples):
    """HOOK for the current PyTorch NPE. Left unwired here because it needs your
    trained model file + the `src` package on the path. On your machine:

        from src.api import load_model
        from src.inference.predict import predict_posterior
        model = load_model()                       # loads results/models/npe_model.pt
        draws = predict_posterior(model, counts, trials, n_samples=num_samples)
        # counts=(N,16) int, trials=(N,4) int  -> reuse test['_counts'],test['trials']

    predict_posterior already returns (N, S, 12) in original param space, so it
    drops straight into shared_eval with no adaptation. Return (draws, single_ms).
    """
    return None, None


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=40000)
    ap.add_argument("--val", type=int, default=300)
    ap.add_argument("--test", type=int, default=2000)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--num-samples", type=int, default=500)
    ap.add_argument("--kind", default="flow_matching", choices=["coupling", "flow_matching"])
    ap.add_argument("--results-dir", default="bayesflow_port/results/compare-run")
    ap.add_argument("--seed", type=int, default=999)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    train_data = sim.simulate(args.train, rng)
    val_data = sim.simulate(args.val, rng)
    test = sim.simulate(args.test, rng)          # the ONE frozen test set

    rows = []

    bf_draws, bf_single, bf_batched = run_bayesflow(
        train_data, val_data, test, args.kind, args.epochs, args.batch_size,
        args.num_samples, args.results_dir)
    # bf_single is already in SECONDS (perf_counter). summarise wants seconds.
    rows.append(ev.summarise(f"bayesflow:{args.kind}", bf_draws, test["theta"],
                             seconds_per_call=bf_single))
    print(f"[bayesflow] batched throughput: {1e3*bf_batched:.4f} ms/matrix")

    be_draws, be_single = run_bespoke(test, args.num_samples)
    if be_draws is not None:
        # be_single in SECONDS (see run_bespoke docstring).
        rows.append(ev.summarise("bespoke:gaussian", be_draws, test["theta"],
                                 seconds_per_call=be_single))

    print()
    ev.print_table(rows)
    print("\nNote: ms_per_call is single-matrix latency (the adaptive-loop budget). "
          "The bespoke Gaussian head is ~0.003 ms/call for reference.")


if __name__ == "__main__":
    main()
