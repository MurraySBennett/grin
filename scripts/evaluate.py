"""Run the simulated-data validation gates on a trained model. From project root:
    python scripts/evaluate.py
"""
import numpy as np, torch, time
from src.config import MODEL_FILE, FIGURES_DIR, MLE_FITS_DIR, DEVICE, DROPOUT
from src.data.generator import GRTDataGenerator
from src.models.network import NPEModel, featurize
from src.inference.predict import predict_posterior, predict_point
from src.inference import evaluate as ev
from src.inference.compare import head_to_head
from src.inference.mle import fit_and_select
from src.api import load_model
import os


def main(n_per_class=1000, seed=999, n_mle=300):
    device = DEVICE if torch.cuda.is_available() else "cpu"
    # model = NPEModel(dropout=DROPOUT)
    # model.load_state_dict(torch.load(MODEL_FILE, map_location=device))
    # model.to(device).eval()
    model = load_model(MODEL_FILE, device=device)

    gen = GRTDataGenerator(n_per_class=n_per_class, seed=seed)
    X, yp, Xt, yc, yl = gen.generate_all_model_cms()
    post = predict_posterior(model, X, Xt, n_samples=800)
    mean, samples = post["mean"].numpy(), post["samples"].numpy()

    rec = ev.recovery_metrics(mean, yp)
    print("recovery:", rec["_aggregate"])
    print("coverage:", ev.coverage_curve(samples, yp))
    ev.plot_calibration(samples, yp, os.path.join(FIGURES_DIR, "calibration.png"))

    sub = np.random.default_rng(0).choice(len(X), n_mle, replace=False)
    picks = [fit_and_select(X[i], Xt[i])[0]["model"] for i in sub]
    print("model-ID:", ev.classification_metrics(picks, [yl[i] for i in sub])["accuracy"])

    h = head_to_head(lambda a, b: (predict_point(model, a, b).numpy(),
                                   _t(lambda: predict_point(model, a, b))), X, Xt, yp, n_mle=n_mle)
    print(f"head-to-head speedup: {h['speedup']:.0f}x  "
          f"(NPE {h['npe']['ms_per_matrix']:.4f} ms vs MLE {h['mle']['ms_per_matrix']:.1f} ms/matrix)")


def _t(fn):
    t0 = time.time(); fn(); return time.time() - t0


if __name__ == "__main__":
    main()
