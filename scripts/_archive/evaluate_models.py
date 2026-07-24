import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    precision_recall_fscore_support, mean_absolute_error,
    brier_score_loss
)
import tensorflow as tf
from tensorflow import keras
from keras.utils import to_categorical

from src.utils.config import (
    DATASET_FILE, MATRIX_FEATURE_FILE, MODEL_RESULTS_DIR,
    MODEL_FILES, FIGURES_DIR
)
from src.utils.custom_losses import classification_loss
from src.utils.GRT_data_generator import GRTDataGenerator
from scripts.train_models import load_model_from_file


# ---------------------------------------------------------------------
# Helper plotting functions (self-contained)
# ---------------------------------------------------------------------
def save_confusion_matrix(y_true, y_pred, class_names, save_path, normalize=True):
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt=".2f" if normalize else "d",
                cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.title("Confusion Matrix" + (" (Normalized)" if normalize else ""))
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_per_class_metrics(metrics_dict, class_names, metric_name, save_path):
    values = [metrics_dict[i] for i in range(len(class_names))]
    plt.figure(figsize=(10, 6))
    sns.barplot(x=class_names, y=values)
    plt.ylabel(metric_name)
    plt.xticks(rotation=45)
    plt.title(f"Per-Class {metric_name}")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_calibration(y_true, y_prob, save_path):
    # true labels (0/1) vs confidence (max softmax prob)
    confidences = np.max(y_prob, axis=1)
    predictions = np.argmax(y_prob, axis=1)
    correctness = (predictions == y_true)

    bins = np.linspace(0.0, 1.0, 11)
    binids = np.digitize(confidences, bins) - 1
    bin_acc = [correctness[binids == i].mean() if np.any(binids == i) else 0
               for i in range(len(bins) - 1)]
    bin_conf = [confidences[binids == i].mean() if np.any(binids == i) else 0
                for i in range(len(bins) - 1)]

    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.plot(bin_conf, bin_acc, marker="o")
    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.title("Calibration Curve")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_regression_scatter(y_true, y_pred, target_names, save_path_prefix):
    for i, name in enumerate(target_names):
        plt.figure(figsize=(6, 6))
        plt.scatter(y_true[:, i], y_pred[:, i], alpha=0.5)
        lims = [
            np.min([y_true[:, i], y_pred[:, i]]),
            np.max([y_true[:, i], y_pred[:, i]])
        ]
        plt.plot(lims, lims, "r--")
        plt.xlabel("True")
        plt.ylabel("Predicted")
        plt.title(f"Regression Scatter: {name}")
        plt.tight_layout()
        plt.savefig(f"{save_path_prefix}_{name}.png", dpi=300)
        plt.close()


def plot_error_hist(y_true, y_pred, target_names, save_path_prefix):
    errors = y_pred - y_true
    for i, name in enumerate(target_names):
        plt.figure(figsize=(6, 4))
        sns.histplot(errors[:, i], bins=30, kde=True)
        plt.xlabel("Prediction Error")
        plt.title(f"Error Distribution: {name}")
        plt.tight_layout()
        plt.savefig(f"{save_path_prefix}_{name}.png", dpi=300)
        plt.close()


def plot_performance_vs_difficulty(y_accuracy_test, y_true_cls, y_pred_cls,
                                   save_path):
    confs = np.max(y_pred_cls, axis=1)
    correct = (np.argmax(y_pred_cls, axis=1) == np.argmax(y_true_cls, axis=1))
    df = pd.DataFrame({
        "dataset_accuracy": y_accuracy_test,
        "confidence": confs,
        "correct": correct.astype(int)
    })
    df["bin"] = pd.qcut(df["dataset_accuracy"], q=5, duplicates="drop")
    grouped = df.groupby("bin").agg({"correct": "mean", "confidence": "mean"})

    grouped.plot(kind="bar", figsize=(8, 6))
    plt.ylabel("Score")
    plt.title("Performance vs Dataset Difficulty")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


# ---------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------
if __name__ == "__main__":
    gen = GRTDataGenerator()

    # --- Load dataset ---
    data = np.load(DATASET_FILE)
    X = np.load(MATRIX_FEATURE_FILE)["X_input"]

    y_params = data["y_params"]
    y_cls_id = data["y_model_cls"]
    y_cls_name = data["y_cls_label"]
    X_trials = data["X_trials"]

    y_means = y_params[:, 2:8]
    y_covs = y_params[:, [9, 13, 17, 21]]
    y_crit = y_params[:, 24:26]
    y_accuracy = np.sum(X[:, [0, 5, 10, 15]], axis=1) / np.sum(X_trials, axis=1)
    y_cls_id = to_categorical(y_cls_id)

    # --- Split (use same random seed as training) ---
    from sklearn.model_selection import train_test_split
    _, test_idx = train_test_split(
        np.arange(len(y_cls_name)),
        test_size=0.2,
        stratify=np.argmax(y_cls_id, axis=1),
        random_state=42
    )

    X_test = X[test_idx]
    y_means_test = y_means[test_idx]
    y_covs_test = y_covs[test_idx]
    y_crit_test = y_crit[test_idx]
    y_cls_test = y_cls_id[test_idx]
    y_cls_name_test = y_cls_name[test_idx]
    y_accuracy_test = y_accuracy[test_idx]

    for MODEL_FILE in MODEL_FILES:
        model_builder, config = load_model_from_file(MODEL_FILE)
        if not model_builder:
            continue

        eval_dir = os.path.join(FIGURES_DIR, "model_eval", config["model_name"])
        os.makedirs(eval_dir, exist_ok=True)

        is_multi_task = config.get("is_multi_task", False)

        if is_multi_task:
            model_path = os.path.join(MODEL_RESULTS_DIR, f"{config['model_name']}_1.h5")
            model = tf.keras.models.load_model(model_path)#, custom_objects=config["losses"])
            y_pred_cls, y_pred_means, y_pred_covs, y_pred_crit = model.predict(X_test)
        else:
            cls_path = os.path.join(MODEL_RESULTS_DIR, f"{config['model_name']}_cls_1.h5")
            reg_path = os.path.join(MODEL_RESULTS_DIR, f"{config['model_name']}_reg_1.h5")
            cls_model = tf.keras.models.load_model(cls_path)#, custom_objects={"classification_loss": classification_loss})
            reg_model = tf.keras.models.load_model(reg_path)#, custom_objects=config["reg_losses"])

            y_pred_cls = cls_model.predict(X_test)
            y_pred_means, y_pred_covs, y_pred_crit = reg_model.predict(X_test)

        # --- Metrics ---
        y_true_cls_ids = np.argmax(y_cls_test, axis=1)
        y_pred_cls_ids = np.argmax(y_pred_cls, axis=1)
        acc = accuracy_score(y_true_cls_ids, y_pred_cls_ids)

        print(f"\n=== {config['model_name']} ===")
        print(f"Classification Accuracy: {acc:.4f}")
        print(classification_report(y_true_cls_ids, y_pred_cls_ids, target_names=gen.model_names))

        mae_means = mean_absolute_error(y_means_test, y_pred_means)
        mae_covs = mean_absolute_error(y_covs_test, y_pred_covs)
        mae_crit = mean_absolute_error(y_crit_test, y_pred_crit)

        print(f"MAE Means: {mae_means:.3f}")
        print(f"MAE Covs: {mae_covs:.3f}")
        print(f"MAE Crit: {mae_crit:.3f}")

        # --- Plots ---
        save_confusion_matrix(y_true_cls_ids, y_pred_cls_ids, gen.model_names,
                              os.path.join(eval_dir, "confusion_matrix.png"))
        save_confusion_matrix(y_true_cls_ids, y_pred_cls_ids, gen.model_names,
                              os.path.join(eval_dir, "confusion_matrix_raw.png"), normalize=False)

        report = precision_recall_fscore_support(y_true_cls_ids, y_pred_cls_ids, labels=range(len(gen.model_names)))
        per_class_acc = report[1]
        per_class_f1 = report[2]
        plot_per_class_metrics(dict(enumerate(per_class_acc)), gen.model_names, "Recall (Accuracy)",
                               os.path.join(eval_dir, "per_class_accuracy.png"))
        plot_per_class_metrics(dict(enumerate(per_class_f1)), gen.model_names, "F1-score",
                               os.path.join(eval_dir, "per_class_f1.png"))

        plot_calibration(y_true_cls_ids, y_pred_cls, os.path.join(eval_dir, "calibration_curve.png"))

        reg_targets = ["mean1", "mean2", "mean3", "mean4", "mean5", "mean6",
                       "cov1", "cov2", "cov3", "cov4", "crit1", "crit2"]
        y_true_reg = np.hstack([y_means_test, y_covs_test, y_crit_test])
        y_pred_reg = np.hstack([y_pred_means, y_pred_covs, y_pred_crit])
        plot_regression_scatter(y_true_reg, y_pred_reg, reg_targets,
                                os.path.join(eval_dir, "scatter"))
        plot_error_hist(y_true_reg, y_pred_reg, reg_targets,
                        os.path.join(eval_dir, "error_hist"))

        plot_performance_vs_difficulty(y_accuracy_test, y_cls_test, y_pred_cls,
                                       os.path.join(eval_dir, "perf_vs_difficulty.png"))

