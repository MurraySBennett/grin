import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import mean_absolute_error, accuracy_score
from sklearn.model_selection import train_test_split
from scipy.stats import multivariate_normal

import numpy as np
import os
from pprint import pprint as pp
import importlib.util
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix

from src.utils.config import *
from src.utils.GRT_data_generator import GRTDataGenerator

def plot_confusion_matrix(y_true_labels, y_pred_labels, all_model_names, save_dir, model_name):
    """
    Plots a confusion matrix for model classifications using string labels.
    """
    # Create a DataFrame for the confusion matrix
    cm = confusion_matrix(y_true_labels, y_pred_labels, labels=all_model_names)
    cm_df = pd.DataFrame(cm, index=all_model_names, columns=all_model_names)
    
    # Normalize the confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_normalized_df = pd.DataFrame(cm_normalized, index=all_model_names, columns=all_model_names)

    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_normalized_df, annot=True, fmt='.2f', cmap='Blues', xticklabels=all_model_names, yticklabels=all_model_names)
    plt.title(f'Normalized Confusion Matrix for {model_name} Classification', fontsize=16)
    plt.ylabel('True Model')
    plt.xlabel('Predicted Model')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, f'class_confusions_{model_name}.png')
    plt.savefig(save_path)
    print(f"Confusion matrix plot saved to: {save_path}")
    
def plot_overall_performance(accuracy, save_dir, model_name):
    """
    Plots a simple bar chart of the overall classification accuracy.
    """
    plt.figure(figsize=(8, 6))
    plt.bar(['Accuracy'], [accuracy * 100], color='skyblue')
    plt.ylim(0, 100)
    plt.ylabel('Percentage')
    plt.title(f'Overall Classification Accuracy for {model_name}', fontsize=16)
    plt.text(0, accuracy * 100 + 1, f'{accuracy * 100:.2f}%', ha='center')
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, f'overall_performance_{model_name}.png')
    plt.savefig(save_path)
    print(f"Overall performance plot saved to: {save_path}")

def run_model_selection(X_input_test, X_test_raw_cms, y_cls_name_test, X_trials_test):
    """
    Evaluates each specialist model on test data and determines the best-fitting
    model for each sample based on parameter predictions.
    """
    all_predictions = []

    print("Loading all specialist models...")
    specialist_models = {}
    for model_name in MODEL_NAMES:
        model_path = os.path.join(MODEL_RESULTS_DIR, "training", f"specialist_model_{model_name}.h5")
        if os.path.exists(model_path):
            specialist_models[model_name] = tf.keras.models.load_model(model_path, compile=False)

    print("Pre-computing predictions for the entire test set...")
    all_specialist_predictions = {}
    for model_name, model in specialist_models.items():
        # Predict on the entire test set in a single, vectorized call
        preds = model.predict(X_input_test, verbose=0)
        # Store predictions in a dictionary for easy access
        all_specialist_predictions[model_name] = preds

    print("Performing model selection for each test sample...")
    
    gen = GRTDataGenerator() # Assuming this is available

    for i in range(X_input_test.shape[0]):
        cm_input = X_test_raw_cms[i]
        true_model_name = y_cls_name_test[i]
        n_trials = int(np.max(X_trials_test[i].flatten()))
        model_fits = {}
        
        for model_name, preds_dict in all_specialist_predictions.items():
            # Correctly extract the precomputed prediction for the current sample
            pred_means = preds_dict['means_output'][i]
            pred_crit = preds_dict['crit_output'][i]

            # Handle model-specific parameter sizes and shapes
            if 'pi' not in model_name:
                pred_covs = preds_dict['cov_output'][i]
                if 'rho1' in model_name:
                    # rho1 model has only one covariance output, expand dims for hstack
                    pred_covs = np.expand_dims(pred_covs, axis=0)
                pred_covs_list = [np.array([[1.0, cov], [cov, 1.0]]) for cov in pred_covs]
            else:
                pred_covs_list = None

            # Correctly pad means based on the model type
            if 'ps_' in model_name:
                # ps_ds predicts 2 means, need to hstack with 4 zeros for the 6-mean array
                pred_means_full = np.zeros(6)
                pred_means_full[[0, 3]] = pred_means
                pred_means = pred_means_full
            elif 'psa' in model_name:
                # psa_ds predicts 4 means, need to hstack with 2 zeros
                pred_means_full = np.zeros(6)
                pred_means_full[[0, 1, 3, 5]] = pred_means
                pred_means = pred_means_full
            elif 'psb' in model_name:
                # psb_ds predicts 4 means, need to hstack with 2 zeros
                pred_means_full = np.zeros(6)
                pred_means_full[[0, 2, 3, 4]] = pred_means
                pred_means = pred_means_full
            else:
                # All other models predict 6 means
                pass

            # Use the trials from the original test data
            generated_cm, _, _ = gen.simulate_cm_from_params(
                means=pred_means, 
                cov_mat=pred_covs_list, 
                crit=pred_crit, 
                n_stimulus_trials=n_trials
            )

            cm_input_flat = cm_input.flatten()
            generated_cm_flat = generated_cm.flatten()
            
            distance = mean_absolute_error(cm_input_flat, generated_cm_flat)
            model_fits[model_name] = -distance

        if model_fits:
            predicted_model_name = max(model_fits, key=model_fits.get)
            all_predictions.append({
                'true_label': true_model_name,
                'predicted_label': predicted_model_name
            })
        else:
            all_predictions.append({
                'true_label': true_model_name,
                'predicted_label': 'N/A'
            })

    return all_predictions


if __name__ == '__main__':
    # --- Data Loading (Re-using logic from evaluation script) ---
    print("Loading full dataset...")
    data = np.load(DATASET_FILE, allow_pickle=True)
    X_raw_cms = data['X'] 
    X_trials = data['X_trials']
    y_cls_name = data['y_cls_label']

    print(f"Loading matrix features from {MATRIX_FEATURE_FILE}...")
    feature_data = np.load(MATRIX_FEATURE_FILE)
    X_input = feature_data['X_input']
    
    # --- Split dataset (train+val vs test) ---
    train_val_idx, test_idx = train_test_split(
        np.arange(len(y_cls_name)),
        test_size=TEST_SPLIT,
        stratify=y_cls_name,
        random_state=42
    )

    X_input_test = X_input[test_idx]
    y_cls_name_test = y_cls_name[test_idx]
    X_test_raw_cms = X_raw_cms[test_idx]
    X_trials_test = X_trials[test_idx]
    
    # --- Run the model selection process ---
    predictions = run_model_selection(
        X_input_test=X_input_test,
        X_test_raw_cms=X_test_raw_cms,
        y_cls_name_test=y_cls_name_test,
        X_trials_test=X_trials_test
    )
    
    # --- Evaluate Classification Performance ---
    true_labels = [p['true_label'] for p in predictions]
    predicted_labels = [p['predicted_label'] for p in predictions]
    
    # Plotting confusion matrix
    all_model_names = MODEL_NAMES
    
    plot_confusion_matrix(
        y_true_labels=true_labels,
        y_pred_labels=predicted_labels,
        all_model_names=all_model_names,
        save_dir=os.path.join(FIGURES_DIR, "classification"),
        model_name="Ensemble_Classifier_Performance"
    )
    
    accuracy = accuracy_score(true_labels, predicted_labels)
    print(f"\nFinal Ensemble Classifier Accuracy: {accuracy:.4f}")

    # Plot the overall accuracy
    plot_overall_performance(accuracy, os.path.join(FIGURES_DIR, "classification"), "Ensemble_Classifier")
