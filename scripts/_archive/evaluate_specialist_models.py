import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from src.utils.config import *
from scipy.stats import skew

   
def plot_regression_performance(y_true_reg, y_pred_reg, save_dir, model_name, param_names):
    """
    Plots subplots for each parameter, showing true vs. predicted values.
    This provides a more granular view to check for parameter-specific biases.
    """
    num_params = y_true_reg.shape[1]
    cols = min(num_params, 4)
    rows = (num_params + cols - 1) // cols
    
    fig, axs = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False)
    fig.suptitle(f'True vs. Predicted Values per Parameter for {model_name}', fontsize=16)
    axs = axs.flatten()

    for i in range(num_params):
        ax = axs[i]
        true_values = y_true_reg[:, i]
        pred_values = y_pred_reg[:, i]
        
        sns.regplot(
            x=true_values, y=pred_values,
            scatter_kws={'alpha':0.3, 's':10},
            line_kws={'color':'red'},
            ax=ax
        )
        
        min_val = min(np.min(true_values), np.min(pred_values))
        max_val = max(np.max(true_values), np.max(pred_values))
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, label='Perfect Prediction')

        # Use the passed-in param_names list
        ax.set_title(f'Parameter {param_names[i]}')
        ax.set_xlabel('True Values')
        ax.set_ylabel('Predicted Values')
        ax.legend()
        ax.grid(False)

    for j in range(num_params, len(axs)):
        fig.delaxes(axs[j])

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    save_path = os.path.join(save_dir, f'param_pred_{model_name}.png')
    plt.savefig(save_path)
    plt.close()
    

def plot_performance_vs_trial_count(y_true, y_pred, X_trials, save_dir, model_name):
    """
    Plots the mean absolute error with standard deviation as a function of trial count,
    binned for clarity.
    """
    # Calculate the absolute error for each sample across all parameters
    abs_error = np.mean(np.abs(y_true - y_pred), axis=1)
    trial_counts = np.sum(X_trials, axis=1) // 4
    
    # Create bins for trial counts (e.g., every 20 trials)
    trial_bins = np.arange(0, np.max(trial_counts) + 20, 20)
    
    binned_means = []
    binned_stds = []
    bin_centers = []
    
    # Loop through bins and calculate mean and std
    for i in range(len(trial_bins) - 1):
        lower_bound = trial_bins[i]
        upper_bound = trial_bins[i+1]
        
        # Filter errors for the current bin
        bin_mask = (trial_counts >= lower_bound) & (trial_counts < upper_bound)
        if np.any(bin_mask):
            errors_in_bin = abs_error[bin_mask]
            binned_means.append(np.mean(errors_in_bin))
            binned_stds.append(np.std(errors_in_bin))
            bin_centers.append((lower_bound + upper_bound) / 2)
        else:
            # Append NaN to avoid plotting empty bins
            binned_means.append(np.nan)
            binned_stds.append(np.nan)
            bin_centers.append((lower_bound + upper_bound) / 2)

    plt.figure(figsize=(12, 8))
    
    # Plot the mean line
    plt.plot(bin_centers, binned_means, marker='.', linestyle='-', color='b', label='Mean Absolute Error')
    
    # Plot the standard deviation as a shaded area
    binned_means = np.array(binned_means)
    binned_stds = np.array(binned_stds)
    plt.fill_between(bin_centers, binned_means - binned_stds, binned_means + binned_stds, color='b', alpha=0.2, label='Standard Deviation')

    plt.title(f'Mean Prediction Error vs. Trial Count for {model_name}', fontsize=16)
    plt.xlabel('Total Trial Count')
    plt.ylabel('Mean Absolute Error')
    plt.ylim(0, 1.5)
    plt.legend()
    plt.grid(False)
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, f'error_vs_trials_binned_{model_name}.png')
    plt.savefig(save_path)
    plt.close()
    print(f"Binned error vs. trial count plot saved to: {save_path}")

def plot_performance_vs_accuracy(y_true, y_pred, y_accuracy, save_dir, model_name):
    """
    Plots the mean absolute error with standard deviation as a function of matrix accuracy,
    binned for clarity.
    """
    # Calculate the absolute error for each sample across all parameters
    abs_error = np.mean(np.abs(y_true - y_pred), axis=1)
    
    # Create bins for accuracy (e.g., every 5%)
    accuracy_bins = np.arange(0, 1.05, 0.05)
    
    binned_means = []
    binned_stds = []
    bin_centers = []
    
    # Loop through bins and calculate mean and std
    for i in range(len(accuracy_bins) - 1):
        lower_bound = accuracy_bins[i]
        upper_bound = accuracy_bins[i+1]
        
        # Filter errors for the current bin
        bin_mask = (y_accuracy >= lower_bound) & (y_accuracy < upper_bound)
        if np.any(bin_mask):
            errors_in_bin = abs_error[bin_mask]
            binned_means.append(np.mean(errors_in_bin))
            binned_stds.append(np.std(errors_in_bin))
            bin_centers.append((lower_bound + upper_bound) / 2)
        else:
            # Append NaN to avoid plotting empty bins
            binned_means.append(np.nan)
            binned_stds.append(np.nan)
            bin_centers.append((lower_bound + upper_bound) / 2)

    plt.figure(figsize=(12, 8))
    
    # Plot the mean line
    plt.plot(bin_centers, binned_means, marker='o', linestyle='-', color='r', label='Mean Absolute Error')
    
    # Plot the standard deviation as a shaded area
    binned_means = np.array(binned_means)
    binned_stds = np.array(binned_stds)
    plt.fill_between(bin_centers, binned_means - binned_stds, binned_means + binned_stds, color='r', alpha=0.2, label='Standard Deviation')

    plt.title(f'Mean Prediction Error vs. Matrix Accuracy for {model_name}', fontsize=16)
    plt.xlabel('Matrix Accuracy')
    plt.ylabel('Mean Absolute Error')
    plt.ylim(0, 1.5)
    plt.legend()
    plt.grid(False)
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, f'error_vs_accuracy_binned_{model_name}.png')
    plt.savefig(save_path)
    plt.close()
    print(f"Binned error vs. accuracy plot saved to: {save_path}")


# --- Main Evaluation Script ---
if __name__ == '__main__':
    # --- Data Loading and Preprocessing (Test Set) ---
    print("Loading full dataset...")
    data = np.load(DATASET_FILE, allow_pickle=True)
    X = data['X']
    X_trials = data['X_trials']
    y_params = data['y_params']
    y_cls_name = data['y_cls_label']
    y_accuracy = np.sum(X[:, [0, 5, 10, 15]], axis=1) / np.sum(X_trials, axis=1)
    

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

    X_test = X_input[test_idx]
    y_params_test = y_params[test_idx]
    y_cls_name_test = y_cls_name[test_idx]
    y_trials_test = X_trials[test_idx]
    y_accuracy_test = y_accuracy[test_idx]

    # --- Loop through each specialist model and evaluate ---
    for model_name in MODEL_NAMES:
        print(f"\n--- Evaluating specialist model for: {model_name} ---")

        # Load the model
        model_path = os.path.join(MODEL_RESULTS_DIR, f"specialist_model_{model_name}.h5")
        if not os.path.exists(model_path):
            print(f"Model file not found for {model_name}. Skipping...")
            continue

        # Split test targets
        y_means_test = y_params_test[:, 2:8]
        y_covs_test = y_params_test[:, [9, 13, 17, 21]]
        y_crit_test = y_params_test[:, 24:26]

        if 'ps_' in model_name:
            y_means_test = y_means_test[:, [0, 3]] # stim1x and stim2y
        elif 'psa' in model_name:
            y_means_test = y_means_test[:, [0, 1, 3, 5]] # stim1x, stim1y, stim2y, stim3y
        elif 'psb' in model_name:
            y_means_test = y_means_test[:, [0, 2, 3, 4]] # stim1x, stim2x, stim2y, stim3x
        if 'rho1' in model_name:
            y_covs_test = y_covs_test[:, 0] # single output, all the same so pick any of 0-3
        y_crit = y_params_test[:, 24:26]

                
        model = tf.keras.models.load_model(model_path)

        # Filter the test data to include only this model class
        mask = y_cls_name_test == model_name
        
        if np.sum(mask) == 0:
            print(f"No test data available for model class {model_name}. Skipping...")
            continue
        
        X_test_filtered = X_test[mask]
        y_means_test_filtered = y_means_test[mask]
        y_covs_test_filtered = y_covs_test[mask]
        y_crit_test_filtered = y_crit_test[mask]
        y_trials_test_filtered = y_trials_test[mask]
        y_accuracy_test_filtered = y_accuracy_test[mask]

        # Make predictions
        preds = model.predict(X_test_filtered)
        y_pred_means = preds['means_output']
        y_pred_crit = preds['crit_output']
        if 'pi' not in model_name:
            y_pred_covs = preds['cov_output']
            if 'rho1' in model_name:
                y_covs_test_filtered = np.expand_dims(y_covs_test_filtered, axis=1)
            y_true_all = np.hstack([y_means_test_filtered, y_covs_test_filtered, y_crit_test_filtered])
            y_pred_all = np.hstack([y_pred_means, y_pred_covs, y_pred_crit])
        else:
            y_true_all = np.hstack([y_means_test_filtered, y_crit_test_filtered])
            y_pred_all = np.hstack([y_pred_means, y_pred_crit])
        
        # Calculate MAE for this specific model on its test data
        mae_means = mean_absolute_error(y_means_test_filtered, y_pred_means)
        mae_crit = mean_absolute_error(y_crit_test_filtered, y_pred_crit)
        if 'pi' not in model_name:
            mae_covs = mean_absolute_error(y_covs_test_filtered, y_pred_covs)
       
        # Plotting the main true vs. predicted regression performance
        plot_regression_performance(
            y_true_all,
            y_pred_all,
            os.path.join(FIGURES_DIR, "specialist_params"),
            model_name,
            PARAM_NAMES
        )

        # Plotting performance vs. trial count
        plot_performance_vs_trial_count(
            y_true_all,
            y_pred_all,
            y_trials_test_filtered,
            os.path.join(FIGURES_DIR, "specialist_trials"),
            model_name
        )

        # Plotting performance vs. matrix accuracy
        plot_performance_vs_accuracy(
            y_true_all,
            y_pred_all,
            y_accuracy_test_filtered,
            os.path.join(FIGURES_DIR, "specialist_acc"),
            model_name
        )
        
    print("\nEvaluation complete for all specialist models.")
