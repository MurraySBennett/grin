import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_absolute_error, accuracy_score, confusion_matrix
import sys
import glob

# Assuming src.utils.config are available
from src.utils.config import *
import math

# Dynamically import custom layers for model loading
try:
    from src.models.customised_gate_control_model import Expert, Gate
except ImportError:
    # Fallback for models without custom layers
    Expert, Gate = None, None

def load_all_predictions():
    """
    Loads and concatenates all prediction CSVs into a single DataFrame.
    """
    print("\nLoading all model predictions...")
    all_predictions = []
    prediction_files = glob.glob(os.path.join(MODEL_RESULTS_DIR, "*_predictions.csv"))
    
    for f in prediction_files:
        df = pd.read_csv(f)
        df['model'] = f.split("\\")[-1].split("_")[0]
        all_predictions.append(df)
        
    if not all_predictions:
        print("No prediction files found. Please run python -m scripts.save_model_predictions first.")
        return pd.DataFrame()
        
    return pd.concat(all_predictions, ignore_index=True)

def get_model_labels(df):
    """
    Dynamically creates a mapping from numerical model IDs to string names
    from the loaded dataframes. This is a more robust approach than relying
    on a global variable.
    """
    # Get all unique model names from the true labels
    unique_model_names = MODEL_NAMES
    # # Sort them to ensure consistent ordering
    # unique_model_names.sort()
    
    # Create the mapping dictionaries
    model_name_to_idx = {name: i for i, name in enumerate(unique_model_names)}
    idx_to_model_name = {i: name for i, name in enumerate(unique_model_names)}
    
    return model_name_to_idx, idx_to_model_name


def plot_performance_vs_trials_and_accuracy(all_predictions_df, output_dir):
    """
    Evaluates model performance across different numbers of trials per stimulus
    for each model, plotting both regression MAE and classification accuracy.
    """
    print("\nEvaluating and plotting performance across different trial counts...")

    model_name_to_idx, idx_to_model_name = get_model_labels(all_predictions_df)

    performance_data = []
    trial_counts = np.arange(0, 1020, 20)
    
    for model_name, model_df in all_predictions_df.groupby('model'):
        for idx, trial_count_bin in enumerate(trial_counts[:-1]):
            df_filtered_trials = model_df[(model_df['trial_count'] > trial_count_bin) & (model_df['trial_count'] <= trial_counts[idx+1])]
            if df_filtered_trials.empty:
                continue
            unique_samples_df = df_filtered_trials.drop_duplicates(subset=['sample_id'])
            if 'predicted_cls' in unique_samples_df.columns and not unique_samples_df['predicted_cls'].isnull().any():
                y_pred_mapped = unique_samples_df['predicted_cls'].map(idx_to_model_name)
                accuracy = accuracy_score(unique_samples_df['model_name'], y_pred_mapped)
            else:
                accuracy = np.nan
            mae = mean_absolute_error(df_filtered_trials['true_value'], df_filtered_trials['predicted_value'])
            
            performance_data.append({
                'model': model_name,
                'trial_count_bin': trial_count_bin,
                'accuracy': accuracy,
                'mae': mae
            })
            
    if not performance_data:
        print("No data to plot for performance vs. trials.")
        return
        
    performance_df = pd.DataFrame(performance_data)
    
    # Now, plot for each model
    for model_name, group_df in performance_df.groupby('model'):
        plt.figure(figsize=(12, 6))
        
        # Plot Accuracy
        plt.subplot(1, 2, 1)
        sns.lineplot(x='trial_count_bin', y='accuracy', data=group_df, errorbar='sd')
        plt.title(f'Accuracy vs. Trial Count for {model_name}', fontsize=12)
        plt.xlabel('Number of Trials per Stimulus', fontsize=10)
        plt.ylabel('Accuracy', fontsize=10)
        plt.ylim(0,1)
        
        # Plot MAE
        plt.subplot(1, 2, 2)
        sns.lineplot(x='trial_count_bin', y='mae', data=group_df, errorbar='sd')
        plt.title(f'MAE vs. Trial Count for {model_name}', fontsize=12)
        plt.xlabel('Number of Trials per Stimulus', fontsize=10)
        plt.ylabel('Mean Absolute Error (MAE)', fontsize=10)
        plt.ylim(0, np.max(group_df['mae'] + 0.5))
        
        plt.tight_layout()
        save_path = os.path.join(output_dir, f'{model_name}_performance_vs_trials.png')
        plt.savefig(save_path)
        plt.close()
        print(f"Performance vs. trial count plot saved to {save_path}")

def plot_confusion_matrices(all_predictions_df, output_dir):
    """
    Creates and saves a confusion matrix for each multi-task model.
    """
    print("\nPlotting confusion matrices...")
    
    unique_models = all_predictions_df['model_name'].unique()
    model_name_to_idx, idx_to_model_name = get_model_labels(all_predictions_df)
    
    for model_name in unique_models:
        model_df = all_predictions_df[all_predictions_df['model_name'] == model_name]
        
        # Check if the model is a multi-task model with classification predictions
        if 'predicted_cls' in model_df.columns and not model_df['predicted_cls'].isnull().any():
            # Get unique samples to avoid double-counting
            unique_samples_df = model_df.drop_duplicates(subset=['sample_id'])
            y_true = unique_samples_df['model_name']
            y_pred_numeric = unique_samples_df['predicted_cls']
            
            # Map the numerical predictions to the actual model names
            y_pred_mapped = y_pred_numeric.map(idx_to_model_name)

            cm = confusion_matrix(y_true, y_pred_mapped, labels=list(model_name_to_idx.keys()))
            
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=list(model_name_to_idx.keys()), yticklabels=list(model_name_to_idx.keys()))
            plt.title(f'Confusion Matrix for {model_name}', fontsize=16)
            plt.xlabel('Predicted Label', fontsize=12)
            plt.ylabel('True Label', fontsize=12)
            plt.tight_layout()
            
            save_path = os.path.join(output_dir, f'{model_name}_confusion_matrix.png')
            plt.savefig(save_path)
            plt.close()
            print(f"Confusion matrix for {model_name} saved to {save_path}")
        else:
            print(f"Skipping confusion matrix for {model_name} as it is not a multi-task model.")

def plot_regression_scatter(all_predictions_df, output_dir):
    """
    Creates scatter plots of predicted vs. true values for each parameter.
    """
    print("\nPlotting regression scatter plots...")
    
    unique_models = all_predictions_df['model_name'].unique()
    unique_params = all_predictions_df['parameter_name'].unique()
    
    for model_name in unique_models:
        model_df = all_predictions_df[all_predictions_df['model_name'] == model_name]
        
        for param_name in unique_params:
            param_df = model_df[model_df['parameter_name'] == param_name]
            
            plt.figure(figsize=(8, 8))
            sns.scatterplot(
                x='true_value',
                y='predicted_value',
                data=param_df,
                alpha=0.6,
                s=20
            )
            
            # Add a diagonal line for perfect prediction
            max_val = max(param_df['true_value'].max(), param_df['predicted_value'].max())
            min_val = min(param_df['true_value'].min(), param_df['predicted_value'].min())
            plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--')
            
            plt.title(f'Predicted vs. True for {param_name} ({model_name})', fontsize=16)
            plt.xlabel(f'True {param_name}', fontsize=12)
            plt.ylabel(f'Predicted {param_name}', fontsize=12)
            plt.tight_layout()
            
            save_path = os.path.join(output_dir, f'{model_name}_{param_name}_scatter.png')
            plt.savefig(save_path)
            plt.close()
            print(f"Scatter plot for {model_name} and {param_name} saved to {save_path}")


if __name__ == '__main__':
    # Make sure the main plots directory exists
    os.makedirs(FIGURES_DIR, exist_ok=True)
    
    # Create subdirectories for the new plots
    trial_count_dir = os.path.join(FIGURES_DIR, "trial_count_performance")
    accuracy_dir = os.path.join(FIGURES_DIR, "accuracy_performance")
    regression_scatter_dir = os.path.join(FIGURES_DIR, "regression_scatter")
    os.makedirs(trial_count_dir, exist_ok=True)
    os.makedirs(accuracy_dir, exist_ok=True)
    os.makedirs(regression_scatter_dir, exist_ok=True)

    # Load all evaluation data
    all_predictions_df = load_all_predictions()
    
    if not all_predictions_df.empty:
        # Evaluate performance vs trial count and plot
        plot_performance_vs_trials_and_accuracy(all_predictions_df, trial_count_dir)
        
    #     # Plot confusion matrices
    #     plot_confusion_matrices(all_predictions_df, accuracy_dir)
        
    #     # Plot regression scatter plots
    #     plot_regression_scatter(all_predictions_df, regression_scatter_dir)
