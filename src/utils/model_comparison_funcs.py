import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_absolute_error, accuracy_score
import sys
from src.utils.config import *
import math

# Dynamically import custom layers for model loading
try:
    from src.models.customised_gate_control_model import Expert, Gate
except ImportError:
    # Fallback for models without custom layers
    Expert, Gate = None, None

def load_all_metrics():
    """
    Loads computational metrics (trainable parameters, training time) for all models
    from a single JSON file.
    """
    metrics_path = os.path.join(METRICS_DIR, 'computational_metrics.json')
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            all_metrics = json.load(f)
        df = pd.DataFrame([all_metrics[model_name] for model_name in all_metrics])
        df['model_name'] = [model_name for model_name in all_metrics]
        return df
    else:
        print(f"Metrics file not found at {metrics_path}")
        return pd.DataFrame()

def prepare_test_data(data_files, test_split=0.2):
    """
    Loads and prepares the test data.
    
    This function has been updated to no longer scale the regression targets,
    as the models now predict the raw parameter values directly.
    """
    print("Preparing test data...")
    all_data = []
    for f in data_files:
        path = os.path.join(DATA_DIR, f)
        if os.path.exists(path):
            df = pd.read_csv(path)
            all_data.append(df)
        else:
            print(f"Warning: Data file not found at {path}")
    
    if not all_data:
        sys.exit("Error: No data files found. Exiting.")

    full_df = pd.concat(all_data, ignore_index=True)
    
    # Extract features and targets
    X = full_df[['stimulus', 'stimulus_trial_count', 'stimulus_trial', 'num_trials', 'num_classes', 'num_dimensions', 'beta_0', 'beta_1', 'var_0', 'var_1']].to_numpy()
    y_reg = full_df[['beta_0', 'beta_1']].to_numpy()
    y_cls = full_df['stimulus'].to_numpy()
    
    # Split data into train and test sets
    X_train, X_test, y_reg_train, y_reg_test, y_cls_train, y_cls_test = train_test_split(
        X, y_reg, y_cls, test_size=test_split, random_state=42
    )
    
    # We no longer scale the targets, as the model will predict raw values.
    
    print("Test data prepared.")
    return X_test, y_reg_test, y_cls_test

def evaluate_final_performance(X_test, y_reg_test, y_cls_test):
    """
    Evaluates the performance of each model on the full test set.
    
    This function has been updated to remove the un-normalization step, as the
    model's predictions are now in the correct un-normalized range.
    """
    print("\nEvaluating final performance on the full test set...")
    performance_data = []
    models = [f for f in os.listdir(PRETRAINED_MODELS_DIR) if f.endswith('.h5')]
    
    for model_file in models:
        model_name = os.path.splitext(model_file)[0]
        model_path = os.path.join(PRETRAINED_MODELS_DIR, model_file)
        
        try:
            # Load the model with custom objects
            custom_objects = {}
            if Expert:
                custom_objects['Expert'] = Expert
            if Gate:
                custom_objects['Gate'] = Gate
            
            model = keras.models.load_model(model_path, custom_objects=custom_objects)
            print(f"Successfully loaded model: {model_name}")
            
            # Make predictions
            # The multi-task model returns two outputs, so we need to handle that.
            y_pred = model.predict(X_test[:, :NUM_FEATURES], verbose=0)
            
            if isinstance(y_pred, list):
                y_cls_pred = y_pred[0]
                y_reg_pred = y_pred[1]
            else:
                y_reg_pred = y_pred
                # A single output model won't have classification, so we can't calculate accuracy.
                y_cls_pred = None

            # Calculate metrics
            mae = mean_absolute_error(y_reg_test, y_reg_pred)
            accuracy = accuracy_score(y_cls_test, np.argmax(y_cls_pred, axis=1)) if y_cls_pred is not None else np.nan
            
            performance_data.append({
                'model_name': model_name,
                'accuracy': accuracy,
                'mae': mae
            })
            
        except Exception as e:
            print(f"Error loading or evaluating model {model_name}: {e}")
            
    return pd.DataFrame(performance_data)

def evaluate_performance_vs_trials(model, X_test, y_reg_test, y_cls_test):
    """
    Evaluates model performance across different numbers of trials per stimulus.
    
    This function has been updated to remove the un-normalization step, as the
    model's predictions are now in the correct un-normalized range.
    """
    print("\nEvaluating performance across different trial counts...")
    trial_counts = np.arange(20, 1020, 20)  # Bins from 20 to 1000
    performance_data = []

    # Extract the total trial count for each stimulus from the test set
    stimulus_trial_counts = X_test[:, 1]

    for trial_count_bin in trial_counts:
        # Filter data for stimuli with at least the current trial count
        mask = stimulus_trial_counts >= trial_count_bin
        
        if np.sum(mask) == 0:
            continue
        
        X_subset = X_test[mask]
        y_reg_subset = y_reg_test[mask]
        y_cls_subset = y_cls_test[mask]

        if X_subset.shape[0] > 0:
            y_pred = model.predict(X_subset[:, :NUM_FEATURES], verbose=0)
            
            if isinstance(y_pred, list):
                y_cls_pred = y_pred[0]
                y_reg_pred = y_pred[1]
            else:
                y_reg_pred = y_pred
                y_cls_pred = None
            
            # Calculate metrics
            mae = mean_absolute_error(y_reg_subset, y_reg_pred)
            accuracy = accuracy_score(y_cls_subset, np.argmax(y_cls_pred, axis=1)) if y_cls_pred is not None else np.nan
            
            performance_data.append({
                'trial_count_bin': trial_count_bin,
                'accuracy': accuracy,
                'mae': mae
            })
    
    return pd.DataFrame(performance_data)


def plot_comparison_metrics(df, x_col, y_col, title, y_label, file_name):
    """
    Creates and saves a bar plot for model comparison.
    """
    plt.figure(figsize=(10, 6))
    sns.barplot(x=x_col, y=y_col, data=df, palette='viridis')
    plt.title(title, fontsize=16)
    plt.xlabel('Model Name', fontsize=12)
    plt.ylabel(y_label, fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    save_path = os.path.join(PLOTS_DIR, f'{file_name}_comparison.png')
    plt.savefig(save_path)
    plt.close()
    print(f"Plot saved to {save_path}")
    
def plot_performance_vs_trials(df, model_name):
    """
    Creates and saves line plots for performance vs. trial count.
    """
    plt.figure(figsize=(12, 6))

    # Plot Accuracy
    plt.subplot(1, 2, 1)
    sns.lineplot(x='trial_count_bin', y='accuracy', data=df)
    plt.title(f'Classification Accuracy vs. Trial Count for {model_name}', fontsize=12)
    plt.xlabel('Number of Trials per Stimulus', fontsize=10)
    plt.ylabel('Accuracy', fontsize=10)
    plt.grid(True)
    
    # Plot MAE
    plt.subplot(1, 2, 2)
    sns.lineplot(x='trial_count_bin', y='mae', data=df)
    plt.title(f'Regression MAE vs. Trial Count for {model_name}', fontsize=12)
    plt.xlabel('Number of Trials per Stimulus', fontsize=10)
    plt.ylabel('Mean Absolute Error (MAE)', fontsize=10)
    plt.grid(True)
    
    plt.tight_layout()
    
    save_path = os.path.join(PLOTS_DIR, f'{model_name}_performance_vs_trials.png')
    plt.savefig(save_path)
    plt.close()
    print(f"Performance vs. trial count plot saved to {save_path}")
