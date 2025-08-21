import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import sys
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_absolute_error, accuracy_score
from sklearn.model_selection import train_test_split
import importlib.util
import math

# Dynamically import custom layers and functions
try:
    from src.utils.config import *
    from src.utils.model_comparison_funcs import load_all_metrics, prepare_test_data, evaluate_final_performance, evaluate_performance_vs_trials, plot_comparison_metrics, plot_performance_vs_trials
except ImportError:
    print("Warning: Could not import utility functions from src.utils. Make sure the package is installed.")
    sys.exit()

# Dynamically import custom layers for model loading
try:
    from src.models.customised_gate_control_model import Expert, Gate
except ImportError:
    # Fallback for models without custom layers
    Expert, Gate = None, None


def main():
    """
    Main function to compare model performance and generate plots.
    This version has been updated to no longer handle target normalization.
    """
    # Prepare test data without normalization
    X_test, y_reg_test, y_cls_test = prepare_test_data([DATASET_FILE], TEST_SPLIT)
    
    # Load computational metrics
    metrics_df = load_all_metrics()
    
    # Evaluate final performance on the full test set
    performance_df = evaluate_final_performance(X_test, y_reg_test, y_cls_test)
    
    # Merge metrics and performance data for a single comparison dataframe
    comparison_df = pd.merge(metrics_df, performance_df, on='model_name')

    print("\n--- Final Model Comparison Summary ---")
    print(comparison_df.to_string(index=False))

    # Plot comparison metrics
    plot_comparison_metrics(comparison_df, 'model_name', 'training_time_seconds',
                            'Model Training Time Comparison', 'Training Time (seconds)',
                            'training_time')
    plot_comparison_metrics(comparison_df, 'model_name', 'trainable_parameters',
                            'Model Size Comparison', 'Number of Trainable Parameters',
                            'model_size')
    plot_comparison_metrics(comparison_df, 'model_name', 'accuracy',
                            'Final Test Accuracy Comparison', 'Test Accuracy',
                            'final_accuracy')
    plot_comparison_metrics(comparison_df, 'model_name', 'mae',
                            'Final Test Mean Absolute Error Comparison', 'Test Mean Absolute Error',
                            'final_mae')

    print("\n--- Evaluating performance vs. trial counts ---")
    models = [f for f in os.listdir(PRETRAINED_MODELS_DIR) if f.endswith('.h5')]
    for model_file in models:
        model_name = os.path.splitext(model_file)[0]
        model_path = os.path.join(PRETRAINED_MODELS_DIR, model_file)
        try:
            # Load the model with custom objects if they exist
            custom_objects = {}
            if Expert:
                custom_objects['Expert'] = Expert
            if Gate:
                custom_objects['Gate'] = Gate

            model = keras.models.load_model(model_path, custom_objects=custom_objects)
            
            # Evaluate performance vs. trial count and plot
            trials_df = evaluate_performance_vs_trials(model, X_test, y_reg_test, y_cls_test)
            plot_performance_vs_trials(trials_df, model_name)
        except Exception as e:
            print(f"Error loading or evaluating model {model_name} for trial count analysis: {e}")

if __name__ == '__main__':
    main()
