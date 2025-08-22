import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import sys
import math
from sklearn.model_selection import train_test_split
import importlib.util

from src.utils.config import *
# Import the custom layers from the model file
from src.models.customised_gate_control_model import Expert, Gate
from src.models.independent_separate_param_loss import scale_corr_loss, convert_scales_corr_to_cov, make_regression_losses
from src.utils.param_training_funcs import regression_multi_output_loss

def load_model_from_file(module_name):
    """Dynamically loads the model builder and config from a specified Python file."""
    try:
        model_module = importlib.import_module(f"src.models.{module_name}")
        return getattr(model_module, f"get_{module_name.lower()}_config")()
    except (ModuleNotFoundError, AttributeError) as e:
        print(f"Error loading model from '{module_name}': {e}")
        return None, None

def prepare_validation_data(datasets):
    """
    Re-creates the combined validation set from the pre-training datasets.
    This is necessary to evaluate the model's performance after training is complete.
    """
    X_combined = []
    X_trials_combined = []
    y_params_combined = []

    for file_path in datasets:
        if not os.path.exists(file_path):
            print(f"Error: Dataset not found at {file_path}. Please run the data generation script first.")
            sys.exit(1)

        data = np.load(file_path)
        X_combined.append(data['X'])
        X_trials_combined.append(data['X_trials'])
        y_params_combined.append(data['y_params'])

    X_current = np.vstack(X_combined)
    X_trials_current = np.vstack(X_trials_combined)
    y_params_current = np.vstack(y_params_combined)

    # Preprocess the data just as in the training script
    X_proportions = X_current / (np.repeat(X_trials_current, 4, axis=1) + EPSILON)
    X_trials_log = np.log(X_trials_current + EPSILON)
    X_pretrain_current = np.hstack([X_proportions, X_trials_log])
    
    # Split the combined data into train and validation sets
    (X_train_pre, X_val_pre, y_reg_train_pre, y_reg_val_pre) = train_test_split(
        X_pretrain_current, y_params_current, test_size=0.2, random_state=42
    )

    # Return the un-normalized validation data
    return X_val_pre, y_reg_val_pre

def print_summary_statistics(true_params, pred_params, model_name, param_names):
    """
    Prints summary statistics (mean, std, and MSE) for true and predicted parameters,
    using provided parameter names.
    """
    print(f"\n--- Summary Statistics for {model_name} ---")
    num_pred_params = pred_params.shape[1]
    num_true_params = true_params.shape[1]

    if num_pred_params != num_true_params or num_true_params != len(param_names):
        print(f"\nWarning: Mismatch between number of predicted parameters ({num_pred_params}), number of true parameters ({num_true_params}), and number of parameter names ({len(param_names)}).")
        print("Using the number of predicted parameters for the loop and slicing the parameter names accordingly.")
        
    # We will slice param_names to match the number of predicted parameters
    param_names_sliced = param_names[:num_pred_params]
    true_params_sliced = true_params[:, :num_pred_params]
    
    for i in range(num_pred_params):
        true_p = true_params_sliced[:, i]
        pred_p = pred_params[:, i]
        
        # Calculate statistics
        mse = np.mean((true_p - pred_p)**2)
        true_mean = np.mean(true_p)
        true_std = np.std(true_p)
        pred_mean = np.mean(pred_p)
        pred_std = np.std(pred_p)
        
        print(f"\n{param_names_sliced[i]}:")
        print(f"  MSE: {mse:.6f}")
        print(f"  True: Mean={true_mean:.4f}, Std={true_std:.4f}")
        print(f"  Predicted: Mean={pred_mean:.4f}, Std={pred_std:.4f}")


       
        
def main():
    """Main function to load data and generate plots."""

    # Define the datasets to load
    PRETRAIN_DATASETS = [
        os.path.join(SIMULATED_DATA_DIR, "vary_m_dataset.npz"),
        os.path.join(SIMULATED_DATA_DIR, "vary_mv_dataset.npz"),
        os.path.join(SIMULATED_DATA_DIR, "vary_mc_dataset.npz"),
        os.path.join(SIMULATED_DATA_DIR, "vary_mvc_dataset.npz")
    ]
    
    # 1. Prepare the validation data
    X_val, y_val = prepare_validation_data(PRETRAIN_DATASETS)
    
    # Get the model name to visualize from user input
    for model_name in MODEL_FILES:
        model_builder, config = load_model_from_file(model_name)
        
        # Define the output directory and ensure it exists
        output_dir = os.path.join(FIGURES_DIR, 'pretraining', config['model_name'])
        os.makedirs(output_dir, exist_ok=True)
        
        # 2. Load the pre-training history
        history_path = os.path.join(PRETRAINED_MODEL_RESULTS_DIR, f"{config['model_name']}_pretraining_history.json")
        if not os.path.exists(history_path):
            print(f"Error: History file not found at {history_path}. Please ensure pre-training has been run for this model.")
            sys.exit(1)
            
        with open(history_path, 'r') as f:
            history = json.load(f)
            
        # 3. Plot the training history (loss over epochs)
        plt.figure(figsize=(10, 6))
        # Safely check for multi-task-specific keys
        is_multi_task = config.get('is_multi_task', False)
        if is_multi_task:
            if 'regression_output_loss' in history:
                plt.plot(history['regression_output_loss'], label='Regression Training Loss')
                plt.plot(history['val_regression_output_loss'], label='Regression Validation Loss')
            if 'classification_output_loss' in history:
                plt.plot(history['classification_output_loss'], label='Classification Training Loss')
                plt.plot(history['val_classification_output_loss'], label='Classification Validation Loss')
        else:
            plt.plot(history['loss'], label='Training Loss')
            plt.plot(history['val_loss'], label='Validation Loss')
        
        plt.title(f'Training and Validation Loss for {config["model_name"]} Model')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.tight_layout()
        history_plot_path = os.path.join(output_dir, f"{config['model_name']}_loss_history.png")
        plt.savefig(history_plot_path)
        plt.show()

        # 4. Load the best-trained model
        model_path = os.path.join(PRETRAINED_MODELS_DIR, f"{config['model_name']}_pretrained.h5")
        if not os.path.exists(model_path):
            print(f"Error: Model file not found at {model_path}. Please ensure pre-training has been run for this model.")
            sys.exit(1)
        
        try:
            # Pass custom objects to the load_model function
            custom_objects = {
                'Expert': Expert,
                'Gate': Gate,
                'scale_corr_loss': scale_corr_loss,
                'make_regression_losses': make_regression_losses,
                'regression_multi_output_loss': regression_multi_output_loss
            }
            model = keras.models.load_model(model_path, custom_objects=custom_objects)
        except Exception as e:
            print(f"Error loading model: {e}")
            print("This script assumes a simple Keras model. For multi-task models, you may need a custom loading function.")
            sys.exit(1)
            
        # 5. Make predictions and plot true vs. predicted values
        print("\n--- DEBUGGING PREDICTIONS ---")
        print(f"Input data shape for prediction: {X_val.shape}")
        
        y_pred_list = model.predict(X_val)

        
        print(f"Type of model.predict output: {type(y_pred_list)}")
        print(f"Length of model.predict output: {len(y_pred_list) if isinstance(y_pred_list, list) else 'N/A'}")

        # Check if the output is a list and get the correct prediction array
        if config['model_name'] == "independent_separate_param_loss":
            if isinstance(y_pred_list, dict):
                y_means_pred = y_pred_list['means_out']
                y_scales_pred = y_pred_list['scales_corr_out']
                y_crit_pred = y_pred_list['crit_out']
            elif isinstance(y_pred_list, list):
                y_means_pred, y_scales_pred, y_crit_pred = y_pred_list
            else:
                raise ValueError("Unexpected output type for independent_separate_param_loss model")
            y_pred = convert_scales_corr_to_cov(np.hstack([y_means_pred, y_scales_pred, y_crit_pred]))
        else:
            if isinstance(y_pred_list, list):
                is_multi_task = config.get('is_multi_task', False)
                if is_multi_task:
                    # The second output is the regression output for the multi-task model
                    y_pred = y_pred_list[1]
                    print(f"Regression output shape: {y_pred.shape}")
                else:
                    # For a single-output model, the prediction is the first element of the list
                    y_pred = y_pred_list[0]
                    if config['model_name'] == "independent_separate_param_loss":
                        y_pred = convert_scales_corr_to_cov(y_pred)
                    print(f"Single-task model output shape: {y_pred.shape}")
            else:
                y_pred = y_pred_list
                print(f"Regression output shape: {y_pred.shape}")
        
        # Check if the prediction array is 1-dimensional and reshape if needed
        # This handles cases where a single regression output is returned as a 1D array
        if y_pred.ndim == 1:
            y_pred = np.expand_dims(y_pred, axis=1)
            print(f"Reshaped prediction output shape: {y_pred.shape}")
        
        # We now have the true values (y_val) and predicted values (y_pred)
        num_outputs = y_pred.shape[1]
        
        # Plot all parameters in a grid
        num_cols = 4  # Set a constant number of columns
        num_rows = math.ceil(num_outputs / num_cols)
        
        fig, axs = plt.subplots(num_rows, num_cols, figsize=(num_cols * 4, num_rows * 4), squeeze=False)
        fig.suptitle(f'True vs. Predicted Values for {config["model_name"]}', fontsize=16)
        
        for i in range(num_outputs):
            ax = axs.flatten()[i]
            
            # Use a scatter plot to show the relationship
            ax.scatter(y_val[:, i], y_pred[:, i], alpha=0.5)
            
            # Plot the ideal y=x line without a label
            line_min = min(y_val[:, i].min(), y_pred[:, i].min())
            line_max = max(y_val[:, i].max(), y_pred[:, i].max())
            ax.plot([line_min, line_max], [line_min, line_max], 'r--')
            
            # Set label inside the plot area and use the parameter names
            ax.text(0.05, 0.95, f'{PARAM_NAMES[i]}', transform=ax.transAxes, fontsize=12, verticalalignment='top')
            ax.set_xlabel('True Value')
            ax.set_ylabel('Predicted Value')
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.95)
        regression_plot_path = os.path.join(output_dir, f"{config['model_name']}_regression_performance.png")
        plt.savefig(regression_plot_path)
        plt.show()

        # Print summary statistics for all parameters
        print_summary_statistics(y_val, y_pred, config['model_name'], PARAM_NAMES)

if __name__ == '__main__':
    main()
