import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import seaborn as sns
import pandas as pd
from sklearn.metrics import mean_absolute_error, accuracy_score
from src.utils.config import PARAM_NAMES, MODEL_NAMES

def plot_history(history, save_dir, model_name):
    """Plots training/validation metrics."""
    fig, axs = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(f'Training and Validation Metrics for {model_name}', fontsize=16)
    axs[0].plot(history.history['loss'], label='Training Loss')
    axs[0].plot(history.history['val_loss'], label='Validation Loss')
    axs[0].set_title('Loss')
    axs[0].set_ylabel('Loss')
    axs[0].set_xlabel('Epoch')
    axs[0].legend()

    # Metric plot (accuracy or MAE)
    if 'accuracy' in history.history:
        axs[1].plot(history.history['accuracy'], label='Training Accuracy')
        axs[1].plot(history.history['val_accuracy'], label='Validation Accuracy')
        axs[1].set_title('Accuracy')
        axs[1].set_ylabel('Accuracy')
    elif 'mae' in history.history:
        axs[1].plot(history.history['mae'], label='Training MAE')
        axs[1].plot(history.history['val_mae'], label='Validation MAE')
        axs[1].set_title('MAE')
        axs[1].set_ylabel('MAE')
    elif 'mean_absolute_error' in history.history:
        axs[1].plot(history.history['mean_absolute_error'], label='Training MAE')
        axs[1].plot(history.history['val_mean_absolute_error'], label='Validation MAE')
        axs[1].set_title('MAE')
        axs[1].set_ylabel('MAE')
    axs[1].set_xlabel('Epoch')
    axs[1].legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_path = os.path.join(save_dir, f'training_history_{model_name}.png')
    plt.savefig(save_path)
    print(f"Training history plot saved to: {save_path}")

def plot_confusion_matrix(y_true_cls, y_pred_cls, model_names, save_dir, model_name):
    """Plots a confusion matrix for model classifications."""
    y_true_labels = np.argmax(y_true_cls, axis=1)
    y_pred_labels = np.argmax(y_pred_cls, axis=1)
    cm = tf.math.confusion_matrix(y_true_labels, y_pred_labels).numpy()
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', xticklabels=model_names, yticklabels=model_names)
    plt.title(f'Normalized Confusion Matrix for {model_name} Classification')
    plt.ylabel('True Model')
    plt.xlabel('Predicted Model')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, f'class_confusions_{model_name}.png')
    plt.savefig(save_path)
    print(f"Confusion matrix plot saved to: {save_path}")

def plot_regression_performance(y_true_reg, y_pred_reg, save_dir, model_name):
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

        ax.set_title(f'Parameter {PARAM_NAMES[i]}')
        ax.set_xlabel('True Values')
        ax.set_ylabel('Predicted Values')
        ax.legend()
        ax.grid(False)

    for j in range(num_params, len(axs)):
        fig.delaxes(axs[j])

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    save_path = os.path.join(save_dir, f'param_pred_{model_name}.png')
    plt.savefig(save_path)
    print(f"Regression performance per parameter plot saved to: {save_path}")
    # plt.show()

def plot_regression_error_per_class(y_true_reg, y_pred_reg, y_pred_cls, model_names, save_dir, model_name):
    """
    Plots regression error per parameter, grouped by the model's predicted class.
    """
    y_pred_labels = np.argmax(y_pred_cls, axis=1)
    
    # Calculate the absolute error for each parameter
    abs_error = np.abs(y_true_reg - y_pred_reg)
    num_params = abs_error.shape[1]
    
    # Create a DataFrame for easy plotting with seaborn
    data = []
    for i in range(num_params):
        for j in range(len(abs_error)):
            data.append({
                'Error': abs_error[j, i],
                'Parameter': f'Parameter {i+1}',
                'Predicted Class': model_names[y_pred_labels[j]]
            })
    
    df = pd.DataFrame(data)
    
    plt.figure(figsize=(15, 8))
    sns.boxplot(x='Parameter', y='Error', hue='Predicted Class', data=df)
    plt.title(f'Regression Error per Predicted Class for {model_name}', fontsize=16)
    plt.xlabel('Parameter')
    plt.ylabel('Absolute Error')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, f'regression_error_per_class_{model_name}.png')
    plt.savefig(save_path)
    print(f"Regression error per class plot saved to: {save_path}")

def plot_parameter_distributions(y_true_reg, y_pred_reg, save_dir, model_name):
    """
    Plots the distribution of true vs. predicted values for each parameter.
    """
    num_params = y_true_reg.shape[1]
    cols = min(num_params, 4)
    rows = (num_params + cols - 1) // cols
    
    fig, axs = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False)
    fig.suptitle(f'Distribution of True vs. Predicted Parameters for {model_name}', fontsize=16)
    axs = axs.flatten()
    
    for i in range(num_params):
        sns.kdeplot(y_true_reg[:, i], ax=axs[i], label='True Values', fill=True)
        sns.kdeplot(y_pred_reg[:, i], ax=axs[i], label='Predicted Values', fill=True)
        axs[i].set_title(f'Parameter {i+1}')
        axs[i].set_xlabel('Value')
        axs[i].set_ylabel('Density')
        axs[i].legend()
        
    for j in range(num_params, len(axs)):
        fig.delaxes(axs[j])
        
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    save_path = os.path.join(save_dir, f'parameter_distributions_{model_name}.png')
    plt.savefig(save_path)
    print(f"Parameter distribution plots saved to: {save_path}")

def plot_uncertainty_vs_error(y_true_reg, y_pred_reg, y_pred_reg_std, save_dir, model_name):
    """Plots the relationship between prediction error and uncertainty."""
    # Calculate absolute error for each parameter
    absolute_error = np.abs(y_true_reg - y_pred_reg)
    
    # Flatten the arrays to plot all parameters together
    error_flat = absolute_error.flatten()
    std_flat = y_pred_reg_std.flatten()

    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=std_flat, y=error_flat, alpha=0.3)
    plt.title(f'Prediction Error vs. Uncertainty for {model_name}')
    plt.xlabel('Predicted Uncertainty (Standard Deviation)')
    plt.ylabel('Absolute Prediction Error')
    
    # Add a diagonal line for reference (Error = Uncertainty)
    min_val = min(np.min(std_flat), np.min(error_flat))
    max_val = max(np.max(std_flat), np.max(error_flat))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Error = Uncertainty')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, f'uncertainty_vs_error_{model_name}.png')
    plt.savefig(save_path)
    print(f"Uncertainty vs. error plot saved to: {save_path}")
    # plt.show()

def plot_uncertainty_distribution(y_pred_reg_std, save_dir, model_name):
    """
    Plots a histogram of the predicted uncertainty (standard deviation) values.
    """
    plt.figure(figsize=(10, 6))
    sns.histplot(y_pred_reg_std.flatten(), kde=True, bins=50)
    plt.title(f'Distribution of Predicted Uncertainty for {model_name}', fontsize=16)
    plt.xlabel('Predicted Uncertainty (Standard Deviation)')
    plt.ylabel('Frequency')
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, f'uncertainty_distribution_{model_name}.png')
    plt.savefig(save_path)
    print(f"Uncertainty distribution plot saved to: {save_path}")
    

def evaluate_and_plot(model, history, config, X_val, y_cls_val, y_reg_val):
    """
    Handles the evaluation and plotting for a trained model.
    """
    # Make predictions
    y_pred = model.predict(X_val, verbose=0)
    
    if isinstance(y_pred, list):
        y_pred_cls = y_pred[0]
        y_pred_reg = y_pred[1]
    else:
        y_pred_reg = y_pred
        y_pred_cls = None

    final_accuracy = accuracy_score(np.argmax(y_cls_val, axis=1), np.argmax(y_pred_cls, axis=1))
    final_mae = mean_absolute_error(y_reg_val, y_pred_reg)
    
    print("\n--- Final Model Performance on Validation Set ---")
    print(f"Classification Accuracy: {final_accuracy:.4f}")
    print(f"Regression MAE: {final_mae:.4f}")

    plot_history(history, FIGURES_DIR, config['model_name'], config.get('is_bayesian', False))
    plot_confusion_matrix(y_cls_val, y_pred_cls, model_names, FIGURES_DIR, config['model_name'])
    plot_regression_performance(y_reg_val, y_pred_reg, y_cls_val, model_names, FIGURES_DIR, config['model_name'])
    
    plot_regression_error_per_class(y_reg_val, y_pred_reg, y_pred_cls, model_names, FIGURES_DIR, config['model_name'])
    plot_parameter_distributions(y_reg_val, y_pred_reg, FIGURES_DIR, config['model_name'])

    if y_pred_reg_std is not None and (config.get('is_bayesian', False) or config.get('mc_dropout', False)):
        plot_uncertainty_vs_error(y_reg_val, y_pred_reg, y_pred_reg_std, FIGURES_DIR, config['model_name'])
        plot_uncertainty_distribution(y_pred_reg_std, FIGURES_DIR, config['model_name'])
    
    
    save_path = os.path.join(MODELS_DIR, f"{config['model_name']}_predictions.npz")
    save_data = {
        'y_pred_cls': y_pred_cls,
        'y_pred_reg': y_pred_reg
    }
    if y_pred_reg_std is not None:
        save_data['y_pred_reg_std'] = y_pred_reg_std
    np.savez(save_path, **save_data)
    print(f"Model predictions saved to: {save_path}")
