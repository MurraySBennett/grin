import numpy as np
import matplotlib.pyplot as plt
import os
from src.utils.config import DATASET_FILE, MODEL_NAMES

def load_data(file_path):
    """
    Loads the numpy dataset from a specified file path.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} was not found.")
    data = np.load(file_path, allow_pickle=True)
    return data

def plot_parameters(data, accuracy_bins=[(0.0, 0.5), (0.5, 1.0)]):
    """
    Plots the distribution of all 26 parameters for each GRT model,
    color-coded by accuracy bin.
    """
    X = data['X']
    X_trials = data['X_trials']
    y_params = data['y_params']
    y_cls_label = data['y_cls_label']
    
    # Calculate grand mean accuracy for each matrix
    total_trials = np.sum(X_trials, axis=1)
    correct_counts = np.array([np.diag(matrix.reshape(4, 4)) for matrix in X])
    correct_counts_sum = np.sum(correct_counts, axis=1)
    grand_mean_accuracies = np.divide(
        correct_counts_sum, total_trials, 
        out=np.zeros_like(correct_counts_sum, dtype=float), 
        where=total_trials != 0
    )
    
    for model_name in MODEL_NAMES:
        print(f"Generating plots for model: {model_name}")
        
        # Filter data for the current model
        model_mask = (y_cls_label == model_name)
        model_params = y_params[model_mask]
        model_accuracies = grand_mean_accuracies[model_mask]
        
        num_params = model_params.shape[1]
        param_labels = get_param_labels(num_params)
        
        fig, axes = plt.subplots(nrows=6, ncols=5, figsize=(20, 24))
        fig.suptitle(f'Parameter Distributions for {model_name} Model', fontsize=24, y=1.02)
        axes = axes.flatten()

        for i in range(num_params):
            ax = axes[i]
            
            for bin_min, bin_max in accuracy_bins:
                bin_mask = (model_accuracies > bin_min) & (model_accuracies <= bin_max)
                if np.sum(bin_mask) > 0:
                    params_in_bin = model_params[bin_mask, i]
                    
                    if len(params_in_bin) > 0:
                        ax.hist(params_in_bin, bins=20, alpha=0.6, label=f'Acc: ({bin_min}-{bin_max}]')
            
            ax.set_title(param_labels[i], fontsize=12)
            ax.set_xlabel('Parameter Value')
            ax.set_ylabel('Frequency')
            # ax.legend()
            # ax.grid(True, which='both', linestyle='--', linewidth=0.5)

        # Hide any unused subplots
        for i in range(num_params, len(axes)):
            fig.delaxes(axes[i])
            
        plt.tight_layout(rect=[0, 0, 1, 0.98])
        plt.show()

def get_param_labels(num_params):
    """
    Generates meaningful labels for the 26 parameters based on their structure.
    8 means, 16 covariance terms, 2 critical values.
    """
    labels = []
    # Means (4 stimuli * 2 dimensions)
    for i in range(4):
        labels.append(f'Mean Stim {i+1} (Dim 1)')
        labels.append(f'Mean Stim {i+1} (Dim 2)')
    
    # Covariance matrices (4 matrices, each 2x2)
    for i in range(4):
        labels.append(f'Cov Stim {i+1} [0,0]')
        labels.append(f'Cov Stim {i+1} [0,1]')
        labels.append(f'Cov Stim {i+1} [1,0]')
        labels.append(f'Cov Stim {i+1} [1,1]')
        
    # Critical values
    labels.append('Crit Value (Dim 1)')
    labels.append('Crit Value (Dim 2)')
    
    return labels[:num_params]

if __name__ == "__main__":
    try:
        grt_data = load_data(DATASET_FILE)
        plot_parameters(grt_data)
    except FileNotFoundError as e:
        print(e)
        print("Please ensure you have run the GRTDataGenerator script with the --full or --all flag to create the dataset.")
        print("Example: python your_data_generator_script.py --all")
