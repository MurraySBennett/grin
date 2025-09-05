import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from src.utils.config import *
from src.utils.pltStyle_8bit import eight_bit_style

plt.rcParams.update(eight_bit_style)

try:
    data = np.load(DATASET_FILE, allow_pickle=True)
    cms = data['X']
    model_labels = data['y_cls_label']
    print(f"Data loaded from {DATASET_FILE}. Found {len(cms)} confusion matrices.")
except FileNotFoundError:
    print(f"Error: The data file {DATASET_FILE} was not found. Please run the generator script first.")
    exit()

def calculate_accuracy_and_trials(cm_flat):
    cm = cm_flat.reshape(4, 4)
    correct_counts = np.diag(cm)
    total_trials_per_stim = np.sum(cm, axis=1)
    stimulus_accuracies = correct_counts / np.where(total_trials_per_stim == 0, 1, total_trials_per_stim)
    mean_accuracy = np.mean(stimulus_accuracies)
    total_trials = np.max(total_trials_per_stim)
    return mean_accuracy, total_trials

results = [calculate_accuracy_and_trials(cm) for cm in cms]
accuracies = np.array([res[0] for res in results])
total_trials = np.array([res[1] for res in results])

data_df = pd.DataFrame({
    'model_name': model_labels,
    'accuracy': accuracies,
    'total_trials': total_trials
})

## 1. Descriptive Statistics Printout

# Calculate stats
descriptive_stats = data_df.groupby('model_name')['accuracy'].agg(['mean', 'median', 'min', 'max', 'std', 'count']).reset_index()

# Sort by mean accuracy for a neater display
descriptive_stats = descriptive_stats.sort_values(by='mean', ascending=False)

# Print the results in a readable format
print("\n--- Descriptive Statistics of Model Accuracy ---")
print(descriptive_stats.to_string(index=False, float_format="%.3f"))
print("--------------------------------------------------")

## 2. Scatter Plot Subplots

model_names = sorted(data_df['model_name'].unique())
num_models = len(model_names)
n_cols = 3
n_rows = int(np.ceil(num_models / n_cols))

fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows), sharex=True, sharey=True)
axes = axes.flatten()

for i, model in enumerate(model_names):
    ax = axes[i]
    model_data = data_df[data_df['model_name'] == model]
    sns.scatterplot(
        x='total_trials',
        y='accuracy',
        data=model_data,
        ax=ax,
        alpha=0.6,
        s=10
    )
    ax.set_title(f'Model: {model}', fontsize=12)
    ax.set_xlabel('Total Trials')
    ax.set_ylabel('Mean Accuracy')
    ax.set_ylim(0, 1)

for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

fig.suptitle('Accuracy vs. Total Trials per Confusion Matrix by Model', fontsize=16, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'accuracy_vs_trials_subplots.png'))
plt.show()

## 3. Boxplot for Overall Model Comparison
plt.figure(figsize=(12, 8))
sns.boxplot(x='model_name', y='accuracy', data=data_df)
plt.xticks(rotation=45, ha='right')
plt.title('Distribution of Mean Accuracy by Model Name')
plt.xlabel('Model Name')
plt.ylabel('Mean Accuracy')
plt.ylim((0,1))
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'accuracy_boxplot.png'))
plt.show()


def plot_cm_stats(cms, labels, stat_type):
    """
    Plots the mean or standard deviation of confusion matrices.
    
    Args:
        cms (np.array): A numpy array of flattened confusion matrices.
        labels (np.array): An array of model labels corresponding to each CM.
        stat_type (str): 'mean' or 'std' to determine which statistic to plot.
    """
    df_cms = pd.DataFrame({
        'cm': list(cms.reshape(-1, 4, 4)),
        'model_name': labels
    })
    
    # Group by model name and calculate the mean/std of the matrices
    # grouped_cms = df_cms.groupby('model_name')['cm'].apply(list)
    def cm_prop(cm):
        n_trials = np.sum(cm, axis=1, keepdims=True)
        cm_prop = np.divide(cm, n_trials, out=np.zeros_like(cm, dtype=float), where=n_trials != 0)
        return cm_prop

    df_cms['cm_prop'] = df_cms['cm'].apply(cm_prop) 
    grouped_cms = df_cms.groupby('model_name')['cm_prop']
    if stat_type == 'mean':
        stat_cms = grouped_cms.apply(lambda x: np.mean(np.stack(x.to_numpy()), axis=0))
    elif stat_type == 'std':
        stat_cms = grouped_cms.apply(lambda x: np.std(np.stack(x.to_numpy()), axis=0))
    else:
        raise ValueError("stat_type must be 'mean' or 'std'")

    model_names = sorted(stat_cms.index.tolist())
    num_models = len(model_names)
    n_cols = 4  # Adjusted for better layout
    n_rows = int(np.ceil(num_models / n_cols))
    # Determine the maximum value for consistent color mapping
    vmax = np.max([matrix.max() for matrix in stat_cms.to_list()])

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 4.5 * n_rows), constrained_layout=True)
    axes = axes.flatten()

    for i, model in enumerate(model_names):
        ax = axes[i]
        cm_data = stat_cms[model]
        sns.heatmap(
            cm_data,
            ax=ax,
            annot=True,
            fmt=".2f",
            cmap="cool",
            cbar=False,
            vmax=vmax
        )
        ax.set_aspect('equal', adjustable='box')
        ax.set_title(f'{model}', fontsize=12)
        # ax.set_ylabel('Stimulus')
        # ax.set_xlabel('Response')
        ax.set_xticklabels(['a1b1', 'a2b1', 'a1b2', 'a2b2'])
        ax.set_yticklabels(['A1B1', 'A2B1', 'A1B2', 'A2B2'], rotation=0)

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle(f'Confusion Matrix {stat_type.upper()}', fontsize=16, y=1.02)
    fig.supxlabel('Response', fontsize=16)
    fig.supylabel('Stimulus', fontsize=16)

    # plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, f'simulated_confusion_matrix_{stat_type}.png'))
    plt.show()

plot_cm_stats(cms, model_labels, 'mean')
plot_cm_stats(cms, model_labels, 'std')
