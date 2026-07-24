import tensorflow as tf
from tensorflow import keras
from keras.utils import to_categorical
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import numpy as np
import os
import importlib.util
import sys

from src.utils.model_plotting_funcs import *
from src.utils.GRT_data_generator import GRTDataGenerator
from src.utils.custom_losses import dense_residual_block
from src.utils.config import *

def build_specialist_model(input_shape, model_name):
    """
    Builds a neural network for parameter regression.
    This model is a specialist, trained for a single model class.
    """
    inputs = tf.keras.layers.Input(shape=(input_shape,), name='cm_input')
    x = tf.keras.layers.Dense(256, activation='relu', name='reg_dense1')(inputs)
    x = dense_residual_block(x, 256, activation='relu', dropout_rate=0.2)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)

    outputs = {}
    if 'ps_' in model_name:
        outputs['means_output'] = tf.keras.layers.Dense(2, activation='linear', name='means_output')(x)
    elif 'psa' in model_name or 'psb' in model_name:
        outputs['means_output'] = tf.keras.layers.Dense(4, activation='linear', name='means_output')(x)
    else:
        outputs['means_output'] = tf.keras.layers.Dense(6, activation='linear', name='means_output')(x)

    if 'pi' in model_name:
        pass
    elif 'rho1' in model_name:
        outputs['cov_output'] = tf.keras.layers.Dense(1, activation='tanh', name='cov_output')(x)
    else:
        outputs['cov_output'] = tf.keras.layers.Dense(4, activation='tanh', name='cov_output')(x)
        
    outputs['crit_output'] = tf.keras.layers.Dense(2, activation='linear', name='crit_output')(x)

    model = tf.keras.Model(
        inputs=inputs, 
        outputs=outputs
    )
    return model

def plot_history(history, save_dir, model_name):
    """
    Plots training and validation loss for the overall model and each individual output.
    """
    # Check which output losses are present in the history object
    has_covs = 'cov_output_loss' in history.history
    
    # Define the losses to plot
    losses_to_plot = {
        'Total Loss': 'loss',
        'Means Loss': 'means_output_loss',
        'Crit Loss': 'crit_output_loss'
    }
    if has_covs:
        losses_to_plot['Covs Loss'] = 'cov_output_loss'
        
    num_plots = len(losses_to_plot)
    fig, axs = plt.subplots(1, num_plots, figsize=(6 * num_plots, 6))
    
    # Handle the case where there is only one plot (e.g., for simple models)
    if num_plots == 1:
        axs = [axs]
        
    fig.suptitle(f'Training and Validation Loss for {model_name}', fontsize=16)

    for i, (plot_title, loss_key) in enumerate(losses_to_plot.items()):
        ax = axs[i]
        
        # Plot training loss
        train_loss = history.history[loss_key]
        ax.plot(train_loss, label=f'Training {plot_title}')
        
        # Plot validation loss
        val_loss_key = f'val_{loss_key}'
        if val_loss_key in history.history:
            val_loss = history.history[val_loss_key]
            ax.plot(val_loss, label=f'Validation {plot_title}')
        
        ax.set_title(plot_title)
        ax.set_ylabel('Loss')
        ax.set_ylim((0, 5))
        ax.set_xlabel('Epoch')
        ax.legend()
        ax.grid(False)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_path = os.path.join(save_dir, f'training_history_{model_name}.png')
    plt.savefig(save_path)
    print(f"Training history plot saved to: {save_path}")
    plt.close(fig) # Close the figure to free up memory

    
if __name__ == '__main__':
    if os.path.exists(DATASET_FILE):
        print("Loading pre-existing dataset...")
        data = np.load(DATASET_FILE, allow_pickle=True)
        X = data['X']
        X_trials = data['X_trials']
        y_params = data['y_params']
        y_cls_id = data['y_model_cls']
        y_cls_name = data['y_cls_label']
    else:
        print(f"Generating a new dataset with {NUM_MATRICES_PER_MODEL} matrices per model...")
        gen = GRTDataGenerator(num_matrices=NUM_MATRICES_PER_MODEL, trial_range=TRIALS_RANGE)
        X, y_params, X_trials, y_cls_id, y_cls_name = gen.generate_all_model_cms()
        np.savez(DATASET_FILE, X=X, X_trials=X_trials, y_params=y_params,
                 y_model_cls=y_cls_id, y_cls_label=y_cls_name)
        print("Dataset saved!")
    
    total_trials = X_trials.sum(axis=1) // 4
    y_accuracy = (X[:, 0] + X[:, 5] + X[:, 10] + X[:, 15]) / np.sum(X_trials, axis=1)
    
    if os.path.exists(MATRIX_FEATURE_FILE):
        print(f"Loading {MATRIX_FEATURE_FILE}...")
        feature_data = np.load(MATRIX_FEATURE_FILE)
        X_input = feature_data['X_input']
    else:
        print("Generating matrix features...")
        X_input = compute_matrix_features(X, X_trials)
        np.savez(MATRIX_FEATURE_FILE, X_input=X_input)

    input_shape = X_input.shape[1]

    all_metrics = {}
    for model_class_idx, model_name in enumerate(MODEL_NAMES):
        print(f"\n--- Training specialist model for: {model_name} ---")
        model_mask = y_cls_name == model_name
        trial_mask = total_trials > 50
        accuracy_mask = (y_accuracy > 0.5) & (y_accuracy < 0.9)
        mask = model_mask & trial_mask & accuracy_mask
        X_filtered = X_input[mask]

        y_params_filtered = y_params[mask]

        y_means = y_params_filtered[:, 2:8]
        if 'ps_' in model_name:
            y_means = y_means[:, [0, 3]] # stim1x and stim2y
        elif 'psa' in model_name:
            y_means = y_means[:, [0, 1, 3, 5]] # stim1x, stim1y, stim2y, stim3y
        elif 'psb' in model_name:
            y_means = y_means[:, [0, 2, 3, 4]] # stim1x, stim2x, stim2y, stim3x

        y_covs = y_params_filtered[:, [9, 13, 17, 21]]
        if 'rho1' in model_name:
            y_covs = y_covs[:, 0] # single output, all the same so pick any of 0-3

        y_crit = y_params_filtered[:, 24:26]
        
        X_train, X_val, y_means_train, y_means_val, y_covs_train, y_covs_val, y_crit_train, y_crit_val = train_test_split(
            X_filtered, y_means, y_covs, y_crit, test_size=0.25, random_state=42
        )

        loss_dict = {
            'means_output': 'mae',
            'crit_output': 'mae'
        } 
        if 'pi' not in model_name:
            loss_dict['cov_output'] = 'mae'
        
        loss_weights_dict = {
            'means_output': 1.0,
            'crit_output': 1.0
        }
        if 'pi' not in model_name:
            loss_weights_dict['cov_output'] = 2.0

        model = build_specialist_model(input_shape, model_name)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
            loss=loss_dict,
            loss_weights=loss_weights_dict
        )
        callbacks = [
            ReduceLROnPlateau(monitor='val_loss', factor=RLRP_FACTOR, patience=RLRP_PATIENCE, min_lr=RLRP_MIN_LR, verbose=1),
            EarlyStopping(monitor='val_loss', patience=PATIENCE, min_delta=MIN_DELTA, verbose=1, restore_best_weights=True)
        ]
        model_path = os.path.join(MODEL_RESULTS_DIR, "training", f"specialist_model_{model_name}.h5")
        model_checkpoint = ModelCheckpoint(
            filepath=model_path,
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
        callbacks.append(model_checkpoint)

        
        y_train_dict = {
            'crit_output': y_crit_train,
            'means_output': y_means_train
        }
        y_val_dict = {
            'crit_output': y_crit_val,
            'means_output': y_means_val
        }
        if 'pi' not in model_name:
            y_train_dict['cov_output'] = y_covs_train
            y_val_dict['cov_output'] = y_covs_val

        history = model.fit(
            x=X_train,
            y=y_train_dict,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            validation_data=(X_val, y_val_dict),
            callbacks=callbacks,
            verbose=2
        )
        plot_history(history, os.path.join(FIGURES_DIR, "training"), f"{model_name}_training_history")
        y_val_eval_dict = {
            'means_output': y_means_val,
            'crit_output': y_crit_val
        }
        if 'pi' not in model_name:
            y_val_eval_dict['cov_output'] = y_covs_val
            
        eval_metrics = model.evaluate(
            X_val,
            y_val_eval_dict
        )
        
        metrics_dict = {
            'mae_total': eval_metrics[0],
            'mae_means': eval_metrics[1],
            'mae_crits': eval_metrics[2]
        }
        if 'pi' not in model_name:
            metrics_dict['mae_covs'] = eval_metrics[3]
            
        all_metrics[model_name] = metrics_dict
        print(f"Training complete for {model_name}. Validation MAE: {metrics_dict['mae_total']:.4f}")
    
    metrics_df = pd.DataFrame.from_dict(all_metrics, orient='index')
    metrics_df.to_csv(os.path.join(MODEL_RESULTS_DIR, 'specialist_model_metrics.csv'))
    print("\nAll specialist models have been trained and evaluated.")
