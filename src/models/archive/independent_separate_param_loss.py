import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np


def convert_targets_to_scales_corr(y_params):
    """
    Convert flattened 2x2 covariance matrices (16 values for 4 stimuli)
    into a stable parameterization: (log σx, log σy, atanh ρ).
    
    Input shape: (N, 26) -> [means(8), cov_flat(16), crit(2)]
    Output shape: (N, 22) -> [means(8), scales_corr(12), crit(2)]
    """
    N = y_params.shape[0]

    means = y_params[:, :8]
    covs = y_params[:, 8:24].reshape(N, 4, 2, 2)  # 4 stimuli, 2x2 each
    crits = y_params[:, 24:]

    scales_corr = []
    for i in range(4):
        cov = covs[:, i, :, :]  # shape (N, 2, 2)
        var_x = cov[:, 0, 0]
        var_y = cov[:, 1, 1]
        cov_xy = cov[:, 0, 1]

        sigma_x = np.sqrt(np.clip(var_x, 1e-8, None))
        sigma_y = np.sqrt(np.clip(var_y, 1e-8, None))
        rho = cov_xy / (sigma_x * sigma_y + 1e-8)
        rho = np.clip(rho, -0.999999, 0.999999)

        log_sx = np.log(sigma_x)
        log_sy = np.log(sigma_y)
        fisher_rho = np.arctanh(rho)

        scales_corr.append(np.stack([log_sx, log_sy, fisher_rho], axis=1))

    # Concatenate across the 4 stimuli → shape (N, 12)
    scales_corr = np.concatenate(scales_corr, axis=1)

    return np.hstack([means, scales_corr, crits])


def convert_scales_corr_to_cov(y_pred):
    """
    Convert predicted [means(8), log σx/log σy/atanh ρ (12), crit(2)]
    back into [means(8), cov_flat(16), crit(2)].

    Input: (N, 22)
    Output: (N, 26)
    """
    N = y_pred.shape[0]

    means = y_pred[:, :8]
    scales_corr = y_pred[:, 8:20].reshape(N, 4, 3)  # 4 stimuli, (log_sx, log_sy, fisher_rho)
    crits = y_pred[:, 20:]

    covs_flat = []
    for i in range(4):
        log_sx = scales_corr[:, i, 0]
        log_sy = scales_corr[:, i, 1]
        fisher_rho = scales_corr[:, i, 2]

        sigma_x = np.exp(log_sx)
        sigma_y = np.exp(log_sy)
        rho = np.tanh(fisher_rho)

        cov_xx = sigma_x ** 2
        cov_yy = sigma_y ** 2
        cov_xy = rho * sigma_x * sigma_y

        # Flatten each 2x2 covariance: [xx, xy, yx, yy]
        covs_flat.append(
            np.stack([cov_xx, cov_xy, cov_xy, cov_yy], axis=1)
        )

    covs_flat = np.concatenate(covs_flat, axis=1)  # shape (N, 16)

    return np.hstack([means, covs_flat, crits])


# ==============================
# Utility losses
# ==============================

def scale_corr_loss(y_true, y_pred):
    """
    Loss for covariance parameters.
    y_true and y_pred are in transformed space:
      - log σx
      - log σy
      - Fisher z (atanh of correlation coefficient)
    """
    return tf.reduce_mean(tf.square(y_true - y_pred))


def make_regression_losses():
    """
    Returns losses, loss weights, and metrics dicts for regression model.
    """
    losses = {
        'means_out': 'mse',
        'scales_corr_out': scale_corr_loss,
        'crit_out': 'mse',
    }
    loss_weights = {
        'means_out': 1.0,
        'scales_corr_out': 2.0,  # emphasize covariance learning
        'crit_out': 1.0,
    }
    metrics = {
        'means_out': 'mae',
        'scales_corr_out': 'mae',
        'crit_out': 'mae',
    }
    return losses, loss_weights, metrics


# ==============================
# Models
# ==============================

def build_classification_model(input_shape, num_models, dropout_rate=0.3, activation='tanh'):
    """
    Standalone classification model.
    """
    model_input = keras.Input(shape=(input_shape,), name='cls_input')

    x = layers.Dense(512, activation=activation, name='cls_dense1')(model_input)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(256, activation=activation, name='cls_dense2')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)

    classification_output = layers.Dense(num_models, activation='softmax', name='classification_output')(x)

    return keras.Model(inputs=model_input, outputs=classification_output)


def build_regression_model(input_shape, dropout_rate=0.3, activation='tanh'):
    """
    Standalone regression model for GRT parameter recovery.
    Predicts:
      - means_out: (8,)
      - scales_corr_out: (12,) = [log σx, log σy, Fisher z(ρ)] × 4 stimuli
      - crit_out: (2,)
    """
    reg_input = keras.Input(shape=(input_shape,), name='reg_input')

    x = layers.Dense(512, activation=activation, name='reg_dense1')(reg_input)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(256, activation=activation, name='reg_dense2')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)

    # --- Outputs ---
    means_out = layers.Dense(8, activation='linear', name='means_out')(x)  # 8 means
    scales_corr_out = layers.Dense(12, activation='linear', name='scales_corr_out')(x)  # cov params
    crit_out = layers.Dense(2, activation='linear', name='crit_out')(x)  # 2 criteria

    return keras.Model(inputs=reg_input, outputs=[means_out, scales_corr_out, crit_out])


def build_independent_models(input_shape, num_models, num_params, dropout_rate=0.3, activation='tanh'):
    """
    Entry point for training script.
    Returns two separate models: classification + regression.
    """
    cls_model = build_classification_model(input_shape, num_models, dropout_rate=dropout_rate, activation=activation)
    reg_model = build_regression_model(input_shape, dropout_rate=dropout_rate, activation=activation)
    return cls_model, reg_model


# ==============================
# Config
# ==============================

INDEPENDENT_CONFIG = {
    'losses': {
        'means_out': 'mse',
        'scales_corr_out': scale_corr_loss,
        'crit_out': 'mse',
    },
    'loss_weights': {
        'means_out': 1.0,
        'scales_corr_out': 2.0,
        'crit_out': 1.0,
    },
    'metrics': {
        'means_out': 'mae',
        'scales_corr_out': 'mae',
        'crit_out': 'mae',
    },
    'is_multi_task': False,  # independent training
    'model_name': 'independent_separate_param_loss'
}


def get_independent_separate_param_loss_config():
    """Returns the model builder and config."""
    return build_independent_models, INDEPENDENT_CONFIG
