import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras import layers
import numpy as np
from src.utils.config import MODEL_NAMES

model_to_idx = {name: i for i, name in enumerate(MODEL_NAMES)}
high_cost_groups = [
    ['pi_ps_ds', 'rho1_ps_ds', 'ps_ds'],
    ['pi_psa_ds', 'rho1_psa_ds', 'psa_ds'],
    ['pi_psb_ds', 'rho1_psb_ds', 'psb_ds']
]

num_models = len(MODEL_NAMES)
penalty_matrix = np.full((num_models, num_models), 1.2, dtype=np.float32)
np.fill_diagonal(penalty_matrix, 1.0)

for group in high_cost_groups:
    indices = [model_to_idx[name] for name in group]
    for i in indices:
        for j in indices:
            if i != j:
                penalty_matrix[i, j] = 1.5
penalty_matrix_tf = tf.constant(penalty_matrix, dtype=tf.float32)

def classification_loss(y_true, y_pred):
    """
    Cost-sensitive categorical cross-entropy with misclassification penalties.
    - y_true: one-hot [batch, num_classes]
    - y_pred: predicted softmax [batch, num_classes]
    """
    y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())

    # True class index for each sample
    true_class_idx = tf.argmax(y_true, axis=1)  # [batch]

    # Get the penalty row for each true class
    penalties = tf.gather(penalty_matrix_tf, true_class_idx)  # [batch, num_classes]

    # Standard categorical cross-entropy
    ce = -y_true * tf.math.log(y_pred)  # [batch, num_classes]

    # Apply cost-sensitive weighting to all classes
    weighted_ce = ce * penalties  # penalize misclassification more heavily

    # Sum over classes to get per-sample loss
    loss = tf.reduce_sum(weighted_ce, axis=-1)  # [batch]

    return loss


def get_cov_type(model_name):
    if 'pi_' in model_name:
        return 'pi'
    elif 'rho1_' in model_name:
        return 'rho1'
    else:
        return 'other'

cov_constraint_types = tf.constant([get_cov_type(name) for name in MODEL_NAMES], dtype=tf.string)
def custom_cov_loss(y_true_with_id, y_pred):
    y_true = y_true_with_id[:, :-1]
    y_cls_id = tf.cast(y_true_with_id[:, -1], tf.int32)

    # per-example MAE, shape [batch]
    regression_loss = tf.reduce_mean(tf.abs(y_true - y_pred), axis=1)
    # OR: tf.keras.losses.MeanAbsoluteError(reduction=tf.keras.losses.Reduction.NONE)(y_true, y_pred)

    # use the correct constraint map (see Fix 2)
    sample_constraint_types = tf.gather(cov_constraint_types, y_cls_id)

    penalty = tf.zeros_like(regression_loss)  # shape [batch]

    is_pi = tf.equal(sample_constraint_types, 'pi')
    if tf.reduce_any(is_pi):
        pi_preds = tf.boolean_mask(y_pred, is_pi)                 # [k, D]
        pi_penalty = tf.reduce_sum(tf.abs(pi_preds), axis=1)       # [k]
        indices = tf.cast(tf.where(is_pi), tf.int64)               # [k,1]
        out_shape = tf.cast(tf.shape(penalty), tf.int64)           # [1]
        penalty += tf.scatter_nd(indices, pi_penalty, out_shape)

    is_rho1 = tf.equal(sample_constraint_types, 'rho1')
    if tf.reduce_any(is_rho1):
        rho1_preds = tf.boolean_mask(y_pred, is_rho1)              # [k, D]
        mean_corr = tf.reduce_mean(rho1_preds, axis=1, keepdims=True)
        rho1_penalty = tf.reduce_sum(tf.abs(rho1_preds - mean_corr), axis=1)
        indices = tf.cast(tf.where(is_rho1), tf.int64)
        out_shape = tf.cast(tf.shape(penalty), tf.int64)
        penalty += tf.scatter_nd(indices, rho1_penalty, out_shape)

    return regression_loss + penalty #* 0.5

    
# Define the constraint types for each model
def get_mean_type(model_name):
    if 'ps_' in model_name:
        return 'ps_all'
    if 'psa' in model_name:
        return 'psa'
    if 'psb' in model_name:
        return 'psb'
    return 'other'

mean_constraint_types = tf.constant([get_mean_type(name) for name in MODEL_NAMES], dtype=tf.string)
def custom_means_loss(y_true_with_id, y_pred):
    y_true = y_true_with_id[:, :-1]
    y_cls_id = tf.cast(y_true_with_id[:, -1], tf.int32)

    regression_loss = tf.reduce_mean(tf.abs(y_true - y_pred), axis=1)  # [batch]
    sample_constraint_types = tf.gather(mean_constraint_types, y_cls_id)
    penalty = tf.zeros_like(regression_loss)                           # [batch]

    is_ps = tf.equal(sample_constraint_types, 'ps_all')
    if tf.reduce_any(is_ps):
        ps_preds = tf.boolean_mask(y_pred, is_ps)  # [k, 6]
        ps_penalty = (tf.abs(ps_preds[:, 0] - ps_preds[:, 4])
                      + tf.abs(ps_preds[:, 1])
                      + tf.abs(ps_preds[:, 2])
                      + tf.abs(ps_preds[:, 3] - ps_preds[:, 5]))       # [k]
        indices = tf.cast(tf.where(is_ps), tf.int64)
        out_shape = tf.cast(tf.shape(penalty), tf.int64)
        penalty += tf.scatter_nd(indices, ps_penalty, out_shape)

    is_psa = tf.equal(sample_constraint_types, 'psa')
    if tf.reduce_any(is_psa):
        psa_preds = tf.boolean_mask(y_pred, is_psa)
        psa_penalty = tf.abs(psa_preds[:, 2]) + tf.abs(psa_preds[:, 0] - psa_preds[:, 4])
        indices = tf.cast(tf.where(is_psa), tf.int64)
        out_shape = tf.cast(tf.shape(penalty), tf.int64)
        penalty += tf.scatter_nd(indices, psa_penalty, out_shape)

    is_psb = tf.equal(sample_constraint_types, 'psb')
    if tf.reduce_any(is_psb):
        psb_preds = tf.boolean_mask(y_pred, is_psb)
        psb_penalty = tf.abs(psb_preds[:, 3] - psb_preds[:, 5]) + tf.abs(psb_preds[:, 1])
        indices = tf.cast(tf.where(is_psb), tf.int64)
        out_shape = tf.cast(tf.shape(penalty), tf.int64)
        penalty += tf.scatter_nd(indices, psb_penalty, out_shape)

    return regression_loss + penalty # * 0.5


def custom_mae(y_true, y_pred):
    # Slice off the last column (the class ID) from the true values
    y_true_mae = y_true[:, :-1]
    return tf.keras.losses.mean_absolute_error(y_true_mae, y_pred)

def dense_residual_block(x, units, activation='tanh', dropout_rate=0.3):
    """
    Creates a residual block with two Dense layers.
    Adds input x to the output (skip connection).
    """
    shortcut = x  # save input for residual connection
    
    # First Dense layer
    out = layers.Dense(units, activation=activation)(x)
    out = layers.BatchNormalization()(out)
    out = layers.Dropout(dropout_rate)(out)
    
    # Second Dense layer
    out = layers.Dense(units, activation=activation)(out)
    out = layers.BatchNormalization()(out)
    out = layers.Dropout(dropout_rate)(out)
    
    # Residual connection (input added back)
    out = layers.Add()([out, shortcut])
    
    return out
