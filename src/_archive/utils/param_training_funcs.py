import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def get_covariance_matrices(chol_params):
    """
    Converts unconstrained chol_params into positive semi-definite 2x2 covariance matrices
    for 4 stimuli. Uses softplus for diagonals and tanh to constrain correlations to [-1, 1].
    
    Input: (batch_size, 12) → 4 stimuli × 3 params (log_sx, log_sy, fisher_rho)
    Output: (batch_size, 16) → flattened covariance matrices
    """
    # Reshape: (batch_size, 4, 3)
    chol_params = tf.reshape(chol_params, (-1, 4, 3))

    # Diagonal entries: enforce positivity
    L11 = tf.math.softplus(chol_params[:, :, 0]) + 1e-6
    L22 = tf.math.softplus(chol_params[:, :, 1]) + 1e-6

    # Off-diagonal: constrain correlation via tanh
    rho = tf.tanh(chol_params[:, :, 2])
    L21 = rho * tf.sqrt(L11 * L22)  # L21 scaled to match correlation

    # Build 2x2 lower-triangular matrices
    L = tf.stack([
        tf.stack([L11, tf.zeros_like(L11)], axis=2),
        tf.stack([L21, L22], axis=2)
    ], axis=2)

    # Covariance: C = L @ L^T
    L_T = tf.transpose(L, perm=[0, 1, 3, 2])
    C = tf.matmul(L, L_T)

    # Flatten 4 matrices → 16 values
    return tf.reshape(C, (-1, 16))

    
    
def regression_multi_output_loss(y_true, y_pred, weight_means=1.0, weight_cov=2.0, weight_crit=1.0):
    """
    Multi-output regression loss combining:
    - Means (8 values)
    - Covariance matrices (flattened 4x2x2 -> 16)
    - Crit values (2 values)

    Parameters
    ----------
    y_true : tf.Tensor
        Shape (batch_size, 26) -> [means(8), cov_flat(16), crit(2)]
    y_pred : tf.Tensor
        Shape (batch_size, 26) -> same structure as y_true
    weight_means : float
        Weight for the means component
    weight_cov : float
        Weight for the covariance component
    weight_crit : float
        Weight for the crit component
    """
    # Split components
    means_true = y_true[:, :8]
    cov_true = y_true[:, 8:24]
    crit_true = y_true[:, 24:]

    means_pred = y_pred[:, :8]
    cov_pred = y_pred[:, 8:24]
    crit_pred = y_pred[:, 24:]

    # --- Means loss ---
    loss_means = tf.reduce_mean(tf.square(means_true - means_pred))

    # --- Covariance loss ---
    cov_true_matrices = tf.reshape(cov_true, (-1, 4, 2, 2))
    cov_pred_matrices = tf.reshape(cov_pred, (-1, 4, 2, 2))

    # Variances
    var_true = tf.stack([cov_true_matrices[:, :, 0, 0], cov_true_matrices[:, :, 1, 1]], axis=-1)
    var_pred = tf.stack([cov_pred_matrices[:, :, 0, 0], cov_pred_matrices[:, :, 1, 1]], axis=-1)
    var_loss = tf.reduce_mean(tf.square(var_true - var_pred))

    # Correlations
    cov_xy_true = cov_true_matrices[:, :, 0, 1]
    cov_xy_pred = cov_pred_matrices[:, :, 0, 1]
    rho_true = cov_xy_true / (tf.sqrt(var_true[:, :, 0] * var_true[:, :, 1]) + 1e-6)
    rho_pred = cov_xy_pred / (tf.sqrt(var_pred[:, :, 0] * var_pred[:, :, 1]) + 1e-6)
    rho_loss = tf.reduce_mean(tf.square(rho_true - rho_pred))

    loss_cov = var_loss + rho_loss

    # --- Crit loss ---
    loss_crit = tf.reduce_mean(tf.square(crit_true - crit_pred))

    # --- Weighted sum ---
    total_loss = weight_means * loss_means + weight_cov * loss_cov + weight_crit * loss_crit
    return total_loss