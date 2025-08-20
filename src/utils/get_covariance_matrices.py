import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def get_covariance_matrices(chol_params):
    # Reshape the 12 parameters into 4 sets of 3
    # Shape will be (batch_size, 4, 3)
    chol_params_reshaped = tf.reshape(chol_params, (-1, 4, 3))
    
    # Process each set of 3 parameters to create a 2x2 lower triangular matrix
    L11 = tf.math.softplus(chol_params_reshaped[:, :, 0]) + 1e-6
    L21 = chol_params_reshaped[:, :, 1]
    L22 = tf.math.softplus(chol_params_reshaped[:, :, 2]) + 1e-6

    # Build the 2x2 lower triangular matrices (L) for each of the 4 stimuli
    L = tf.stack([
        tf.stack([L11, tf.zeros_like(L11)], axis=2),
        tf.stack([L21, L22], axis=2)
    ], axis=2)
    
    # Transpose L to get L^T
    L_T = tf.transpose(L, perm=[0, 1, 3, 2])
    
    # Calculate C = L @ L^T
    # This matrix multiplication will produce the 4 valid covariance matrices
    C = tf.matmul(L, L_T)
    
    # Flatten the 4 matrices back into a single vector of 16 values
    return tf.reshape(C, (-1, 16))
