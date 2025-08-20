import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from src.utils.get_covariance_matrices import get_covariance_matrices



def build_gated_multi_task_model(input_shape, num_models, num_params, dropout_rate=0.3, activation='tanh'):
    """
    Builds a multi-task learning model with a dynamic gating mechanism.
    A shared backbone's output is weighted by a gate before feeding into
    the classification and regression heads.
    """
    model_input = keras.Input(shape=(input_shape,), name='model_input')

    # --- Shared Feature Backbone ---
    shared_backbone = layers.Dense(512, activation=activation, name='shared_dense1')(model_input)
    shared_backbone = layers.BatchNormalization()(shared_backbone)
    shared_backbone = layers.Dropout(dropout_rate)(shared_backbone)
    shared_backbone = layers.Dense(256, activation=activation, name='shared_dense2')(shared_backbone)
    shared_backbone = layers.BatchNormalization()(shared_backbone)
    
    # --- Gating Mechanism ---
    # The gate takes the shared backbone features and generates a weight vector
    gate = layers.Dense(256, activation='sigmoid', name='gate')(shared_backbone)
    
    # Apply the gate to the shared backbone output
    gated_features = layers.Multiply()([shared_backbone, gate])
    gated_features = layers.Dropout(dropout_rate)(gated_features)

    # --- Classification Head ---
    cls_head = layers.Dense(128, activation=activation, name='cls_dense1')(gated_features)
    classification_output = layers.Dense(num_models, activation='softmax', name='classification_output')(cls_head)

    # --- Regression Head (Now predicting a valid covariance matrix) ---
    reg_head = layers.Dense(128, activation=activation, name='reg_dense1')(gated_features)

    num_chol_params = 12
    chol_params_output = layers.Dense(num_chol_params, activation='linear', name='chol_params_output')(reg_head)
    cov_matrices_output = layers.Lambda(
        lambda chol_params: get_covariance_matrices(chol_params),
        name='cov_matrices_output'
    )(chol_params_output) 
    
    # --- Output Head for Means (8 parameters) ---
    # The first 8 parameters in your target data correspond to the means
    means_output = layers.Dense(8, activation='linear', name='means_output')(reg_head)

    # --- Output Head for Critical Values (2 parameters) ---
    # The last 2 parameters in your target data correspond to the 'crit' values
    crit_output = layers.Dense(2, activation='linear', name='crit_output')(reg_head)

    # --- Concatenate all outputs ---
    # Combine all three output heads into a single tensor of shape (batch_size, 26)
    regression_output = layers.Concatenate(name='regression_output')(
        [means_output, cov_matrices_output, crit_output]
    )
    
    model = keras.Model(
        inputs=model_input,
        outputs=[classification_output, regression_output]
    )
    return model



GATED_MULTI_TASK_CONFIG = {
    'losses': {
        'classification_output': 'categorical_crossentropy',
        'regression_output': 'mean_squared_error'
    },
    'loss_weights': {
        'classification_output': 1.0,
        'regression_output': 1.0
    },
    'metrics': {
        'classification_output': 'accuracy',
        'regression_output': 'mae'
    },
    'is_multi_task': True,
    'model_name': 'GatedMultiTask',
    'output_names': ['classification_output', 'regression_output']
}

def get_gated_multi_task_model_config():
    """Returns the model builder and config."""
    return build_gated_multi_task_model, GATED_MULTI_TASK_CONFIG