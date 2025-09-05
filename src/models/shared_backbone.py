import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from src.utils.custom_losses import classification_loss, custom_means_loss, custom_cov_loss, custom_mae, dense_residual_block

def build_shared_backbone_model(input_shape, num_models, dropout_rate=0.3, activation='tanh'):
    """
    Builds a multi-task learning model with a single shared backbone that 
    then branches into two separate heads for classification and regression.
    """
    model_input = keras.Input(shape=(input_shape,), name='model_input')

    # --- Shared Backbone ---
    shared_backbone = layers.Dense(512, activation=activation, name='shared_dense1')(model_input)
    shared_backbone = dense_residual_block(shared_backbone, 512, activation, dropout_rate)
    shared_backbone = layers.Dense(256, activation=activation, name='shared_dense2')(shared_backbone)
    shared_backbone = layers.BatchNormalization()(shared_backbone)
    shared_backbone = layers.Dropout(dropout_rate)(shared_backbone)
    shared_backbone = layers.Dense(128, activation=activation, name='shared_dense3')(shared_backbone)
    
    # --- Classification Head ---
    classification_output = layers.Dense(
        num_models, activation='softmax', name='classification_output'
    )(shared_backbone)
    
    # --- Output Heads for Regression ---
    means_output = layers.Dense(6, activation='linear', name='means_output')(shared_backbone)
    cov_output = layers.Dense(4, activation='tanh', name='cov_output')(shared_backbone)
    crit_output = layers.Dense(2, activation='linear', name='crit_output')(shared_backbone)
    # regression_output = layers.Concatenate(name='regression_output')(
    #     [means_output, cov_params_output, crit_output]
    # )

    model = keras.Model(
        inputs=model_input,
        outputs=[classification_output, means_output, cov_output, crit_output] #regression_output]
    )
    return model

SHARED_BACKBONE_CONFIG = {
    'losses': {
        'classification_output': classification_loss, # 'categorical_crossentropy',
        'means_output': custom_means_loss,
        'cov_output': custom_cov_loss,
        'crit_output': 'mae'
    },
    'loss_weights': {
        'classification_output': 1.0,
        'means_output': 1.0,
        'cov_output': 2.0,
        'crit_output': 1.0
    },
    'metrics': {
        'classification_output': 'accuracy',
        'means_output': custom_mae,
        'cov_output': custom_mae,
        'crit_output': 'mae'
    },
    'is_multi_task': True,
    'model_name': 'SharedBackboneMultiTask',
    'output_names': ['classification_output', 'regression_output']
}

def get_shared_backbone_config():
    """Returns the model builder and config."""
    return build_shared_backbone_model, SHARED_BACKBONE_CONFIG
