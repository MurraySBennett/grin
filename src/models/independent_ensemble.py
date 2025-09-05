import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from src.utils.custom_losses import classification_loss, custom_means_loss, custom_cov_loss, custom_mae, dense_residual_block

def build_classification_model(input_shape, num_models, dropout_rate=0.3, activation='tanh'):
    """
    Builds a standalone classification model.
    This model takes the raw data and predicts the model class.
    """
    model_input = keras.Input(shape=(input_shape,), name='cls_input')
    
    x = layers.Dense(512, activation=activation, name='cls_dense1')(model_input)
    x = dense_residual_block(x, 512, activation, dropout_rate)
    x = layers.Dense(256, activation=activation, name='cls_dense2')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(128, activation=activation, name='cls_dense3')(x)
    
    classification_output = layers.Dense(num_models, activation='softmax', name='classification_output')(x)
    
    return keras.Model(inputs=model_input, outputs=classification_output)

def build_regression_model(input_shape, num_models, dropout_rate=0.3, activation='tanh'):
    """
    Builds a standalone regression model that is conditional on the model class.
    This model takes the raw data AND the one-hot encoded model class.
    """
    data_input = keras.Input(shape=(input_shape,), name='data_input')
    class_input = keras.Input(shape=(num_models,), name='class_input')
    x = layers.Concatenate(name='conditional_input')([data_input, class_input])
    
    # --- Regression Backbone ---
    x = layers.Dense(512, activation=activation, name='reg_dense1')(x)
    x = dense_residual_block(x, 512, activation, dropout_rate)
    x = layers.Dense(256, activation=activation, name='reg_dense2')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(128, activation=activation, name='reg_dense3')(x)

    # --- Output Heads for Regression ---
    means_output = layers.Dense(6, activation='linear', name='means_output')(x)
    cov_output = layers.Dense(4, activation='tanh', name='cov_output')(x)
    crit_output = layers.Dense(2, activation='linear', name='crit_output')(x)

    return keras.Model(
        inputs=[data_input, class_input], 
        outputs=[means_output, cov_output, crit_output]
    )

def build_independent_models(input_shape, num_models, dropout_rate=0.3, activation='tanh'):
    """
    This is a wrapper function to return the two models used in the ensemble.
    """
    cls_model = build_classification_model(input_shape, num_models, dropout_rate=dropout_rate, activation=activation)
    reg_model = build_regression_model(input_shape, num_models, dropout_rate=dropout_rate, activation=activation)
    return cls_model, reg_model


INDEPENDENT_ENSEMBLE_CONFIG = {
    'cls_losses': {
        'classification_output': classification_loss,
    },
    'reg_losses': {
        'means_output': custom_means_loss,
        'cov_output': custom_cov_loss,
        'crit_output': 'mae'
    },
    'loss_weights': {
        'means_output': 1.0,
        'cov_output': 2.0,
        'crit_output': 1.0
    },
    'cls_metrics': {
        'classification_output': 'accuracy',
    },
    'reg_metrics': {
        'means_output': custom_mae,
        'cov_output': custom_mae,
        'crit_output': 'mae'
    },
    'is_multi_task': False,
    'model_name': 'IndependentEnsemble'
}

def get_independent_ensemble_config():
    """Returns the model builder and config."""
    return build_independent_ensemble_models, INDEPENDENT_ENSEMBLE_CONFIG
