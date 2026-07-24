# Missing LSTM model files

`config.LSTM_MODEL_FILES` and `scripts/train_lstm_models.py` expect these modules
to live here, but they were not present in the original flat directory:

- standard_lstm.py
- bidirectional_lstm.py
- gru_model.py
- cnn_lstm.py

Each is expected to expose a `get_model_config()` returning `(model_builder, config)`.
Until they are restored, `scripts/train_lstm_models.py` will fail at runtime.
