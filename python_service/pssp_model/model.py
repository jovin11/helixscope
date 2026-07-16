"""CNN-LSTM architecture used by the bundled HelixScope checkpoint.

TensorFlow is imported inside :func:`build_model`, so lightweight web tests and
quality checks do not need to initialize the ML runtime.
"""

from __future__ import annotations


def build_model(
    *,
    sequence_length: int = 400,
    num_features: int = 20,
    num_classes: int = 3,
    learning_rate: float = 0.00015,
):
    """Build and compile the CNN-LSTM architecture.

    Change this function when experimenting with the architecture. Existing
    checkpoints must continue to be loaded with the architecture that produced
    them; the bundled ``best_model_v2.keras`` already contains its graph.
    """

    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import (
        LSTM,
        BatchNormalization,
        Conv1D,
        Dense,
        Dropout,
        Input,
        MaxPooling1D,
        TimeDistributed,
    )
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.regularizers import l2

    model = Sequential(name="helixscope_cnn_lstm")
    model.add(Input(shape=(sequence_length, num_features)))

    for filters in (64, 128, 128, 256):
        model.add(
            Conv1D(
                filters=filters,
                kernel_size=3,
                activation="relu",
                padding="same",
                kernel_regularizer=l2(0.002),
            )
        )
        # A pool size of one intentionally preserves one output per residue.
        model.add(MaxPooling1D(pool_size=1))
        model.add(Dropout(0.2))

    model.add(BatchNormalization())
    for _ in range(3):
        model.add(LSTM(128, return_sequences=True, dropout=0.4))

    model.add(
        TimeDistributed(Dense(64, activation="relu", kernel_regularizer=l2(0.002)))
    )
    model.add(Dropout(0.6))
    model.add(
        TimeDistributed(
            Dense(num_classes, activation="softmax", kernel_regularizer=l2(0.002))
        )
    )
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
