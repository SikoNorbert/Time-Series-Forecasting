import numpy as np
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, LSTM
from tensorflow.keras.callbacks import EarlyStopping

def create_ann_tf(seq_length, n_features):
    return Sequential([
        Flatten(input_shape=(seq_length, n_features)),
        Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(n_features)
    ])

def create_cnn_tf(seq_length, n_features):
    return Sequential([
        Conv1D(16, kernel_size=3, activation='relu', input_shape=(seq_length, n_features), padding='same'),
        MaxPooling1D(2),
        Conv1D(32, kernel_size=3, activation='relu', padding='same'),
        Flatten(),
        tf.keras.layers.Dropout(0.2),
        Dense(n_features)
    ])

def create_rnn_tf(seq_length, n_features):
    return Sequential([
        LSTM(32, input_shape=(seq_length, n_features)),
        tf.keras.layers.Dropout(0.2),
        Dense(n_features)
    ])

def train_evaluate_tf(model, X_train, y_train, X_test, y_test, batch_size=16, epochs=20):
    model.compile(optimizer='adam', loss='mse')
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.2, callbacks=[early_stopping], verbose=1)
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    return rmse, y_pred, history.history['loss'], history.history['val_loss']
