"""
Bi-LSTM Model Architecture for ECG Arrhythmia Classification
Defines the neural network model for 5-class heartbeat classification
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import numpy as np
from typing import Tuple, Optional


def build_bilstm_model(input_shape: Tuple[int, int] = (216, 1), 
                       num_classes: int = 5,
                       lstm_units: list = [128, 64],
                       dropout_rate: float = 0.5,
                       dense_units: int = 32,
                       l2_reg: float = 0.001) -> keras.Model:
    """
    Build Bidirectional LSTM model for ECG classification
    
    Architecture:
        Input -> Bi-LSTM(128) -> Dropout -> Bi-LSTM(64) -> Dropout -> Dense(32) -> Dense(5)
    
    Args:
        input_shape: Shape of input (timesteps, features)
        num_classes: Number of output classes
        lstm_units: List of LSTM units for each layer
        dropout_rate: Dropout rate for regularization
        dense_units: Units in dense layer
        l2_reg: L2 regularization factor
    
    Returns:
        Compiled Keras model
    """
    model = models.Sequential(name='BiLSTM_ECG_Classifier')
    
    # Input layer
    model.add(layers.Input(shape=input_shape, name='ecg_input'))
    
    # First Bidirectional LSTM layer
    model.add(layers.Bidirectional(
        layers.LSTM(
            lstm_units[0],
            return_sequences=True,
            kernel_regularizer=regularizers.l2(l2_reg),
            recurrent_regularizer=regularizers.l2(l2_reg),
            name='lstm_1'
        ),
        name='bidirectional_lstm_1'
    ))
    model.add(layers.Dropout(dropout_rate, name='dropout_1'))
    
    # Second Bidirectional LSTM layer
    model.add(layers.Bidirectional(
        layers.LSTM(
            lstm_units[1],
            return_sequences=False,
            kernel_regularizer=regularizers.l2(l2_reg),
            recurrent_regularizer=regularizers.l2(l2_reg),
            name='lstm_2'
        ),
        name='bidirectional_lstm_2'
    ))
    model.add(layers.Dropout(dropout_rate, name='dropout_2'))
    
    # Dense layer
    model.add(layers.Dense(
        dense_units,
        activation='relu',
        kernel_regularizer=regularizers.l2(l2_reg),
        name='dense_1'
    ))
    
    # Output layer
    model.add(layers.Dense(
        num_classes,
        activation='softmax',
        name='output'
    ))
    
    return model


def build_enhanced_bilstm_model(input_shape: Tuple[int, int] = (216, 1),
                                num_classes: int = 5) -> keras.Model:
    """
    Build enhanced Bi-LSTM model with batch normalization and attention
    
    Args:
        input_shape: Shape of input (timesteps, features)
        num_classes: Number of output classes
    
    Returns:
        Compiled Keras model
    """
    inputs = layers.Input(shape=input_shape, name='ecg_input')
    
    # First Bi-LSTM layer with batch normalization
    x = layers.Bidirectional(
        layers.LSTM(128, return_sequences=True, name='lstm_1')
    )(inputs)
    x = layers.BatchNormalization(name='bn_1')(x)
    x = layers.Dropout(0.5, name='dropout_1')(x)
    
    # Second Bi-LSTM layer
    x = layers.Bidirectional(
        layers.LSTM(64, return_sequences=True, name='lstm_2')
    )(x)
    x = layers.BatchNormalization(name='bn_2')(x)
    x = layers.Dropout(0.5, name='dropout_2')(x)
    
    # Attention mechanism
    attention = layers.Dense(1, activation='tanh', name='attention_score')(x)
    attention = layers.Flatten(name='attention_flatten')(attention)
    attention = layers.Activation('softmax', name='attention_weights')(attention)
    attention = layers.RepeatVector(128, name='attention_repeat')(attention)
    attention = layers.Permute([2, 1], name='attention_permute')(attention)
    
    # Apply attention
    x = layers.Multiply(name='attention_multiply')([x, attention])
    x = layers.Lambda(lambda xin: tf.reduce_sum(xin, axis=1), name='attention_sum')(x)
    
    # Dense layers
    x = layers.Dense(32, activation='relu', name='dense_1')(x)
    x = layers.Dropout(0.3, name='dropout_3')(x)
    
    # Output layer
    outputs = layers.Dense(num_classes, activation='softmax', name='output')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs, name='Enhanced_BiLSTM_ECG')
    
    return model


def compile_model(model: keras.Model,
                  learning_rate: float = 0.001,
                  class_weights: Optional[dict] = None) -> keras.Model:
    """
    Compile model with optimizer and loss function
    
    Args:
        model: Keras model to compile
        learning_rate: Learning rate for Adam optimizer
        class_weights: Class weights for handling imbalance
    
    Returns:
        Compiled model
    """
    # Adam optimizer with learning rate
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    
    # Categorical crossentropy for multi-class classification
    loss = keras.losses.SparseCategoricalCrossentropy()
    
    # Metrics
    metrics = [
        keras.metrics.SparseCategoricalAccuracy(name='accuracy'),
        keras.metrics.SparseCategoricalCrossentropy(name='crossentropy'),
    ]
    
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics
    )
    
    return model


def get_callbacks(model_save_path: str = './models/best_model.h5',
                  patience: int = 10,
                  min_delta: float = 0.001) -> list:
    """
    Get training callbacks
    
    Args:
        model_save_path: Path to save best model
        patience: Patience for early stopping
        min_delta: Minimum change to consider as improvement
    
    Returns:
        List of callbacks
    """
    callbacks = [
        # Save best model
        ModelCheckpoint(
            filepath=model_save_path,
            monitor='val_accuracy',
            mode='max',
            save_best_only=True,
            verbose=1
        ),
        
        # Early stopping
        EarlyStopping(
            monitor='val_loss',
            patience=patience,
            min_delta=min_delta,
            restore_best_weights=True,
            verbose=1
        ),
        
        # Reduce learning rate on plateau
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
    ]
    
    return callbacks


def calculate_class_weights(y_train: np.ndarray, num_classes: int = 5) -> dict:
    """
    Calculate class weights for handling class imbalance
    
    Args:
        y_train: Training labels
        num_classes: Number of classes
    
    Returns:
        Dictionary of class weights
    """
    from sklearn.utils.class_weight import compute_class_weight
    
    # Compute class weights
    classes = np.arange(num_classes)
    weights = compute_class_weight(
        class_weight='balanced',
        classes=classes,
        y=y_train
    )
    
    class_weights = {i: weights[i] for i in range(num_classes)}
    
    print("Class weights:")
    class_names = ['N', 'S', 'V', 'F', 'Q']
    for i, name in enumerate(class_names):
        print(f"  {name}: {class_weights[i]:.3f}")
    
    return class_weights


def print_model_summary(model: keras.Model):
    """Print model architecture summary"""
    print("\nModel Architecture:")
    print("=" * 70)
    model.summary()
    print("=" * 70)
    
    # Count parameters
    total_params = model.count_params()
    print(f"\nTotal parameters: {total_params:,}")
    
    # Estimate model size
    model_size_mb = (total_params * 4) / (1024 ** 2)  # Assuming float32
    print(f"Estimated model size: {model_size_mb:.2f} MB")


def create_model(model_type: str = 'standard',
                 input_shape: Tuple[int, int] = (216, 1),
                 num_classes: int = 5,
                 learning_rate: float = 0.001) -> keras.Model:
    """
    Create and compile model
    
    Args:
        model_type: Type of model ('standard' or 'enhanced')
        input_shape: Input shape
        num_classes: Number of classes
        learning_rate: Learning rate
    
    Returns:
        Compiled Keras model
    """
    if model_type == 'standard':
        model = build_bilstm_model(input_shape, num_classes)
    elif model_type == 'enhanced':
        model = build_enhanced_bilstm_model(input_shape, num_classes)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model = compile_model(model, learning_rate)
    print_model_summary(model)
    
    return model


if __name__ == "__main__":
    print("Building Bi-LSTM model for ECG classification...")
    
    # Create standard model
    model = create_model(model_type='standard', input_shape=(216, 1), num_classes=5)
    
    # Test with random data
    print("\nTesting model with random data...")
    X_test = np.random.randn(10, 216, 1)
    y_pred = model.predict(X_test, verbose=0)
    
    print(f"Input shape: {X_test.shape}")
    print(f"Output shape: {y_pred.shape}")
    print(f"Sample prediction: {y_pred[0]}")
    print(f"Predicted class: {np.argmax(y_pred[0])}")
    
    print("\nModel created successfully!")

