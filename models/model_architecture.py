"""
Bi-LSTM Model Architecture for ECG Arrhythmia Classification
Defines the neural network model for 5-class heartbeat classification
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import tensorflow.keras.backend as K
import numpy as np
from typing import Tuple, Optional


class FocalLoss(keras.losses.Loss):
    """
    Focal Loss for addressing extreme class imbalance
    
    Focuses training on hard-to-classify examples by down-weighting
    easy examples. This is particularly useful for ECG classification
    where some classes (Fusion, Unknown) are severely underrepresented.
    
    Formula: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    
    Args:
        alpha: Weighting factor for class balance (0-1)
        gamma: Focusing parameter (higher = more focus on hard examples)
               Recommended: 2.0 for extreme imbalance
    
    Reference:
        Lin et al. "Focal Loss for Dense Object Detection" (2017)
    """
    def __init__(self, alpha=0.25, gamma=2.0, name='focal_loss', **kwargs):
        # Accept all keras.losses.Loss parameters
        super().__init__(name=name, **kwargs)
        self.alpha = alpha
        self.gamma = gamma
    
    def call(self, y_true, y_pred):
        # Convert y_true to one-hot if needed
        y_true = tf.cast(y_true, tf.int32)
        y_true = tf.one_hot(y_true, depth=tf.shape(y_pred)[-1])
        y_true = tf.cast(y_true, tf.float32)
        
        # Clip predictions to prevent log(0)
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)
        
        # Calculate focal loss
        cross_entropy = -y_true * K.log(y_pred)
        loss = self.alpha * K.pow(1 - y_pred, self.gamma) * cross_entropy
        
        return K.mean(K.sum(loss, axis=-1))
    
    def get_config(self):
        """Return configuration for serialization"""
        config = super().get_config()
        config.update({
            'alpha': self.alpha,
            'gamma': self.gamma
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        """Create instance from configuration"""
        return cls(**config)


class F1ScoreCallback(keras.callbacks.Callback):
    """
    Monitor F1-score for minority classes during training
    
    This callback prints F1-scores for all classes every N epochs
    to help monitor performance on underrepresented classes.
    """
    def __init__(self, validation_data, class_names, print_every=10):
        super().__init__()
        self.validation_data = validation_data
        self.class_names = class_names
        self.print_every = print_every
    
    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.print_every == 0:
            X_val, y_val = self.validation_data
            y_pred = np.argmax(self.model.predict(X_val, verbose=0), axis=1)
            
            from sklearn.metrics import classification_report
            report = classification_report(
                y_val, y_pred,
                target_names=self.class_names,
                output_dict=True,
                zero_division=0
            )
            
            print(f"\n{'='*70}")
            print(f"[Epoch {epoch+1}] F1-Scores by Class:")
            print(f"{'='*70}")
            for class_name in self.class_names:
                f1 = report[class_name]['f1-score']
                recall = report[class_name]['recall']
                precision = report[class_name]['precision']
                print(f"  {class_name:20s} - F1: {f1:.3f} | Precision: {precision:.3f} | Recall: {recall:.3f}")
            print(f"{'='*70}\n")


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


def build_cnn_lstm_model(input_shape: Tuple[int, int] = (216, 1),
                         num_classes: int = 5,
                         l2_reg: float = 0.001) -> keras.Model:
    """
    Build CNN-LSTM hybrid model for ECG classification
    
    This architecture uses CNN layers to extract local morphological features
    (QRS complex, P-wave, T-wave patterns) and LSTM layers to capture temporal
    dependencies between heartbeats. This approach has shown state-of-the-art
    results on MIT-BIH dataset.
    
    Architecture:
        Input (216x1)
        -> Conv1D(64) -> BatchNorm -> ReLU -> SpatialDropout -> MaxPool
        -> Conv1D(128) -> BatchNorm -> ReLU -> SpatialDropout -> MaxPool
        -> Bi-LSTM(64) -> Dropout
        -> Bi-LSTM(32) -> Dropout
        -> Dense(64) -> Dropout
        -> Dense(5, softmax)
    
    Args:
        input_shape: Shape of input (timesteps, features)
        num_classes: Number of output classes
        l2_reg: L2 regularization factor
    
    Returns:
        Keras model
    """
    inputs = layers.Input(shape=input_shape, name='ecg_input')
    
    # First Convolutional Block
    # Extract low-level morphological features
    x = layers.Conv1D(
        filters=64,
        kernel_size=5,
        padding='same',
        kernel_regularizer=regularizers.l2(l2_reg),
        name='conv1d_1'
    )(inputs)
    x = layers.BatchNormalization(name='bn_1')(x)
    x = layers.Activation('relu', name='relu_1')(x)
    x = layers.SpatialDropout1D(0.5, name='spatial_dropout_1')(x)
    x = layers.MaxPooling1D(pool_size=2, name='maxpool_1')(x)
    
    # Second Convolutional Block
    # Extract higher-level features
    x = layers.Conv1D(
        filters=128,
        kernel_size=3,
        padding='same',
        kernel_regularizer=regularizers.l2(l2_reg),
        name='conv1d_2'
    )(x)
    x = layers.BatchNormalization(name='bn_2')(x)
    x = layers.Activation('relu', name='relu_2')(x)
    x = layers.SpatialDropout1D(0.5, name='spatial_dropout_2')(x)
    x = layers.MaxPooling1D(pool_size=2, name='maxpool_2')(x)
    
    # First Bidirectional LSTM
    # Model temporal dependencies
    x = layers.Bidirectional(
        layers.LSTM(
            64,
            return_sequences=True,
            kernel_regularizer=regularizers.l2(l2_reg),
            recurrent_regularizer=regularizers.l2(l2_reg),
            name='lstm_1'
        ),
        name='bidirectional_lstm_1'
    )(x)
    x = layers.Dropout(0.5, name='dropout_1')(x)  # Increased from 0.4
    
    # Second Bidirectional LSTM
    # Further temporal processing
    x = layers.Bidirectional(
        layers.LSTM(
            32,
            return_sequences=False,
            kernel_regularizer=regularizers.l2(l2_reg),
            recurrent_regularizer=regularizers.l2(l2_reg),
            name='lstm_2'
        ),
        name='bidirectional_lstm_2'
    )(x)
    x = layers.Dropout(0.5, name='dropout_2')(x)  # Increased from 0.4
    
    # Dense layer
    x = layers.Dense(
        64,
        activation='relu',
        kernel_regularizer=regularizers.l2(l2_reg),
        name='dense_1'
    )(x)
    x = layers.Dropout(0.4, name='dropout_3')(x)  # Increased from 0.3
    
    # Output layer
    outputs = layers.Dense(num_classes, activation='softmax', name='output')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs, name='CNN_LSTM_ECG')
    
    return model


def build_rescnn_lstm_model(input_shape: Tuple[int, int] = (216, 1),
                             num_classes: int = 5,
                             l2_reg: float = 0.001) -> keras.Model:
    """
    Build ResNet-style CNN-LSTM model with residual connections
    
    This enhanced architecture adds residual (skip) connections to help with
    gradient flow and allow the network to learn identity mappings when needed.
    Residual connections have been shown to improve training of deep networks.
    
    Architecture:
        Input (216x1)
        -> Conv1D(64) -> BatchNorm -> ReLU -> [Residual Block 1]
        -> Conv1D(128) -> BatchNorm -> ReLU -> MaxPool -> [Residual Block 2]
        -> Conv1D(128) -> BatchNorm -> ReLU -> MaxPool
        -> Bi-LSTM(64) -> Dropout
        -> Bi-LSTM(32) -> Dropout
        -> Dense(64) -> Dropout
        -> Dense(5, softmax)
    
    Args:
        input_shape: Shape of input (timesteps, features)
        num_classes: Number of output classes
        l2_reg: L2 regularization factor
    
    Returns:
        Keras model
    """
    inputs = layers.Input(shape=input_shape, name='ecg_input')
    
    # Initial convolution
    x = layers.Conv1D(
        filters=64,
        kernel_size=7,
        padding='same',
        kernel_regularizer=regularizers.l2(l2_reg),
        name='conv1d_init'
    )(inputs)
    x = layers.BatchNormalization(name='bn_init')(x)
    x = layers.Activation('relu', name='relu_init')(x)
    
    # Residual Block 1
    residual = x
    x = layers.Conv1D(
        filters=64,
        kernel_size=5,
        padding='same',
        kernel_regularizer=regularizers.l2(l2_reg),
        name='conv1d_1a'
    )(x)
    x = layers.BatchNormalization(name='bn_1a')(x)
    x = layers.Activation('relu', name='relu_1a')(x)
    x = layers.SpatialDropout1D(0.3, name='spatial_dropout_1a')(x)
    
    x = layers.Conv1D(
        filters=64,
        kernel_size=5,
        padding='same',
        kernel_regularizer=regularizers.l2(l2_reg),
        name='conv1d_1b'
    )(x)
    x = layers.BatchNormalization(name='bn_1b')(x)
    
    # Add residual connection
    x = layers.Add(name='residual_add_1')([x, residual])
    x = layers.Activation('relu', name='relu_1b')(x)
    x = layers.MaxPooling1D(pool_size=2, name='maxpool_1')(x)
    
    # Residual Block 2
    residual = layers.Conv1D(
        filters=128,
        kernel_size=1,
        padding='same',
        kernel_regularizer=regularizers.l2(l2_reg),
        name='conv1d_residual_proj'
    )(x)
    
    x = layers.Conv1D(
        filters=128,
        kernel_size=3,
        padding='same',
        kernel_regularizer=regularizers.l2(l2_reg),
        name='conv1d_2a'
    )(x)
    x = layers.BatchNormalization(name='bn_2a')(x)
    x = layers.Activation('relu', name='relu_2a')(x)
    x = layers.SpatialDropout1D(0.3, name='spatial_dropout_2a')(x)
    
    x = layers.Conv1D(
        filters=128,
        kernel_size=3,
        padding='same',
        kernel_regularizer=regularizers.l2(l2_reg),
        name='conv1d_2b'
    )(x)
    x = layers.BatchNormalization(name='bn_2b')(x)
    
    # Add residual connection
    x = layers.Add(name='residual_add_2')([x, residual])
    x = layers.Activation('relu', name='relu_2b')(x)
    x = layers.MaxPooling1D(pool_size=2, name='maxpool_2')(x)
    
    # Additional conv layer for feature refinement
    x = layers.Conv1D(
        filters=128,
        kernel_size=3,
        padding='same',
        kernel_regularizer=regularizers.l2(l2_reg),
        name='conv1d_3'
    )(x)
    x = layers.BatchNormalization(name='bn_3')(x)
    x = layers.Activation('relu', name='relu_3')(x)
    x = layers.SpatialDropout1D(0.4, name='spatial_dropout_3')(x)
    
    # First Bidirectional LSTM
    x = layers.Bidirectional(
        layers.LSTM(
            64,
            return_sequences=True,
            kernel_regularizer=regularizers.l2(l2_reg),
            recurrent_regularizer=regularizers.l2(l2_reg),
            name='lstm_1'
        ),
        name='bidirectional_lstm_1'
    )(x)
    x = layers.Dropout(0.4, name='dropout_1')(x)
    
    # Second Bidirectional LSTM
    x = layers.Bidirectional(
        layers.LSTM(
            32,
            return_sequences=False,
            kernel_regularizer=regularizers.l2(l2_reg),
            recurrent_regularizer=regularizers.l2(l2_reg),
            name='lstm_2'
        ),
        name='bidirectional_lstm_2'
    )(x)
    x = layers.Dropout(0.4, name='dropout_2')(x)
    
    # Dense layer
    x = layers.Dense(
        64,
        activation='relu',
        kernel_regularizer=regularizers.l2(l2_reg),
        name='dense_1'
    )(x)
    x = layers.Dropout(0.3, name='dropout_3')(x)
    
    # Output layer
    outputs = layers.Dense(num_classes, activation='softmax', name='output')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs, name='ResCNN_LSTM_ECG')
    
    return model


def compile_model(model: keras.Model,
                  learning_rate: float = 0.001,
                  use_focal_loss: bool = True,
                  class_weights: Optional[dict] = None) -> keras.Model:
    """
    Compile model with optimizer and loss function
    
    Args:
        model: Keras model to compile
        learning_rate: Learning rate for Adam optimizer
        use_focal_loss: Use Focal Loss instead of standard cross-entropy
                       (Recommended for handling extreme class imbalance)
        class_weights: Class weights for handling imbalance
    
    Returns:
        Compiled model
    """
    # Adam optimizer with learning rate
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    
    # Use Focal Loss for better handling of class imbalance
    # This significantly improves performance on minority classes (F, Q)
    if use_focal_loss:
        loss = FocalLoss(alpha=0.25, gamma=2.0)
        print("✓ Using Focal Loss for extreme class imbalance handling")
    else:
        loss = keras.losses.SparseCategoricalCrossentropy()
        print("✓ Using standard Cross-Entropy loss")
    
    # Metrics
    metrics = [
        keras.metrics.SparseCategoricalAccuracy(name='accuracy'),
    ]
    
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics
    )
    
    return model


def get_callbacks(model_save_path: str = './models/best_model.h5',
                  patience: int = 10,
                  min_delta: float = 0.001,
                  validation_data: Optional[tuple] = None,
                  class_names: Optional[list] = None) -> list:
    """
    Get training callbacks
    
    Args:
        model_save_path: Path to save best model
        patience: Patience for early stopping
        min_delta: Minimum change to consider as improvement
        validation_data: Tuple of (X_val, y_val) for F1 monitoring
        class_names: List of class names for F1 monitoring
    
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
    
    # Add F1-score monitoring if validation data provided
    if validation_data is not None and class_names is not None:
        callbacks.append(
            F1ScoreCallback(
                validation_data=validation_data,
                class_names=class_names,
                print_every=10
            )
        )
    
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
        model_type: Type of model architecture
            - 'standard': Basic Bi-LSTM (original)
            - 'enhanced': Bi-LSTM with attention mechanism
            - 'cnn_lstm': CNN-LSTM hybrid (RECOMMENDED for best performance)
            - 'rescnn_lstm': ResNet-style CNN-LSTM with residual connections
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
    elif model_type == 'cnn_lstm':
        model = build_cnn_lstm_model(input_shape, num_classes)
    elif model_type == 'rescnn_lstm':
        model = build_rescnn_lstm_model(input_shape, num_classes)
    else:
        raise ValueError(f"Unknown model type: {model_type}. "
                        f"Choose from: 'standard', 'enhanced', 'cnn_lstm', 'rescnn_lstm'")
    
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


