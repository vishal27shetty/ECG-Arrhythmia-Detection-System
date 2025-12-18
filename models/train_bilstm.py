"""
Training Script for Bi-LSTM ECG Arrhythmia Classifier
Trains the model on MIT-BIH dataset and evaluates performance
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from imblearn.over_sampling import SMOTE
import tensorflow as tf
from tensorflow import keras
import pickle
import json
from datetime import datetime

# Import custom modules
from model_architecture import create_model, get_callbacks, calculate_class_weights, print_model_summary
from dataset_preparation import MITBIHDataset, INDEX_TO_CLASS


class ECGModelTrainer:
    """
    Trainer class for ECG arrhythmia classification model
    """
    
    def __init__(self, data_dir: str = './data/mit_bih', 
                 model_dir: str = './models',
                 results_dir: str = './results'):
        """
        Initialize trainer
        
        Args:
            data_dir: Directory containing MIT-BIH data
            model_dir: Directory to save models
            results_dir: Directory to save results and plots
        """
        self.data_dir = data_dir
        self.model_dir = model_dir
        self.results_dir = results_dir
        
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(results_dir, exist_ok=True)
        
        self.model = None
        self.history = None
        self.class_names = ['N', 'S', 'V', 'F', 'Q']
        self.class_full_names = ['Normal', 'Supraventricular', 'Ventricular', 'Fusion', 'Unknown']
    
    def load_data(self, balance_strategy: str = 'hybrid') -> tuple:
        """
        Load and prepare dataset with advanced balancing
        
        Args:
            balance_strategy: Balancing strategy
                - 'none': No balancing
                - 'smote': SMOTE oversampling
                - 'hybrid': SMOTE + undersampling majority
                - 'weights': Use class weights only (no resampling)
        
        Returns:
            Tuple of (X_train, y_train, X_val, y_val, X_test, y_test)
        """
        print("Loading MIT-BIH dataset...")
        
        dataset = MITBIHDataset(data_dir=self.data_dir, beat_length=216)
        
        # Check if prepared data exists
        prepared_file = os.path.join(self.data_dir, 'prepared_data.pkl')
        
        if os.path.exists(prepared_file):
            print("Loading prepared data from file...")
            X_train, y_train, X_test, y_test = dataset.load_prepared_data()
        else:
            print("Preparing data from scratch...")
            print("This will download MIT-BIH database if not present...")
            
            # Download if necessary
            if not os.path.exists(os.path.join(self.data_dir, '100.dat')):
                dataset.download_all_records()
            
            # Prepare data
            X_train, y_train, X_test, y_test = dataset.prepare_train_test_split()
            dataset.save_prepared_data(X_train, y_train, X_test, y_test)
        
        # Print original distribution
        from collections import Counter
        print("\nOriginal training class distribution:")
        counter = Counter(y_train)
        for class_idx in sorted(counter.keys()):
            count = counter[class_idx]
            pct = (count / len(y_train)) * 100
            print(f"  {self.class_names[class_idx]}: {count} ({pct:.1f}%)")
        
        # Split training data into train and validation BEFORE balancing
        # Use stratification but handle classes with too few samples
        try:
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, test_size=0.15, random_state=42, stratify=y_train
            )
        except ValueError:
            # If stratification fails due to too few samples in some classes
            print("\n⚠️ Warning: Some classes have too few samples for stratification")
            print("   Performing split without stratification...")
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, test_size=0.15, random_state=42
            )
        
        print(f"\nDataset sizes before balancing:")
        print(f"  Training: {len(X_train)}")
        print(f"  Validation: {len(X_val)}")
        print(f"  Test: {len(X_test)}")
        
        # Apply balancing strategy
        if balance_strategy == 'smote':
            X_train, y_train = self._apply_smote(X_train, y_train)
        elif balance_strategy == 'hybrid':
            X_train, y_train = self._apply_hybrid_balancing(X_train, y_train)
        elif balance_strategy == 'weights':
            print("\nUsing class weights only (no resampling)")
        elif balance_strategy == 'none':
            print("\nNo balancing applied")
        else:
            raise ValueError(f"Unknown balance_strategy: {balance_strategy}")
        
        return X_train, y_train, X_val, y_val, X_test, y_test
    
    def _apply_smote(self, X_train, y_train):
        """Apply SMOTE with adaptive k_neighbors"""
        print("\nApplying SMOTE for class balancing...")
        
        from collections import Counter
        counter = Counter(y_train)
        
        # Find minimum samples in any class
        min_samples = min(counter.values())
        
        # Adjust k_neighbors based on smallest class
        k_neighbors = min(5, max(1, min_samples - 1))
        
        if k_neighbors < 5:
            print(f"⚠️ Warning: Using k_neighbors={k_neighbors} due to small class sizes")
        
        X_train_2d = X_train.reshape(X_train.shape[0], -1)
        
        try:
            from imblearn.over_sampling import SMOTE
            smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
            X_train_balanced, y_train_balanced = smote.fit_resample(X_train_2d, y_train)
            
            X_train = X_train_balanced.reshape(-1, 216, 1)
            y_train = y_train_balanced
            
            print(f"✅ After SMOTE: {len(X_train)} training samples")
            
            # Print new class distribution
            counter = Counter(y_train)
            print("New class distribution:")
            for class_idx in sorted(counter.keys()):
                count = counter[class_idx]
                pct = (count / len(y_train)) * 100
                print(f"  {self.class_names[class_idx]}: {count} ({pct:.1f}%)")
        
        except Exception as e:
            print(f"⚠️ SMOTE failed: {str(e)}")
            print("   Continuing without SMOTE...")
        
        return X_train, y_train
    
    def _apply_hybrid_balancing(self, X_train, y_train):
        """Apply hybrid: undersample majority + SMOTE minorities"""
        print("\nApplying hybrid balancing (undersample + SMOTE)...")
        
        from collections import Counter
        from imblearn.over_sampling import SMOTE
        from imblearn.under_sampling import RandomUnderSampler
        from imblearn.pipeline import Pipeline
        
        counter = Counter(y_train)
        min_samples = min(counter.values())
        
        # Strategy 1: First undersample the majority class (N) to reduce dominance
        # Target: Keep N at ~40% of dataset instead of 90%
        majority_target = int(len(y_train) * 0.4)
        
        # Strategy 2: Then oversample minorities to match a reasonable target
        # For very small classes, set a minimum target
        min_target = max(1000, min_samples * 10)
        
        # Define sampling strategy
        undersample_strategy = {0: majority_target}  # Class N (Normal)
        
        # Calculate SMOTE target for each minority class
        oversample_strategy = {}
        for class_idx, count in counter.items():
            if class_idx != 0:  # Not the majority class
                if count < min_target:
                    oversample_strategy[class_idx] = min_target
        
        X_train_2d = X_train.reshape(X_train.shape[0], -1)
        
        try:
            # Adjust k_neighbors
            k_neighbors = min(5, max(1, min_samples - 1))
            
            # Create pipeline: undersample then oversample
            if oversample_strategy:
                pipeline = Pipeline([
                    ('undersample', RandomUnderSampler(sampling_strategy=undersample_strategy, random_state=42)),
                    ('oversample', SMOTE(sampling_strategy=oversample_strategy, random_state=42, k_neighbors=k_neighbors))
                ])
            else:
                # Only undersample if SMOTE not needed
                pipeline = Pipeline([
                    ('undersample', RandomUnderSampler(sampling_strategy=undersample_strategy, random_state=42))
                ])
            
            X_train_balanced, y_train_balanced = pipeline.fit_resample(X_train_2d, y_train)
            
            X_train = X_train_balanced.reshape(-1, 216, 1)
            y_train = y_train_balanced
            
            print(f"✅ After hybrid balancing: {len(X_train)} training samples")
            
            # Print new class distribution
            counter = Counter(y_train)
            print("New class distribution:")
            for class_idx in sorted(counter.keys()):
                count = counter[class_idx]
                pct = (count / len(y_train)) * 100
                print(f"  {self.class_names[class_idx]}: {count} ({pct:.1f}%)")
        
        except Exception as e:
            print(f"⚠️ Hybrid balancing failed: {str(e)}")
            print("   Falling back to SMOTE only...")
            return self._apply_smote(X_train, y_train)
        
        return X_train, y_train
    
    def train(self, X_train, y_train, X_val, y_val,
              model_type: str = 'standard',
              epochs: int = 50,
              batch_size: int = 128,
              learning_rate: float = 0.001,
              use_class_weights: bool = True):
        """
        Train the model
        
        Args:
            X_train: Training data
            y_train: Training labels
            X_val: Validation data
            y_val: Validation labels
            model_type: Type of model architecture
            epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            use_class_weights: Whether to use class weights
        """
        print("\n" + "="*70)
        print("Starting Model Training")
        print("="*70)
        
        # Create model
        self.model = create_model(
            model_type=model_type,
            input_shape=(216, 1),
            num_classes=5,
            learning_rate=learning_rate
        )
        
        # Calculate class weights
        class_weights = None
        if use_class_weights:
            class_weights = calculate_class_weights(y_train, num_classes=5)
        
        # Get callbacks
        model_save_path = os.path.join(self.model_dir, 'best_model.h5')
        callbacks = get_callbacks(model_save_path=model_save_path, patience=10)
        
        # Train model
        print(f"\nTraining for {epochs} epochs with batch size {batch_size}...")
        
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            class_weight=class_weights,
            callbacks=callbacks,
            verbose=1
        )
        
        print("\nTraining completed!")
        
        # Save final model
        final_model_path = os.path.join(self.model_dir, 'trained_model.h5')
        self.model.save(final_model_path)
        print(f"Model saved to {final_model_path}")
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate model on test set
        
        Args:
            X_test: Test data
            y_test: Test labels
        
        Returns:
            Dictionary with evaluation metrics
        """
        print("\n" + "="*70)
        print("Evaluating Model on Test Set")
        print("="*70)
        
        # Predict
        y_pred_proba = self.model.predict(X_test, verbose=0)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        # Calculate metrics
        test_loss, test_acc, _ = self.model.evaluate(X_test, y_test, verbose=0)
        
        print(f"\nTest Accuracy: {test_acc*100:.2f}%")
        print(f"Test Loss: {test_loss:.4f}")
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(
            y_test, y_pred,
            target_names=self.class_full_names,
            digits=4
        ))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Save results
        results = {
            'test_accuracy': float(test_acc),
            'test_loss': float(test_loss),
            'confusion_matrix': cm.tolist(),
            'classification_report': classification_report(
                y_test, y_pred,
                target_names=self.class_full_names,
                output_dict=True
            )
        }
        
        results_file = os.path.join(self.results_dir, 'evaluation_results.json')
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to {results_file}")
        
        return results, y_pred, y_pred_proba
    
    def plot_training_history(self):
        """Plot training history"""
        if self.history is None:
            print("No training history available")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Accuracy plot
        axes[0].plot(self.history.history['accuracy'], label='Train Accuracy')
        axes[0].plot(self.history.history['val_accuracy'], label='Val Accuracy')
        axes[0].set_title('Model Accuracy', fontsize=14)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Loss plot
        axes[1].plot(self.history.history['loss'], label='Train Loss')
        axes[1].plot(self.history.history['val_loss'], label='Val Loss')
        axes[1].set_title('Model Loss', fontsize=14)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        plot_path = os.path.join(self.results_dir, 'training_history.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Training history plot saved to {plot_path}")
        plt.close()
    
    def plot_confusion_matrix(self, cm):
        """Plot confusion matrix"""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Normalize confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        im = ax.imshow(cm_normalized, interpolation='nearest', cmap='Blues')
        ax.figure.colorbar(im, ax=ax)
        
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               xticklabels=self.class_names,
               yticklabels=self.class_names,
               title='Confusion Matrix (Normalized)',
               ylabel='True Label',
               xlabel='Predicted Label')
        
        # Add text annotations
        thresh = cm_normalized.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, f'{cm_normalized[i, j]:.2f}\n({cm[i, j]})',
                       ha="center", va="center",
                       color="white" if cm_normalized[i, j] > thresh else "black",
                       fontsize=10)
        
        plt.tight_layout()
        
        plot_path = os.path.join(self.results_dir, 'confusion_matrix.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix plot saved to {plot_path}")
        plt.close()


def main():
    """Main training function"""
    print("="*70)
    print("ECG Arrhythmia Classification - Model Training")
    print("="*70)
    
    # Configuration
    config = {
        'model_type': 'standard',  # or 'enhanced'
        'epochs': 50,
        'batch_size': 128,
        'learning_rate': 0.001,
        'balance_strategy': 'hybrid',  # 'none', 'smote', 'hybrid', 'weights'
        'use_class_weights': True
    }
    
    print("\nConfiguration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    print("\n" + "="*70)
    print("IMPORTANT: Addressing Class Imbalance")
    print("="*70)
    print("The MIT-BIH dataset is highly imbalanced:")
    print("  • Normal beats (N): ~90% of data")
    print("  • Minority classes (S, V, F, Q): ~10% combined")
    print("\nBalancing strategies:")
    print("  • 'hybrid': Undersample N + Oversample minorities (RECOMMENDED)")
    print("  • 'smote': Only oversample minorities")
    print("  • 'weights': Use class weights without resampling")
    print("  • 'none': No balancing (not recommended)")
    print(f"\nUsing: {config['balance_strategy']}")
    print("="*70 + "\n")
    
    # Initialize trainer
    trainer = ECGModelTrainer(
        data_dir='./data/mit_bih',
        model_dir='./models',
        results_dir='./results'
    )
    
    # Load data
    X_train, y_train, X_val, y_val, X_test, y_test = trainer.load_data(
        balance_strategy=config['balance_strategy']
    )
    
    # Train model
    trainer.train(
        X_train, y_train, X_val, y_val,
        model_type=config['model_type'],
        epochs=config['epochs'],
        batch_size=config['batch_size'],
        learning_rate=config['learning_rate'],
        use_class_weights=config['use_class_weights']
    )
    
    # Plot training history
    trainer.plot_training_history()
    
    # Evaluate on test set
    results, y_pred, y_pred_proba = trainer.evaluate(X_test, y_test)
    
    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    trainer.plot_confusion_matrix(cm)
    
    print("\n" + "="*70)
    print("Training and Evaluation Complete!")
    print("="*70)
    print(f"\nFinal Test Accuracy: {results['test_accuracy']*100:.2f}%")
    print(f"\nCheck the './results' directory for detailed results and plots.")
    print(f"Model saved in './models' directory.")


if __name__ == "__main__":
    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    main()

