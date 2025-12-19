"""
MIT-BIH Arrhythmia Database Preparation
Downloads and preprocesses the dataset for training
"""

import os
import numpy as np
import wfdb
from typing import Dict, List, Tuple
from collections import Counter
import pickle


# MIT-BIH arrhythmia class mapping (AAMI standard)
AAMI_CLASSES = {
    'N': ['N', 'L', 'R', 'e', 'j'],  # Normal beat
    'S': ['A', 'a', 'J', 'S'],        # Supraventricular ectopic beat
    'V': ['V', 'E'],                   # Ventricular ectopic beat
    'F': ['F'],                        # Fusion beat
    'Q': ['/', 'f', 'Q']              # Unknown beat
}

# Reverse mapping: annotation symbol -> class
ANNOTATION_TO_CLASS = {}
for class_label, annotations in AAMI_CLASSES.items():
    for ann in annotations:
        ANNOTATION_TO_CLASS[ann] = class_label

CLASS_TO_INDEX = {'N': 0, 'S': 1, 'V': 2, 'F': 3, 'Q': 4}
INDEX_TO_CLASS = {v: k for k, v in CLASS_TO_INDEX.items()}


class MITBIHDataset:
    """
    MIT-BIH Arrhythmia Database handler
    Downloads, processes, and prepares data for training
    """
    
    def __init__(self, data_dir: str = './data/mit_bih', beat_length: int = 216):
        """
        Initialize dataset handler
        
        Args:
            data_dir: Directory to store MIT-BIH data
            beat_length: Length of each beat segment in samples
        """
        self.data_dir = data_dir
        self.beat_length = beat_length
        self.sampling_rate = 360  # MIT-BIH sampling rate
        
        # Beat extraction window (asymmetric around R-peak)
        self.pre_r = int(0.25 * beat_length)
        self.post_r = beat_length - self.pre_r
        
        # MIT-BIH record numbers
        self.all_records = [
            100, 101, 102, 103, 104, 105, 106, 107, 108, 109,
            111, 112, 113, 114, 115, 116, 117, 118, 119, 121,
            122, 123, 124, 200, 201, 202, 203, 205, 207, 208,
            209, 210, 212, 213, 214, 215, 217, 219, 220, 221,
            222, 223, 228, 230, 231, 232, 233, 234
        ]
        
        # Standard train/test split (DS1/DS2)
        self.train_records = [
            101, 106, 108, 109, 112, 114, 115, 116, 118, 119,
            122, 124, 201, 203, 205, 207, 208, 209, 215, 220, 223, 230
        ]
        
        self.test_records = [
            100, 103, 105, 111, 113, 117, 121, 123, 200, 202,
            210, 212, 213, 214, 217, 219, 221, 222, 228, 231, 232, 233, 234
        ]
        
        os.makedirs(data_dir, exist_ok=True)
    
    def download_record(self, record_number: int) -> bool:
        """
        Download a single MIT-BIH record
        
        Args:
            record_number: Record number (e.g., 100)
        
        Returns:
            True if successful, False otherwise
        """
        record_name = str(record_number)
        
        try:
            print(f"Downloading record {record_name}...")
            wfdb.dl_database('mitdb', self.data_dir, [record_name])
            return True
        except Exception as e:
            print(f"Error downloading record {record_name}: {e}")
            return False
    
    def download_all_records(self):
        """Download all MIT-BIH records"""
        print(f"Downloading {len(self.all_records)} MIT-BIH records...")
        
        for record_num in self.all_records:
            self.download_record(record_num)
        
        print("Download complete!")
    
    def load_record(self, record_number: int) -> Tuple[np.ndarray, wfdb.Annotation]:
        """
        Load a single MIT-BIH record with annotations
        
        Args:
            record_number: Record number
        
        Returns:
            Tuple of (signal, annotation)
        """
        record_name = str(record_number)
        record_path = os.path.join(self.data_dir, record_name)
        
        # Load signal
        record = wfdb.rdrecord(record_path)
        signal = record.p_signal[:, 0]  # Use first channel (MLII or V1)
        
        # Load annotations
        annotation = wfdb.rdann(record_path, 'atr')
        
        return signal, annotation
    
    def extract_beats_from_record(self, record_number: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract labeled beats from a record
        
        Args:
            record_number: Record number
        
        Returns:
            Tuple of (beats, labels)
            beats: Array of shape (n_beats, beat_length)
            labels: Array of class indices
        """
        signal, annotation = self.load_record(record_number)
        
        beats = []
        labels = []
        
        # Extract beats around each annotated R-peak
        for i, (sample, symbol) in enumerate(zip(annotation.sample, annotation.symbol)):
            # Map annotation to AAMI class
            if symbol not in ANNOTATION_TO_CLASS:
                continue  # Skip unknown annotations
            
            class_label = ANNOTATION_TO_CLASS[symbol]
            
            # Define beat window
            start = sample - self.pre_r
            end = sample + self.post_r
            
            # Check bounds
            if start >= 0 and end <= len(signal):
                beat = signal[start:end]
                
                # Ensure exact length
                if len(beat) == self.beat_length:
                    beats.append(beat)
                    labels.append(CLASS_TO_INDEX[class_label])
        
        return np.array(beats), np.array(labels)
    
    def prepare_dataset(self, records: List[int], normalize: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare dataset from multiple records
        
        Args:
            records: List of record numbers
            normalize: Whether to normalize beats
        
        Returns:
            Tuple of (X, y)
            X: Beats array, shape (n_beats, beat_length, 1)
            y: Labels array, shape (n_beats,)
        """
        all_beats = []
        all_labels = []
        
        for record_num in records:
            try:
                print(f"Processing record {record_num}...")
                beats, labels = self.extract_beats_from_record(record_num)
                
                if len(beats) > 0:
                    all_beats.append(beats)
                    all_labels.append(labels)
                    
                    # Print class distribution for this record
                    counter = Counter(labels)
                    print(f"  Found {len(beats)} beats: {dict(counter)}")
            
            except Exception as e:
                print(f"  Error processing record {record_num}: {e}")
        
        # Concatenate all beats
        X = np.concatenate(all_beats, axis=0)
        y = np.concatenate(all_labels, axis=0)
        
        # Normalize each beat independently
        if normalize:
            X_normalized = np.zeros_like(X)
            for i in range(len(X)):
                beat = X[i]
                mean = np.mean(beat)
                std = np.std(beat)
                if std > 1e-6:
                    X_normalized[i] = (beat - mean) / std
                else:
                    X_normalized[i] = beat - mean
            X = X_normalized
        
        # Reshape for LSTM input: (samples, timesteps, features)
        X = X.reshape(-1, self.beat_length, 1)
        
        return X, y
    
    def prepare_train_test_split(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare standard train/test split (DS1/DS2)
        
        Returns:
            Tuple of (X_train, y_train, X_test, y_test)
        """
        print("Preparing training set (DS1)...")
        X_train, y_train = self.prepare_dataset(self.train_records)
        
        print("\nPreparing test set (DS2)...")
        X_test, y_test = self.prepare_dataset(self.test_records)
        
        print(f"\nDataset summary:")
        print(f"Training set: {X_train.shape[0]} beats")
        print(f"Test set: {X_test.shape[0]} beats")
        
        # Print class distributions
        print("\nTraining set class distribution:")
        train_counter = Counter(y_train)
        for class_idx in sorted(train_counter.keys()):
            class_name = INDEX_TO_CLASS[class_idx]
            count = train_counter[class_idx]
            percentage = (count / len(y_train)) * 100
            print(f"  {class_name}: {count} ({percentage:.1f}%)")
        
        print("\nTest set class distribution:")
        test_counter = Counter(y_test)
        for class_idx in sorted(test_counter.keys()):
            class_name = INDEX_TO_CLASS[class_idx]
            count = test_counter[class_idx]
            percentage = (count / len(y_test)) * 100
            print(f"  {class_name}: {count} ({percentage:.1f}%)")
        
        return X_train, y_train, X_test, y_test
    
    def save_prepared_data(self, X_train, y_train, X_test, y_test, filename: str = 'prepared_data.pkl'):
        """Save prepared data to file"""
        filepath = os.path.join(self.data_dir, filename)
        
        data = {
            'X_train': X_train,
            'y_train': y_train,
            'X_test': X_test,
            'y_test': y_test,
            'beat_length': self.beat_length,
            'sampling_rate': self.sampling_rate,
            'class_mapping': CLASS_TO_INDEX
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"Saved prepared data to {filepath}")
    
    def load_prepared_data(self, filename: str = 'prepared_data.pkl'):
        """Load prepared data from file"""
        filepath = os.path.join(self.data_dir, filename)
        
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        print(f"Loaded prepared data from {filepath}")
        return data['X_train'], data['y_train'], data['X_test'], data['y_test']


def main():
    """Main function to prepare MIT-BIH dataset"""
    print("MIT-BIH Arrhythmia Database Preparation")
    print("=" * 50)
    
    # Initialize dataset
    dataset = MITBIHDataset(data_dir='./data/mit_bih', beat_length=216)
    
    # Download records (only need to do this once)
    print("\nStep 1: Downloading MIT-BIH records...")
    print("This may take several minutes...")
    dataset.download_all_records()
    
    # Prepare train/test split
    print("\nStep 2: Preparing train/test split...")
    X_train, y_train, X_test, y_test = dataset.prepare_train_test_split()
    
    # Save prepared data
    print("\nStep 3: Saving prepared data...")
    dataset.save_prepared_data(X_train, y_train, X_test, y_test)
    
    print("\nDataset preparation complete!")
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")


if __name__ == "__main__":
    main()


