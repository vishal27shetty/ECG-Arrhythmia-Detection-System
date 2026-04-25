"""
Batch ECG Processing Module
Processes entire ECG recording after acquisition completes
More accurate than real-time processing
"""

import numpy as np
from tensorflow import keras
from collections import Counter
from typing import Dict, List, Tuple
import time

from preprocessing.filters import ECGFilter
from preprocessing.signal_processing import PanTompkinsDetector, BeatSegmenter


class BatchECGProcessor:
    """
    Process complete ECG recording after acquisition
    
    This approach is more accurate than real-time processing because:
    - Can use the entire signal context
    - No edge effects from streaming
    - Better peak detection with lookahead
    - No duplicate detections
    """
    
    def __init__(self, model_path: str, sampling_rate: int = 360):
        """
        Initialize batch processor
        
        Args:
            model_path: Path to trained model
            sampling_rate: ECG sampling rate in Hz
        """
        self.sampling_rate = sampling_rate
        
        # Load model
        print(f"Loading model from {model_path}...")
        try:
            from models.model_architecture import FocalLoss
            self.model = keras.models.load_model(
                model_path,
                custom_objects={'FocalLoss': FocalLoss}
            )
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Loading with custom objects failed, trying without: {e}")
            self.model = keras.models.load_model(model_path)
        
        # Initialize processing components
        self.ecg_filter = ECGFilter(sampling_rate=sampling_rate)
        self.peak_detector = PanTompkinsDetector(sampling_rate=sampling_rate)
        self.beat_segmenter = BeatSegmenter(sampling_rate=sampling_rate, beat_length=216)
        
        # Class names
        self.class_names = ['N', 'S', 'V', 'F', 'Q']
        self.class_full_names = ['Normal', 'Supraventricular', 'Ventricular', 'Fusion', 'Unknown']
    
    def _validate_signal_quality(self, ecg_signal: np.ndarray) -> Dict:
        """Check signal quality before processing"""
        quality = {
            'is_valid': True,
            'warnings': [],
            'metrics': {}
        }
        
        # Check signal statistics
        mean_val = np.mean(ecg_signal)
        std_val = np.std(ecg_signal)
        signal_range = np.max(ecg_signal) - np.min(ecg_signal)
        
        quality['metrics']['mean'] = float(mean_val)
        quality['metrics']['std'] = float(std_val)
        quality['metrics']['range'] = float(signal_range)
        quality['metrics']['snr_estimate'] = float(signal_range / (std_val + 1e-6))
        
        # Check for flat signal
        if std_val < 1.0:
            quality['is_valid'] = False
            quality['warnings'].append("Signal is too flat (std < 1.0) - possible disconnected electrodes")
        
        # Check for saturation
        if signal_range < 10:
            quality['warnings'].append("Very low signal range - check electrode contact")
        
        # Check for excessive noise
        if std_val > 500:
            quality['warnings'].append("High noise level detected - check electrode placement")
        
        return quality
    
    def process_recording(self, ecg_signal: np.ndarray) -> Dict:
        """
        Process complete ECG recording
        
        Args:
            ecg_signal: Complete ECG signal array
        
        Returns:
            Dictionary with all classification results
        """
        print(f"\n{'='*70}")
        print("BATCH PROCESSING ECG RECORDING")
        print(f"{'='*70}")
        print(f"Signal length: {len(ecg_signal)} samples ({len(ecg_signal)/self.sampling_rate:.1f} seconds)")
        
        # Validate signal quality first
        quality = self._validate_signal_quality(ecg_signal)
        print(f"\nSignal Quality Metrics:")
        print(f"  Mean: {quality['metrics']['mean']:.2f}")
        print(f"  Std Dev: {quality['metrics']['std']:.2f}")
        print(f"  Range: {quality['metrics']['range']:.2f}")
        print(f"  SNR Estimate: {quality['metrics']['snr_estimate']:.2f}")
        
        if quality['warnings']:
            print("\n⚠️  Signal Quality Warnings:")
            for warning in quality['warnings']:
                print(f"  - {warning}")
        
        if not quality['is_valid']:
            return {
                'error': 'Signal quality too poor for analysis',
                'quality': quality,
                'total_beats': 0,
                'processing_time': 0
            }
        
        start_time = time.time()
        
        # Step 1: Apply filters
        print("\n[1/5] Applying filters...")
        filtered = self.ecg_filter.apply_all_filters(ecg_signal)
        print("✓ Filtering complete")
        
        # Step 2: Detect R-peaks
        print("\n[2/5] Detecting R-peaks...")
        r_peaks = self.peak_detector.detect_peaks(filtered)
        print(f"✓ Detected {len(r_peaks)} R-peaks")
        
        # Expected beats based on duration (assume 40-100 BPM normal range)
        duration_sec = len(ecg_signal) / self.sampling_rate
        expected_min_beats = int((40 / 60) * duration_sec)  # 40 BPM
        expected_max_beats = int((100 / 60) * duration_sec)  # 100 BPM
        
        print(f"Expected beats for {duration_sec:.1f}s: {expected_min_beats}-{expected_max_beats}")
        
        if len(r_peaks) == 0:
            return {
                'error': 'No R-peaks detected - check signal quality and electrode placement',
                'quality': quality,
                'total_beats': 0,
                'processing_time': time.time() - start_time
            }
        
        if len(r_peaks) < expected_min_beats / 2:
            print(f"⚠️  WARNING: Very few peaks detected ({len(r_peaks)} vs expected {expected_min_beats}-{expected_max_beats})")
            print(f"⚠️  This suggests poor signal quality or incorrect electrode placement")
        
        # Refine peaks
        r_peaks = self.peak_detector.refine_peaks(ecg_signal, r_peaks)
        print(f"✓ Refined to {len(r_peaks)} peaks")
        
        # Step 3: Segment beats
        print("\n[3/5] Segmenting beats...")
        beats, valid_indices = self.beat_segmenter.segment_beats(filtered, r_peaks)
        print(f"✓ Segmented {len(beats)} valid beats")
        
        if len(beats) == 0:
            return {
                'error': 'No valid beats segmented',
                'total_beats': 0,
                'r_peaks_detected': len(r_peaks),
                'processing_time': time.time() - start_time
            }
        
        # Step 4: Normalize beats
        print("\n[4/5] Normalizing beats...")
        beats_normalized = self.beat_segmenter.normalize_beats(beats, method='zscore')
        X = beats_normalized.reshape(-1, 216, 1)
        print(f"✓ Prepared {len(X)} beats for classification")
        
        # Step 5: Classify all beats
        print("\n[5/5] Classifying beats...")
        predictions = self.model.predict(X, verbose=0)
        print("✓ Classification complete")
        
        # Process results
        classifications = []
        for i in range(len(predictions)):
            pred = predictions[i]
            class_idx = np.argmax(pred)
            confidence = pred[class_idx]
            
            # Calculate timestamp (approximate based on peak location)
            peak_sample = r_peaks[valid_indices[i]]
            timestamp = peak_sample / self.sampling_rate
            
            classification = {
                'beat_index': i,
                'peak_sample': int(peak_sample),
                'timestamp': float(timestamp),
                'class': self.class_names[class_idx],
                'class_full': self.class_full_names[class_idx],
                'confidence': float(confidence),
                'probabilities': pred.tolist()
            }
            classifications.append(classification)
        
        # Calculate statistics
        class_counts = Counter([c['class'] for c in classifications])
        confidence_values = [c['confidence'] for c in classifications]
        
        # Calculate heart rate
        intervals = []
        for i in range(1, min(len(r_peaks), 100)):
            interval = (r_peaks[i] - r_peaks[i-1]) / self.sampling_rate
            if 0.3 < interval < 2.0:
                intervals.append(60.0 / interval)
        
        processing_time = time.time() - start_time
        
        results = {
            'total_beats': len(classifications),
            'r_peaks_detected': len(r_peaks),
            'valid_beats_segmented': len(beats),
            'signal_duration': float(len(ecg_signal) / self.sampling_rate),
            'class_distribution': {
                self.class_full_names[self.class_names.index(k)]: int(v)
                for k, v in class_counts.items()
            },
            'class_percentages': {
                self.class_full_names[self.class_names.index(k)]: float(v / len(classifications) * 100)
                for k, v in class_counts.items()
            },
            'confidence': {
                'mean': float(np.mean(confidence_values)),
                'std': float(np.std(confidence_values)),
                'min': float(np.min(confidence_values)),
                'max': float(np.max(confidence_values))
            },
            'heart_rate': {
                'mean': float(np.mean(intervals)) if intervals else 0,
                'std': float(np.std(intervals)) if intervals else 0,
                'min': float(np.min(intervals)) if intervals else 0,
                'max': float(np.max(intervals)) if intervals else 0
            },
            'processing_time': processing_time,
            'classifications': classifications
        }
        
        print(f"\n{'='*70}")
        print("PROCESSING COMPLETE")
        print(f"{'='*70}")
        print(f"Total Beats Classified: {results['total_beats']}")
        print(f"Mean Heart Rate: {results['heart_rate']['mean']:.1f} BPM")
        print(f"Mean Confidence: {results['confidence']['mean']*100:.1f}%")
        print(f"Processing Time: {processing_time:.2f} seconds")
        print(f"{'='*70}\n")
        
        return results


if __name__ == "__main__":
    # Test with synthetic data
    print("Testing Batch Processor...")
    
    # Generate 10 seconds of synthetic ECG at 360 Hz
    fs = 360
    duration = 10
    t = np.linspace(0, duration, fs * duration)
    ecg = np.sin(2 * np.pi * 1.2 * t) + 0.2 * np.random.randn(len(t))
    
    processor = BatchECGProcessor(model_path='./models/best_model.h5')
    results = processor.process_recording(ecg)
    
    print(f"\nTest Results:")
    print(f"Detected: {results['total_beats']} beats")
    print(f"Classes: {results['class_distribution']}")

