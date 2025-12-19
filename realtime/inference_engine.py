"""
Real-Time Inference Engine for ECG Arrhythmia Detection
Integrates trained model with live data stream for real-time classification
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
import time
import queue
import threading
from collections import deque, Counter
from typing import Optional, Dict, List
from datetime import datetime

# Import custom modules
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocessing.filters import ECGFilter
from preprocessing.signal_processing import PanTompkinsDetector, BeatSegmenter

# Import custom loss function
try:
    from models.model_architecture import FocalLoss
except ImportError:
    # Fallback if running from different directory
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "model_architecture",
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "model_architecture.py")
    )
    model_arch = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model_arch)
    FocalLoss = model_arch.FocalLoss


# Class names
CLASS_NAMES = ['N', 'S', 'V', 'F', 'Q']
CLASS_FULL_NAMES = ['Normal', 'Supraventricular', 'Ventricular', 'Fusion', 'Unknown']


class AlertManager:
    """
    Manages arrhythmia alerts based on classification results
    Implements state machine to prevent alert fatigue
    """
    
    def __init__(self):
        """Initialize alert manager"""
        self.alert_history = []
        self.alert_state = {
            'V': {'consecutive': 0, 'per_minute': 0, 'last_alert': 0},
            'S': {'consecutive': 0, 'per_minute': 0, 'last_alert': 0},
        }
        self.recent_beats = deque(maxlen=100)  # Track last 100 beats (approx 1 minute at 75 BPM)
        self.min_alert_interval = 5.0  # Minimum seconds between same alert type
        
    def add_beat(self, class_label: str, confidence: float, timestamp: float) -> Optional[Dict]:
        """
        Add classified beat and check for alert conditions
        
        Args:
            class_label: Predicted class ('N', 'S', 'V', 'F', 'Q')
            confidence: Prediction confidence (0-1)
            timestamp: Unix timestamp
        
        Returns:
            Alert dictionary if alert triggered, None otherwise
        """
        # CRITICAL FIX: Only add beat if confidence is reasonable
        # This prevents false alarms from low-confidence predictions
        MIN_CONFIDENCE_FOR_ALERTS = 0.60  # 60% minimum confidence
        
        if confidence >= MIN_CONFIDENCE_FOR_ALERTS:
            self.recent_beats.append({'class': class_label, 'time': timestamp, 'confidence': confidence})
        
        alert = None
        
        # Check ventricular beats (only if high confidence)
        if class_label == 'V' and confidence >= MIN_CONFIDENCE_FOR_ALERTS:
            self.alert_state['V']['consecutive'] += 1
            
            # Count HIGH-CONFIDENCE V beats in last minute
            v_count = sum(1 for beat in self.recent_beats 
                         if beat['class'] == 'V' and beat.get('confidence', 0) >= MIN_CONFIDENCE_FOR_ALERTS)
            self.alert_state['V']['per_minute'] = v_count
            
            # Alert conditions for ventricular beats (stricter thresholds)
            if (self.alert_state['V']['consecutive'] >= 3 or v_count >= 10):
                if timestamp - self.alert_state['V']['last_alert'] >= self.min_alert_interval:
                    alert = {
                        'type': 'CRITICAL',
                        'class': 'Ventricular',
                        'message': f'Ventricular beats detected ({v_count}/min, {self.alert_state["V"]["consecutive"]} consecutive)',
                        'timestamp': timestamp,
                        'confidence': confidence
                    }
                    self.alert_state['V']['last_alert'] = timestamp
                    self.alert_history.append(alert)
        else:
            self.alert_state['V']['consecutive'] = 0
        
        # Check supraventricular beats (only if high confidence)
        if class_label == 'S' and confidence >= MIN_CONFIDENCE_FOR_ALERTS:
            self.alert_state['S']['consecutive'] += 1
            
            # Alert if sustained episode (>5 consecutive)
            if self.alert_state['S']['consecutive'] >= 7:  # Increased threshold
                if timestamp - self.alert_state['S']['last_alert'] >= self.min_alert_interval:
                    alert = {
                        'type': 'WARNING',
                        'class': 'Supraventricular',
                        'message': f'Sustained supraventricular episode ({self.alert_state["S"]["consecutive"]} consecutive)',
                        'timestamp': timestamp,
                        'confidence': confidence
                    }
                    self.alert_state['S']['last_alert'] = timestamp
                    self.alert_history.append(alert)
        else:
            self.alert_state['S']['consecutive'] = 0
        
        # Check for unknown beats (only if very high confidence)
        if class_label == 'Q' and confidence > 0.75:
            alert = {
                'type': 'INFO',
                'class': 'Unknown',
                'message': 'Unknown beat pattern detected - requires review',
                'timestamp': timestamp,
                'confidence': confidence
            }
        
        return alert
    
    def get_recent_alerts(self, count: int = 10) -> List[Dict]:
        """Get most recent alerts"""
        return list(self.alert_history[-count:])
    
    def get_alert_statistics(self) -> Dict:
        """Get alert statistics"""
        counter = Counter([a['class'] for a in self.alert_history])
        return {
            'total_alerts': len(self.alert_history),
            'by_class': dict(counter),
            'recent_beats': len(self.recent_beats)
        }


class RealtimeInferenceEngine:
    """
    Real-time inference engine for ECG classification
    Processes live ECG stream and performs beat-by-beat classification
    """
    
    def __init__(self, model_path: str, sampling_rate: int = 360, beat_length: int = 216):
        """
        Initialize inference engine
        
        Args:
            model_path: Path to trained model (.h5 file)
            sampling_rate: ECG sampling rate in Hz
            beat_length: Beat segment length in samples
        """
        self.sampling_rate = sampling_rate
        self.beat_length = beat_length
        
        # Load model with custom objects
        print(f"Loading model from {model_path}...")
        try:
            # Try loading with custom objects (for models trained with Focal Loss)
            self.model = keras.models.load_model(
                model_path,
                custom_objects={'FocalLoss': FocalLoss}
            )
            print("Model loaded successfully (with Focal Loss)!")
        except Exception as e:
            # Fallback: try loading without custom objects (for older models)
            print(f"Warning: Could not load with Focal Loss ({str(e)})")
            print("Attempting to load without custom objects...")
            self.model = keras.models.load_model(model_path)
            print("Model loaded successfully!")
        
        # Initialize processing components
        self.ecg_filter = ECGFilter(sampling_rate=sampling_rate)
        self.peak_detector = PanTompkinsDetector(sampling_rate=sampling_rate)
        self.beat_segmenter = BeatSegmenter(sampling_rate=sampling_rate, beat_length=beat_length)
        
        # Alert manager
        self.alert_manager = AlertManager()
        
        # Data buffers
        self.raw_buffer = deque(maxlen=5000)  # Buffer ~14 seconds at 360 Hz
        self.filtered_buffer = deque(maxlen=5000)
        self.last_processed_sample = 0
        
        # Results
        self.classification_results = deque(maxlen=1000)
        self.detected_peaks = []
        
        # Classification smoothing (prevent flip-flopping)
        self.recent_predictions = deque(maxlen=5)  # Track last 5 predictions
        
        # Statistics
        self.total_beats_classified = 0
        self.inference_times = deque(maxlen=100)
        
        # Thread-safe queue for results
        self.results_queue = queue.Queue(maxsize=100)
        
    def add_samples(self, samples: np.ndarray, leads_off: bool = False):
        """
        Add new ECG samples to buffer
        
        Args:
            samples: Array of ECG values
            leads_off: Whether electrodes are disconnected
        """
        if leads_off:
            return  # Don't process if leads are off
        
        for sample in samples:
            self.raw_buffer.append(sample)
    
    def process_buffer(self):
        """
        Process buffered ECG data
        Detect beats and perform classification
        """
        if len(self.raw_buffer) < 360:  # Need at least 1 second of data
            return
        
        # Get recent data
        recent_data = np.array(list(self.raw_buffer))
        
        # Apply filters
        filtered = self.ecg_filter.apply_all_filters(recent_data)
        
        # Detect R-peaks
        r_peaks = self.peak_detector.detect_peaks(filtered)
        
        if len(r_peaks) == 0:
            return
        
        # Refine peaks
        r_peaks = self.peak_detector.refine_peaks(recent_data, r_peaks)
        
        # Segment beats
        beats, valid_indices = self.beat_segmenter.segment_beats(filtered, r_peaks)
        
        if len(beats) == 0:
            return
        
        # Normalize beats
        beats_normalized = self.beat_segmenter.normalize_beats(beats, method='zscore')
        
        # Reshape for model input
        X = beats_normalized.reshape(-1, self.beat_length, 1)
        
        # Perform inference
        start_time = time.time()
        predictions = self.model.predict(X, verbose=0)
        inference_time = (time.time() - start_time) * 1000  # milliseconds
        
        self.inference_times.append(inference_time)
        
        # Process predictions
        for i, pred in enumerate(predictions):
            class_idx = np.argmax(pred)
            confidence = pred[class_idx]
            raw_class_label = CLASS_NAMES[class_idx]
            
            # SMOOTHING: Apply majority voting on recent predictions if confidence is low
            self.recent_predictions.append({'class': raw_class_label, 'confidence': confidence})
            
            # Use smoothing if current confidence is low (<60%)
            if confidence < 0.60 and len(self.recent_predictions) >= 3:
                # Majority vote from recent high-confidence predictions
                high_conf_recent = [p['class'] for p in self.recent_predictions 
                                   if p['confidence'] >= 0.60]
                if high_conf_recent:
                    from collections import Counter
                    majority_class = Counter(high_conf_recent).most_common(1)[0][0]
                    class_label = majority_class
                    # Mark as smoothed
                    is_smoothed = True
                else:
                    class_label = raw_class_label
                    is_smoothed = False
            else:
                class_label = raw_class_label
                is_smoothed = False
            
            # Get final class index
            class_idx = CLASS_NAMES.index(class_label) if class_label in CLASS_NAMES else class_idx
            
            result = {
                'timestamp': time.time(),
                'class': class_label,
                'class_full': CLASS_FULL_NAMES[class_idx],
                'confidence': float(confidence),
                'probabilities': pred.tolist(),
                'beat_data': beats[i].tolist(),
                'smoothed': is_smoothed,
                'raw_class': raw_class_label
            }
            
            self.classification_results.append(result)
            self.total_beats_classified += 1
            
            # Check for alerts (use smoothed classification)
            alert = self.alert_manager.add_beat(class_label, confidence, time.time())
            if alert:
                result['alert'] = alert
            
            # Add to results queue
            try:
                self.results_queue.put_nowait(result)
            except queue.Full:
                # Remove oldest result
                try:
                    self.results_queue.get_nowait()
                    self.results_queue.put_nowait(result)
                except queue.Empty:
                    pass
    
    def get_latest_result(self) -> Optional[Dict]:
        """Get latest classification result"""
        try:
            return self.results_queue.get_nowait()
        except queue.Empty:
            return None
    
    def get_statistics(self) -> Dict:
        """Get inference engine statistics"""
        avg_inference_time = np.mean(self.inference_times) if self.inference_times else 0
        
        # Class distribution
        recent_classes = [r['class'] for r in list(self.classification_results)[-100:]]
        class_dist = Counter(recent_classes)
        
        return {
            'total_beats': self.total_beats_classified,
            'avg_inference_time_ms': float(avg_inference_time),
            'buffer_size': len(self.raw_buffer),
            'class_distribution': dict(class_dist),
            'alert_stats': self.alert_manager.get_alert_statistics()
        }
    
    def get_recent_classifications(self, count: int = 10) -> List[Dict]:
        """Get recent classification results"""
        return list(self.classification_results)[-count:]


def test_inference_engine():
    """Test inference engine with synthetic data"""
    print("Testing Inference Engine...")
    
    # Create dummy model for testing
    from models.model_architecture import create_model
    model = create_model(model_type='standard')
    model.save('test_model.h5')
    
    # Initialize engine
    engine = RealtimeInferenceEngine(model_path='test_model.h5')
    
    # Generate synthetic ECG
    fs = 360
    t = np.linspace(0, 5, fs * 5)
    ecg = np.sin(2 * np.pi * 1.2 * t) + 0.1 * np.random.randn(len(t))
    
    # Add samples
    engine.add_samples(ecg)
    
    # Process
    engine.process_buffer()
    
    # Get results
    stats = engine.get_statistics()
    print(f"Processed {stats['total_beats']} beats")
    print(f"Average inference time: {stats['avg_inference_time_ms']:.2f} ms")
    
    # Clean up
    os.remove('test_model.h5')
    
    print("Test completed!")


if __name__ == "__main__":
    test_inference_engine()


