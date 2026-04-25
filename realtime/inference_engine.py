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
import json
from collections import deque, Counter
from typing import Optional, Dict, List
from datetime import datetime
import os

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
        self.buffer_position = 0  # Track absolute position in stream
        self.last_processed_position = 0  # Last position we processed peaks from
        
        # Results
        self.classification_results = deque(maxlen=1000)
        self.detected_peaks = []  # Store already processed peak positions
        self.processed_peak_positions = set()  # Track which peak positions we've already classified
        
        # Classification smoothing (prevent flip-flopping)
        self.recent_predictions = deque(maxlen=5)  # Track last 5 predictions
        
        # Statistics
        self.total_beats_classified = 0
        self.inference_times = deque(maxlen=100)
        
        # Session tracking
        self.session_start_time = None
        self.session_id = None
        
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
            self.buffer_position += 1
    
    def process_buffer(self):
        """
        Process buffered ECG data
        Detect beats and perform classification
        Only processes NEW beats since last call to avoid duplicate classifications
        """
        if len(self.raw_buffer) < 360:  # Need at least 1 second of data
            return
        
        # Calculate how much new data we have
        new_samples = self.buffer_position - self.last_processed_position
        
        # Process frequently to catch all beats (at least 10 samples = ~0.03 seconds)
        # This ensures we don't miss beats in real-time streaming
        if new_samples < 10:
            return
        
        # Debug: Print buffer state
        if new_samples >= 10:
            print(f"[PROCESS] New samples: {new_samples}, Buffer size: {len(self.raw_buffer)}, Position: {self.buffer_position}")
        
        # Get recent data (last 3 seconds for context)
        buffer_len = min(len(self.raw_buffer), 1080)  # 3 seconds max
        recent_data = np.array(list(self.raw_buffer))[-buffer_len:]
        
        # Calculate offset (where this data starts in absolute position)
        data_start_position = self.buffer_position - len(recent_data)
        
        # Apply filters
        filtered = self.ecg_filter.apply_all_filters(recent_data)
        
        # Detect R-peaks (indices relative to recent_data)
        r_peaks = self.peak_detector.detect_peaks(filtered)
        
        print(f"[PEAKS] Detected {len(r_peaks)} peaks in {len(recent_data)} samples")
        
        if len(r_peaks) == 0:
            return
        
        # Refine peaks
        r_peaks = self.peak_detector.refine_peaks(recent_data, r_peaks)
        
        # Convert peaks to absolute positions
        absolute_peaks = [data_start_position + peak for peak in r_peaks]
        
        # Filter out peaks we've already processed
        # Use a simple duplicate check based on position
        new_peaks = []
        new_peak_indices = []
        
        for i, abs_peak in enumerate(absolute_peaks):
            # Check if this peak is far enough from any already processed peak
            is_duplicate = False
            for processed_peak in self.processed_peak_positions:
                if abs(abs_peak - processed_peak) < 100:  # Within 100 samples (~0.28s) = duplicate
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                new_peaks.append(abs_peak)
                new_peak_indices.append(r_peaks[i])
                self.processed_peak_positions.add(abs_peak)
        
        print(f"[FILTER] Total peaks: {len(absolute_peaks)}, New peaks: {len(new_peaks)}, Already processed: {len(self.processed_peak_positions)}")
        
        if len(new_peak_indices) == 0:
            # No new peaks to process
            self.last_processed_position = self.buffer_position
            print(f"[SKIP] No new peaks to classify")
            return
        
        # Segment only the NEW beats
        beats, valid_indices = self.beat_segmenter.segment_beats(filtered, new_peak_indices)
        
        print(f"[SEGMENT] Segmented {len(beats)} beats from {len(new_peak_indices)} peaks")
        
        if len(beats) == 0:
            self.last_processed_position = self.buffer_position
            print(f"[SKIP] No valid beat segments extracted")
            return
        
        # Update last processed position
        self.last_processed_position = self.buffer_position
        
        # Clean up old peak positions from the set to prevent memory growth
        # Remove positions older than buffer size (5000 samples)
        if len(self.processed_peak_positions) > 200:  # Keep only recent peaks
            min_position_to_keep = self.buffer_position - 5000
            self.processed_peak_positions = {p for p in self.processed_peak_positions if p >= min_position_to_keep}
        
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
    
    def start_session(self):
        """Start a new monitoring session"""
        self.session_start_time = time.time()
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        print(f"Session started: {self.session_id}")
    
    def generate_session_analysis(self) -> Dict:
        """
        Generate comprehensive session analysis
        
        Returns:
            Dictionary containing detailed session statistics and analysis
        """
        if not self.session_start_time:
            return {'error': 'No active session'}
        
        session_duration = time.time() - self.session_start_time
        
        # Classification statistics
        all_classifications = list(self.classification_results)
        class_counts = Counter([r['class'] for r in all_classifications])
        
        # Confidence statistics by class
        confidence_by_class = {}
        for class_name in CLASS_NAMES:
            class_confidences = [r['confidence'] for r in all_classifications if r['class'] == class_name]
            if class_confidences:
                confidence_by_class[class_name] = {
                    'mean': float(np.mean(class_confidences)),
                    'std': float(np.std(class_confidences)),
                    'min': float(np.min(class_confidences)),
                    'max': float(np.max(class_confidences))
                }
            else:
                confidence_by_class[class_name] = None
        
        # Alert analysis
        alert_stats = self.alert_manager.get_alert_statistics()
        recent_alerts = self.alert_manager.get_recent_alerts(count=100)
        
        # Calculate heart rate statistics
        heart_rates = []
        if len(all_classifications) >= 2:
            for i in range(1, min(len(all_classifications), 100)):
                time_diff = all_classifications[i]['timestamp'] - all_classifications[i-1]['timestamp']
                if 0.2 < time_diff < 2.0:  # Reasonable beat-to-beat interval
                    bpm = 60.0 / time_diff
                    if 40 < bpm < 200:  # Physiologically reasonable
                        heart_rates.append(bpm)
        
        # Smoothing statistics
        smoothed_count = sum(1 for r in all_classifications if r.get('smoothed', False))
        
        # Performance metrics
        avg_inference_time = float(np.mean(self.inference_times)) if self.inference_times else 0
        
        analysis = {
            'session_info': {
                'session_id': self.session_id,
                'start_time': datetime.fromtimestamp(self.session_start_time).strftime('%Y-%m-%d %H:%M:%S'),
                'end_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'duration_seconds': float(session_duration),
                'duration_formatted': f"{int(session_duration // 60)}m {int(session_duration % 60)}s"
            },
            'classification_summary': {
                'total_beats': self.total_beats_classified,
                'class_distribution': {
                    CLASS_FULL_NAMES[CLASS_NAMES.index(k)]: int(v) 
                    for k, v in class_counts.items()
                },
                'class_percentages': {
                    CLASS_FULL_NAMES[CLASS_NAMES.index(k)]: float(v / self.total_beats_classified * 100)
                    for k, v in class_counts.items()
                } if self.total_beats_classified > 0 else {}
            },
            'confidence_analysis': {
                'by_class': {CLASS_FULL_NAMES[CLASS_NAMES.index(k)]: v 
                           for k, v in confidence_by_class.items() if v},
                'overall_mean': float(np.mean([r['confidence'] for r in all_classifications])) if all_classifications else 0,
                'low_confidence_count': sum(1 for r in all_classifications if r['confidence'] < 0.60),
                'high_confidence_count': sum(1 for r in all_classifications if r['confidence'] >= 0.80)
            },
            'heart_rate_analysis': {
                'mean_bpm': float(np.mean(heart_rates)) if heart_rates else 0,
                'std_bpm': float(np.std(heart_rates)) if heart_rates else 0,
                'min_bpm': float(np.min(heart_rates)) if heart_rates else 0,
                'max_bpm': float(np.max(heart_rates)) if heart_rates else 0,
                'variability': float(np.std(heart_rates)) if heart_rates else 0
            },
            'alert_summary': {
                'total_alerts': alert_stats['total_alerts'],
                'alerts_by_type': alert_stats.get('by_class', {}),
                'alert_details': recent_alerts
            },
            'processing_performance': {
                'avg_inference_time_ms': avg_inference_time,
                'smoothed_classifications': smoothed_count,
                'smoothing_percentage': float(smoothed_count / self.total_beats_classified * 100) if self.total_beats_classified > 0 else 0
            },
            'quality_indicators': self._assess_session_quality(all_classifications, heart_rates, alert_stats)
        }
        
        return analysis
    
    def _assess_session_quality(self, classifications: List[Dict], heart_rates: List[float], alert_stats: Dict) -> Dict:
        """
        Assess overall session quality
        
        Returns quality indicators and recommendations
        """
        issues = []
        recommendations = []
        overall_quality = "Good"
        
        # Check confidence levels
        if classifications:
            avg_confidence = np.mean([r['confidence'] for r in classifications])
            low_conf_pct = sum(1 for r in classifications if r['confidence'] < 0.60) / len(classifications) * 100
            
            if avg_confidence < 0.50:
                overall_quality = "Poor"
                issues.append("Very low average confidence (<50%)")
                recommendations.append("Check electrode placement and signal quality")
                recommendations.append("Consider retraining model with 'balance_strategy=weights'")
            elif avg_confidence < 0.65:
                overall_quality = "Fair"
                issues.append(f"Low average confidence ({avg_confidence*100:.1f}%)")
                recommendations.append("Improve electrode contact")
                recommendations.append("Minimize movement during recording")
            
            if low_conf_pct > 50:
                issues.append(f"High percentage of low-confidence predictions ({low_conf_pct:.1f}%)")
                recommendations.append("Clean skin before electrode placement")
                recommendations.append("Check for loose connections")
        
        # Check heart rate
        if heart_rates:
            avg_hr = np.mean(heart_rates)
            hr_std = np.std(heart_rates)
            
            if avg_hr > 150 or avg_hr < 40:
                issues.append(f"Abnormal heart rate detected ({avg_hr:.1f} BPM)")
                recommendations.append("Verify R-peak detection is working correctly")
                if avg_hr > 150:
                    recommendations.append("May be detecting noise as R-peaks")
            
            if hr_std > 30:
                issues.append(f"High heart rate variability (std: {hr_std:.1f})")
                recommendations.append("Check for motion artifacts")
                recommendations.append("Ensure stable electrode contact")
        
        # Check alert frequency
        if alert_stats['total_alerts'] > 50:
            issues.append("Excessive alerts triggered")
            recommendations.append("Review alert thresholds")
            recommendations.append("Verify signal quality")
        
        # Overall assessment
        if len(issues) == 0:
            overall_quality = "Excellent"
            recommendations.append("Signal quality is good - continue current setup")
        elif len(issues) >= 3:
            overall_quality = "Poor"
        
        return {
            'overall_quality': overall_quality,
            'issues_detected': issues,
            'recommendations': recommendations
        }
    
    def save_session_log(self, log_dir: str = './logs') -> str:
        """
        Save comprehensive session log to file
        
        Args:
            log_dir: Directory to save logs
        
        Returns:
            Path to saved log file
        """
        # Create logs directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)
        
        # Generate analysis
        analysis = self.generate_session_analysis()
        
        # Add raw classification data
        analysis['raw_classifications'] = [
            {
                'timestamp': r['timestamp'],
                'time_formatted': datetime.fromtimestamp(r['timestamp']).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3],
                'class': r['class'],
                'class_full': r['class_full'],
                'confidence': r['confidence'],
                'probabilities': r['probabilities'],
                'smoothed': r.get('smoothed', False),
                'raw_class': r.get('raw_class', r['class'])
            }
            for r in list(self.classification_results)
        ]
        
        # Save to JSON file
        session_id = self.session_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"ecg_session_{session_id}.json"
        log_path = os.path.join(log_dir, log_filename)
        
        with open(log_path, 'w') as f:
            json.dump(analysis, f, indent=2)
        
        print(f"\n{'='*70}")
        print(f"Session log saved: {log_path}")
        print(f"{'='*70}")
        
        # Also save a human-readable summary
        summary_filename = f"ecg_session_{session_id}_summary.txt"
        summary_path = os.path.join(log_dir, summary_filename)
        self._save_human_readable_summary(analysis, summary_path)
        
        return log_path
    
    def _save_human_readable_summary(self, analysis: Dict, filepath: str):
        """Save human-readable summary text file"""
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write("ECG ARRHYTHMIA DETECTION - SESSION SUMMARY\n")
            f.write("="*70 + "\n\n")
            
            # Session Info
            f.write("SESSION INFORMATION\n")
            f.write("-" * 70 + "\n")
            for key, value in analysis['session_info'].items():
                f.write(f"{key.replace('_', ' ').title()}: {value}\n")
            f.write("\n")
            
            # Classification Summary
            f.write("CLASSIFICATION SUMMARY\n")
            f.write("-" * 70 + "\n")
            f.write(f"Total Beats Analyzed: {analysis['classification_summary']['total_beats']}\n\n")
            
            f.write("Beat Distribution:\n")
            for class_name, count in analysis['classification_summary']['class_distribution'].items():
                pct = analysis['classification_summary']['class_percentages'].get(class_name, 0)
                f.write(f"  {class_name:20s}: {count:5d} beats ({pct:5.1f}%)\n")
            f.write("\n")
            
            # Heart Rate
            f.write("HEART RATE ANALYSIS\n")
            f.write("-" * 70 + "\n")
            hr = analysis['heart_rate_analysis']
            f.write(f"Mean Heart Rate: {hr['mean_bpm']:.1f} BPM\n")
            f.write(f"Range: {hr['min_bpm']:.1f} - {hr['max_bpm']:.1f} BPM\n")
            f.write(f"Variability (std): {hr['std_bpm']:.1f} BPM\n\n")
            
            # Confidence
            f.write("CONFIDENCE ANALYSIS\n")
            f.write("-" * 70 + "\n")
            f.write(f"Overall Mean Confidence: {analysis['confidence_analysis']['overall_mean']*100:.1f}%\n")
            f.write(f"High Confidence (>=80%): {analysis['confidence_analysis']['high_confidence_count']} beats\n")
            f.write(f"Low Confidence (<60%): {analysis['confidence_analysis']['low_confidence_count']} beats\n\n")
            
            # Alerts
            f.write("ALERT SUMMARY\n")
            f.write("-" * 70 + "\n")
            f.write(f"Total Alerts: {analysis['alert_summary']['total_alerts']}\n")
            if analysis['alert_summary']['alerts_by_type']:
                f.write("Alerts by Type:\n")
                for alert_type, count in analysis['alert_summary']['alerts_by_type'].items():
                    f.write(f"  {alert_type}: {count}\n")
            f.write("\n")
            
            # Quality Assessment
            f.write("QUALITY ASSESSMENT\n")
            f.write("-" * 70 + "\n")
            quality = analysis['quality_indicators']
            f.write(f"Overall Quality: {quality['overall_quality']}\n\n")
            
            if quality['issues_detected']:
                f.write("Issues Detected:\n")
                for issue in quality['issues_detected']:
                    f.write(f"  [!] {issue}\n")
                f.write("\n")
            
            if quality['recommendations']:
                f.write("Recommendations:\n")
                for rec in quality['recommendations']:
                    f.write(f"  - {rec}\n")
            
            f.write("\n" + "="*70 + "\n")
        
        print(f"Human-readable summary saved: {filepath}")


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


