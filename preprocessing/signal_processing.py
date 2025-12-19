"""
ECG Signal Processing and Feature Extraction
Implements R-peak detection, beat segmentation, and feature extraction
"""

import numpy as np
from scipy import signal
from typing import List, Tuple, Optional
import warnings


class PanTompkinsDetector:
    """
    R-peak detection using Pan-Tompkins algorithm
    Reference: Pan J, Tompkins WJ. "A Real-Time QRS Detection Algorithm" (1985)
    """
    
    def __init__(self, sampling_rate: int = 360):
        """
        Initialize Pan-Tompkins detector
        
        Args:
            sampling_rate: Sampling frequency in Hz
        """
        self.fs = sampling_rate
        
        # Detection parameters
        self.integration_window = int(0.150 * self.fs)  # 150 ms window
        
        # Thresholds (adaptive)
        self.threshold_i1 = 0.0
        self.threshold_i2 = 0.0
        self.spki = 0.0
        self.npki = 0.0
        
    def detect_peaks(self, ecg_signal: np.ndarray) -> np.ndarray:
        """
        Detect R-peaks in ECG signal using Pan-Tompkins algorithm
        
        Args:
            ecg_signal: Filtered ECG signal
        
        Returns:
            Array of R-peak indices
        """
        if len(ecg_signal) < 100:
            return np.array([], dtype=int)
        
        # Step 1: Bandpass filter (5-15 Hz) - emphasize QRS complex
        b, a = signal.butter(2, [5.0/(self.fs/2), 15.0/(self.fs/2)], btype='band')
        filtered = signal.filtfilt(b, a, ecg_signal)
        
        # Step 2: Derivative filter (emphasize slope information)
        diff = np.diff(filtered)
        
        # Step 3: Squaring (emphasize higher frequencies)
        squared = diff ** 2
        
        # Step 4: Moving window integration
        window = np.ones(self.integration_window) / self.integration_window
        integrated = np.convolve(squared, window, mode='same')
        
        # Step 5: Find peaks in integrated signal
        # Use simple peak detection with minimum distance
        min_distance = int(0.2 * self.fs)  # Minimum 200ms between beats (max 300 BPM)
        
        peaks = self._find_peaks_adaptive(integrated, min_distance)
        
        return peaks
    
    def _find_peaks_adaptive(self, signal_data: np.ndarray, min_distance: int) -> np.ndarray:
        """
        Find peaks with adaptive thresholding
        
        Args:
            signal_data: Processed signal
            min_distance: Minimum samples between peaks
        
        Returns:
            Array of peak indices
        """
        peaks = []
        
        # Initialize thresholds
        threshold = 0.3 * np.max(signal_data[:min(1000, len(signal_data))])
        
        i = min_distance
        while i < len(signal_data) - min_distance:
            # Check if current point is a peak
            if signal_data[i] > threshold:
                # Find local maximum in window
                window_start = max(0, i - min_distance // 2)
                window_end = min(len(signal_data), i + min_distance // 2)
                window = signal_data[window_start:window_end]
                
                local_max_idx = np.argmax(window) + window_start
                
                if local_max_idx == i or len(peaks) == 0 or (i - peaks[-1]) >= min_distance:
                    peaks.append(local_max_idx)
                    
                    # Update adaptive threshold
                    threshold = 0.5 * signal_data[local_max_idx] + 0.5 * threshold
                    
                    # Skip forward
                    i = local_max_idx + min_distance
                    continue
            
            i += 1
        
        return np.array(peaks, dtype=int)
    
    def refine_peaks(self, ecg_signal: np.ndarray, peaks: np.ndarray, window: int = 25) -> np.ndarray:
        """
        Refine R-peak locations to actual maximum in original signal
        
        Args:
            ecg_signal: Original ECG signal
            peaks: Initial peak locations
            window: Search window around each peak
        
        Returns:
            Refined peak locations
        """
        refined_peaks = []
        
        for peak in peaks:
            # Define search window
            start = max(0, peak - window)
            end = min(len(ecg_signal), peak + window)
            
            # Find maximum in window
            window_data = ecg_signal[start:end]
            local_max = np.argmax(window_data)
            refined_peak = start + local_max
            
            refined_peaks.append(refined_peak)
        
        return np.array(refined_peaks, dtype=int)


class BeatSegmenter:
    """
    Segment ECG signal into individual heartbeats
    Extract fixed-length windows around R-peaks
    """
    
    def __init__(self, sampling_rate: int = 360, beat_length: int = 216):
        """
        Initialize beat segmenter
        
        Args:
            sampling_rate: Sampling frequency in Hz
            beat_length: Length of extracted beat in samples (default: 216 for 600ms at 360Hz)
        """
        self.fs = sampling_rate
        self.beat_length = beat_length
        
        # Asymmetric window (more samples after R-peak)
        self.pre_r = int(0.25 * beat_length)   # 25% before R-peak
        self.post_r = beat_length - self.pre_r  # 75% after R-peak
    
    def segment_beats(self, ecg_signal: np.ndarray, r_peaks: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract individual beats around R-peaks
        
        Args:
            ecg_signal: ECG signal
            r_peaks: Array of R-peak indices
        
        Returns:
            Tuple of (beats_array, valid_peak_indices)
            beats_array: Shape (n_beats, beat_length)
            valid_peak_indices: Original indices of valid peaks
        """
        beats = []
        valid_indices = []
        
        for idx, peak in enumerate(r_peaks):
            # Define beat window
            start = peak - self.pre_r
            end = peak + self.post_r
            
            # Check if window is within signal bounds
            if start >= 0 and end <= len(ecg_signal):
                beat = ecg_signal[start:end]
                
                # Ensure exact length (handle edge cases)
                if len(beat) == self.beat_length:
                    beats.append(beat)
                    valid_indices.append(idx)
        
        if len(beats) == 0:
            return np.array([]), np.array([], dtype=int)
        
        return np.array(beats), np.array(valid_indices, dtype=int)
    
    def normalize_beats(self, beats: np.ndarray, method: str = 'zscore') -> np.ndarray:
        """
        Normalize each beat independently
        
        Args:
            beats: Array of beats, shape (n_beats, beat_length)
            method: Normalization method ('zscore', 'minmax')
        
        Returns:
            Normalized beats array
        """
        if len(beats) == 0:
            return beats
        
        normalized_beats = np.zeros_like(beats)
        
        for i, beat in enumerate(beats):
            if method == 'zscore':
                mean = np.mean(beat)
                std = np.std(beat)
                if std > 1e-6:
                    normalized_beats[i] = (beat - mean) / std
                else:
                    normalized_beats[i] = beat - mean
            
            elif method == 'minmax':
                min_val = np.min(beat)
                max_val = np.max(beat)
                if max_val > min_val:
                    normalized_beats[i] = (beat - min_val) / (max_val - min_val)
                else:
                    normalized_beats[i] = beat
        
        return normalized_beats


class HeartRateCalculator:
    """
    Calculate heart rate and RR intervals from R-peaks
    """
    
    def __init__(self, sampling_rate: int = 360):
        """
        Initialize heart rate calculator
        
        Args:
            sampling_rate: Sampling frequency in Hz
        """
        self.fs = sampling_rate
    
    def calculate_hr(self, r_peaks: np.ndarray) -> float:
        """
        Calculate average heart rate in BPM
        
        Args:
            r_peaks: Array of R-peak indices
        
        Returns:
            Heart rate in beats per minute
        """
        if len(r_peaks) < 2:
            return 0.0
        
        # Calculate RR intervals in samples
        rr_intervals = np.diff(r_peaks)
        
        # Convert to seconds
        rr_seconds = rr_intervals / self.fs
        
        # Calculate average heart rate
        avg_rr = np.mean(rr_seconds)
        
        if avg_rr > 0:
            hr = 60.0 / avg_rr
        else:
            hr = 0.0
        
        return hr
    
    def calculate_rr_intervals(self, r_peaks: np.ndarray) -> np.ndarray:
        """
        Calculate RR intervals in milliseconds
        
        Args:
            r_peaks: Array of R-peak indices
        
        Returns:
            Array of RR intervals in ms
        """
        if len(r_peaks) < 2:
            return np.array([])
        
        rr_samples = np.diff(r_peaks)
        rr_ms = (rr_samples / self.fs) * 1000.0
        
        return rr_ms
    
    def calculate_hrv(self, r_peaks: np.ndarray) -> dict:
        """
        Calculate basic heart rate variability metrics
        
        Args:
            r_peaks: Array of R-peak indices
        
        Returns:
            Dictionary with HRV metrics
        """
        rr_intervals = self.calculate_rr_intervals(r_peaks)
        
        if len(rr_intervals) < 2:
            return {
                'mean_rr': 0.0,
                'sdnn': 0.0,
                'rmssd': 0.0,
                'pnn50': 0.0
            }
        
        # Mean RR interval
        mean_rr = np.mean(rr_intervals)
        
        # SDNN: Standard deviation of RR intervals
        sdnn = np.std(rr_intervals)
        
        # RMSSD: Root mean square of successive differences
        successive_diffs = np.diff(rr_intervals)
        rmssd = np.sqrt(np.mean(successive_diffs ** 2))
        
        # pNN50: Percentage of successive RR differences > 50ms
        nn50 = np.sum(np.abs(successive_diffs) > 50)
        pnn50 = (nn50 / len(successive_diffs)) * 100.0 if len(successive_diffs) > 0 else 0.0
        
        return {
            'mean_rr': mean_rr,
            'sdnn': sdnn,
            'rmssd': rmssd,
            'pnn50': pnn50
        }


def process_ecg_signal(ecg_signal: np.ndarray, 
                       sampling_rate: int = 360,
                       beat_length: int = 216) -> dict:
    """
    Complete ECG signal processing pipeline
    
    Args:
        ecg_signal: Raw/filtered ECG signal
        sampling_rate: Sampling frequency
        beat_length: Desired beat length in samples
    
    Returns:
        Dictionary with processed data:
            - r_peaks: R-peak locations
            - beats: Segmented beats
            - heart_rate: Average heart rate (BPM)
            - rr_intervals: RR intervals (ms)
            - hrv_metrics: HRV metrics
    """
    # Detect R-peaks
    detector = PanTompkinsDetector(sampling_rate)
    r_peaks = detector.detect_peaks(ecg_signal)
    r_peaks = detector.refine_peaks(ecg_signal, r_peaks)
    
    # Segment beats
    segmenter = BeatSegmenter(sampling_rate, beat_length)
    beats, valid_indices = segmenter.segment_beats(ecg_signal, r_peaks)
    beats_normalized = segmenter.normalize_beats(beats, method='zscore')
    
    # Calculate heart rate metrics
    hr_calc = HeartRateCalculator(sampling_rate)
    heart_rate = hr_calc.calculate_hr(r_peaks)
    rr_intervals = hr_calc.calculate_rr_intervals(r_peaks)
    hrv_metrics = hr_calc.calculate_hrv(r_peaks)
    
    return {
        'r_peaks': r_peaks,
        'beats': beats_normalized,
        'raw_beats': beats,
        'heart_rate': heart_rate,
        'rr_intervals': rr_intervals,
        'hrv_metrics': hrv_metrics,
        'num_beats': len(beats)
    }


def test_signal_processing():
    """Test function for signal processing"""
    print("Testing ECG signal processing...")
    
    # Create synthetic ECG signal
    fs = 360
    duration = 10
    t = np.linspace(0, duration, fs * duration)
    
    # Simulate ECG with multiple heartbeats
    hr = 75  # beats per minute
    beat_interval = 60.0 / hr
    ecg = np.zeros_like(t)
    
    for i in range(int(duration / beat_interval)):
        peak_time = i * beat_interval + 0.5
        peak_idx = int(peak_time * fs)
        if peak_idx < len(ecg):
            # Simulate QRS complex
            for j in range(-20, 40):
                if 0 <= peak_idx + j < len(ecg):
                    ecg[peak_idx + j] += np.exp(-((j/10)**2))
    
    # Add noise
    ecg += 0.1 * np.random.randn(len(ecg))
    
    # Process signal
    result = process_ecg_signal(ecg, sampling_rate=fs)
    
    print(f"Detected {result['num_beats']} beats")
    print(f"Heart rate: {result['heart_rate']:.1f} BPM")
    print(f"Mean RR interval: {result['hrv_metrics']['mean_rr']:.1f} ms")
    print(f"SDNN: {result['hrv_metrics']['sdnn']:.1f} ms")
    print(f"Beats shape: {result['beats'].shape}")
    
    print("Signal processing test completed successfully!")


if __name__ == "__main__":
    test_signal_processing()


