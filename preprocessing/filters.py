"""
Digital Signal Processing Filters for ECG Data
Implements Butterworth bandpass and notch filters for noise removal
"""

import numpy as np
from scipy import signal
from typing import Optional, Tuple


class ECGFilter:
    """
    ECG signal filtering using digital filters
    Removes baseline wander, high-frequency noise, and powerline interference
    """
    
    def __init__(self, sampling_rate: int = 360, powerline_freq: int = 60):
        """
        Initialize ECG filter
        
        Args:
            sampling_rate: Sampling frequency in Hz (default: 360 Hz for MIT-BIH)
            powerline_freq: Powerline frequency in Hz (50 or 60)
        """
        self.fs = sampling_rate
        self.powerline_freq = powerline_freq
        
        # Design filters
        self._design_bandpass_filter()
        self._design_notch_filter()
        
        # Filter states for real-time filtering
        self.bandpass_zi: Optional[np.ndarray] = None
        self.notch_zi: Optional[np.ndarray] = None
        
    def _design_bandpass_filter(self):
        """
        Design Butterworth bandpass filter (0.5-40 Hz)
        Removes baseline wander and high-frequency noise
        """
        # Filter specifications
        lowcut = 0.5   # Hz - removes baseline wander
        highcut = 40.0  # Hz - removes high-frequency noise
        order = 4       # Filter order
        
        # Normalize frequencies
        nyquist = 0.5 * self.fs
        low = lowcut / nyquist
        high = highcut / nyquist
        
        # Design Butterworth bandpass filter
        self.bandpass_b, self.bandpass_a = signal.butter(
            order, [low, high], btype='band'
        )
        
        print(f"Designed Butterworth bandpass filter: {lowcut}-{highcut} Hz")
    
    def _design_notch_filter(self):
        """
        Design notch filter for powerline interference removal
        Removes 50 Hz or 60 Hz noise
        """
        # Filter specifications
        freq = self.powerline_freq  # Frequency to remove
        quality = 30.0              # Quality factor (higher = narrower notch)
        
        # Normalize frequency
        nyquist = 0.5 * self.fs
        w0 = freq / nyquist
        
        # Design notch filter
        self.notch_b, self.notch_a = signal.iirnotch(w0, quality)
        
        print(f"Designed notch filter: {freq} Hz")
    
    def apply_bandpass(self, data: np.ndarray) -> np.ndarray:
        """
        Apply bandpass filter to ECG data (batch processing)
        
        Args:
            data: ECG signal array
        
        Returns:
            Filtered ECG signal
        """
        if len(data) < 10:
            return data
        
        # Apply zero-phase filtering (filtfilt)
        filtered = signal.filtfilt(self.bandpass_b, self.bandpass_a, data)
        return filtered
    
    def apply_notch(self, data: np.ndarray) -> np.ndarray:
        """
        Apply notch filter to ECG data (batch processing)
        
        Args:
            data: ECG signal array
        
        Returns:
            Filtered ECG signal
        """
        if len(data) < 10:
            return data
        
        # Apply zero-phase filtering
        filtered = signal.filtfilt(self.notch_b, self.notch_a, data)
        return filtered
    
    def apply_all_filters(self, data: np.ndarray) -> np.ndarray:
        """
        Apply all filters in sequence (batch processing)
        
        Args:
            data: Raw ECG signal array
        
        Returns:
            Fully filtered ECG signal
        """
        # Apply bandpass filter first
        filtered = self.apply_bandpass(data)
        
        # Then apply notch filter
        filtered = self.apply_notch(filtered)
        
        return filtered
    
    def apply_moving_average(self, data: np.ndarray, window_size: int = 5) -> np.ndarray:
        """
        Apply moving average smoothing
        
        Args:
            data: ECG signal array
            window_size: Size of moving average window
        
        Returns:
            Smoothed ECG signal
        """
        if len(data) < window_size:
            return data
        
        # Use uniform convolution for moving average
        window = np.ones(window_size) / window_size
        smoothed = np.convolve(data, window, mode='same')
        
        return smoothed
    
    def reset_filter_states(self):
        """Reset filter states for real-time processing"""
        self.bandpass_zi = signal.lfilter_zi(self.bandpass_b, self.bandpass_a)
        self.notch_zi = signal.lfilter_zi(self.notch_b, self.notch_a)
    
    def filter_realtime_sample(self, sample: float) -> float:
        """
        Filter single sample in real-time (stateful filtering)
        
        Args:
            sample: Single ECG sample value
        
        Returns:
            Filtered sample value
        """
        # Initialize filter states on first call
        if self.bandpass_zi is None:
            self.reset_filter_states()
            # Scale initial conditions
            self.bandpass_zi *= sample
            self.notch_zi *= sample
        
        # Apply bandpass filter
        filtered_bp, self.bandpass_zi = signal.lfilter(
            self.bandpass_b, self.bandpass_a, [sample], zi=self.bandpass_zi
        )
        
        # Apply notch filter
        filtered_notch, self.notch_zi = signal.lfilter(
            self.notch_b, self.notch_a, filtered_bp, zi=self.notch_zi
        )
        
        return filtered_notch[0]
    
    def filter_realtime_batch(self, samples: np.ndarray) -> np.ndarray:
        """
        Filter batch of samples in real-time (stateful filtering)
        
        Args:
            samples: Array of ECG samples
        
        Returns:
            Array of filtered samples
        """
        # Initialize filter states on first call
        if self.bandpass_zi is None:
            self.reset_filter_states()
            if len(samples) > 0:
                self.bandpass_zi *= samples[0]
                self.notch_zi *= samples[0]
        
        # Apply bandpass filter
        filtered_bp, self.bandpass_zi = signal.lfilter(
            self.bandpass_b, self.bandpass_a, samples, zi=self.bandpass_zi
        )
        
        # Apply notch filter
        filtered_notch, self.notch_zi = signal.lfilter(
            self.notch_b, self.notch_a, filtered_bp, zi=self.notch_zi
        )
        
        return filtered_notch


class AdaptiveBaselineRemoval:
    """
    Adaptive baseline removal using median filtering
    More robust than simple high-pass for drift removal
    """
    
    def __init__(self, window_size: int = 200):
        """
        Initialize adaptive baseline removal
        
        Args:
            window_size: Size of median filter window (in samples)
        """
        self.window_size = window_size
    
    def remove_baseline(self, data: np.ndarray) -> np.ndarray:
        """
        Remove baseline wander using median filtering
        
        Args:
            data: ECG signal with baseline wander
        
        Returns:
            ECG signal with baseline removed
        """
        if len(data) < self.window_size:
            return data - np.median(data)
        
        # Estimate baseline using median filter
        baseline = signal.medfilt(data, kernel_size=self.window_size if self.window_size % 2 == 1 else self.window_size + 1)
        
        # Remove baseline
        corrected = data - baseline
        
        return corrected


def normalize_signal(data: np.ndarray, method: str = 'zscore') -> np.ndarray:
    """
    Normalize ECG signal
    
    Args:
        data: ECG signal array
        method: Normalization method ('zscore', 'minmax', 'robust')
    
    Returns:
        Normalized signal
    """
    if len(data) == 0:
        return data
    
    if method == 'zscore':
        # Zero mean, unit variance
        mean = np.mean(data)
        std = np.std(data)
        if std > 0:
            normalized = (data - mean) / std
        else:
            normalized = data - mean
    
    elif method == 'minmax':
        # Scale to [0, 1]
        min_val = np.min(data)
        max_val = np.max(data)
        if max_val > min_val:
            normalized = (data - min_val) / (max_val - min_val)
        else:
            normalized = np.zeros_like(data)
    
    elif method == 'robust':
        # Use median and IQR (robust to outliers)
        median = np.median(data)
        q75, q25 = np.percentile(data, [75, 25])
        iqr = q75 - q25
        if iqr > 0:
            normalized = (data - median) / iqr
        else:
            normalized = data - median
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    return normalized


def test_filters():
    """Test function for ECG filters"""
    print("Testing ECG filters...")
    
    # Create synthetic ECG-like signal
    fs = 360  # Sampling rate
    duration = 10  # seconds
    t = np.linspace(0, duration, fs * duration)
    
    # Simulate ECG with noise
    ecg = np.sin(2 * np.pi * 1.2 * t)  # Simulated heartbeat
    baseline = 0.5 * np.sin(2 * np.pi * 0.1 * t)  # Baseline wander
    powerline = 0.2 * np.sin(2 * np.pi * 60 * t)  # 60 Hz noise
    noise = 0.1 * np.random.randn(len(t))  # Random noise
    
    noisy_signal = ecg + baseline + powerline + noise
    
    # Test filters
    ecg_filter = ECGFilter(sampling_rate=fs, powerline_freq=60)
    
    # Apply filters
    filtered = ecg_filter.apply_all_filters(noisy_signal)
    
    print(f"Original signal: mean={np.mean(noisy_signal):.3f}, std={np.std(noisy_signal):.3f}")
    print(f"Filtered signal: mean={np.mean(filtered):.3f}, std={np.std(filtered):.3f}")
    
    # Test normalization
    normalized = normalize_signal(filtered, method='zscore')
    print(f"Normalized signal: mean={np.mean(normalized):.3f}, std={np.std(normalized):.3f}")
    
    print("Filter test completed successfully!")


if __name__ == "__main__":
    test_filters()






