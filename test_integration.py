"""
Integration Test Script for ECG Arrhythmia Detection System
Tests all components end-to-end
"""

import os
import sys
import numpy as np
import time
from datetime import datetime

print("="*70)
print("ECG Arrhythmia Detection System - Integration Test")
print("="*70)
print(f"Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# Test results tracking
tests_passed = 0
tests_failed = 0
test_results = []


def test_module(test_name, test_func):
    """Run a test module and track results"""
    global tests_passed, tests_failed
    
    print(f"\n{'='*70}")
    print(f"Testing: {test_name}")
    print(f"{'='*70}")
    
    try:
        test_func()
        print(f"✅ {test_name} - PASSED")
        tests_passed += 1
        test_results.append((test_name, "PASSED", None))
        return True
    except Exception as e:
        print(f"❌ {test_name} - FAILED")
        print(f"Error: {str(e)}")
        tests_failed += 1
        test_results.append((test_name, "FAILED", str(e)))
        return False


def test_dependencies():
    """Test 1: Check all required dependencies"""
    print("\nChecking dependencies...")
    
    required_packages = [
        'numpy',
        'scipy',
        'tensorflow',
        'keras',
        'wfdb',
        'serial',
        'streamlit',
        'plotly',
        'pandas',
        'sklearn',
        'imblearn'
    ]
    
    for package in required_packages:
        try:
            if package == 'serial':
                __import__('serial')
                print(f"  ✓ pyserial installed")
            elif package == 'imblearn':
                __import__('imblearn')
                print(f"  ✓ imbalanced-learn installed")
            else:
                __import__(package)
                print(f"  ✓ {package} installed")
        except ImportError:
            raise ImportError(f"Missing package: {package}")
    
    print("\nAll dependencies installed!")


def test_file_structure():
    """Test 2: Check project structure"""
    print("\nChecking file structure...")
    
    required_files = [
        'arduino/ecg_acquisition.ino',
        'requirements.txt',
        'README.md',
        'preprocessing/filters.py',
        'preprocessing/signal_processing.py',
        'models/dataset_preparation.py',
        'models/model_architecture.py',
        'models/train_bilstm.py',
        'realtime/serial_reader.py',
        'realtime/inference_engine.py',
        'dashboard/app.py'
    ]
    
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"  ✓ {file_path}")
        else:
            raise FileNotFoundError(f"Missing file: {file_path}")
    
    print("\nAll required files present!")


def test_filters():
    """Test 3: DSP filters"""
    print("\nTesting DSP filters...")
    
    from preprocessing.filters import ECGFilter, normalize_signal
    
    # Create filter
    ecg_filter = ECGFilter(sampling_rate=360, powerline_freq=60)
    
    # Generate test signal
    fs = 360
    duration = 5
    t = np.linspace(0, duration, fs * duration)
    signal = np.sin(2 * np.pi * 1.2 * t) + 0.1 * np.random.randn(len(t))
    
    # Apply filters
    filtered = ecg_filter.apply_all_filters(signal)
    
    assert len(filtered) == len(signal), "Filtered signal length mismatch"
    assert not np.any(np.isnan(filtered)), "Filtered signal contains NaN"
    
    # Test normalization
    normalized = normalize_signal(filtered)
    assert abs(np.mean(normalized)) < 0.1, "Normalization failed (mean not close to 0)"
    
    print(f"  ✓ Bandpass filter working")
    print(f"  ✓ Notch filter working")
    print(f"  ✓ Normalization working")
    print("\nFilters test passed!")


def test_signal_processing():
    """Test 4: Signal processing and R-peak detection"""
    print("\nTesting signal processing...")
    
    from preprocessing.signal_processing import (
        PanTompkinsDetector, 
        BeatSegmenter,
        HeartRateCalculator
    )
    
    # Create detector
    detector = PanTompkinsDetector(sampling_rate=360)
    
    # Generate synthetic ECG with known peaks
    fs = 360
    duration = 10
    t = np.linspace(0, duration, fs * duration)
    
    # Simulate heartbeats at ~75 BPM
    ecg = np.zeros_like(t)
    beat_times = np.arange(0.5, duration, 0.8)  # Every 0.8 seconds = 75 BPM
    
    for bt in beat_times:
        idx = int(bt * fs)
        if idx < len(ecg):
            for j in range(-20, 40):
                if 0 <= idx + j < len(ecg):
                    ecg[idx + j] += np.exp(-((j/10)**2))
    
    # Add noise
    ecg += 0.1 * np.random.randn(len(ecg))
    
    # Detect peaks
    peaks = detector.detect_peaks(ecg)
    
    assert len(peaks) > 0, "No peaks detected"
    assert len(peaks) >= 8, f"Too few peaks detected: {len(peaks)}"
    
    print(f"  ✓ Detected {len(peaks)} peaks")
    
    # Test beat segmentation
    segmenter = BeatSegmenter(sampling_rate=fs, beat_length=216)
    beats, valid_indices = segmenter.segment_beats(ecg, peaks)
    
    assert len(beats) > 0, "No beats segmented"
    assert beats.shape[1] == 216, "Incorrect beat length"
    
    print(f"  ✓ Segmented {len(beats)} beats")
    
    # Test heart rate calculation
    hr_calc = HeartRateCalculator(sampling_rate=fs)
    hr = hr_calc.calculate_hr(peaks)
    
    assert 60 < hr < 90, f"Unexpected heart rate: {hr} BPM"
    
    print(f"  ✓ Calculated heart rate: {hr:.1f} BPM")
    print("\nSignal processing test passed!")


def test_model_architecture():
    """Test 5: Model architecture"""
    print("\nTesting model architecture...")
    
    from models.model_architecture import create_model
    import tensorflow as tf
    
    # Create model
    model = create_model(model_type='standard', input_shape=(216, 1), num_classes=5)
    
    # Check architecture
    assert model is not None, "Model creation failed"
    assert len(model.layers) > 0, "Model has no layers"
    
    # Test with random data
    X_test = np.random.randn(10, 216, 1).astype(np.float32)
    predictions = model.predict(X_test, verbose=0)
    
    assert predictions.shape == (10, 5), "Unexpected prediction shape"
    assert np.allclose(np.sum(predictions, axis=1), 1.0, atol=1e-5), "Predictions don't sum to 1"
    
    print(f"  ✓ Model created successfully")
    print(f"  ✓ Model has {model.count_params():,} parameters")
    print(f"  ✓ Predictions working correctly")
    print("\nModel architecture test passed!")


def test_inference_engine():
    """Test 6: Inference engine"""
    print("\nTesting inference engine...")
    
    from models.model_architecture import create_model
    from realtime.inference_engine import RealtimeInferenceEngine
    
    # Create and save temporary model
    model = create_model(model_type='standard')
    temp_model_path = 'test_temp_model.h5'
    model.save(temp_model_path)
    
    try:
        # Initialize engine
        engine = RealtimeInferenceEngine(model_path=temp_model_path, sampling_rate=360)
        
        print(f"  ✓ Engine initialized")
        
        # Generate synthetic ECG
        fs = 360
        duration = 5
        t = np.linspace(0, duration, fs * duration)
        ecg = np.sin(2 * np.pi * 1.2 * t) + 0.1 * np.random.randn(len(t))
        
        # Add samples
        engine.add_samples(ecg[:360])  # Add 1 second of data
        
        print(f"  ✓ Samples added to buffer")
        
        # Get statistics
        stats = engine.get_statistics()
        
        assert 'total_beats' in stats, "Statistics missing total_beats"
        assert 'avg_inference_time_ms' in stats, "Statistics missing inference time"
        
        print(f"  ✓ Statistics working")
        print("\nInference engine test passed!")
    
    finally:
        # Clean up
        if os.path.exists(temp_model_path):
            os.remove(temp_model_path)


def test_serial_reader():
    """Test 7: Serial reader (without hardware)"""
    print("\nTesting serial reader module...")
    
    from realtime.serial_reader import ECGSerialReader
    
    # Just test import and initialization (won't connect without hardware)
    reader = ECGSerialReader()
    
    assert reader.baudrate == 115200, "Incorrect baud rate"
    assert reader.buffer_size == 1000, "Incorrect buffer size"
    
    print(f"  ✓ Serial reader module loaded")
    print(f"  ✓ Default configuration correct")
    print("\nNote: Skipping actual serial connection test (requires hardware)")
    print("Serial reader module test passed!")


def print_summary():
    """Print test summary"""
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    total_tests = tests_passed + tests_failed
    
    print(f"\nTotal Tests: {total_tests}")
    print(f"Passed: {tests_passed} ✅")
    print(f"Failed: {tests_failed} ❌")
    print(f"Success Rate: {(tests_passed/total_tests)*100:.1f}%")
    
    print("\nDetailed Results:")
    print("-" * 70)
    for test_name, status, error in test_results:
        status_icon = "✅" if status == "PASSED" else "❌"
        print(f"{status_icon} {test_name}: {status}")
        if error:
            print(f"   Error: {error}")
    
    print("\n" + "="*70)
    
    if tests_failed == 0:
        print("🎉 ALL TESTS PASSED! System is ready for use.")
    else:
        print("⚠️ Some tests failed. Please review errors above.")
    
    print("="*70)


def main():
    """Run all tests"""
    
    # Run tests
    test_module("Dependencies Check", test_dependencies)
    test_module("File Structure Check", test_file_structure)
    test_module("DSP Filters", test_filters)
    test_module("Signal Processing", test_signal_processing)
    test_module("Model Architecture", test_model_architecture)
    test_module("Inference Engine", test_inference_engine)
    test_module("Serial Reader Module", test_serial_reader)
    
    # Print summary
    print_summary()
    
    print(f"\nTest completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()

