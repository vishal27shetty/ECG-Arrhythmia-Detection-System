"""
Streamlit Dashboard for Real-Time ECG Monitoring and Arrhythmia Detection
Live visualization of ECG waveform, classification results, and alerts

Runs on both desktop and Raspberry Pi.  Set the environment variable
    ECGPI=1
to activate Pi-optimised defaults (smaller buffers, .tflite model, 50 Hz
powerline filter, /dev/ttyUSB0 serial port).
"""

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
import time
from collections import deque
from datetime import datetime
import sys
import os
import platform

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from realtime.serial_reader import ECGSerialReader
from realtime.batch_processor import BatchECGProcessor

# ---------- Pi detection ----------
IS_PI = (
    os.environ.get('ECGPI', '0') == '1'
    or (platform.system() == 'Linux' and platform.machine().startswith('aarch64'))
    or (platform.system() == 'Linux' and platform.machine().startswith('arm'))
)

# Tunable defaults that differ between desktop and Pi
PI_DEFAULTS = {
    'buffer_maxlen': 7200,      # ~20 s at 360 Hz (saves RAM)
    'model_path': './models/best_model.tflite',
    'serial_port': '',          # auto-detect (Uno=/dev/ttyACM0, CH340=/dev/ttyUSB0)
    'refresh_default': 250,     # slower refresh to ease CPU
    'session_default': 30,
}
DESKTOP_DEFAULTS = {
    'buffer_maxlen': 36000,     # ~100 s at 360 Hz
    'model_path': './models/best_model.h5',
    'serial_port': '',
    'refresh_default': 100,
    'session_default': 30,
}
CFG = PI_DEFAULTS if IS_PI else DESKTOP_DEFAULTS


# Page configuration
st.set_page_config(
    page_title="ECG Arrhythmia Monitor",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .big-font {
        font-size: 24px !important;
        font-weight: bold;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .alert-critical {
        background-color: #ff4b4b;
        color: white;
        padding: 15px;
        border-radius: 5px;
        font-weight: bold;
        margin: 10px 0;
    }
    .alert-warning {
        background-color: #ffa500;
        color: white;
        padding: 15px;
        border-radius: 5px;
        font-weight: bold;
        margin: 10px 0;
    }
    .alert-info {
        background-color: #4b7bff;
        color: white;
        padding: 15px;
        border-radius: 5px;
        font-weight: bold;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)


# Initialize session state
if 'serial_reader' not in st.session_state:
    st.session_state.serial_reader = None
if 'batch_processor' not in st.session_state:
    st.session_state.batch_processor = None
if 'ecg_buffer' not in st.session_state:
    st.session_state.ecg_buffer = deque(maxlen=CFG['buffer_maxlen'])
if 'time_buffer' not in st.session_state:
    st.session_state.time_buffer = deque(maxlen=CFG['buffer_maxlen'])
if 'complete_ecg_signal' not in st.session_state:
    st.session_state.complete_ecg_signal = []
if 'is_running' not in st.session_state:
    st.session_state.is_running = False
if 'start_time' not in st.session_state:
    st.session_state.start_time = 0
if 'session_duration' not in st.session_state:
    st.session_state.session_duration = 30  # Default 30 seconds
if 'last_session_results' not in st.session_state:
    st.session_state.last_session_results = None


def initialize_system(serial_port, model_path, session_duration):
    """Initialize serial reader and batch processor"""
    try:
        # Initialize serial reader
        reader = ECGSerialReader(port=serial_port)
        if not reader.connect():
            detail = reader.last_error or "Unknown serial error"
            return False, f"Failed to connect to Arduino. {detail}"
        reader.start_reading()
        st.session_state.serial_reader = reader
        
        # Initialize batch processor
        processor = BatchECGProcessor(model_path=model_path, sampling_rate=360)
        st.session_state.batch_processor = processor
        
        # Reset buffers
        st.session_state.ecg_buffer.clear()
        st.session_state.time_buffer.clear()
        st.session_state.complete_ecg_signal = []
        
        st.session_state.is_running = True
        st.session_state.start_time = time.time()
        st.session_state.session_duration = session_duration
        
        return True, f"System initialized - Recording for {session_duration} seconds..."
    except Exception as e:
        return False, f"Error initializing system: {str(e)}"


def stop_system():
    """Stop system and process complete recording"""
    results = None
    
    try:
        # Stop serial reading first
        if st.session_state.serial_reader:
            st.session_state.serial_reader.stop_reading()
            st.session_state.serial_reader = None
        
        # Get complete ECG signal
        complete_signal = np.array(st.session_state.complete_ecg_signal)
        
        duration_sec = len(complete_signal) / 360
        
        # Check minimum duration
        if len(complete_signal) < 360:  # Less than 1 second
            st.error("⚠️ Recording too short for analysis (< 1 second)")
            st.session_state.is_running = False
            return None
        
        if len(complete_signal) < 5400:  # Less than 15 seconds
            st.warning(f"⚠️ Short recording ({duration_sec:.1f}s). Recommend at least 15 seconds for reliable analysis.")
        
        # Display signal statistics
        st.info(f"📊 Signal collected: {len(complete_signal)} samples ({duration_sec:.1f}s)")
        st.write(f"  - Mean: {np.mean(complete_signal):.2f}")
        st.write(f"  - Std Dev: {np.std(complete_signal):.2f}")
        st.write(f"  - Range: {np.max(complete_signal) - np.min(complete_signal):.2f}")
        
        # Process the complete recording
        if st.session_state.batch_processor:
            st.info(f"🔄 Processing {len(complete_signal)/360:.1f} seconds of ECG data...")
            results = st.session_state.batch_processor.process_recording(complete_signal)
            
            # Check for errors
            if 'error' in results:
                st.error(f"❌ Processing failed: {results['error']}")
                if 'quality' in results:
                    st.warning("Signal Quality Issues:")
                    for warning in results['quality'].get('warnings', []):
                        st.write(f"  - {warning}")
                st.session_state.is_running = False
                return None
            
            # Check if too few beats detected
            duration_sec = len(complete_signal) / 360
            expected_min_beats = int((40 / 60) * duration_sec)
            
            if results['total_beats'] < expected_min_beats / 2:
                st.error(f"⚠️ Very few beats detected: {results['total_beats']} beats in {duration_sec:.1f}s")
                st.warning("""
                **Possible causes:**
                1. Poor electrode contact - check connections
                2. Incorrect electrode placement - follow standard lead II placement
                3. Signal noise - ensure cables are not moving
                4. Low signal amplitude - check if ECG sensor is powered correctly
                
                **Recommendations:**
                - Ensure electrodes are firmly attached to skin
                - Use electrode gel if available
                - Keep cables still during recording
                - Try a longer recording (30-60 seconds recommended)
                """)
            
            # Store results
            st.session_state.last_session_results = results
            
            # Save to file
            import json
            from datetime import datetime
            session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            results['session_id'] = session_id
            
            import os
            os.makedirs('./logs', exist_ok=True)
            log_path = f'./logs/ecg_batch_{session_id}.json'
            with open(log_path, 'w') as f:
                json.dump(results, f, indent=2)
            
            if results['total_beats'] > 0:
                st.success(f"✅ Processing complete! Analyzed {results['total_beats']} beats")
                st.info(f"📁 Results saved to: {log_path}")
            else:
                st.warning(f"⚠️ Processing complete but no beats detected. Check signal quality.")
                st.info(f"📁 Results saved to: {log_path}")
        
    except Exception as e:
        st.error(f"Error processing recording: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
    
    st.session_state.batch_processor = None
    st.session_state.is_running = False
    
    return results


def update_data():
    """Update data from serial reader - just collect, don't classify"""
    if not st.session_state.is_running:
        return
    
    reader = st.session_state.serial_reader
    
    if reader is None:
        return
    
    # Check if session duration exceeded
    elapsed_time = time.time() - st.session_state.start_time
    if elapsed_time >= st.session_state.session_duration:
        # Auto-stop and process
        st.warning(f"⏱️ Session duration ({st.session_state.session_duration}s) reached - stopping and processing...")
        stop_system()
        st.rerun()
        return
    
    # Read ALL available samples from serial (don't limit to just 10!)
    # At 360 Hz with 200ms refresh, we should get ~72 samples per update
    samples = reader.get_samples(count=200, timeout=0.01)  # Get up to 200 samples
    
    if samples:
        for sample in samples:
            if not sample['leads_off']:
                # Add to display buffer
                st.session_state.ecg_buffer.append(sample['ecg'])
                st.session_state.time_buffer.append(time.time() - st.session_state.start_time)
                
                # Add to complete signal for batch processing
                st.session_state.complete_ecg_signal.append(sample['ecg'])
    
    # If still not getting enough samples, warn about serial issues
    elapsed_time = time.time() - st.session_state.start_time
    if elapsed_time > 2.0:  # After 2 seconds
        expected_samples = int(elapsed_time * 360)
        actual_samples = len(st.session_state.complete_ecg_signal)
        if actual_samples < expected_samples * 0.5:  # Less than 50%
            # Try to drain more from the queue
            extra_samples = reader.get_samples(count=500, timeout=0.001)
            if extra_samples:
                for sample in extra_samples:
                    if not sample['leads_off']:
                        st.session_state.ecg_buffer.append(sample['ecg'])
                        st.session_state.time_buffer.append(time.time() - st.session_state.start_time)
                        st.session_state.complete_ecg_signal.append(sample['ecg'])


def plot_ecg_waveform():
    """Plot real-time ECG waveform"""
    if len(st.session_state.ecg_buffer) == 0:
        return go.Figure()
    
    ecg_data = np.array(list(st.session_state.ecg_buffer))
    time_data = np.array(list(st.session_state.time_buffer))
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=time_data,
        y=ecg_data,
        mode='lines',
        name='ECG',
        line=dict(color='red', width=2)
    ))
    
    fig.update_layout(
        title='Live ECG Waveform',
        xaxis_title='Time (seconds)',
        yaxis_title='Amplitude',
        height=400,
        showlegend=False,
        hovermode='x unified'
    )
    
    return fig


def plot_class_distribution(results=None):
    """Plot classification distribution from batch results"""
    fig = go.Figure()
    
    if results and 'class_distribution' in results:
        class_names = list(results['class_distribution'].keys())
        counts = list(results['class_distribution'].values())
        
        colors = {
            'Normal': '#00cc00',
            'Supraventricular': '#ffaa00',
            'Ventricular': '#ff0000',
            'Fusion': '#ff00ff',
            'Unknown': '#888888'
        }
        
        color_list = [colors.get(name, '#cccccc') for name in class_names]
        
        fig = go.Figure(data=[go.Pie(
            labels=class_names,
            values=counts,
            marker=dict(colors=color_list),
            hole=0.3
        )])
        
        fig.update_layout(
            title='Beat Classification Distribution',
            height=350
        )
    
    return fig


def main():
    """Main dashboard application"""
    
    # Title
    st.title("❤️ ECG Arrhythmia Detection System")
    st.markdown("---")
    
    # Sidebar - System Control
    with st.sidebar:
        st.header("System Control")
        
        if IS_PI:
            st.caption("Running in **Raspberry Pi** mode")
        
        # Configuration
        serial_port = st.text_input("Serial Port", value=CFG['serial_port'],
                                    help="Leave empty for auto-detect. Pi: /dev/ttyUSB0 or /dev/ttyACM0")
        model_path = st.text_input("Model Path", value=CFG['model_path'])
        session_duration = st.slider("Session Duration (seconds)", 15, 300, CFG['session_default'], step=5,
                                     disabled=st.session_state.is_running,
                                     help="Recording will auto-stop after this duration. Minimum 15s recommended for reliable analysis.")
        
        # Start/Stop buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("▶️ Start", disabled=st.session_state.is_running):
                port = serial_port.strip() if serial_port else None
                success, message = initialize_system(port, model_path, session_duration)
                if success:
                    st.success(message)
                    st.rerun()
                else:
                    st.error(message)
        
        with col2:
            if st.button("⏹️ Stop", disabled=not st.session_state.is_running):
                results = stop_system()
                st.rerun()
        
        st.markdown("---")
        
        # System Status
        st.header("System Status")
        
        if st.session_state.is_running:
            elapsed = time.time() - st.session_state.start_time
            remaining = st.session_state.session_duration - elapsed
            st.success(f"Recording... {elapsed:.1f}s / {st.session_state.session_duration}s")
            st.progress(min(1.0, elapsed / st.session_state.session_duration))
            st.info(f"⏱️ Time Remaining: {remaining:.1f}s")
            
            # Connection status
            if st.session_state.serial_reader:
                stats = st.session_state.serial_reader.get_stats()
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Connection", "✅ Connected")
                with col2:
                    st.metric("Samples Collected", stats['total_samples'])
                with col3:
                    st.metric("Signal Length", f"{len(st.session_state.complete_ecg_signal)/360:.1f}s")
        else:
            st.warning("Stopped")
        
        st.markdown("---")
        
        # Auto-refresh control
        refresh_rate = st.slider("Refresh Rate (ms)", 50, 500, CFG['refresh_default'], step=50,
                                 help="Faster refresh = better data collection. Pi: 200-300ms recommended.")
    
    # Main content area
    if st.session_state.is_running:
        # Update data
        update_data()
        
        st.header("Live ECG Signal")
        
        # Check if we're getting enough data
        elapsed = time.time() - st.session_state.start_time
        expected_samples = int(elapsed * 360)  # 360 Hz
        actual_samples = len(st.session_state.complete_ecg_signal)
        sample_rate_pct = (actual_samples / max(1, expected_samples)) * 100 if expected_samples > 0 else 0
        
        if sample_rate_pct < 50:
            st.error(f"⚠️ LOW DATA RATE: Only receiving {sample_rate_pct:.1f}% of expected samples!")
            st.warning("Check Arduino connection and serial port settings.")
        elif sample_rate_pct < 80:
            st.warning(f"⚠️ Reduced data rate: {sample_rate_pct:.1f}% of expected samples")
        else:
            st.info("💚 Recording in progress... Click STOP when ready to analyze.")
        
        # ECG Waveform - Just show the live signal
        st.plotly_chart(plot_ecg_waveform(), width='stretch', key='ecg_waveform_chart')
        
        # Signal statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            signal_std = np.std(list(st.session_state.ecg_buffer)[-360:]) if len(st.session_state.ecg_buffer) > 360 else 0
            signal_quality = "✅ Good" if signal_std > 10 else "⚠️ Low"
            st.metric("Signal Quality", signal_quality)
        with col2:
            st.metric("Samples Collected", f"{actual_samples}")
        with col3:
            st.metric("Data Rate", f"{sample_rate_pct:.0f}%")
        
        # Auto-refresh
        time.sleep(refresh_rate / 1000.0)
        st.rerun()
    
    else:
        # Check if there are batch processing results to display
        if 'last_session_results' in st.session_state and st.session_state.last_session_results:
            st.success("✅ Analysis Complete!")
            
            results = st.session_state.last_session_results
            
            # Session Summary
            st.header("Session Analysis Results")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Duration", f"{results['signal_duration']:.1f}s")
            with col2:
                avg_hr = results['heart_rate']['mean']
                st.metric("Avg Heart Rate", f"{avg_hr:.0f} BPM" if avg_hr > 0 else "N/A")
            with col3:
                avg_conf = results['confidence']['mean'] * 100
                st.metric("Avg Confidence", f"{avg_conf:.1f}%")
            
            st.markdown("---")
            
            # Classification Distribution
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Beat Classification Distribution")
                
                # Define all classes with icons
                all_classes = {
                    'Normal': '✅',
                    'Supraventricular': '⚠️',
                    'Ventricular': '⚠️',
                    'Fusion': '⚠️',
                    'Unknown': '❓'
                }
                
                # Show all classes with percentages only (no beat counts)
                st.write("---")
                for class_name, icon in all_classes.items():
                    pct = results['class_percentages'].get(class_name, 0)
                    st.write(f"{icon} **{class_name}**: {pct:.1f}%")
            
            with col2:
                st.subheader("Confidence Analysis")
                conf = results['confidence']
                st.write(f"**Mean**: {conf['mean']*100:.1f}%")
                st.write(f"**Std Dev**: {conf['std']*100:.1f}%")
                st.write(f"**Range**: {conf['min']*100:.1f}% - {conf['max']*100:.1f}%")
            
            # Visual distribution chart
            st.plotly_chart(plot_class_distribution(results), width='stretch', key='results_class_distribution')
            
            st.markdown("---")
            
            # Heart Rate Analysis
            if results['heart_rate']['mean'] > 0:
                st.subheader("❤️ Heart Rate Analysis")
                hr = results['heart_rate']
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Mean HR", f"{hr['mean']:.0f} BPM")
                with col2:
                    st.metric("Std Dev", f"{hr['std']:.0f} BPM")
                with col3:
                    st.metric("Min HR", f"{hr['min']:.0f} BPM")
                with col4:
                    st.metric("Max HR", f"{hr['max']:.0f} BPM")
                
                st.markdown("---")
            
            # Processing Performance
            st.subheader("⚡ Processing Performance")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("R-peaks Detected", results['r_peaks_detected'])
            with col2:
                st.metric("Valid Beats", results['valid_beats_segmented'])
            with col3:
                st.metric("Processing Time", f"{results['processing_time']:.2f}s")
            
            st.markdown("---")
            
            # Button to clear results
            if st.button("Start New Session"):
                st.session_state.last_session_results = None
                st.rerun()
        
        else:
            # Welcome screen
            st.info("👈 Configure settings and click 'Start' to begin recording")
            
            platform_note = ""
            if IS_PI:
                platform_note = """
            ### Raspberry Pi Mode Active
            - Using **TFLite** runtime for lightweight inference
            - Reduced buffer sizes to save RAM
            - Default serial port: auto-detect (`/dev/ttyACM0` or `/dev/ttyUSB0`)
            - Tip: use `ECGPI=1 streamlit run dashboard/app.py` to force Pi mode
            """
            
            st.markdown(f"""
            ### How It Works:
            
            **During Recording:**
            - System displays LIVE ECG waveform only
            - No classification during recording (faster, no duplicates)
            - Fixed duration session (30-300 seconds)
            
            **After Stopping:**
            - Batch processes the ENTIRE recording at once
            - More accurate R-peak detection
            - Complete beat-by-beat classification
            - Comprehensive analysis report
            - Results saved to JSON log files
            {platform_note}
            ### Instructions:
            1. **Connect Hardware**: Arduino with AD8232 via USB
            2. **Set Duration**: Choose recording length (60s recommended)
            3. **Start Recording**: Click Start button
            4. **Wait**: System auto-stops after duration
            5. **View Results**: Complete analysis displayed automatically
            
            ### Safety Notice:
            ⚠️ For educational and research purposes ONLY.
            Do not use for clinical diagnosis.
            """)


if __name__ == "__main__":
    main()


