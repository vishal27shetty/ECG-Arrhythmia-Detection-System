"""
Streamlit Dashboard for Real-Time ECG Monitoring and Arrhythmia Detection
Live visualization of ECG waveform, classification results, and alerts
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

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from realtime.serial_reader import ECGSerialReader
from realtime.inference_engine import RealtimeInferenceEngine


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
if 'inference_engine' not in st.session_state:
    st.session_state.inference_engine = None
if 'ecg_buffer' not in st.session_state:
    st.session_state.ecg_buffer = deque(maxlen=3600)  # 10 seconds at 360 Hz
if 'time_buffer' not in st.session_state:
    st.session_state.time_buffer = deque(maxlen=3600)
if 'classification_history' not in st.session_state:
    st.session_state.classification_history = []
if 'alert_history' not in st.session_state:
    st.session_state.alert_history = []
if 'heart_rate_buffer' not in st.session_state:
    st.session_state.heart_rate_buffer = deque(maxlen=100)
if 'is_running' not in st.session_state:
    st.session_state.is_running = False
if 'start_time' not in st.session_state:
    st.session_state.start_time = 0


def initialize_system(serial_port, model_path):
    """Initialize serial reader and inference engine"""
    try:
        # Initialize serial reader
        reader = ECGSerialReader(port=serial_port)
        if not reader.connect():
            return False, "Failed to connect to Arduino"
        reader.start_reading()
        st.session_state.serial_reader = reader
        
        # Initialize inference engine
        engine = RealtimeInferenceEngine(model_path=model_path)
        st.session_state.inference_engine = engine
        
        st.session_state.is_running = True
        st.session_state.start_time = time.time()
        
        return True, "System initialized successfully"
    except Exception as e:
        return False, f"Error initializing system: {str(e)}"


def stop_system():
    """Stop system and close connections"""
    if st.session_state.serial_reader:
        st.session_state.serial_reader.stop_reading()
        st.session_state.serial_reader = None
    
    st.session_state.inference_engine = None
    st.session_state.is_running = False


def update_data():
    """Update data from serial reader and inference engine"""
    if not st.session_state.is_running:
        return
    
    reader = st.session_state.serial_reader
    engine = st.session_state.inference_engine
    
    if reader is None or engine is None:
        return
    
    # Read new samples from serial
    samples = reader.get_samples(count=10, timeout=0.1)
    
    if samples:
        for sample in samples:
            if not sample['leads_off']:
                # Add to buffer
                st.session_state.ecg_buffer.append(sample['ecg'])
                st.session_state.time_buffer.append(time.time() - st.session_state.start_time)
                
                # Add to inference engine
                engine.add_samples(np.array([sample['ecg']]))
        
        # Process buffer for classification
        engine.process_buffer()
        
        # Get classification results
        result = engine.get_latest_result()
        if result:
            st.session_state.classification_history.append(result)
            
            # Check for alerts
            if 'alert' in result:
                st.session_state.alert_history.append(result['alert'])
            
            # Calculate heart rate
            if len(st.session_state.classification_history) >= 2:
                recent_beats = st.session_state.classification_history[-10:]
                time_diff = recent_beats[-1]['timestamp'] - recent_beats[0]['timestamp']
                if time_diff > 0:
                    hr = (len(recent_beats) / time_diff) * 60
                    st.session_state.heart_rate_buffer.append(hr)


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


def plot_class_distribution():
    """Plot classification distribution"""
    if len(st.session_state.classification_history) == 0:
        return go.Figure()
    
    recent_classifications = st.session_state.classification_history[-100:]
    classes = [c['class'] for c in recent_classifications]
    class_names = ['Normal', 'Supraventricular', 'Ventricular', 'Fusion', 'Unknown']
    class_labels = ['N', 'S', 'V', 'F', 'Q']
    
    counts = [classes.count(label) for label in class_labels]
    
    colors = ['#00cc00', '#ffaa00', '#ff0000', '#ff00ff', '#888888']
    
    fig = go.Figure(data=[go.Pie(
        labels=class_names,
        values=counts,
        marker=dict(colors=colors),
        hole=0.3
    )])
    
    fig.update_layout(
        title='Beat Classification Distribution (Last 100 beats)',
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
        
        # Configuration
        serial_port = st.text_input("Serial Port", value="", help="Leave empty for auto-detect")
        model_path = st.text_input("Model Path", value="./models/best_model.h5")
        
        # Start/Stop buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("▶️ Start", disabled=st.session_state.is_running):
                success, message = initialize_system(serial_port if serial_port else None, model_path)
                if success:
                    st.success(message)
                else:
                    st.error(message)
        
        with col2:
            if st.button("⏹️ Stop", disabled=not st.session_state.is_running):
                stop_system()
                st.info("System stopped")
        
        st.markdown("---")
        
        # System Status
        st.header("System Status")
        
        if st.session_state.is_running:
            st.success("🟢 Running")
            
            # Connection status
            if st.session_state.serial_reader:
                stats = st.session_state.serial_reader.get_stats()
                st.metric("Serial Connection", "Connected")
                st.metric("Total Samples", stats['total_samples'])
                st.metric("Queue Size", stats['queue_size'])
            
            # Inference status
            if st.session_state.inference_engine:
                stats = st.session_state.inference_engine.get_statistics()
                st.metric("Beats Classified", stats['total_beats'])
                st.metric("Avg Inference Time", f"{stats['avg_inference_time_ms']:.1f} ms")
        else:
            st.warning("🔴 Stopped")
        
        st.markdown("---")
        
        # Auto-refresh control
        refresh_rate = st.slider("Refresh Rate (ms)", 100, 1000, 200)
    
    # Main content area
    if st.session_state.is_running:
        # Update data
        update_data()
        
        # Top row - Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if len(st.session_state.heart_rate_buffer) > 0:
                hr = np.mean(list(st.session_state.heart_rate_buffer)[-10:])
                st.metric("Heart Rate", f"{hr:.0f} BPM", delta=None)
            else:
                st.metric("Heart Rate", "-- BPM")
        
        with col2:
            total_beats = len(st.session_state.classification_history)
            st.metric("Total Beats", total_beats)
        
        with col3:
            if st.session_state.classification_history:
                latest = st.session_state.classification_history[-1]
                st.metric("Latest Class", latest['class_full'])
            else:
                st.metric("Latest Class", "--")
        
        with col4:
            confidence = 0
            if st.session_state.classification_history:
                confidence = st.session_state.classification_history[-1]['confidence'] * 100
            st.metric("Confidence", f"{confidence:.1f}%")
        
        # Alert Panel
        if st.session_state.alert_history:
            latest_alert = st.session_state.alert_history[-1]
            alert_type = latest_alert['type']
            
            if alert_type == 'CRITICAL':
                st.markdown(f"""
                <div class="alert-critical">
                    🚨 CRITICAL ALERT: {latest_alert['message']}
                </div>
                """, unsafe_allow_html=True)
            elif alert_type == 'WARNING':
                st.markdown(f"""
                <div class="alert-warning">
                    ⚠️ WARNING: {latest_alert['message']}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="alert-info">
                    ℹ️ INFO: {latest_alert['message']}
                </div>
                """, unsafe_allow_html=True)
        
        # ECG Waveform
        st.plotly_chart(plot_ecg_waveform(), use_container_width=True)
        
        # Bottom row - Charts
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Recent classifications table
            st.subheader("Recent Classifications")
            if st.session_state.classification_history:
                recent = st.session_state.classification_history[-10:]
                df = pd.DataFrame([{
                    'Time': datetime.fromtimestamp(r['timestamp']).strftime('%H:%M:%S'),
                    'Class': r['class_full'],
                    'Confidence': f"{r['confidence']*100:.1f}%"
                } for r in reversed(recent)])
                st.dataframe(df, use_container_width=True, hide_index=True)
            else:
                st.info("No classifications yet")
        
        with col2:
            # Class distribution pie chart
            st.plotly_chart(plot_class_distribution(), use_container_width=True)
        
        # Alert History
        with st.expander("Alert History", expanded=False):
            if st.session_state.alert_history:
                df_alerts = pd.DataFrame([{
                    'Time': datetime.fromtimestamp(a['timestamp']).strftime('%H:%M:%S'),
                    'Type': a['type'],
                    'Message': a['message']
                } for a in reversed(st.session_state.alert_history[-20:])])
                st.dataframe(df_alerts, use_container_width=True, hide_index=True)
            else:
                st.info("No alerts")
        
        # Auto-refresh
        time.sleep(refresh_rate / 1000.0)
        st.rerun()
    
    else:
        # Welcome screen
        st.info("👈 Configure settings and click 'Start' to begin monitoring")
        
        st.markdown("""
        ### Instructions:
        1. **Connect Hardware**: Ensure Arduino with AD8232 is connected via USB
        2. **Configure Settings**: Set serial port (or leave empty for auto-detect)
        3. **Load Model**: Specify path to trained model (.h5 file)
        4. **Start Monitoring**: Click the Start button
        5. **Attach Electrodes**: Place ECG electrodes on patient
        
        ### Safety Notice:
        This system is for educational and research purposes only.
        Do not use for clinical diagnosis without proper validation and regulatory approval.
        """)


if __name__ == "__main__":
    main()


