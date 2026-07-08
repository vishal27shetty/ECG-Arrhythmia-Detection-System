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

# ---------------------------------------------------------------------------
# Premium Dark Theme CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    /* ---------- Google Font ---------- */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500&display=swap');

    /* ---------- Root variables ---------- */
    :root {
        --bg-primary: #0a0e1a;
        --bg-secondary: #111827;
        --bg-card: rgba(17, 24, 39, 0.7);
        --bg-card-hover: rgba(30, 41, 59, 0.8);
        --border-color: rgba(99, 102, 241, 0.2);
        --border-glow: rgba(99, 102, 241, 0.4);
        --text-primary: #f1f5f9;
        --text-secondary: #94a3b8;
        --text-muted: #64748b;
        --accent-indigo: #6366f1;
        --accent-cyan: #06b6d4;
        --accent-emerald: #10b981;
        --accent-rose: #f43f5e;
        --accent-amber: #f59e0b;
        --accent-violet: #8b5cf6;
        --gradient-primary: linear-gradient(135deg, #6366f1 0%, #06b6d4 100%);
        --gradient-rose: linear-gradient(135deg, #f43f5e 0%, #fb923c 100%);
        --gradient-emerald: linear-gradient(135deg, #10b981 0%, #06b6d4 100%);
        --shadow-lg: 0 20px 40px rgba(0, 0, 0, 0.4);
        --shadow-glow: 0 0 30px rgba(99, 102, 241, 0.15);
    }

    /* ---------- Global Overrides ---------- */
    .stApp {
        background: var(--bg-primary) !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
        color: var(--text-primary) !important;
    }

    /* ---------- Scrollbar ---------- */
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: var(--bg-primary); }
    ::-webkit-scrollbar-thumb { background: var(--accent-indigo); border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: var(--accent-cyan); }

    /* ---------- Sidebar ---------- */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f1629 0%, #111827 50%, #0f1629 100%) !important;
        border-right: 1px solid var(--border-color) !important;
    }
    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: var(--text-primary) !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em;
    }
    section[data-testid="stSidebar"] hr {
        border-color: var(--border-color) !important;
        margin: 1rem 0 !important;
    }

    /* ---------- Header area ---------- */
    header[data-testid="stHeader"] {
        background: transparent !important;
    }

    /* ---------- Main block container ---------- */
    .block-container {
        padding-top: 2rem !important;
        max-width: 1400px !important;
    }

    /* ---------- Metric cards ---------- */
    [data-testid="stMetric"] {
        background: var(--bg-card) !important;
        backdrop-filter: blur(12px) !important;
        -webkit-backdrop-filter: blur(12px) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 16px !important;
        padding: 1.1rem 1.2rem !important;
        box-shadow: var(--shadow-glow) !important;
        transition: transform 0.2s ease, border-color 0.2s ease !important;
    }
    [data-testid="stMetric"]:hover {
        transform: translateY(-2px) !important;
        border-color: var(--border-glow) !important;
    }
    [data-testid="stMetricLabel"] {
        color: var(--text-secondary) !important;
        font-size: 0.8rem !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.06em !important;
    }
    [data-testid="stMetricValue"] {
        color: var(--text-primary) !important;
        font-weight: 700 !important;
        font-size: 1.5rem !important;
    }

    /* ---------- Buttons ---------- */
    .stButton > button {
        background: var(--gradient-primary) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.6rem 1.5rem !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
        letter-spacing: 0.02em !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.3) !important;
    }
    .stButton > button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 25px rgba(99, 102, 241, 0.5) !important;
    }
    .stButton > button:disabled {
        background: linear-gradient(135deg, #374151 0%, #4b5563 100%) !important;
        color: #6b7280 !important;
        box-shadow: none !important;
        cursor: not-allowed !important;
    }

    /* ---------- Input fields ---------- */
    .stTextInput input, .stSelectbox select {
        background: var(--bg-secondary) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 10px !important;
        color: var(--text-primary) !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.85rem !important;
    }
    .stTextInput input:focus {
        border-color: var(--accent-indigo) !important;
        box-shadow: 0 0 0 2px rgba(99, 102, 241, 0.2) !important;
    }

    /* ---------- Slider ---------- */
    .stSlider [data-testid="stTickBarMin"],
    .stSlider [data-testid="stTickBarMax"] {
        color: var(--text-muted) !important;
        font-size: 0.75rem !important;
    }

    /* ---------- Progress bar ---------- */
    .stProgress > div > div {
        background: var(--gradient-primary) !important;
        border-radius: 8px !important;
    }
    .stProgress > div {
        background: rgba(99, 102, 241, 0.1) !important;
        border-radius: 8px !important;
    }

    /* ---------- Alerts ---------- */
    .stAlert, [data-testid="stAlert"] {
        border-radius: 12px !important;
        border: none !important;
        font-weight: 500 !important;
    }

    /* ---------- Markdown typography ---------- */
    .stMarkdown h1 {
        font-weight: 800 !important;
        letter-spacing: -0.03em !important;
    }
    .stMarkdown h2 {
        font-weight: 700 !important;
        letter-spacing: -0.02em !important;
        color: var(--text-primary) !important;
    }
    .stMarkdown h3 {
        font-weight: 600 !important;
        color: var(--text-primary) !important;
    }
    .stMarkdown p, .stMarkdown li {
        color: var(--text-secondary) !important;
        line-height: 1.7 !important;
    }
    .stMarkdown hr {
        border-color: var(--border-color) !important;
        margin: 1.5rem 0 !important;
    }

    /* ---------- Animated heartbeat header ---------- */
    @keyframes heartbeat {
        0%, 100% { transform: scale(1); }
        15% { transform: scale(1.15); }
        30% { transform: scale(1); }
        45% { transform: scale(1.1); }
        60% { transform: scale(1); }
    }
    @keyframes pulseGlow {
        0%, 100% { text-shadow: 0 0 10px rgba(244, 63, 94, 0.3); }
        50% { text-shadow: 0 0 30px rgba(244, 63, 94, 0.6), 0 0 60px rgba(244, 63, 94, 0.2); }
    }
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    @keyframes slideIn {
        from { opacity: 0; transform: translateX(-10px); }
        to { opacity: 1; transform: translateX(0); }
    }
    @keyframes blink {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.3; }
    }

    .dashboard-header {
        text-align: center;
        padding: 1.5rem 0 1rem 0;
        animation: fadeInUp 0.6s ease-out;
    }
    .dashboard-header .heart-icon {
        font-size: 2.2rem;
        display: inline-block;
        animation: heartbeat 1.5s ease-in-out infinite;
        margin-right: 0.3rem;
    }
    .dashboard-header .title-text {
        font-size: 2.2rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        background: linear-gradient(135deg, #f43f5e, #fb923c, #f43f5e, #a855f7);
        background-size: 300% 300%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        animation: gradientShift 4s ease infinite;
    }
    .dashboard-subtitle {
        text-align: center;
        color: var(--text-muted);
        font-size: 0.9rem;
        font-weight: 400;
        margin-top: -0.3rem;
        margin-bottom: 1.5rem;
        letter-spacing: 0.05em;
    }

    /* ---------- Status badge ---------- */
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 14px;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.04em;
    }
    .status-recording {
        background: rgba(16, 185, 129, 0.15);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }
    .status-recording .dot {
        width: 8px; height: 8px;
        background: #34d399;
        border-radius: 50%;
        animation: blink 1.2s ease-in-out infinite;
    }
    .status-stopped {
        background: rgba(100, 116, 139, 0.15);
        color: #94a3b8;
        border: 1px solid rgba(100, 116, 139, 0.3);
    }
    .status-disconnected {
        background: rgba(244, 63, 94, 0.15);
        color: #fb7185;
        border: 1px solid rgba(244, 63, 94, 0.3);
    }
    .status-disconnected .dot {
        width: 8px; height: 8px;
        background: #fb7185;
        border-radius: 50%;
        animation: blink 0.6s ease-in-out infinite;
    }

    /* ---------- Glass panel ---------- */
    .glass-panel {
        background: var(--bg-card);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid var(--border-color);
        border-radius: 20px;
        padding: 1.5rem;
        box-shadow: var(--shadow-lg);
        margin-bottom: 1.2rem;
        animation: fadeInUp 0.5s ease-out;
    }

    /* ---------- Classification result cards ---------- */
    .class-card {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0.75rem 1rem;
        border-radius: 12px;
        margin-bottom: 0.5rem;
        transition: transform 0.15s ease;
        animation: slideIn 0.4s ease-out;
    }
    .class-card:hover { transform: translateX(4px); }
    .class-card .class-label {
        font-weight: 600;
        font-size: 0.9rem;
    }
    .class-card .class-pct {
        font-weight: 700;
        font-size: 1rem;
        font-family: 'JetBrains Mono', monospace;
    }
    .class-normal {
        background: rgba(16, 185, 129, 0.1);
        border-left: 3px solid #10b981;
    }
    .class-normal .class-label { color: #34d399; }
    .class-normal .class-pct { color: #6ee7b7; }
    .class-supra {
        background: rgba(245, 158, 11, 0.1);
        border-left: 3px solid #f59e0b;
    }
    .class-supra .class-label { color: #fbbf24; }
    .class-supra .class-pct { color: #fcd34d; }
    .class-vent {
        background: rgba(244, 63, 94, 0.1);
        border-left: 3px solid #f43f5e;
    }
    .class-vent .class-label { color: #fb7185; }
    .class-vent .class-pct { color: #fda4af; }
    .class-fusion {
        background: rgba(168, 85, 247, 0.1);
        border-left: 3px solid #a855f7;
    }
    .class-fusion .class-label { color: #c084fc; }
    .class-fusion .class-pct { color: #d8b4fe; }
    .class-unknown {
        background: rgba(100, 116, 139, 0.1);
        border-left: 3px solid #64748b;
    }
    .class-unknown .class-label { color: #94a3b8; }
    .class-unknown .class-pct { color: #cbd5e1; }

    /* ---------- Welcome feature cards ---------- */
    .feature-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
        gap: 1rem;
        margin: 1.5rem 0;
    }
    .feature-card {
        background: var(--bg-card);
        border: 1px solid var(--border-color);
        border-radius: 16px;
        padding: 1.3rem;
        transition: all 0.3s ease;
        animation: fadeInUp 0.5s ease-out;
    }
    .feature-card:hover {
        border-color: var(--border-glow);
        transform: translateY(-3px);
        box-shadow: var(--shadow-glow);
    }
    .feature-card .feature-icon {
        font-size: 1.8rem;
        margin-bottom: 0.5rem;
    }
    .feature-card .feature-title {
        font-size: 1rem;
        font-weight: 700;
        color: var(--text-primary);
        margin-bottom: 0.3rem;
    }
    .feature-card .feature-desc {
        font-size: 0.82rem;
        color: var(--text-muted);
        line-height: 1.5;
    }

    /* ---------- Section headers ---------- */
    .section-header {
        display: flex;
        align-items: center;
        gap: 0.6rem;
        margin-bottom: 1rem;
        animation: fadeInUp 0.4s ease-out;
    }
    .section-header .section-icon {
        font-size: 1.3rem;
    }
    .section-header .section-title {
        font-size: 1.2rem;
        font-weight: 700;
        color: var(--text-primary);
        letter-spacing: -0.01em;
    }

    /* ---------- Confidence bar ---------- */
    .conf-bar-track {
        background: rgba(99, 102, 241, 0.1);
        border-radius: 8px;
        height: 10px;
        overflow: hidden;
        margin: 0.4rem 0;
    }
    .conf-bar-fill {
        height: 100%;
        border-radius: 8px;
        transition: width 0.6s ease;
    }

    /* ---------- Stat row ---------- */
    .stat-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.55rem 0;
        border-bottom: 1px solid rgba(99, 102, 241, 0.08);
    }
    .stat-row:last-child { border-bottom: none; }
    .stat-label {
        color: var(--text-muted);
        font-size: 0.85rem;
        font-weight: 500;
    }
    .stat-value {
        color: var(--text-primary);
        font-size: 0.95rem;
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
    }

    /* ---------- Alert overrides for dark theme ---------- */
    .disconnect-banner {
        background: linear-gradient(135deg, rgba(244,63,94,0.12) 0%, rgba(251,146,60,0.08) 100%);
        border: 1px solid rgba(244,63,94,0.3);
        border-radius: 14px;
        padding: 1rem 1.3rem;
        margin-bottom: 1rem;
        animation: fadeInUp 0.3s ease-out;
    }
    .disconnect-banner .banner-title {
        color: #fb7185;
        font-weight: 700;
        font-size: 0.95rem;
        margin-bottom: 0.2rem;
    }
    .disconnect-banner .banner-desc {
        color: #fda4af;
        font-size: 0.82rem;
        font-weight: 400;
    }

    .recording-banner {
        background: linear-gradient(135deg, rgba(16,185,129,0.1) 0%, rgba(6,182,212,0.06) 100%);
        border: 1px solid rgba(16,185,129,0.25);
        border-radius: 14px;
        padding: 1rem 1.3rem;
        margin-bottom: 1rem;
    }
    .recording-banner .banner-title {
        color: #34d399;
        font-weight: 700;
        font-size: 0.95rem;
    }

    /* ---------- Divider ---------- */
    .custom-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, var(--border-color), transparent);
        margin: 1.5rem 0;
        border: none;
    }

    /* ---------- Hide default Streamlit header/footer ---------- */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
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
if 'leads_off_active' not in st.session_state:
    st.session_state.leads_off_active = False
if 'last_update_time' not in st.session_state:
    st.session_state.last_update_time = time.time()


# ── Business Logic (unchanged) ───────────────────────────────────────────────

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
        st.session_state.last_update_time = time.time()
        st.session_state.session_duration = session_duration

        return True, f"System initialized — Recording for {session_duration}s"
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

    # Track update intervals to compute flatline samples if disconnected
    current_real_time = time.time()
    elapsed_since_update = current_real_time - st.session_state.get('last_update_time', current_real_time)
    st.session_state.last_update_time = current_real_time

    # Check connection status
    stats = reader.get_stats()
    is_connected = stats.get('is_connected', False)

    if not is_connected:
        # Device disconnected - generate flat line at 0 (baseline) at 360 Hz
        num_flat_samples = int(elapsed_since_update * 360)
        # Cap to prevent memory/buffer overflow if UI lags or browser is out of focus
        num_flat_samples = min(max(1, num_flat_samples), 200)

        for _ in range(num_flat_samples):
            current_time = time.time() - st.session_state.start_time
            st.session_state.ecg_buffer.append(0)
            st.session_state.time_buffer.append(current_time)

        st.session_state.leads_off_active = False
        return

    # Read ALL available samples from serial (don't limit to just 10!)
    # At 360 Hz with 200ms refresh, we should get ~72 samples per update
    samples = reader.get_samples(count=200, timeout=0.01)  # Get up to 200 samples

    # Track whether any sample in this batch has leads off
    any_leads_off = False

    if samples:
        for sample in samples:
            current_time = time.time() - st.session_state.start_time
            if not sample['leads_off']:
                # Normal sample - add to display buffer and complete signal
                st.session_state.ecg_buffer.append(sample['ecg'])
                st.session_state.time_buffer.append(current_time)

                # Add to complete signal for batch processing
                st.session_state.complete_ecg_signal.append(sample['ecg'])
            else:
                # Leads are off - show flat line at 0 (baseline)
                # This makes disconnection clearly visible on the waveform
                any_leads_off = True
                st.session_state.ecg_buffer.append(0)
                st.session_state.time_buffer.append(current_time)
                # Do NOT add to complete_ecg_signal - only real data goes to analysis

    # Update leads-off status for UI feedback
    st.session_state.leads_off_active = any_leads_off

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
                    current_time = time.time() - st.session_state.start_time
                    if not sample['leads_off']:
                        st.session_state.ecg_buffer.append(sample['ecg'])
                        st.session_state.time_buffer.append(current_time)
                        st.session_state.complete_ecg_signal.append(sample['ecg'])
                    else:
                        st.session_state.ecg_buffer.append(0)
                        st.session_state.time_buffer.append(current_time)


# ── Plotly Charts (redesigned) ────────────────────────────────────────────────

def plot_ecg_waveform():
    """Plot real-time ECG waveform with premium dark styling"""
    fig = go.Figure()

    if len(st.session_state.ecg_buffer) == 0:
        # Empty state placeholder
        fig.update_layout(
            height=420,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(10, 14, 26, 0.9)',
            font=dict(family='Inter', color='#64748b'),
            annotations=[dict(
                text="Waiting for ECG signal...",
                xref="paper", yref="paper", x=0.5, y=0.5,
                showarrow=False, font=dict(size=16, color='#475569')
            )]
        )
        return fig

    ecg_data = np.array(list(st.session_state.ecg_buffer))
    time_data = np.array(list(st.session_state.time_buffer))

    # Main ECG trace with cyan/teal colour and glow effect
    fig.add_trace(go.Scatter(
        x=time_data,
        y=ecg_data,
        mode='lines',
        name='ECG',
        line=dict(color='#06b6d4', width=2, shape='spline', smoothing=0.3),
        fill='tozeroy',
        fillcolor='rgba(6, 182, 212, 0.05)',
        hovertemplate='<b>Time</b>: %{x:.2f}s<br><b>Amplitude</b>: %{y:.0f}<extra></extra>'
    ))

    fig.update_layout(
        height=420,
        margin=dict(l=50, r=20, t=40, b=50),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(10, 14, 26, 0.9)',
        font=dict(family='Inter', color='#94a3b8', size=12),
        title=dict(
            text='<b>Live ECG Waveform</b>',
            font=dict(size=16, color='#e2e8f0'),
            x=0.02, xanchor='left'
        ),
        xaxis=dict(
            title=dict(text='Time (s)', font=dict(size=12, color='#64748b')),
            gridcolor='rgba(99, 102, 241, 0.06)',
            gridwidth=1,
            zerolinecolor='rgba(99, 102, 241, 0.15)',
            linecolor='rgba(99, 102, 241, 0.2)',
            tickfont=dict(color='#64748b', size=11),
            showgrid=True,
            dtick=1,
        ),
        yaxis=dict(
            title=dict(text='Amplitude', font=dict(size=12, color='#64748b')),
            gridcolor='rgba(99, 102, 241, 0.06)',
            gridwidth=1,
            zerolinecolor='rgba(244, 63, 94, 0.2)',
            zerolinewidth=1,
            linecolor='rgba(99, 102, 241, 0.2)',
            tickfont=dict(color='#64748b', size=11),
            showgrid=True,
        ),
        showlegend=False,
        hovermode='x unified',
        hoverlabel=dict(
            bgcolor='#1e293b',
            bordercolor='rgba(99, 102, 241, 0.3)',
            font=dict(color='#e2e8f0', family='JetBrains Mono', size=12)
        ),
    )

    return fig


def plot_class_distribution(results=None):
    """Plot classification distribution donut chart with dark styling"""
    fig = go.Figure()

    if results and 'class_distribution' in results:
        class_names = list(results['class_distribution'].keys())
        counts = list(results['class_distribution'].values())

        colors = {
            'Normal': '#10b981',
            'Supraventricular': '#f59e0b',
            'Ventricular': '#f43f5e',
            'Fusion': '#a855f7',
            'Unknown': '#64748b'
        }

        color_list = [colors.get(name, '#475569') for name in class_names]

        fig = go.Figure(data=[go.Pie(
            labels=class_names,
            values=counts,
            marker=dict(
                colors=color_list,
                line=dict(color='#0a0e1a', width=3)
            ),
            hole=0.55,
            textinfo='percent+label',
            textfont=dict(size=12, color='#e2e8f0', family='Inter'),
            hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>',
            hoverlabel=dict(
                bgcolor='#1e293b',
                bordercolor='rgba(99,102,241,0.3)',
                font=dict(color='#e2e8f0', family='Inter')
            ),
        )])

        # Centre annotation
        total = sum(counts)
        fig.add_annotation(
            text=f'<b>{total}</b><br><span style="font-size:11px;color:#64748b">beats</span>',
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=24, color='#e2e8f0', family='Inter'),
        )

        fig.update_layout(
            height=370,
            margin=dict(l=10, r=10, t=40, b=20),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(family='Inter', color='#94a3b8'),
            title=dict(
                text='<b>Beat Classification</b>',
                font=dict(size=15, color='#e2e8f0'),
                x=0.02, xanchor='left'
            ),
            showlegend=True,
            legend=dict(
                font=dict(size=12, color='#94a3b8'),
                bgcolor='rgba(0,0,0,0)',
                borderwidth=0,
                orientation='h',
                yanchor='bottom', y=-0.15, xanchor='center', x=0.5
            )
        )

    return fig


def plot_heart_rate_gauge(hr_mean):
    """Create a heart rate gauge chart"""
    # Determine colour zone
    if hr_mean < 60:
        bar_color = '#06b6d4'   # Bradycardia – cyan
        status = 'Low'
    elif hr_mean <= 100:
        bar_color = '#10b981'   # Normal – emerald
        status = 'Normal'
    else:
        bar_color = '#f43f5e'   # Tachycardia – rose
        status = 'Elevated'

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=hr_mean,
        number=dict(
            suffix=' BPM',
            font=dict(size=28, color='#e2e8f0', family='Inter')
        ),
        gauge=dict(
            axis=dict(
                range=[30, 180],
                tickwidth=1,
                tickcolor='#334155',
                tickfont=dict(color='#64748b', size=10),
            ),
            bar=dict(color=bar_color, thickness=0.3),
            bgcolor='rgba(17,24,39,0.5)',
            borderwidth=0,
            steps=[
                dict(range=[30, 60], color='rgba(6,182,212,0.08)'),
                dict(range=[60, 100], color='rgba(16,185,129,0.08)'),
                dict(range=[100, 180], color='rgba(244,63,94,0.08)'),
            ],
            threshold=dict(
                line=dict(color=bar_color, width=3),
                thickness=0.8,
                value=hr_mean
            )
        ),
    ))

    fig.update_layout(
        height=220,
        margin=dict(l=30, r=30, t=30, b=10),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Inter', color='#94a3b8'),
    )

    return fig


def plot_confidence_gauge(conf_mean):
    """Create a confidence gauge chart"""
    pct = conf_mean * 100
    if pct >= 70:
        bar_color = '#10b981'
    elif pct >= 50:
        bar_color = '#f59e0b'
    else:
        bar_color = '#f43f5e'

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pct,
        number=dict(
            suffix='%',
            font=dict(size=28, color='#e2e8f0', family='Inter')
        ),
        gauge=dict(
            axis=dict(
                range=[0, 100],
                tickwidth=1,
                tickcolor='#334155',
                tickfont=dict(color='#64748b', size=10),
            ),
            bar=dict(color=bar_color, thickness=0.3),
            bgcolor='rgba(17,24,39,0.5)',
            borderwidth=0,
            steps=[
                dict(range=[0, 50], color='rgba(244,63,94,0.06)'),
                dict(range=[50, 70], color='rgba(245,158,11,0.06)'),
                dict(range=[70, 100], color='rgba(16,185,129,0.06)'),
            ],
            threshold=dict(
                line=dict(color=bar_color, width=3),
                thickness=0.8,
                value=pct
            )
        ),
    ))

    fig.update_layout(
        height=220,
        margin=dict(l=30, r=30, t=30, b=10),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Inter', color='#94a3b8'),
    )

    return fig


# ── Main Application ─────────────────────────────────────────────────────────

def main():
    """Main dashboard application"""

    # ── Animated Header ───────────────────────────────────────────────────
    st.markdown("""
        <div class="dashboard-header">
            <span class="heart-icon">❤️</span>
            <span class="title-text">ECG Arrhythmia Detection System</span>
        </div>
        <div class="dashboard-subtitle">Real-time cardiac monitoring &amp; AI-powered arrhythmia classification</div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    # ── Sidebar ───────────────────────────────────────────────────────────
    with st.sidebar:
        # Sidebar branding
        st.markdown("""
            <div style="text-align:center; padding:0.5rem 0 0.8rem 0;">
                <span style="font-size:1.6rem;">🫀</span>
                <div style="font-size:1.1rem; font-weight:700; color:#e2e8f0; margin-top:0.2rem;">Control Panel</div>
                <div style="font-size:0.72rem; color:#64748b; letter-spacing:0.08em; text-transform:uppercase;">System Configuration</div>
            </div>
        """, unsafe_allow_html=True)

        if IS_PI:
            st.markdown("""
                <div style="text-align:center; margin-bottom:0.5rem;">
                    <span class="status-badge status-recording">
                        <span class="dot"></span> Raspberry Pi Mode
                    </span>
                </div>
            """, unsafe_allow_html=True)

        st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

        # Configuration
        st.markdown('<div class="section-header"><span class="section-icon">⚙️</span><span class="section-title">Configuration</span></div>', unsafe_allow_html=True)

        serial_port = st.text_input("Serial Port", value=CFG['serial_port'],
                                    help="Leave empty for auto-detect. Pi: /dev/ttyUSB0 or /dev/ttyACM0")
        model_path = st.text_input("Model Path", value=CFG['model_path'])
        session_duration = st.slider("Session Duration (s)", 15, 300, CFG['session_default'], step=5,
                                     disabled=st.session_state.is_running,
                                     help="Recording will auto-stop after this duration. Minimum 15s recommended.")

        st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

        # Start/Stop buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("▶ Start", disabled=st.session_state.is_running, use_container_width=True):
                port = serial_port.strip() if serial_port else None
                success, message = initialize_system(port, model_path, session_duration)
                if success:
                    st.success(message)
                    st.rerun()
                else:
                    st.error(message)

        with col2:
            if st.button("⏹ Stop", disabled=not st.session_state.is_running, use_container_width=True):
                results = stop_system()
                st.rerun()

        st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

        # ── System Status ─────────────────────────────────────────────────
        st.markdown('<div class="section-header"><span class="section-icon">📡</span><span class="section-title">System Status</span></div>', unsafe_allow_html=True)

        if st.session_state.is_running:
            elapsed = time.time() - st.session_state.start_time
            remaining = max(0, st.session_state.session_duration - elapsed)
            progress_pct = min(1.0, elapsed / st.session_state.session_duration)

            # Connection status
            is_sidebar_connected = True
            if st.session_state.serial_reader:
                stats = st.session_state.serial_reader.get_stats()
                is_sidebar_connected = stats.get('is_connected', False)

            if is_sidebar_connected:
                st.markdown(f"""
                    <div style="text-align:center; margin-bottom:0.6rem;">
                        <span class="status-badge status-recording">
                            <span class="dot"></span> RECORDING
                        </span>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    <div style="text-align:center; margin-bottom:0.6rem;">
                        <span class="status-badge status-disconnected">
                            <span class="dot"></span> DISCONNECTED
                        </span>
                    </div>
                """, unsafe_allow_html=True)

            st.progress(progress_pct)

            # Time stats
            st.markdown(f"""
                <div class="glass-panel" style="padding:0.9rem 1rem;">
                    <div class="stat-row">
                        <span class="stat-label">⏱ Elapsed</span>
                        <span class="stat-value">{elapsed:.1f}s</span>
                    </div>
                    <div class="stat-row">
                        <span class="stat-label">⏳ Remaining</span>
                        <span class="stat-value">{remaining:.1f}s</span>
                    </div>
                    <div class="stat-row">
                        <span class="stat-label">📶 Samples</span>
                        <span class="stat-value">{stats.get('total_samples', 0) if st.session_state.serial_reader else 0}</span>
                    </div>
                    <div class="stat-row">
                        <span class="stat-label">📏 Signal</span>
                        <span class="stat-value">{len(st.session_state.complete_ecg_signal)/360:.1f}s</span>
                    </div>
                </div>
            """, unsafe_allow_html=True)

        else:
            st.markdown("""
                <div style="text-align:center; margin-bottom:0.8rem;">
                    <span class="status-badge status-stopped">● IDLE</span>
                </div>
            """, unsafe_allow_html=True)

        st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

        # Refresh rate control
        st.markdown('<div class="section-header"><span class="section-icon">🔄</span><span class="section-title">Refresh Rate</span></div>', unsafe_allow_html=True)
        refresh_rate = st.slider("Refresh (ms)", 50, 500, CFG['refresh_default'], step=50,
                                 help="Faster refresh = better data. Pi: 200-300ms recommended.")

    # ══════════════════════════════════════════════════════════════════════
    # MAIN CONTENT AREA
    # ══════════════════════════════════════════════════════════════════════

    if st.session_state.is_running:
        # Update data
        update_data()

        # Check connection status
        is_connected = True
        if st.session_state.serial_reader:
            stats = st.session_state.serial_reader.get_stats()
            is_connected = stats.get('is_connected', False)

        # ── Warning Banners ───────────────────────────────────────────────
        if not is_connected:
            st.markdown("""
                <div class="disconnect-banner">
                    <div class="banner-title">⚠️ DEVICE DISCONNECTED</div>
                    <div class="banner-desc">USB/Serial connection lost. Waveform shows flat line. Check USB cable and Arduino power.</div>
                </div>
            """, unsafe_allow_html=True)
        elif st.session_state.leads_off_active:
            st.markdown("""
                <div class="disconnect-banner">
                    <div class="banner-title">🔌 LEADS DISCONNECTED</div>
                    <div class="banner-desc">Electrodes are not connected. Signal shows flat line. Reconnect electrodes to resume recording.</div>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
                <div class="recording-banner">
                    <div class="banner-title">💚 Recording in progress — Click STOP when ready to analyze</div>
                </div>
            """, unsafe_allow_html=True)

        # ── Live ECG Waveform ─────────────────────────────────────────────
        st.markdown('<div class="section-header"><span class="section-icon">💓</span><span class="section-title">Live ECG Waveform</span></div>', unsafe_allow_html=True)
        st.plotly_chart(plot_ecg_waveform(), use_container_width=True, key='ecg_waveform_chart')

        # ── Signal Stats Bar ──────────────────────────────────────────────
        elapsed = time.time() - st.session_state.start_time
        expected_samples = int(elapsed * 360)
        actual_samples = len(st.session_state.complete_ecg_signal)
        sample_rate_pct = (actual_samples / max(1, expected_samples)) * 100 if expected_samples > 0 else 0
        signal_std = np.std(list(st.session_state.ecg_buffer)[-360:]) if len(st.session_state.ecg_buffer) > 360 else 0

        col1, col2, col3 = st.columns(3)
        with col1:
            signal_quality = "✅ Good" if signal_std > 10 else "⚠️ Low"
            st.metric("Signal Quality", signal_quality)
        with col2:
            st.metric("Samples Collected", f"{actual_samples:,}")
        with col3:
            st.metric("Data Rate", f"{sample_rate_pct:.0f}%")

        # ── Data rate warnings (only when connected) ──────────────────────
        if is_connected and sample_rate_pct < 50 and elapsed > 2:
            st.warning(f"⚠️ Low data rate ({sample_rate_pct:.1f}%). Check Arduino connection.")
        elif not is_connected:
            st.caption("Reconnect the device to resume data collection.")

        # Auto-refresh
        time.sleep(refresh_rate / 1000.0)
        st.rerun()

    else:
        # ══════════════════════════════════════════════════════════════════
        # RESULTS or WELCOME
        # ══════════════════════════════════════════════════════════════════
        if 'last_session_results' in st.session_state and st.session_state.last_session_results:
            results = st.session_state.last_session_results

            # ── Success banner ────────────────────────────────────────────
            st.markdown("""
                <div class="glass-panel" style="text-align:center; border-color:rgba(16,185,129,0.3);">
                    <div style="font-size:2rem; margin-bottom:0.3rem;">✅</div>
                    <div style="font-size:1.3rem; font-weight:800; color:#34d399;">Analysis Complete</div>
                    <div style="font-size:0.85rem; color:#64748b; margin-top:0.2rem;">Session processed successfully</div>
                </div>
            """, unsafe_allow_html=True)

            # ── Top-level metrics ─────────────────────────────────────────
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Duration", f"{results['signal_duration']:.1f}s")
            with col2:
                avg_hr = results['heart_rate']['mean']
                st.metric("Avg Heart Rate", f"{avg_hr:.0f} BPM" if avg_hr > 0 else "N/A")
            with col3:
                avg_conf = results['confidence']['mean'] * 100
                st.metric("Avg Confidence", f"{avg_conf:.1f}%")
            with col4:
                st.metric("Total Beats", f"{results['total_beats']}")

            st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

            # ── Classification + Confidence side by side ──────────────────
            col_left, col_right = st.columns([1.1, 0.9])

            with col_left:
                st.markdown('<div class="section-header"><span class="section-icon">🧬</span><span class="section-title">Beat Classification</span></div>', unsafe_allow_html=True)

                class_css_map = {
                    'Normal':           ('class-normal',  '✅'),
                    'Supraventricular': ('class-supra',   '⚡'),
                    'Ventricular':      ('class-vent',    '🔴'),
                    'Fusion':           ('class-fusion',  '🟣'),
                    'Unknown':          ('class-unknown', '❓'),
                }

                cards_html = ""
                for class_name, (css_class, icon) in class_css_map.items():
                    pct = results['class_percentages'].get(class_name, 0)
                    count = results['class_distribution'].get(class_name, 0)
                    cards_html += f"""
                        <div class="class-card {css_class}">
                            <span class="class-label">{icon} {class_name}</span>
                            <span class="class-pct">{pct:.1f}%</span>
                        </div>
                    """

                st.markdown(f'<div class="glass-panel" style="padding:1rem;">{cards_html}</div>', unsafe_allow_html=True)

            with col_right:
                st.markdown('<div class="section-header"><span class="section-icon">🎯</span><span class="section-title">Confidence Analysis</span></div>', unsafe_allow_html=True)

                conf = results['confidence']

                st.markdown(f"""
                    <div class="glass-panel" style="padding:1rem;">
                        <div class="stat-row">
                            <span class="stat-label">Mean</span>
                            <span class="stat-value">{conf['mean']*100:.1f}%</span>
                        </div>
                        <div class="stat-row">
                            <span class="stat-label">Std Dev</span>
                            <span class="stat-value">{conf['std']*100:.1f}%</span>
                        </div>
                        <div class="stat-row">
                            <span class="stat-label">Min</span>
                            <span class="stat-value">{conf['min']*100:.1f}%</span>
                        </div>
                        <div class="stat-row">
                            <span class="stat-label">Max</span>
                            <span class="stat-value">{conf['max']*100:.1f}%</span>
                        </div>
                    </div>
                """, unsafe_allow_html=True)

                st.plotly_chart(plot_confidence_gauge(conf['mean']), use_container_width=True, key='confidence_gauge')

            st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

            # ── Distribution chart ────────────────────────────────────────
            st.plotly_chart(plot_class_distribution(results), use_container_width=True, key='results_class_distribution')

            st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

            # ── Heart Rate Analysis ───────────────────────────────────────
            if results['heart_rate']['mean'] > 0:
                st.markdown('<div class="section-header"><span class="section-icon">❤️</span><span class="section-title">Heart Rate Analysis</span></div>', unsafe_allow_html=True)
                hr = results['heart_rate']

                col_gauge, col_stats = st.columns([1, 1])

                with col_gauge:
                    st.plotly_chart(plot_heart_rate_gauge(hr['mean']), use_container_width=True, key='hr_gauge')

                with col_stats:
                    st.markdown(f"""
                        <div class="glass-panel" style="padding:1.1rem;">
                            <div class="stat-row">
                                <span class="stat-label">Mean HR</span>
                                <span class="stat-value" style="color:#34d399;">{hr['mean']:.0f} BPM</span>
                            </div>
                            <div class="stat-row">
                                <span class="stat-label">Std Dev</span>
                                <span class="stat-value">{hr['std']:.0f} BPM</span>
                            </div>
                            <div class="stat-row">
                                <span class="stat-label">Min HR</span>
                                <span class="stat-value" style="color:#06b6d4;">{hr['min']:.0f} BPM</span>
                            </div>
                            <div class="stat-row">
                                <span class="stat-label">Max HR</span>
                                <span class="stat-value" style="color:#fb7185;">{hr['max']:.0f} BPM</span>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)

                st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

            # ── Processing Performance ────────────────────────────────────
            st.markdown('<div class="section-header"><span class="section-icon">⚡</span><span class="section-title">Processing Performance</span></div>', unsafe_allow_html=True)

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("R-peaks Detected", results['r_peaks_detected'])
            with col2:
                st.metric("Valid Beats", results['valid_beats_segmented'])
            with col3:
                st.metric("Processing Time", f"{results['processing_time']:.2f}s")

            st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

            # New Session button
            col_btn = st.columns([1, 1, 1])
            with col_btn[1]:
                if st.button("🔄 Start New Session", use_container_width=True):
                    st.session_state.last_session_results = None
                    st.rerun()

        else:
            # ── Welcome Screen ────────────────────────────────────────────
            st.markdown("""
                <div class="glass-panel" style="text-align:center; padding:2rem;">
                    <div style="font-size:3.5rem; margin-bottom:0.5rem;">🫀</div>
                    <div style="font-size:1.4rem; font-weight:800; color:#e2e8f0; margin-bottom:0.3rem;">
                        Welcome to ECG Monitor
                    </div>
                    <div style="font-size:0.9rem; color:#64748b; max-width:500px; margin:0 auto;">
                        Connect your Arduino with AD8232 sensor, configure settings in the sidebar, and start recording.
                    </div>
                </div>
            """, unsafe_allow_html=True)

            # Feature cards
            st.markdown("""
                <div class="feature-grid">
                    <div class="feature-card">
                        <div class="feature-icon">📡</div>
                        <div class="feature-title">Live Waveform</div>
                        <div class="feature-desc">Real-time ECG visualization at 360 Hz with automatic leads-off detection.</div>
                    </div>
                    <div class="feature-card">
                        <div class="feature-icon">🧠</div>
                        <div class="feature-title">AI Classification</div>
                        <div class="feature-desc">Deep learning model classifies each heartbeat into 5 categories (AAMI standard).</div>
                    </div>
                    <div class="feature-card">
                        <div class="feature-icon">📊</div>
                        <div class="feature-title">Batch Analysis</div>
                        <div class="feature-desc">Complete recording processed at once for higher accuracy R-peak detection.</div>
                    </div>
                    <div class="feature-card">
                        <div class="feature-icon">🍓</div>
                        <div class="feature-title">Pi Compatible</div>
                        <div class="feature-desc">Optimised TFLite inference for Raspberry Pi with reduced memory footprint.</div>
                    </div>
                </div>
            """, unsafe_allow_html=True)

            st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

            # Instructions
            st.markdown('<div class="section-header"><span class="section-icon">📋</span><span class="section-title">Quick Start Guide</span></div>', unsafe_allow_html=True)

            col_l, col_r = st.columns(2)

            with col_l:
                st.markdown("""
                    <div class="glass-panel">
                        <div style="font-weight:700; color:#e2e8f0; margin-bottom:0.6rem; font-size:0.95rem;">🔧 Setup</div>
                        <div class="stat-row"><span class="stat-label">1</span><span class="stat-value" style="font-size:0.85rem; font-family:Inter;">Connect Arduino + AD8232 via USB</span></div>
                        <div class="stat-row"><span class="stat-label">2</span><span class="stat-value" style="font-size:0.85rem; font-family:Inter;">Attach electrodes to patient</span></div>
                        <div class="stat-row"><span class="stat-label">3</span><span class="stat-value" style="font-size:0.85rem; font-family:Inter;">Set duration (60s recommended)</span></div>
                        <div class="stat-row"><span class="stat-label">4</span><span class="stat-value" style="font-size:0.85rem; font-family:Inter;">Click ▶ Start to begin</span></div>
                    </div>
                """, unsafe_allow_html=True)

            with col_r:
                st.markdown("""
                    <div class="glass-panel">
                        <div style="font-weight:700; color:#e2e8f0; margin-bottom:0.6rem; font-size:0.95rem;">📖 How It Works</div>
                        <div class="stat-row"><span class="stat-label">Record</span><span class="stat-value" style="font-size:0.85rem; font-family:Inter;">Live ECG waveform display</span></div>
                        <div class="stat-row"><span class="stat-label">Process</span><span class="stat-value" style="font-size:0.85rem; font-family:Inter;">Batch R-peak detection</span></div>
                        <div class="stat-row"><span class="stat-label">Classify</span><span class="stat-value" style="font-size:0.85rem; font-family:Inter;">Beat-by-beat AI analysis</span></div>
                        <div class="stat-row"><span class="stat-label">Report</span><span class="stat-value" style="font-size:0.85rem; font-family:Inter;">Comprehensive results &amp; logs</span></div>
                    </div>
                """, unsafe_allow_html=True)

            # Safety notice
            st.markdown("""
                <div style="text-align:center; margin-top:1.5rem; padding:0.8rem; border-radius:12px; background:rgba(245,158,11,0.06); border:1px solid rgba(245,158,11,0.2);">
                    <span style="font-size:0.82rem; color:#fbbf24; font-weight:500;">
                        ⚠️ For educational and research purposes ONLY. Do not use for clinical diagnosis.
                    </span>
                </div>
            """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
