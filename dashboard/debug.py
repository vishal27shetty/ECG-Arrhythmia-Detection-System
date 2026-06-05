"""
Serial Debug Dashboard for ECG Arduino Connection
Shows raw serial output in a terminal-like UI for debugging
the AD8232 sensor connection on Raspberry Pi or desktop.

Usage:
    streamlit run dashboard/debug.py
"""

import streamlit as st
import serial
import serial.tools.list_ports
import time
import threading
import collections
import numpy as np
import plotly.graph_objects as go
import sys
import os
import platform

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ---------- Pi detection ----------
IS_PI = (
    os.environ.get('ECGPI', '0') == '1'
    or (platform.system() == 'Linux' and platform.machine().startswith('aarch64'))
    or (platform.system() == 'Linux' and platform.machine().startswith('arm'))
)

# Page configuration
st.set_page_config(
    page_title="ECG Serial Debug",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------------------------
# Premium Dark Debug Theme CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

    :root {
        --bg-primary: #0a0e1a;
        --bg-secondary: #111827;
        --bg-card: rgba(17, 24, 39, 0.7);
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
        --terminal-bg: #0c1222;
        --terminal-border: rgba(6, 182, 212, 0.2);
        --terminal-green: #4ade80;
        --terminal-cyan: #22d3ee;
        --terminal-yellow: #facc15;
        --terminal-red: #fb7185;
    }

    .stApp {
        background: var(--bg-primary) !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
        color: var(--text-primary) !important;
    }

    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: var(--bg-primary); }
    ::-webkit-scrollbar-thumb { background: var(--accent-indigo); border-radius: 3px; }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f1629 0%, #111827 50%, #0f1629 100%) !important;
        border-right: 1px solid var(--border-color) !important;
    }
    header[data-testid="stHeader"] { background: transparent !important; }
    .block-container { padding-top: 2rem !important; max-width: 1400px !important; }

    [data-testid="stMetric"] {
        background: var(--bg-card) !important;
        backdrop-filter: blur(12px) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 16px !important;
        padding: 1rem 1.1rem !important;
        box-shadow: 0 0 30px rgba(99, 102, 241, 0.1) !important;
    }
    [data-testid="stMetricLabel"] {
        color: var(--text-secondary) !important;
        font-size: 0.78rem !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.06em !important;
    }
    [data-testid="stMetricValue"] {
        color: var(--text-primary) !important;
        font-weight: 700 !important;
    }

    .stButton > button {
        background: linear-gradient(135deg, #6366f1, #06b6d4) !important;
        color: #fff !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.6rem 1.5rem !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.3) !important;
    }
    .stButton > button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 25px rgba(99, 102, 241, 0.5) !important;
    }
    .stButton > button:disabled {
        background: linear-gradient(135deg, #374151, #4b5563) !important;
        color: #6b7280 !important;
        box-shadow: none !important;
    }

    .stTextInput input {
        background: var(--bg-secondary) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 10px !important;
        color: var(--text-primary) !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.85rem !important;
    }

    .stProgress > div > div { background: linear-gradient(135deg, #6366f1, #06b6d4) !important; border-radius: 8px !important; }
    .stProgress > div { background: rgba(99, 102, 241, 0.1) !important; border-radius: 8px !important; }

    .stMarkdown h1 { font-weight: 800 !important; letter-spacing: -0.03em !important; }
    .stMarkdown h2 { font-weight: 700 !important; color: var(--text-primary) !important; }
    .stMarkdown hr { border-color: var(--border-color) !important; }

    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    @keyframes blink {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.3; }
    }
    @keyframes terminalCursor {
        0%, 100% { border-right-color: #4ade80; }
        50% { border-right-color: transparent; }
    }

    .debug-header {
        text-align: center;
        padding: 1rem 0 0.5rem 0;
        animation: fadeInUp 0.6s ease-out;
    }
    .debug-header .icon { font-size: 2rem; }
    .debug-header .title {
        font-size: 2rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        background: linear-gradient(135deg, #06b6d4, #6366f1, #a855f7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .debug-subtitle {
        text-align: center;
        color: var(--text-muted);
        font-size: 0.85rem;
        margin-bottom: 1.2rem;
        letter-spacing: 0.04em;
    }

    .custom-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, var(--border-color), transparent);
        margin: 1.2rem 0;
        border: none;
    }

    .glass-panel {
        background: var(--bg-card);
        backdrop-filter: blur(16px);
        border: 1px solid var(--border-color);
        border-radius: 20px;
        padding: 1.2rem;
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.3);
        margin-bottom: 1rem;
        animation: fadeInUp 0.5s ease-out;
    }

    .section-header {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        margin-bottom: 0.8rem;
    }
    .section-header .section-icon { font-size: 1.2rem; }
    .section-header .section-title {
        font-size: 1.1rem;
        font-weight: 700;
        color: var(--text-primary);
    }

    .stat-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.5rem 0;
        border-bottom: 1px solid rgba(99, 102, 241, 0.08);
    }
    .stat-row:last-child { border-bottom: none; }
    .stat-label { color: var(--text-muted); font-size: 0.82rem; font-weight: 500; }
    .stat-value { color: var(--text-primary); font-size: 0.9rem; font-weight: 700; font-family: 'JetBrains Mono', monospace; }

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
    .status-connected {
        background: rgba(16, 185, 129, 0.15);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }
    .status-disconnected {
        background: rgba(244, 63, 94, 0.15);
        color: #fb7185;
        border: 1px solid rgba(244, 63, 94, 0.3);
    }
    .status-idle {
        background: rgba(100, 116, 139, 0.15);
        color: #94a3b8;
        border: 1px solid rgba(100, 116, 139, 0.3);
    }
    .dot-blink {
        width: 8px; height: 8px;
        border-radius: 50%;
        animation: blink 1s ease-in-out infinite;
    }

    /* Terminal window */
    .terminal-window {
        background: var(--terminal-bg);
        border: 1px solid var(--terminal-border);
        border-radius: 16px;
        overflow: hidden;
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.5), inset 0 1px 0 rgba(6, 182, 212, 0.05);
        animation: fadeInUp 0.5s ease-out;
    }
    .terminal-titlebar {
        background: linear-gradient(180deg, #1a2332 0%, #151d2e 100%);
        padding: 0.6rem 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
        border-bottom: 1px solid rgba(6, 182, 212, 0.1);
    }
    .terminal-dot {
        width: 12px; height: 12px;
        border-radius: 50%;
    }
    .terminal-dot-red { background: #f43f5e; }
    .terminal-dot-yellow { background: #f59e0b; }
    .terminal-dot-green { background: #10b981; }
    .terminal-title {
        color: #64748b;
        font-size: 0.78rem;
        font-weight: 500;
        font-family: 'JetBrains Mono', monospace;
        margin-left: 0.5rem;
        letter-spacing: 0.03em;
    }
    .terminal-body {
        padding: 1rem;
        max-height: 500px;
        overflow-y: auto;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.82rem;
        line-height: 1.7;
    }
    .terminal-line {
        display: flex;
        gap: 0.6rem;
        padding: 1px 0;
    }
    .terminal-line:hover {
        background: rgba(6, 182, 212, 0.04);
    }
    .terminal-lineno {
        color: #334155;
        min-width: 3.5ch;
        text-align: right;
        user-select: none;
        font-size: 0.75rem;
    }
    .terminal-content { flex: 1; }

    /* Line colour coding */
    .line-normal { color: #4ade80; }
    .line-leadsoff { color: #fb7185; }
    .line-comment { color: #64748b; font-style: italic; }
    .line-error { color: #f43f5e; }
    .line-info { color: #22d3ee; }

    /* Parsed value highlights */
    .val-timestamp { color: #64748b; }
    .val-ecg { color: #4ade80; font-weight: 600; }
    .val-lo-ok { color: #34d399; }
    .val-lo-off { color: #fb7185; font-weight: 600; }

    /* Port card */
    .port-card {
        background: rgba(17, 24, 39, 0.6);
        border: 1px solid rgba(99, 102, 241, 0.15);
        border-radius: 12px;
        padding: 0.7rem 1rem;
        margin-bottom: 0.4rem;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.82rem;
        transition: all 0.2s ease;
    }
    .port-card:hover {
        border-color: rgba(99, 102, 241, 0.4);
        background: rgba(17, 24, 39, 0.8);
    }
    .port-name { color: #06b6d4; font-weight: 600; }
    .port-desc { color: #64748b; font-size: 0.75rem; }

    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── Session State ─────────────────────────────────────────────────────────────

if 'debug_serial' not in st.session_state:
    st.session_state.debug_serial = None
if 'debug_running' not in st.session_state:
    st.session_state.debug_running = False
if 'debug_lines' not in st.session_state:
    st.session_state.debug_lines = collections.deque(maxlen=500)
if 'debug_parsed' not in st.session_state:
    st.session_state.debug_parsed = collections.deque(maxlen=200)
if 'debug_ecg_buffer' not in st.session_state:
    st.session_state.debug_ecg_buffer = collections.deque(maxlen=2000)
if 'debug_time_buffer' not in st.session_state:
    st.session_state.debug_time_buffer = collections.deque(maxlen=2000)
if 'debug_stats' not in st.session_state:
    st.session_state.debug_stats = {
        'total_lines': 0,
        'valid_samples': 0,
        'leads_off_count': 0,
        'errors': 0,
        'start_time': 0,
        'last_ecg': 0,
        'last_lo_plus': 0,
        'last_lo_minus': 0,
        'last_timestamp': 0,
        'min_ecg': 9999,
        'max_ecg': 0,
    }
if 'debug_auto_scroll' not in st.session_state:
    st.session_state.debug_auto_scroll = True
if 'debug_show_raw' not in st.session_state:
    st.session_state.debug_show_raw = True


# ── Helper Functions ──────────────────────────────────────────────────────────

def list_ports():
    """List all available serial ports with descriptions."""
    return [(p.device, p.description, p.manufacturer or '') for p in serial.tools.list_ports.comports()]


def connect_debug(port, baudrate=115200):
    """Connect to the serial port for debugging."""
    try:
        if st.session_state.debug_serial and st.session_state.debug_serial.is_open:
            st.session_state.debug_serial.close()

        conn = serial.Serial(port=port, baudrate=baudrate, timeout=0.5)
        time.sleep(2)  # Wait for Arduino reset
        conn.reset_input_buffer()

        st.session_state.debug_serial = conn
        st.session_state.debug_running = True
        st.session_state.debug_lines.clear()
        st.session_state.debug_parsed.clear()
        st.session_state.debug_ecg_buffer.clear()
        st.session_state.debug_time_buffer.clear()
        st.session_state.debug_stats = {
            'total_lines': 0,
            'valid_samples': 0,
            'leads_off_count': 0,
            'errors': 0,
            'start_time': time.time(),
            'last_ecg': 0,
            'last_lo_plus': 0,
            'last_lo_minus': 0,
            'last_timestamp': 0,
            'min_ecg': 9999,
            'max_ecg': 0,
        }
        return True, f"Connected to {port}"
    except Exception as e:
        return False, str(e)


def disconnect_debug():
    """Disconnect from serial port."""
    st.session_state.debug_running = False
    if st.session_state.debug_serial and st.session_state.debug_serial.is_open:
        st.session_state.debug_serial.close()
    st.session_state.debug_serial = None


def read_serial_data():
    """Read available data from serial port and parse it."""
    conn = st.session_state.debug_serial
    if conn is None or not conn.is_open:
        return

    try:
        lines_read = 0
        max_lines_per_update = 100

        while conn.in_waiting > 0 and lines_read < max_lines_per_update:
            raw_line = conn.readline().decode('utf-8', errors='replace').strip()
            if not raw_line:
                continue

            lines_read += 1
            st.session_state.debug_stats['total_lines'] += 1
            elapsed = time.time() - st.session_state.debug_stats['start_time']

            # Store raw line with metadata
            line_entry = {
                'raw': raw_line,
                'time': elapsed,
                'type': 'unknown',
                'parsed': None,
            }

            if raw_line.startswith('#'):
                line_entry['type'] = 'comment'
            else:
                parts = raw_line.split(',')
                if len(parts) == 4:
                    try:
                        ts = int(parts[0])
                        ecg = int(parts[1])
                        lo_plus = int(parts[2])
                        lo_minus = int(parts[3])

                        leads_off = (lo_plus == 1 or lo_minus == 1)
                        line_entry['type'] = 'leadsoff' if leads_off else 'normal'
                        line_entry['parsed'] = {
                            'timestamp': ts,
                            'ecg': ecg,
                            'lo_plus': lo_plus,
                            'lo_minus': lo_minus,
                            'leads_off': leads_off,
                        }

                        # Update stats
                        stats = st.session_state.debug_stats
                        stats['valid_samples'] += 1
                        stats['last_ecg'] = ecg
                        stats['last_lo_plus'] = lo_plus
                        stats['last_lo_minus'] = lo_minus
                        stats['last_timestamp'] = ts
                        if leads_off:
                            stats['leads_off_count'] += 1
                        if ecg < stats['min_ecg']:
                            stats['min_ecg'] = ecg
                        if ecg > stats['max_ecg']:
                            stats['max_ecg'] = ecg

                        # Buffer for mini plot
                        st.session_state.debug_ecg_buffer.append(ecg if not leads_off else 0)
                        st.session_state.debug_time_buffer.append(elapsed)

                        # Store parsed
                        st.session_state.debug_parsed.append(line_entry['parsed'])

                    except ValueError:
                        line_entry['type'] = 'error'
                        st.session_state.debug_stats['errors'] += 1
                else:
                    line_entry['type'] = 'error'
                    st.session_state.debug_stats['errors'] += 1

            st.session_state.debug_lines.append(line_entry)

    except serial.SerialException as e:
        st.session_state.debug_lines.append({
            'raw': f'[SERIAL ERROR] {e}',
            'time': time.time() - st.session_state.debug_stats['start_time'],
            'type': 'error',
            'parsed': None,
        })
        st.session_state.debug_stats['errors'] += 1
    except Exception as e:
        st.session_state.debug_lines.append({
            'raw': f'[ERROR] {e}',
            'time': time.time() - st.session_state.debug_stats['start_time'],
            'type': 'error',
            'parsed': None,
        })


def render_terminal(lines, show_raw=True, max_display=150):
    """Render terminal output as styled HTML."""
    display_lines = list(lines)[-max_display:]

    if not display_lines:
        return """
            <div class="terminal-window">
                <div class="terminal-titlebar">
                    <div class="terminal-dot terminal-dot-red"></div>
                    <div class="terminal-dot terminal-dot-yellow"></div>
                    <div class="terminal-dot terminal-dot-green"></div>
                    <span class="terminal-title">serial-monitor — /dev/ttyUSB0</span>
                </div>
                <div class="terminal-body" style="text-align:center; padding:3rem; color:#334155;">
                    Waiting for serial data...
                </div>
            </div>
        """

    port_name = ''
    if st.session_state.debug_serial:
        port_name = st.session_state.debug_serial.port or ''

    body_lines = []
    for i, entry in enumerate(display_lines):
        lineno = len(lines) - len(display_lines) + i + 1
        raw = entry['raw']
        t = entry['time']
        line_type = entry['type']

        if show_raw:
            # Show raw line with colour coding
            if line_type == 'comment':
                css = 'line-comment'
            elif line_type == 'leadsoff':
                css = 'line-leadsoff'
            elif line_type == 'error':
                css = 'line-error'
            elif line_type == 'normal':
                css = 'line-normal'
            else:
                css = 'line-info'

            # Escape HTML
            display_text = raw.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')

            body_lines.append(
                f'<div class="terminal-line">'
                f'<span class="terminal-lineno">{lineno}</span>'
                f'<span class="terminal-content {css}">'
                f'<span class="val-timestamp">[{t:7.2f}s]</span> {display_text}'
                f'</span></div>'
            )
        else:
            # Show parsed view
            p = entry.get('parsed')
            if line_type == 'comment':
                display_text = raw.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                body_lines.append(
                    f'<div class="terminal-line">'
                    f'<span class="terminal-lineno">{lineno}</span>'
                    f'<span class="terminal-content line-comment">{display_text}</span>'
                    f'</div>'
                )
            elif p:
                ecg_css = 'val-ecg'
                lo_plus_css = 'val-lo-ok' if p['lo_plus'] == 0 else 'val-lo-off'
                lo_minus_css = 'val-lo-ok' if p['lo_minus'] == 0 else 'val-lo-off'
                leads_status = '✅' if not p['leads_off'] else '❌'

                body_lines.append(
                    f'<div class="terminal-line">'
                    f'<span class="terminal-lineno">{lineno}</span>'
                    f'<span class="terminal-content">'
                    f'<span class="val-timestamp">[{t:7.2f}s]</span> '
                    f'ts=<span class="val-timestamp">{p["timestamp"]}</span> '
                    f'ecg=<span class="{ecg_css}">{p["ecg"]:4d}</span> '
                    f'LO+=<span class="{lo_plus_css}">{p["lo_plus"]}</span> '
                    f'LO-=<span class="{lo_minus_css}">{p["lo_minus"]}</span> '
                    f'{leads_status}'
                    f'</span></div>'
                )
            elif line_type == 'error':
                display_text = raw.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                body_lines.append(
                    f'<div class="terminal-line">'
                    f'<span class="terminal-lineno">{lineno}</span>'
                    f'<span class="terminal-content line-error">{display_text}</span>'
                    f'</div>'
                )

    body_html = '\n'.join(body_lines)

    return f"""
        <div class="terminal-window">
            <div class="terminal-titlebar">
                <div class="terminal-dot terminal-dot-red"></div>
                <div class="terminal-dot terminal-dot-yellow"></div>
                <div class="terminal-dot terminal-dot-green"></div>
                <span class="terminal-title">serial-monitor — {port_name or 'not connected'}</span>
            </div>
            <div class="terminal-body" id="terminal-scroll">{body_html}</div>
        </div>
    """


def plot_debug_ecg():
    """Mini ECG plot for debug view."""
    fig = go.Figure()

    if len(st.session_state.debug_ecg_buffer) > 0:
        ecg = list(st.session_state.debug_ecg_buffer)
        t = list(st.session_state.debug_time_buffer)

        fig.add_trace(go.Scatter(
            x=t, y=ecg,
            mode='lines',
            line=dict(color='#06b6d4', width=1.5),
            fill='tozeroy',
            fillcolor='rgba(6, 182, 212, 0.04)',
            hovertemplate='<b>%{x:.2f}s</b>: %{y}<extra></extra>',
        ))

    fig.update_layout(
        height=200,
        margin=dict(l=40, r=10, t=25, b=30),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(10, 14, 26, 0.9)',
        font=dict(family='JetBrains Mono', color='#64748b', size=10),
        title=dict(text='<b>Live Signal Preview</b>', font=dict(size=12, color='#94a3b8'), x=0.02),
        xaxis=dict(
            gridcolor='rgba(99,102,241,0.05)',
            zerolinecolor='rgba(99,102,241,0.1)',
            tickfont=dict(size=9),
        ),
        yaxis=dict(
            gridcolor='rgba(99,102,241,0.05)',
            zerolinecolor='rgba(244,63,94,0.15)',
            tickfont=dict(size=9),
        ),
        showlegend=False,
        hovermode='x unified',
        hoverlabel=dict(bgcolor='#1e293b', bordercolor='rgba(99,102,241,0.3)', font=dict(color='#e2e8f0', family='JetBrains Mono', size=11)),
    )
    return fig


# ── Main Application ─────────────────────────────────────────────────────────

def main():
    # ── Header ────────────────────────────────────────────────────────────
    st.markdown("""
        <div class="debug-header">
            <span class="icon">🔧</span>
            <span class="title">Serial Debug Monitor</span>
        </div>
        <div class="debug-subtitle">Raw Arduino / AD8232 serial output viewer for ECG connection debugging</div>
    """, unsafe_allow_html=True)
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    # ── Sidebar ───────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("""
            <div style="text-align:center; padding:0.5rem 0 0.6rem 0;">
                <span style="font-size:1.5rem;">🛠️</span>
                <div style="font-size:1rem; font-weight:700; color:#e2e8f0; margin-top:0.2rem;">Debug Panel</div>
                <div style="font-size:0.7rem; color:#64748b; letter-spacing:0.08em; text-transform:uppercase;">Serial Configuration</div>
            </div>
        """, unsafe_allow_html=True)
        st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

        # Available ports
        st.markdown('<div class="section-header"><span class="section-icon">🔌</span><span class="section-title">Ports</span></div>', unsafe_allow_html=True)

        ports = list_ports()
        if ports:
            ports_html = ""
            for device, desc, mfg in ports:
                mfg_str = f" ({mfg})" if mfg else ""
                ports_html += f'<div class="port-card"><span class="port-name">{device}</span><br><span class="port-desc">{desc}{mfg_str}</span></div>'
            st.markdown(ports_html, unsafe_allow_html=True)
        else:
            st.markdown('<div class="port-card"><span class="port-desc">No serial ports detected</span></div>', unsafe_allow_html=True)

        st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

        # Connection settings
        st.markdown('<div class="section-header"><span class="section-icon">⚙️</span><span class="section-title">Connection</span></div>', unsafe_allow_html=True)

        default_port = ''
        if ports:
            default_port = ports[0][0]
        if IS_PI and not default_port:
            default_port = '/dev/ttyUSB0'

        serial_port = st.text_input("Serial Port", value=default_port)
        baudrate = st.selectbox("Baud Rate", [9600, 19200, 38400, 57600, 115200], index=4)

        col1, col2 = st.columns(2)
        with col1:
            if st.button("▶ Connect", disabled=st.session_state.debug_running, use_container_width=True):
                if serial_port.strip():
                    ok, msg = connect_debug(serial_port.strip(), baudrate)
                    if ok:
                        st.success(msg)
                        st.rerun()
                    else:
                        st.error(msg)
                else:
                    st.error("Enter a port name")
        with col2:
            if st.button("⏹ Disconnect", disabled=not st.session_state.debug_running, use_container_width=True):
                disconnect_debug()
                st.rerun()

        st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

        # Display settings
        st.markdown('<div class="section-header"><span class="section-icon">🖥️</span><span class="section-title">Display</span></div>', unsafe_allow_html=True)

        view_mode = st.radio("View Mode", ["Raw Output", "Parsed Values"], horizontal=True, label_visibility='collapsed')
        st.session_state.debug_show_raw = (view_mode == "Raw Output")

        refresh_ms = st.slider("Refresh (ms)", 100, 1000, 200, step=50)
        max_lines = st.slider("Max Lines", 50, 500, 150, step=50)

        st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

        # Connection status
        st.markdown('<div class="section-header"><span class="section-icon">📡</span><span class="section-title">Status</span></div>', unsafe_allow_html=True)
        if st.session_state.debug_running:
            st.markdown("""
                <div style="text-align:center;">
                    <span class="status-badge status-connected">
                        <div class="dot-blink" style="background:#34d399;"></div> CONNECTED
                    </span>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
                <div style="text-align:center;">
                    <span class="status-badge status-idle">● IDLE</span>
                </div>
            """, unsafe_allow_html=True)

        if st.session_state.debug_running:
            if st.button("🗑️ Clear Terminal", use_container_width=True):
                st.session_state.debug_lines.clear()
                st.session_state.debug_parsed.clear()
                st.session_state.debug_ecg_buffer.clear()
                st.session_state.debug_time_buffer.clear()
                st.rerun()

    # ── Main Content ──────────────────────────────────────────────────────
    if st.session_state.debug_running:
        # Read new data
        read_serial_data()
        stats = st.session_state.debug_stats

        # ── Live Metrics Bar ──────────────────────────────────────────────
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        with c1:
            elapsed = time.time() - stats['start_time']
            st.metric("Uptime", f"{elapsed:.0f}s")
        with c2:
            st.metric("Samples", f"{stats['valid_samples']:,}")
        with c3:
            rate = stats['valid_samples'] / max(0.1, elapsed)
            st.metric("Rate", f"{rate:.0f} Hz")
        with c4:
            st.metric("Last ECG", f"{stats['last_ecg']}")
        with c5:
            lo_status = "✅ ON" if (stats['last_lo_plus'] == 0 and stats['last_lo_minus'] == 0) else "❌ OFF"
            st.metric("Leads", lo_status)
        with c6:
            st.metric("Errors", f"{stats['errors']}")

        st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

        # ── Terminal Window ───────────────────────────────────────────────
        terminal_html = render_terminal(
            st.session_state.debug_lines,
            show_raw=st.session_state.debug_show_raw,
            max_display=max_lines,
        )
        st.markdown(terminal_html, unsafe_allow_html=True)

        st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

        # ── Bottom row: Mini ECG + Stats ──────────────────────────────────
        col_plot, col_stats = st.columns([1.5, 1])

        with col_plot:
            st.plotly_chart(plot_debug_ecg(), use_container_width=True, key='debug_ecg_mini')

        with col_stats:
            ecg_range = stats['max_ecg'] - stats['min_ecg'] if stats['max_ecg'] > stats['min_ecg'] else 0
            lo_pct = (stats['leads_off_count'] / max(1, stats['valid_samples'])) * 100

            st.markdown(f"""
                <div class="glass-panel" style="padding:0.9rem 1rem;">
                    <div style="font-weight:700; color:#e2e8f0; margin-bottom:0.5rem; font-size:0.9rem;">📊 Signal Statistics</div>
                    <div class="stat-row">
                        <span class="stat-label">ECG Range</span>
                        <span class="stat-value">{stats['min_ecg']} – {stats['max_ecg']}</span>
                    </div>
                    <div class="stat-row">
                        <span class="stat-label">ADC Span</span>
                        <span class="stat-value">{ecg_range}</span>
                    </div>
                    <div class="stat-row">
                        <span class="stat-label">Leads-off %</span>
                        <span class="stat-value" style="color:{'#fb7185' if lo_pct > 10 else '#34d399'};">{lo_pct:.1f}%</span>
                    </div>
                    <div class="stat-row">
                        <span class="stat-label">Last Timestamp</span>
                        <span class="stat-value">{stats['last_timestamp']}</span>
                    </div>
                    <div class="stat-row">
                        <span class="stat-label">Total Lines</span>
                        <span class="stat-value">{stats['total_lines']}</span>
                    </div>
                    <div class="stat-row">
                        <span class="stat-label">LO+ Pin</span>
                        <span class="stat-value" style="color:{'#34d399' if stats['last_lo_plus']==0 else '#fb7185'};">{stats['last_lo_plus']}</span>
                    </div>
                    <div class="stat-row">
                        <span class="stat-label">LO− Pin</span>
                        <span class="stat-value" style="color:{'#34d399' if stats['last_lo_minus']==0 else '#fb7185'};">{stats['last_lo_minus']}</span>
                    </div>
                </div>
            """, unsafe_allow_html=True)

        # Auto-refresh
        time.sleep(refresh_ms / 1000.0)
        st.rerun()

    else:
        # ── Welcome / Not Connected ───────────────────────────────────────
        st.markdown("""
            <div class="glass-panel" style="text-align:center; padding:2.5rem;">
                <div style="font-size:3rem; margin-bottom:0.5rem;">🔌</div>
                <div style="font-size:1.3rem; font-weight:800; color:#e2e8f0; margin-bottom:0.3rem;">
                    Connect to Serial Port
                </div>
                <div style="font-size:0.85rem; color:#64748b; max-width:450px; margin:0 auto; line-height:1.6;">
                    Select a serial port in the sidebar and click <b style="color:#06b6d4;">▶ Connect</b> to begin monitoring
                    raw Arduino output from the AD8232 ECG sensor.
                </div>
            </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

        # Expected format info
        st.markdown('<div class="section-header"><span class="section-icon">📋</span><span class="section-title">Expected Arduino Output Format</span></div>', unsafe_allow_html=True)

        st.markdown("""
            <div class="terminal-window">
                <div class="terminal-titlebar">
                    <div class="terminal-dot terminal-dot-red"></div>
                    <div class="terminal-dot terminal-dot-yellow"></div>
                    <div class="terminal-dot terminal-dot-green"></div>
                    <span class="terminal-title">expected-output-format</span>
                </div>
                <div class="terminal-body">
                    <div class="terminal-line">
                        <span class="terminal-lineno">1</span>
                        <span class="terminal-content line-comment"># ECG Acquisition System Ready</span>
                    </div>
                    <div class="terminal-line">
                        <span class="terminal-lineno">2</span>
                        <span class="terminal-content line-comment"># Format: timestamp,ecg_value,lo_plus,lo_minus</span>
                    </div>
                    <div class="terminal-line">
                        <span class="terminal-lineno">3</span>
                        <span class="terminal-content line-normal">0,512,0,0</span>
                    </div>
                    <div class="terminal-line">
                        <span class="terminal-lineno">4</span>
                        <span class="terminal-content line-normal">1,523,0,0</span>
                    </div>
                    <div class="terminal-line">
                        <span class="terminal-lineno">5</span>
                        <span class="terminal-content line-normal">2,498,0,0</span>
                    </div>
                    <div class="terminal-line">
                        <span class="terminal-lineno">6</span>
                        <span class="terminal-content line-leadsoff">3,0,1,1</span>
                        <span style="color:#64748b; font-size:0.75rem; margin-left:0.5rem;">← leads off</span>
                    </div>
                    <div class="terminal-line">
                        <span class="terminal-lineno">7</span>
                        <span class="terminal-content line-normal">4,510,0,0</span>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

        # Pin configuration reference
        col_l, col_r = st.columns(2)
        with col_l:
            st.markdown("""
                <div class="glass-panel">
                    <div style="font-weight:700; color:#e2e8f0; margin-bottom:0.5rem; font-size:0.9rem;">🔧 Pin Configuration</div>
                    <div class="stat-row"><span class="stat-label">AD8232 OUTPUT</span><span class="stat-value">→ A0</span></div>
                    <div class="stat-row"><span class="stat-label">AD8232 LO+</span><span class="stat-value">→ Pin 10</span></div>
                    <div class="stat-row"><span class="stat-label">AD8232 LO−</span><span class="stat-value">→ Pin 11</span></div>
                    <div class="stat-row"><span class="stat-label">AD8232 GND</span><span class="stat-value">→ GND</span></div>
                    <div class="stat-row"><span class="stat-label">AD8232 3.3V</span><span class="stat-value">→ 3.3V</span></div>
                </div>
            """, unsafe_allow_html=True)

        with col_r:
            st.markdown("""
                <div class="glass-panel">
                    <div style="font-weight:700; color:#e2e8f0; margin-bottom:0.5rem; font-size:0.9rem;">📡 Serial Settings</div>
                    <div class="stat-row"><span class="stat-label">Baud Rate</span><span class="stat-value">115200</span></div>
                    <div class="stat-row"><span class="stat-label">Sampling Rate</span><span class="stat-value">360 Hz</span></div>
                    <div class="stat-row"><span class="stat-label">Data Format</span><span class="stat-value">CSV</span></div>
                    <div class="stat-row"><span class="stat-label">Fields</span><span class="stat-value">ts, ecg, LO+, LO−</span></div>
                    <div class="stat-row"><span class="stat-label">ADC Range</span><span class="stat-value">0 – 1023</span></div>
                </div>
            """, unsafe_allow_html=True)

        # Troubleshooting
        st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
        st.markdown('<div class="section-header"><span class="section-icon">🩺</span><span class="section-title">Debug Checklist</span></div>', unsafe_allow_html=True)

        st.markdown("""
            <div class="glass-panel">
                <div class="stat-row"><span class="stat-label">1</span><span class="stat-value" style="font-family:Inter; font-size:0.85rem;">Is the Arduino powered? (check LED on board)</span></div>
                <div class="stat-row"><span class="stat-label">2</span><span class="stat-value" style="font-family:Inter; font-size:0.85rem;">Is USB cable data-capable? (not charge-only)</span></div>
                <div class="stat-row"><span class="stat-label">3</span><span class="stat-value" style="font-family:Inter; font-size:0.85rem;">Is the serial port visible in the Ports list?</span></div>
                <div class="stat-row"><span class="stat-label">4</span><span class="stat-value" style="font-family:Inter; font-size:0.85rem;">Is the baud rate set to 115200?</span></div>
                <div class="stat-row"><span class="stat-label">5</span><span class="stat-value" style="font-family:Inter; font-size:0.85rem;">Is the 3.5mm jack fully inserted into AD8232?</span></div>
                <div class="stat-row"><span class="stat-label">6</span><span class="stat-value" style="font-family:Inter; font-size:0.85rem;">Are LO+ and LO- showing 0 (leads connected)?</span></div>
                <div class="stat-row"><span class="stat-label">7</span><span class="stat-value" style="font-family:Inter; font-size:0.85rem;">On Pi: did you run <code style="color:#06b6d4;">sudo usermod -aG dialout $USER</code>?</span></div>
            </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
