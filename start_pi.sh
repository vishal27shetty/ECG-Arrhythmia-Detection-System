#!/usr/bin/env bash
# -----------------------------------------------------------
# ECG Arrhythmia Detection — Raspberry Pi startup script
#
# Usage:
#   chmod +x start_pi.sh
#   ./start_pi.sh              # auto-detect serial port
#   ./start_pi.sh /dev/ttyACM0 # specify port
# -----------------------------------------------------------

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Activate venv if present
if [ -d "venv/bin" ]; then
    source venv/bin/activate
fi

# Grant serial access (Pi often needs this)
SERIAL_PORT="${1:-/dev/ttyUSB0}"
if [ -e "$SERIAL_PORT" ]; then
    sudo chmod 666 "$SERIAL_PORT" 2>/dev/null || true
fi

export ECGPI=1

echo "============================================"
echo "  ECG Arrhythmia Monitor — Raspberry Pi"
echo "============================================"
echo "  Serial port : $SERIAL_PORT"
echo "  Model       : models/best_model.tflite"
echo "  Dashboard   : http://$(hostname -I | awk '{print $1}'):8501"
echo "============================================"

exec streamlit run dashboard/app.py \
    --server.port 8501 \
    --server.headless true \
    --browser.gatherUsageStats false
