# Raspberry Pi Deployment Guide

Step-by-step guide to run the ECG Arrhythmia Detection System on a Raspberry Pi.

---

## Prerequisites

### Hardware You Need

| Item | Details |
|------|---------|
| Raspberry Pi | **Pi 4 (2 GB+ RAM)** or **Pi 5** recommended |
| microSD Card | 16 GB+ (Class 10 / A1 or faster) |
| Power Supply | USB-C — 5V/3A for Pi 4, 5V/5A for Pi 5 |
| Arduino Uno | With USB-B cable |
| AD8232 ECG Module | With 3-lead electrode cable and disposable electrodes |
| Network | Wi-Fi or Ethernet (to access dashboard from phone/laptop) |
| Desktop/Laptop | One-time use to train the model and convert it |
| Heatsink / Fan (optional) | Recommended for long recording sessions |

### Which Raspberry Pi OS to Download

Go to https://www.raspberrypi.com/software/operating-systems/

Download **Raspberry Pi OS Lite (64-bit)** — this is all you need.

| Field | Value |
|-------|-------|
| Version | **Raspberry Pi OS Lite (64-bit)** |
| Based on | Debian Trixie (13) |
| Release date | 21 Apr 2026 |
| Kernel | 6.12 |
| Download size | ~551 MB |
| Storage on card | ~3.1 GB |
| Compatible boards | Pi 3B, 3B+, 3A+, **4B**, 400, **5**, 500, CM4, Zero 2 W |

**Why Lite (no desktop)?** The ECG dashboard is a web app — you open it in a
browser on your phone or laptop at `http://<PI_IP>:8501`. The Pi itself does
not need a monitor, keyboard, or desktop environment. Lite saves ~3 GB of
storage and keeps RAM free for processing.

**Why 64-bit?** The `tflite-runtime` package and NumPy/SciPy run faster on
64-bit ARM (aarch64). 32-bit will work but is slower and may have dependency
issues.

> If you want to plug a monitor into the Pi and use its own browser, download
> **Raspberry Pi OS with Desktop (64-bit)** instead (1,285 MB).

### Software Versions

- Raspberry Pi OS Trixie 64-bit (April 2026)
- Python 3.13+ (ships with Trixie)
- Streamlit 1.28+
- tflite-runtime

---

## Step 1 — Train the Model (on your Desktop)

Skip this if you already have `models/best_model.h5`.

```bash
# On your desktop/laptop (needs TensorFlow)
python models/dataset_preparation.py   # download MIT-BIH data
python models/train_bilstm.py          # train the model
```

This produces `models/best_model.h5`.

---

## Step 2 — Convert Model to TFLite (on your Desktop)

The Pi cannot run full TensorFlow. Convert the model to the lightweight
TFLite format:

```bash
python convert_model_to_tflite.py
```

This reads `models/best_model.h5` and creates `models/best_model.tflite`.

Verify the file was created:

```bash
ls -lh models/best_model.tflite
```

---

## Step 3 — Prepare the Raspberry Pi

### 3.1 Flash the OS

1. Download and install [Raspberry Pi Imager](https://www.raspberrypi.com/software/)
2. Open Imager and click **Choose OS**
3. Select **Raspberry Pi OS (other)** → **Raspberry Pi OS Lite (64-bit)**
4. Click the **gear icon** (⚙️) before writing — configure these settings:
   - **Enable SSH** — check "Use password authentication"
   - **Set username and password** — e.g. `pi` / your-password
   - **Configure Wi-Fi** — enter your SSID and password
   - **Set locale** — your timezone and keyboard layout
5. Click **Write** and wait for it to finish
6. Insert the microSD card into the Pi and power it on
7. Wait ~60 seconds for first boot to complete

### 3.2 Find the Pi's IP address

```bash
# From the Pi terminal, or check your router admin page
hostname -I
```

Note this IP — you will use it to SSH in and to open the dashboard.

### 3.3 Update the system

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3-venv python3-pip git libatlas-base-dev
```

`libatlas-base-dev` is required by NumPy/SciPy on ARM.

---

## Step 4 — Copy the Project to the Pi

From your desktop, copy the entire project folder:

```bash
scp -r "FInal Year Project" pi@<PI_IP>:~/ecg-project
```

Or clone from GitHub if you pushed the `pi` branch:

```bash
ssh pi@<PI_IP>
git clone -b pi https://github.com/<your-username>/ECG-Arrhythmia-Detection-System.git ~/ecg-project
```

---

## Step 5 — Install Python Dependencies

```bash
ssh pi@<PI_IP>
cd ~/ecg-project

# Create a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install Pi-specific (lightweight) requirements
pip install -r requirements_pi.txt
```

This installs only what the Pi needs:

| Package | Purpose |
|---------|---------|
| numpy | Array math |
| scipy | Signal filtering |
| pyserial | Arduino serial communication |
| streamlit | Web dashboard |
| plotly | ECG waveform charts |
| pandas | Data handling |
| tflite-runtime | Lightweight model inference |

No TensorFlow, no training libraries — keeps the install small and fast.

---

## Step 6 — Wire the Hardware

### Wiring Diagram

```
┌──────────────┐   USB Cable   ┌──────────────┐
│ Raspberry Pi │──────────────►│ Arduino Uno  │
└──────────────┘               └──────┬───────┘
                                      │
                               ┌──────┴───────┐
                               │   AD8232     │
                               │  ECG Module  │
                               └──────┬───────┘
                                      │
                               3-lead electrodes
                               on patient chest
```

### Pin Connections (Arduino ↔ AD8232)

| AD8232 Pin | Arduino Pin | Notes |
|------------|-------------|-------|
| GND | GND | Common ground |
| 3.3V | 3.3V | **Do NOT use 5V** |
| OUTPUT | A0 | Analog ECG signal |
| LO+ | D10 | Leads-off detection |
| LO- | D11 | Leads-off detection |

### Electrode Placement (Standard Lead II)

| Electrode | Position |
|-----------|----------|
| RA (Right Arm) | Below right clavicle |
| LA (Left Arm) | Below left clavicle |
| RL (Right Leg) | Lower right abdomen (reference) |

---

## Step 7 — Upload Arduino Sketch

1. On any computer, open Arduino IDE
2. Open `arduino/ecg_acquisition.ino`
3. Connect the Arduino via USB
4. Select **Tools → Board → Arduino Uno**
5. Select the correct **Port**
6. Click **Upload**
7. Disconnect from the computer and plug into the Pi's USB port

---

## Step 8 — Run the Dashboard

### Option A: Use the startup script (recommended)

```bash
cd ~/ecg-project
chmod +x start_pi.sh
./start_pi.sh
```

The script will:
- Activate the virtual environment
- Set serial port permissions
- Set the `ECGPI=1` environment variable
- Start Streamlit in headless mode

### Option B: Run manually

```bash
cd ~/ecg-project
source venv/bin/activate
export ECGPI=1
streamlit run dashboard/app.py --server.headless true --server.port 8501
```

### Option C: Specify a serial port

```bash
./start_pi.sh /dev/ttyACM0
```

---

## Step 9 — Open the Dashboard

On any phone, tablet, or laptop on the same network, open:

```
http://<PI_IP>:8501
```

For example: `http://192.168.1.42:8501`

You should see the ECG Arrhythmia Monitor dashboard with a
"Running in Raspberry Pi mode" indicator.

---

## Step 10 — Record and Analyse

1. **Set duration** in the sidebar (30–60 seconds recommended)
2. Click **Start** — live ECG waveform appears
3. Wait for the timer to finish (or click **Stop** early)
4. View the **full analysis** — beat classification, heart rate, confidence
5. Results are saved to `logs/ecg_batch_<timestamp>.json`

---

## Troubleshooting

### Serial port not found

```bash
# List connected USB devices
ls /dev/ttyUSB* /dev/ttyACM* 2>/dev/null

# If nothing shows, check dmesg
dmesg | tail -20

# Grant your user serial access (permanent fix, requires reboot)
sudo usermod -aG dialout $USER
sudo reboot
```

### Permission denied on serial port

```bash
sudo chmod 666 /dev/ttyUSB0
# or use the start_pi.sh script which does this automatically
```

### tflite-runtime won't install

```bash
# Try the official pip index for your Pi/Python version
pip install --extra-index-url https://google-coral.github.io/py-repo/ tflite-runtime
```

### Dashboard is slow / laggy

- Increase the refresh rate slider to 300–500 ms in the sidebar
- Close other programs on the Pi
- Use a Pi 4 with 4 GB RAM or a Pi 5
- Attach a heatsink or fan — thermal throttling slows everything down

### No ECG signal / flat waveform

- Verify the Arduino power LED is on
- Check AD8232 is powered from **3.3V** (not 5V)
- Replace electrodes if they are dry or old
- Clean skin with alcohol before attaching electrodes
- Make sure cables are not moving during recording

### "Model not found" error

- Verify the file exists: `ls models/best_model.tflite`
- If missing, re-run `python convert_model_to_tflite.py` on your desktop
  and copy the file over:

```bash
scp models/best_model.tflite pi@<PI_IP>:~/ecg-project/models/
```

---

## Optional: Auto-start on Boot

To start the dashboard automatically when the Pi boots:

```bash
# Create a systemd service
sudo tee /etc/systemd/system/ecg-monitor.service << 'EOF'
[Unit]
Description=ECG Arrhythmia Monitor
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/ecg-project
ExecStart=/home/pi/ecg-project/start_pi.sh
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

# Enable and start
sudo systemctl enable ecg-monitor.service
sudo systemctl start ecg-monitor.service

# Check status
sudo systemctl status ecg-monitor.service
```

Now the dashboard will start automatically every time the Pi powers on.

---

## Quick Reference

| Action | Command |
|--------|---------|
| Start dashboard | `./start_pi.sh` |
| Start with specific port | `./start_pi.sh /dev/ttyACM0` |
| Check serial devices | `ls /dev/ttyUSB* /dev/ttyACM*` |
| View logs | `ls logs/` |
| Stop dashboard | `Ctrl+C` in terminal |
| Check Pi IP | `hostname -I` |
| Dashboard URL | `http://<PI_IP>:8501` |
