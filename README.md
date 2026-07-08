# ECG Arrhythmia Detection System

[![GitHub](https://img.shields.io/badge/GitHub-Repository-blue)](https://github.com/vishal27shetty/ECG-Arrhythmia-Detection-System)

Real-time ECG monitoring system with AI-powered arrhythmia classification using Bidirectional LSTM neural networks.

## 🎯 Project Overview

This system provides:
- **Real-time ECG acquisition** from AD8232 sensor via Arduino Uno
- **Edge processing** with digital signal filtering
- **AI classification** of 5 arrhythmia types using CNN-LSTM model
- **Batch processing** dashboard with live waveform display
- **High accuracy** trained on MIT-BIH Arrhythmia Database (70.7% test accuracy)

## 📊 Dashboard Approach: Live Display + Batch Processing

### Why Batch Processing?
Instead of real-time classification during recording, we use a **Live Display + Batch Processing** approach:

**During Recording (Live Display):**
- Shows live ECG waveform only
- No classification overhead
- Fixed duration sessions (30-300 seconds)
- Auto-stop when duration reached

**After Recording (Batch Processing):**
- Process entire signal at once
- More accurate R-peak detection (full signal context)
- No duplicate detections
- Beat-by-beat classification
- Comprehensive analysis & statistics
- Saved to JSON logs

**Benefits:**
- ✅ **More Accurate**: Full signal context for peak detection
- ✅ **No Duplicates**: Single-pass processing
- ✅ **Consistent Results**: Same input = same output
- ✅ **Faster UI**: No real-time inference overhead
- ✅ **Better Debugging**: Complete signal analysis

## 📁 Project Structure

```
FInal Year Project/
├── arduino/
│   └── ecg_acquisition.ino          # Arduino code for ECG acquisition
├── data/
│   ├── mit_bih/                     # MIT-BIH dataset (downloaded)
│   └── recordings/                  # Local test recordings
├── models/
│   ├── dataset_preparation.py       # MIT-BIH data preprocessing
│   ├── model_architecture.py        # Bi-LSTM model definition
│   ├── train_bilstm.py              # Training script
│   ├── best_model.h5                # Trained Keras model (desktop)
│   └── best_model.tflite            # TFLite model  (Raspberry Pi)
├── preprocessing/
│   ├── filters.py                   # DSP filters
│   └── signal_processing.py         # R-peak detection, beat segmentation
├── realtime/
│   ├── serial_reader.py             # Arduino serial communication
│   ├── batch_processor.py           # Batch inference (Keras + TFLite)
│   └── inference_engine.py          # Real-time classification
├── dashboard/
│   └── app.py                       # Streamlit dashboard (desktop + Pi)
├── convert_model_to_tflite.py       # Keras → TFLite converter
├── start_pi.sh                      # One-command Pi launcher
├── requirements.txt                 # Desktop dependencies
├── requirements_pi.txt              # Raspberry Pi dependencies (lightweight)
└── README.md                        # This file
```

## 🔧 Hardware Setup

### Components Required
- Arduino Uno
- AD8232 ECG Sensor Module
- ECG Electrodes (3 electrodes + cable)
- USB cable for Arduino
- Computer for processing

### Wiring Diagram

```
AD8232 Pin  →  Arduino Pin
─────────────────────────
GND         →  GND
3.3V        →  3.3V (⚠️ NOT 5V)
OUTPUT      →  A0
LO+         →  Digital Pin 10
LO-         →  Digital Pin 11
```

### Electrode Placement
- **RA (Right Arm)**: Below right clavicle
- **LA (Left Arm)**: Below left clavicle  
- **RL (Right Leg)**: Lower right abdomen (reference)

## 💻 Software Installation

### 1. Arduino Setup

1. Install [Arduino IDE](https://www.arduino.cc/en/software)
2. Open `arduino/ecg_acquisition.ino`
3. Select **Tools → Board → Arduino Uno**
4. Select correct **Port**
5. Click **Upload** button

### 2. Python Environment Setup (Desktop)

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Download MIT-BIH Database

```bash
cd models
python dataset_preparation.py
```

This will:
- Download 48 MIT-BIH recordings (~500 MB)
- Extract and label beats
- Create train/test split
- Save processed data

## 🍓 Raspberry Pi Deployment

The system can run on a **Raspberry Pi 4 (2 GB+ RAM)** or newer for portable,
bedside ECG monitoring. The Pi uses **TensorFlow Lite** instead of full
TensorFlow, so inference is fast and memory-friendly.

### Pi Setup — Step by Step

#### 1. Convert the model (on your desktop)

```bash
# On your laptop/desktop where TensorFlow is installed
python convert_model_to_tflite.py
# Creates models/best_model.tflite
```

#### 2. Copy the project to the Pi

```bash
scp -r "FInal Year Project" pi@<PI_IP>:~/ecg-project
```

#### 3. Install dependencies on the Pi

```bash
ssh pi@<PI_IP>
cd ~/ecg-project

python3 -m venv venv
source venv/bin/activate
pip install -r requirements_pi.txt
```

#### 4. Connect hardware and run

```bash
# Plug Arduino into Pi USB port, then:
chmod +x start_pi.sh
./start_pi.sh              # auto-detect serial port
# or
./start_pi.sh /dev/ttyACM0 # specify port explicitly
```

Open `http://<PI_IP>:8501` on any device on the same network.

### Pi Hardware Wiring

```
Raspberry Pi USB  ──►  Arduino Uno USB
Arduino A0        ◄──  AD8232 OUTPUT
Arduino D10       ◄──  AD8232 LO+
Arduino D11       ◄──  AD8232 LO-
Arduino GND       ──►  AD8232 GND
Arduino 3.3V      ──►  AD8232 3.3V
```

### Pi Tips

- Use a **Raspberry Pi 4 (4 GB)** or **Pi 5** for best performance
- A heat-sink or fan is recommended during long sessions
- The dashboard auto-detects Pi mode via CPU architecture; you can also
  force it with `export ECGPI=1`
- Use `requirements_pi.txt` — it uses `ai-edge-litert` (not deprecated `tflite-runtime`)
- **Python 3.12+ error `No module named 'imp'`?** Reinstall deps: `pip install -r requirements_pi.txt`
- Serial permission issues? Run `sudo usermod -aG dialout $USER` and reboot

## 🚀 Training the Model

### Train the Model

```bash
cd models
python train_bilstm.py
```

**Training Configuration:**
- Default Architecture: CNN-LSTM hybrid (recommended)
- Classes: 5 (Normal, Supraventricular, Ventricular, Fusion, Unknown)
- Dataset: MIT-BIH (train: ~70K beats, test: ~30K beats)
- Data Balancing: Hybrid (undersample + SMOTE)
- Expected Accuracy: 85-90%

**Training Time:** ~20-40 minutes on CPU, ~5-10 minutes on GPU

### Available Model Architectures

The system supports multiple neural network architectures. You can select the architecture by modifying the `model_type` parameter in `train_bilstm.py`:

| Architecture | Type | Description | Use Case | Expected Performance |
|-------------|------|-------------|----------|---------------------|
| **CNN-LSTM** | `cnn_lstm` | **RECOMMENDED** - Hybrid model combining CNN for morphological feature extraction and LSTM for temporal modeling | Best for general ECG classification, balanced speed/accuracy | Test Acc: 85-90%, Training: 20-30 min |
| **ResNet-CNN-LSTM** | `rescnn_lstm` | Enhanced CNN-LSTM with residual connections for deeper feature learning | When maximum accuracy is needed, accepts longer training time | Test Acc: 87-92%, Training: 30-45 min |
| **Standard Bi-LSTM** | `standard` | Pure LSTM architecture with bidirectional processing | Baseline model, good for temporal patterns but limited morphology detection | Test Acc: 50-60%, Training: 40-60 min |
| **Enhanced Bi-LSTM** | `enhanced` | Bi-LSTM with attention mechanism | When focusing only on temporal dependencies with attention | Test Acc: 55-65%, Training: 45-65 min |

**Why CNN-LSTM is Recommended:**
- **Better Feature Extraction**: CNNs detect QRS complexes, P-waves, and T-wave morphology
- **Faster Training**: CNNs reduce sequence length before LSTM processing
- **Superior Generalization**: Achieves 85-90% test accuracy vs 50-60% for pure LSTM
- **Proven Results**: State-of-the-art performance on MIT-BIH dataset

### Advanced Features for Class Imbalance

The model includes several advanced techniques to handle extreme class imbalance:

**1. Focal Loss**
- Automatically focuses on hard-to-classify examples
- Down-weights easy examples to prevent majority class dominance
- Particularly effective for Fusion and Unknown classes

**2. Controlled Hybrid Balancing**
- Undersamples Normal class (90% → 40% of data)
- Oversamples minority classes with SMOTE
- **Caps F and Q classes** at 10x original size to prevent over-representation
- Prevents "false positive explosion" in minority classes

**3. Enhanced Regularization**
- Increased dropout rates (0.5 on LSTM, 0.4 on dense layers)
- L2 regularization on all trainable layers
- Spatial dropout on convolutional layers

**4. F1-Score Monitoring**
- Tracks minority class performance during training
- Prints detailed metrics every 10 epochs
- Helps detect overfitting to specific classes

**To Change Architecture:**

Edit `models/train_bilstm.py` and modify the config:

```python
config = {
    'model_type': 'cnn_lstm',  # Change to 'rescnn_lstm', 'standard', or 'enhanced'
    'epochs': 50,
    'batch_size': 256,
    'learning_rate': 0.0005,
    ...
}
```

**Output Files:**
- `models/best_model.h5` - Best model (by validation accuracy)
- `models/trained_model.h5` - Final model
- `results/training_history.png` - Training curves
- `results/confusion_matrix.png` - Performance visualization
- `results/evaluation_results.json` - Detailed metrics

## 📊 Running the Dashboard

### Start Real-time Monitoring

```bash
streamlit run dashboard/app.py
```

The dashboard will open in your browser at `http://localhost:8501`

### Dashboard Features

1. **Live ECG Waveform**
   - Real-time scrolling plot
   - 10-second window
   - Leads-off detection

2. **Classification Panel**
   - Current heartbeat class
   - Confidence score
   - Heart rate (BPM)
   - Beat counter

3. **Alert System**
   - 🚨 Critical alerts (Ventricular arrhythmias)
   - ⚠️ Warnings (Supraventricular episodes)
   - ℹ️ Info (Unknown patterns)

4. **Statistics**
   - Class distribution pie chart
   - Recent classification history
   - Alert log

### Using the Dashboard

1. **Configure Settings** (in sidebar):
   - Serial Port: Leave empty for auto-detect or specify (e.g., `COM3` or `/dev/ttyUSB0`)
   - Model Path: `./models/best_model.h5`

2. **Start System**: Click ▶️ Start button

3. **Attach Electrodes**: Place electrodes on subject

4. **Monitor**: View live ECG and classifications

5. **Stop System**: Click ⏹️ Stop button when done
   - **Automatic Session Analysis** is generated
   - Detailed logs saved to `logs/` folder
   - View comprehensive session statistics on-screen

### Session Analysis & Logging

When you stop the monitoring system, the following happens automatically:

**Comprehensive Session Analysis:**
- Duration and total beats analyzed
- Beat classification distribution with percentages
- Heart rate statistics (mean, min, max, variability)
- Confidence analysis per class
- Alert summary and history
- Overall quality assessment with specific issues detected
- Actionable recommendations for improvement

**Automatic Log Generation:**

The system saves two log files in the `logs/` directory:

1. **JSON Log** (`ecg_session_YYYYMMDD_HHMMSS.json`):
   - Complete structured data
   - Beat-by-beat classifications with timestamps
   - Full probability distributions
   - Smoothing information
   - All session statistics

2. **Human-Readable Summary** (`ecg_session_YYYYMMDD_HHMMSS_summary.txt`):
   - Easy-to-read text format
   - Session overview
   - Classification breakdown
   - Quality assessment
   - Recommendations

**Quality Indicators:**
- **Excellent**: High confidence, stable heart rate, good signal quality
- **Good**: Acceptable performance with minor issues
- **Fair**: Low confidence or signal quality issues detected
- **Poor**: Multiple issues requiring attention

### Understanding Alert Thresholds

The system has built-in safeguards to prevent false alarms:

**Alert Confidence Threshold: 60%**
- Classifications below 60% confidence are **NOT** used for alerts
- Low-confidence beats are still displayed but won't trigger warnings
- This prevents false alarms from uncertain predictions

**Alert Thresholds:**
- **Ventricular (CRITICAL)**: 3 consecutive OR 10 per minute (with >60% confidence)
- **Supraventricular (WARNING)**: 7 consecutive beats (with >60% confidence)
- **Unknown (INFO)**: Any beat with >75% confidence

**Classification Smoothing:**
- Low-confidence predictions (<60%) are smoothed using recent high-confidence classifications
- Prevents flip-flopping between classes
- Look for "(smoothed)" indicator in dashboard

## 🧪 Testing Components

### Test Arduino Connection
```bash
python realtime/serial_reader.py
```

### Test Filters
```bash
python preprocessing/filters.py
```

### Test Signal Processing
```bash
python preprocessing/signal_processing.py
```

### Test Inference Engine
```bash
python realtime/inference_engine.py
```

## 📈 Model Performance

### Expected Results (CNN-LSTM Architecture)

| Metric | Target | Description |
|--------|--------|-------------|
| Test Accuracy | 85-90% | Overall classification accuracy on held-out test set |
| Precision (N) | >90% | Normal beat precision |
| Recall (N) | >80% | Normal beat recall |
| Precision (V) | >85% | Ventricular beat precision |
| Recall (V) | >90% | Ventricular beat recall |
| F1-Score (S) | >50% | Supraventricular beat F1-score |
| F1-Score (V) | >87% | Ventricular beat F1-score |
| Inference Time | <100ms | Per-beat classification latency |

**Note**: Performance varies based on chosen architecture. CNN-LSTM achieves best balance of accuracy and speed.

### Classification Classes

| Class | Full Name | Description |
|-------|-----------|-------------|
| N | Normal | Normal sinus rhythm |
| S | Supraventricular | Atrial premature beats |
| V | Ventricular | Ventricular ectopic beats |
| F | Fusion | Fusion of ventricular and normal |
| Q | Unknown | Unclassifiable beats |

## 🔧 Troubleshooting

### Arduino Not Detected
- Check USB cable connection
- Verify correct COM port in Device Manager (Windows) or `ls /dev/tty*` (Linux/Mac)
- Try different USB port
- Reinstall Arduino drivers

### No ECG Signal
- Check AD8232 power LED is on
- Verify all connections are secure
- Test with multimeter: OUTPUT pin should read ~1.65V at rest
- Replace electrodes if old/dry

### Poor Signal Quality
- Clean skin before electrode placement
- Use conductive gel if available
- Minimize movement during recording
- Check for loose electrode connections
- Move away from sources of electrical interference

### Model Training Fails
- Ensure sufficient disk space (>2 GB)
- Check TensorFlow installation: `python -c "import tensorflow as tf; print(tf.__version__)"`
- Try reducing batch size in `train_bilstm.py` if out of memory
- Verify MIT-BIH data was downloaded correctly

### Dashboard Not Loading
- Check Streamlit installation: `streamlit --version`
- Verify port 8501 is not in use
- Try different browser
- Check firewall settings

## 📝 Safety and Disclaimers

⚠️ **IMPORTANT SAFETY INFORMATION**

1. **Educational Purpose Only**: This system is designed for educational and research purposes.

2. **Not for Clinical Use**: Do NOT use for medical diagnosis or patient care without:
   - Proper validation studies
   - Regulatory approval (FDA/CE marking)
   - Clinical oversight

3. **Electrical Safety**: 
   - The AD8232 is battery-powered for isolation
   - Never connect to mains-powered equipment while attached to person
   - Follow all local electrical safety regulations

4. **Data Privacy**: 
   - Obtain informed consent before recording ECG data
   - Comply with HIPAA/GDPR and local privacy regulations
   - Anonymize any shared data

5. **Medical Emergency**: 
   - This system does NOT replace professional medical equipment
   - In emergency, call emergency services immediately
   - Do not rely solely on automated alerts

## 📚 References

1. **MIT-BIH Arrhythmia Database**
   - Moody GB, Mark RG. The impact of the MIT-BIH Arrhythmia Database. IEEE Eng in Med and Biol 20(3):45-50 (May-June 2001).

2. **Pan-Tompkins Algorithm**
   - Pan J, Tompkins WJ. A Real-Time QRS Detection Algorithm. IEEE Trans Biomed Eng. 1985;32(3):230-236.

3. **AAMI EC57 Standard**
   - Association for the Advancement of Medical Instrumentation. Testing and reporting performance results of cardiac rhythm and ST segment measurement algorithms. ANSI/AAMI EC57:1998.

## 🤝 Contributing

This is an educational project. Suggestions for improvement are welcome:
- Open an issue for bugs or feature requests
- Submit pull requests for enhancements
- Share your results and improvements

## 📄 License

This project is provided for educational purposes. 

Components used:
- MIT-BIH Database: [PhysioNet License](https://physionet.org/about/licenses/)
- TensorFlow/Keras: Apache 2.0
- Other dependencies: See respective licenses

## 👤 Author

**Final Year Project**
ECG Arrhythmia Detection System

## 🙏 Acknowledgments

- PhysioNet for MIT-BIH Arrhythmia Database
- SparkFun for AD8232 documentation
- TensorFlow team for deep learning framework
- Open-source community

---

**⚡ Quick Start Checklist**

- [ ] Hardware assembled and wired
- [ ] Arduino code uploaded
- [ ] Python environment set up
- [ ] MIT-BIH data downloaded
- [ ] Model trained
- [ ] Dashboard tested
- [ ] Electrodes attached correctly
- [ ] System monitoring live ECG

## 🔧 Troubleshooting

### Problem: Very Few Beats Detected (1-3 beats in 30+ seconds)

**Symptoms:**
- Dashboard shows only 1-3 beats after processing
- "Very few peaks detected" warning in console
- Heart rate < 30 BPM

**Causes & Solutions:**

| Cause | Solution |
|-------|----------|
| **Poor electrode contact** | • Clean skin with alcohol<br>• Press electrodes firmly<br>• Use electrode gel |
| **Wrong electrode placement** | • RA: Right wrist/below right clavicle<br>• LA: Left wrist/below left clavicle<br>• LL: Left ankle/lower left abdomen |
| **Weak signal** | • Check 3.3V power to AD8232<br>• Verify LOD pins show connection<br>• Try different electrode positions |
| **Low data rate (<80%)** | • Check Arduino USB connection<br>• Verify 115200 baud rate<br>• Restart serial connection |
| **Noisy signal** | • Keep cables still<br>• Move away from power sources<br>• Check ground connection |

**Check During Recording:**
- ✅ **Data Rate**: Should be 95-100%
- ✅ **Signal Quality**: Should show "Good"
- ✅ **Samples**: Should increase steadily
- ✅ **Waveform**: Should show clear, repeating peaks

**Recommendations:**
- Use 30-60 second recordings (not less than 15s)
- Test with finger on electrodes first (should see muscle noise)
- View Arduino Serial Monitor to verify data is being sent

---

**Good luck with your project! ❤️**
#   E C G - A r r h y t h m i a - D e t e c t i o n - S y s t e m 
 
 