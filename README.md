# ECG Arrhythmia Detection System

Real-time ECG monitoring system with AI-powered arrhythmia classification using Bidirectional LSTM neural networks.

## 🎯 Project Overview

This system provides:
- **Real-time ECG acquisition** from AD8232 sensor via Arduino Uno
- **Edge processing** with digital signal filtering
- **AI classification** of 5 arrhythmia types using Bi-LSTM model
- **Live visualization** dashboard with alerts
- **High accuracy** trained on MIT-BIH Arrhythmia Database

## 📁 Project Structure

```
FInal Year Project/
├── arduino/
│   └── ecg_acquisition.ino         # Arduino code for ECG acquisition
├── data/
│   ├── mit_bih/                    # MIT-BIH dataset (downloaded)
│   └── recordings/                 # Local test recordings
├── models/
│   ├── dataset_preparation.py      # MIT-BIH data preprocessing
│   ├── model_architecture.py       # Bi-LSTM model definition
│   ├── train_bilstm.py             # Training script
│   └── trained_model.h5            # Trained model (after training)
├── preprocessing/
│   ├── filters.py                  # DSP filters
│   └── signal_processing.py        # R-peak detection, beat segmentation
├── realtime/
│   ├── serial_reader.py            # Arduino serial communication
│   └── inference_engine.py         # Real-time classification
├── dashboard/
│   └── app.py                      # Streamlit dashboard
├── requirements.txt                # Python dependencies
└── README.md                       # This file
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

### 2. Python Environment Setup

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

## 🚀 Training the Model

### Train Bi-LSTM Model

```bash
cd models
python train_bilstm.py
```

**Training Configuration:**
- Architecture: Bi-LSTM (128 + 64 units)
- Classes: 5 (Normal, Supraventricular, Ventricular, Fusion, Unknown)
- Dataset: MIT-BIH (train: ~70K beats, test: ~30K beats)
- Data Balancing: SMOTE oversampling
- Expected Accuracy: >95%

**Training Time:** ~30-60 minutes on CPU, ~5-10 minutes on GPU

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

### Expected Results

| Metric | Target | Description |
|--------|--------|-------------|
| Accuracy | >95% | Overall classification accuracy |
| Precision (V) | >90% | Ventricular beat precision |
| Recall (V) | >85% | Ventricular beat recall |
| F1-Score (N) | >97% | Normal beat F1-score |
| Inference Time | <100ms | Per-beat classification latency |

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

**Need Help?** Check the Troubleshooting section above.

**Good luck with your project! ❤️**
#   E C G - A r r h y t h m i a - D e t e c t i o n - S y s t e m  
 