/*
 * ECG Acquisition with AD8232 Sensor
 * Sampling Rate: 360 Hz (MIT-BIH standard)
 * 
 * Pin Configuration:
 * - AD8232 OUTPUT -> Arduino A0
 * - AD8232 LO+ -> Arduino Pin 10
 * - AD8232 LO- -> Arduino Pin 11
 * - AD8232 GND -> Arduino GND
 * - AD8232 3.3V -> Arduino 3.3V
 * 
 * Serial Output Format: timestamp,ecg_value,lo_plus,lo_minus
 */

// Pin definitions
const int ECG_PIN = A0;        // ECG analog output
const int LO_PLUS_PIN = 10;    // Leads-off detection +
const int LO_MINUS_PIN = 11;   // Leads-off detection -
const int LED_PIN = 13;        // Heartbeat indicator LED

// Sampling configuration
const unsigned long SAMPLING_RATE = 360;  // Hz
const unsigned long SAMPLING_INTERVAL = 1000000UL / SAMPLING_RATE;  // microseconds
unsigned long lastSampleTime = 0;
unsigned long timestamp = 0;

// LED indicator for heartbeat visualization
const int ECG_THRESHOLD = 600;  // Threshold for LED flash (adjust based on signal)
bool ledState = false;
int lastEcgValue = 0;

void setup() {
  // Initialize serial communication at high baud rate
  Serial.begin(115200);
  
  // Configure pins
  pinMode(LO_PLUS_PIN, INPUT);
  pinMode(LO_MINUS_PIN, INPUT);
  pinMode(LED_PIN, OUTPUT);
  
  // Wait for serial connection
  while (!Serial) {
    ; // Wait for serial port to connect
  }
  
  // Send startup message
  Serial.println("# ECG Acquisition System Ready");
  Serial.println("# Format: timestamp,ecg_value,lo_plus,lo_minus");
  
  // Initialize timing
  lastSampleTime = micros();
}

void loop() {
  unsigned long currentTime = micros();
  
  // Check if it's time to sample (360 Hz)
  if (currentTime - lastSampleTime >= SAMPLING_INTERVAL) {
    lastSampleTime = currentTime;
    
    // Read leads-off detection pins
    int loPlus = digitalRead(LO_PLUS_PIN);
    int loMinus = digitalRead(LO_MINUS_PIN);
    
    // Read ECG value
    int ecgValue = 0;
    
    // Only read ECG if electrodes are connected
    if (loPlus == 0 && loMinus == 0) {
      ecgValue = analogRead(ECG_PIN);
      
      // Simple peak detection for LED indicator
      if (ecgValue > ECG_THRESHOLD && lastEcgValue <= ECG_THRESHOLD) {
        digitalWrite(LED_PIN, HIGH);
        ledState = true;
      } else if (ledState && (currentTime % 100000 < 50000)) {
        digitalWrite(LED_PIN, LOW);
        ledState = false;
      }
      
      lastEcgValue = ecgValue;
    } else {
      // Leads are off - turn off LED
      digitalWrite(LED_PIN, LOW);
      ledState = false;
      lastEcgValue = 0;
    }
    
    // Send data via serial: timestamp,ecg_value,lo_plus,lo_minus
    Serial.print(timestamp);
    Serial.print(",");
    Serial.print(ecgValue);
    Serial.print(",");
    Serial.print(loPlus);
    Serial.print(",");
    Serial.println(loMinus);
    
    timestamp++;
  }
}


