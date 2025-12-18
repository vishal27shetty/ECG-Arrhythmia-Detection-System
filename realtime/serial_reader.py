"""
Serial Reader for Arduino ECG Data Acquisition
Handles real-time data streaming from Arduino via USB
"""

import serial
import serial.tools.list_ports
import threading
import queue
import time
import numpy as np
from typing import Optional, Tuple, List


class ECGSerialReader:
    """
    Reads ECG data from Arduino via serial connection
    Provides thread-safe access to real-time ECG samples
    """
    
    def __init__(self, port: Optional[str] = None, baudrate: int = 115200, buffer_size: int = 1000):
        """
        Initialize serial reader
        
        Args:
            port: Serial port name (auto-detect if None)
            baudrate: Serial communication speed (must match Arduino)
            buffer_size: Maximum number of samples to buffer
        """
        self.port = port
        self.baudrate = baudrate
        self.buffer_size = buffer_size
        
        # Thread-safe queue for data
        self.data_queue = queue.Queue(maxsize=buffer_size)
        
        # Serial connection
        self.serial_conn: Optional[serial.Serial] = None
        self.is_running = False
        self.reader_thread: Optional[threading.Thread] = None
        
        # Statistics
        self.total_samples = 0
        self.dropped_samples = 0
        self.leads_off_count = 0
        
    def auto_detect_port(self) -> Optional[str]:
        """
        Automatically detect Arduino serial port
        
        Returns:
            Port name if found, None otherwise
        """
        ports = serial.tools.list_ports.comports()
        
        # Look for common Arduino identifiers
        arduino_keywords = ['arduino', 'ch340', 'usb', 'acm']
        
        for port in ports:
            port_info = (port.device + port.description + port.manufacturer).lower()
            for keyword in arduino_keywords:
                if keyword in port_info:
                    print(f"Auto-detected Arduino on port: {port.device}")
                    return port.device
        
        # If no Arduino found, return first available port
        if ports:
            print(f"Using first available port: {ports[0].device}")
            return ports[0].device
        
        return None
    
    def connect(self) -> bool:
        """
        Connect to Arduino via serial
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            # Auto-detect port if not specified
            if self.port is None:
                self.port = self.auto_detect_port()
                if self.port is None:
                    print("Error: No serial ports found")
                    return False
            
            # Open serial connection
            self.serial_conn = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=1.0,
                write_timeout=1.0
            )
            
            # Wait for Arduino to reset
            time.sleep(2)
            
            # Flush any existing data
            self.serial_conn.reset_input_buffer()
            
            print(f"Connected to Arduino on {self.port} at {self.baudrate} baud")
            return True
            
        except serial.SerialException as e:
            print(f"Error connecting to serial port: {e}")
            return False
    
    def start_reading(self):
        """Start reading data from Arduino in background thread"""
        if self.serial_conn is None or not self.serial_conn.is_open:
            print("Error: Serial connection not established")
            return
        
        self.is_running = True
        self.reader_thread = threading.Thread(target=self._read_loop, daemon=True)
        self.reader_thread.start()
        print("Started reading ECG data...")
    
    def _read_loop(self):
        """Background thread that reads serial data continuously"""
        while self.is_running:
            try:
                if self.serial_conn and self.serial_conn.in_waiting > 0:
                    # Read line from serial
                    line = self.serial_conn.readline().decode('utf-8', errors='ignore').strip()
                    
                    # Skip comments and empty lines
                    if not line or line.startswith('#'):
                        continue
                    
                    # Parse data: timestamp,ecg_value,lo_plus,lo_minus
                    parts = line.split(',')
                    if len(parts) == 4:
                        try:
                            timestamp = int(parts[0])
                            ecg_value = int(parts[1])
                            lo_plus = int(parts[2])
                            lo_minus = int(parts[3])
                            
                            # Check if leads are connected
                            leads_off = (lo_plus == 1 or lo_minus == 1)
                            
                            if leads_off:
                                self.leads_off_count += 1
                            
                            # Create data sample
                            sample = {
                                'timestamp': timestamp,
                                'ecg': ecg_value,
                                'leads_off': leads_off,
                                'time': time.time()
                            }
                            
                            # Add to queue (non-blocking)
                            try:
                                self.data_queue.put_nowait(sample)
                                self.total_samples += 1
                            except queue.Full:
                                # Queue is full, drop oldest sample
                                try:
                                    self.data_queue.get_nowait()
                                    self.data_queue.put_nowait(sample)
                                    self.dropped_samples += 1
                                except queue.Empty:
                                    pass
                        
                        except ValueError as e:
                            # Invalid data format
                            pass
                
                else:
                    # No data available, short sleep to prevent busy waiting
                    time.sleep(0.001)
                    
            except Exception as e:
                print(f"Error reading serial data: {e}")
                time.sleep(0.1)
    
    def get_sample(self, timeout: float = 1.0) -> Optional[dict]:
        """
        Get next ECG sample from queue
        
        Args:
            timeout: Maximum time to wait for sample (seconds)
        
        Returns:
            Sample dictionary or None if timeout
        """
        try:
            return self.data_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def get_samples(self, count: int, timeout: float = 1.0) -> List[dict]:
        """
        Get multiple ECG samples from queue
        
        Args:
            count: Number of samples to retrieve
            timeout: Maximum time to wait for all samples
        
        Returns:
            List of sample dictionaries
        """
        samples = []
        start_time = time.time()
        
        while len(samples) < count:
            remaining_time = timeout - (time.time() - start_time)
            if remaining_time <= 0:
                break
            
            sample = self.get_sample(timeout=remaining_time)
            if sample is not None:
                samples.append(sample)
        
        return samples
    
    def get_buffer(self) -> np.ndarray:
        """
        Get all available samples in buffer as numpy array
        
        Returns:
            Array of ECG values
        """
        samples = []
        while not self.data_queue.empty():
            try:
                sample = self.data_queue.get_nowait()
                samples.append(sample['ecg'])
            except queue.Empty:
                break
        
        return np.array(samples) if samples else np.array([])
    
    def stop_reading(self):
        """Stop reading data and close connection"""
        self.is_running = False
        
        if self.reader_thread:
            self.reader_thread.join(timeout=2.0)
        
        if self.serial_conn and self.serial_conn.is_open:
            self.serial_conn.close()
        
        print(f"Stopped reading. Total samples: {self.total_samples}, Dropped: {self.dropped_samples}")
    
    def get_stats(self) -> dict:
        """Get reader statistics"""
        return {
            'total_samples': self.total_samples,
            'dropped_samples': self.dropped_samples,
            'leads_off_count': self.leads_off_count,
            'queue_size': self.data_queue.qsize(),
            'is_running': self.is_running
        }
    
    def __enter__(self):
        """Context manager entry"""
        self.connect()
        self.start_reading()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.stop_reading()


def test_serial_reader():
    """Test function for serial reader"""
    print("Testing ECG Serial Reader...")
    print("Make sure Arduino is connected and running ecg_acquisition.ino")
    
    reader = ECGSerialReader()
    
    if not reader.connect():
        print("Failed to connect to Arduino")
        return
    
    reader.start_reading()
    
    try:
        print("\nReading 100 samples...")
        for i in range(100):
            sample = reader.get_sample(timeout=1.0)
            if sample:
                status = "LEADS OFF" if sample['leads_off'] else "OK"
                print(f"Sample {i}: ECG={sample['ecg']}, Status={status}")
            else:
                print("Timeout waiting for sample")
        
        print("\nReader statistics:")
        stats = reader.get_stats()
        for key, value in stats.items():
            print(f"  {key}: {value}")
    
    finally:
        reader.stop_reading()


if __name__ == "__main__":
    test_serial_reader()

