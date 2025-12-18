"""
Setup script for ECG Arrhythmia Detection System
Run this script to verify installation and setup
"""

import os
import sys
import subprocess

def print_header(text):
    """Print formatted header"""
    print("\n" + "="*70)
    print(text)
    print("="*70 + "\n")


def check_python_version():
    """Check Python version"""
    print("Checking Python version...")
    version = sys.version_info
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8 or higher is required")
        print(f"   Current version: {version.major}.{version.minor}.{version.micro}")
        return False
    
    print(f"✅ Python {version.major}.{version.minor}.{version.micro}")
    return True


def install_dependencies():
    """Install Python dependencies"""
    print("\nInstalling Python dependencies...")
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install dependencies")
        return False


def create_directories():
    """Create necessary directories"""
    print("\nCreating project directories...")
    
    directories = [
        'data/mit_bih',
        'data/recordings',
        'models',
        'results',
        'logs'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Created: {directory}")
    
    return True


def verify_arduino_code():
    """Verify Arduino code exists"""
    print("\nVerifying Arduino code...")
    
    if os.path.exists('arduino/ecg_acquisition.ino'):
        print("✅ Arduino code found")
        print("   Upload this file to your Arduino Uno using Arduino IDE")
        return True
    else:
        print("❌ Arduino code not found")
        return False


def main():
    """Main setup function"""
    print_header("ECG Arrhythmia Detection System - Setup")
    
    success = True
    
    # Check Python version
    if not check_python_version():
        success = False
    
    # Install dependencies
    if success:
        if not install_dependencies():
            success = False
    
    # Create directories
    if success:
        if not create_directories():
            success = False
    
    # Verify Arduino code
    if success:
        if not verify_arduino_code():
            success = False
    
    # Print summary
    print_header("Setup Summary")
    
    if success:
        print("🎉 Setup completed successfully!\n")
        print("Next steps:")
        print("1. Upload arduino/ecg_acquisition.ino to your Arduino Uno")
        print("2. Connect AD8232 sensor to Arduino as per wiring diagram")
        print("3. Run: python models/dataset_preparation.py (to download MIT-BIH data)")
        print("4. Run: python models/train_bilstm.py (to train the model)")
        print("5. Run: streamlit run dashboard/app.py (to start the dashboard)")
        print("\nFor testing without hardware:")
        print("- Run: python test_integration.py")
    else:
        print("❌ Setup encountered errors. Please fix them and try again.\n")
    
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

