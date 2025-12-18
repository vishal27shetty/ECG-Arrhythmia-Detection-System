"""
Quick Start Script for ECG Arrhythmia Detection System
Interactive script to guide users through the workflow
"""

import os
import sys
import subprocess


def print_header(text):
    """Print formatted header"""
    print("\n" + "="*70)
    print(text)
    print("="*70 + "\n")


def print_menu(title, options):
    """Print menu"""
    print(f"\n{title}")
    print("-" * 50)
    for i, option in enumerate(options, 1):
        print(f"{i}. {option}")
    print("0. Exit")
    print()


def run_script(script_path, description):
    """Run a Python script"""
    print(f"\n{description}...")
    print("-" * 70)
    
    try:
        result = subprocess.run([sys.executable, script_path], check=True)
        print("\n✅ Completed successfully!")
        return True
    except subprocess.CalledProcessError:
        print("\n❌ Failed to complete")
        return False
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user")
        return False


def check_file_exists(filepath, description):
    """Check if a file exists"""
    if os.path.exists(filepath):
        print(f"✅ {description} found")
        return True
    else:
        print(f"❌ {description} not found")
        return False


def main():
    """Main menu"""
    print_header("ECG Arrhythmia Detection System - Quick Start")
    
    print("Welcome to the ECG Arrhythmia Detection System!")
    print("This script will help you set up and run the system.\n")
    
    while True:
        options = [
            "Run Setup (Install dependencies & create directories)",
            "Test Components (Run integration tests)",
            "Download & Prepare MIT-BIH Dataset",
            "Train Bi-LSTM Model",
            "Test Arduino Connection",
            "Launch Dashboard (Real-time monitoring)",
            "View System Status"
        ]
        
        print_menu("Main Menu - Select an option:", options)
        
        try:
            choice = input("Enter your choice (0-7): ").strip()
            
            if choice == "0":
                print("\nExiting... Goodbye!")
                break
            
            elif choice == "1":
                # Run setup
                run_script("setup.py", "Running setup")
            
            elif choice == "2":
                # Run tests
                run_script("test_integration.py", "Running integration tests")
            
            elif choice == "3":
                # Download dataset
                if not os.path.exists('data/mit_bih'):
                    os.makedirs('data/mit_bih', exist_ok=True)
                
                print("\n⚠️ Warning: This will download ~500 MB of data")
                confirm = input("Continue? (y/n): ").strip().lower()
                
                if confirm == 'y':
                    run_script("models/dataset_preparation.py", "Downloading MIT-BIH dataset")
                else:
                    print("Cancelled")
            
            elif choice == "4":
                # Train model
                print("\n⚠️ Warning: Training may take 30-60 minutes")
                confirm = input("Continue? (y/n): ").strip().lower()
                
                if confirm == 'y':
                    run_script("models/train_bilstm.py", "Training Bi-LSTM model")
                else:
                    print("Cancelled")
            
            elif choice == "5":
                # Test Arduino
                print("\n⚠️ Make sure Arduino is connected via USB")
                input("Press Enter to continue...")
                run_script("realtime/serial_reader.py", "Testing Arduino connection")
            
            elif choice == "6":
                # Launch dashboard
                print("\n⚠️ Make sure:")
                print("  1. Arduino is connected")
                print("  2. Model is trained (best_model.h5 exists)")
                print("  3. AD8232 electrodes are ready")
                
                confirm = input("\nReady to launch? (y/n): ").strip().lower()
                
                if confirm == 'y':
                    print("\nLaunching dashboard...")
                    print("Dashboard will open in your browser at http://localhost:8501")
                    print("Press Ctrl+C to stop\n")
                    
                    try:
                        subprocess.run(["streamlit", "run", "dashboard/app.py"])
                    except KeyboardInterrupt:
                        print("\n\nDashboard stopped")
                else:
                    print("Cancelled")
            
            elif choice == "7":
                # System status
                print_header("System Status")
                
                print("Project Structure:")
                check_file_exists("arduino/ecg_acquisition.ino", "Arduino code")
                check_file_exists("requirements.txt", "Requirements file")
                check_file_exists("README.md", "Documentation")
                
                print("\nData:")
                check_file_exists("data/mit_bih/100.dat", "MIT-BIH dataset")
                check_file_exists("data/mit_bih/prepared_data.pkl", "Prepared dataset")
                
                print("\nModels:")
                check_file_exists("models/best_model.h5", "Trained model (best)")
                check_file_exists("models/trained_model.h5", "Trained model (final)")
                
                print("\nResults:")
                check_file_exists("results/training_history.png", "Training history plot")
                check_file_exists("results/confusion_matrix.png", "Confusion matrix")
                check_file_exists("results/evaluation_results.json", "Evaluation results")
                
                input("\nPress Enter to continue...")
            
            else:
                print("Invalid choice. Please try again.")
        
        except KeyboardInterrupt:
            print("\n\nExiting... Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")
            input("\nPress Enter to continue...")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nExiting... Goodbye!")
    except Exception as e:
        print(f"\n❌ Fatal error: {str(e)}")
        sys.exit(1)

