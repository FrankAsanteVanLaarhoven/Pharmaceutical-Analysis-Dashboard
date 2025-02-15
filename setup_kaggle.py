import os
from pathlib import Path
import json
import subprocess
import sys
from config import config

def setup_kaggle():
    """Setup Kaggle credentials and download dataset"""
    try:
        # Setup Kaggle credentials using config
        config.setup_kaggle()
        
        # Create data directory
        config.DATA_DIR.mkdir(exist_ok=True)
        
        # Download dataset
        subprocess.run([
            'kaggle', 'datasets', 'download',
            '-d', 'singhnavjot2062001/11000-medicine-details',
            '-p', str(config.DATA_DIR)
        ], check=True)
        
        # Unzip dataset
        subprocess.run([
            'unzip', '-q', f'{config.DATA_DIR}/11000-medicine-details.zip',
            '-d', str(config.DATA_DIR)
        ], check=True)
        
        print("Dataset downloaded successfully")
        
    except Exception as e:
        print(f"Error in Kaggle setup: {e}")
        sys.exit(1)

def verify_imports():
    """Verify all required packages are installed"""
    required_packages = [
        'streamlit',
        'pandas',
        'numpy',
        'sklearn',
        'xgboost',
        'lightgbm',
        'catboost',
        'optuna',
        'mlflow'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"Missing packages: {', '.join(missing_packages)}")
        print("Please install missing packages using: pip install -r requirements.txt")
        sys.exit(1)
    else:
        print("All core packages imported successfully")

if __name__ == "__main__":
    # Verify imports first
    verify_imports()
    
    # Setup Kaggle and download dataset
    setup_kaggle() 