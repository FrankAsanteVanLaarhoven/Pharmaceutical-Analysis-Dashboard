import subprocess
import sys
from pathlib import Path

def install_dependencies():
    """Install dependencies in the correct order with proper conflict resolution"""
    try:
        # First uninstall potentially conflicting packages
        subprocess.run([
            sys.executable, '-m', 'pip', 'uninstall', '-y',
            'numba', 'ydata-profiling'
        ], check=True)
        
        # Install base requirements
        subprocess.run([
            sys.executable, '-m', 'pip', 'install',
            'wheel>=0.38.4',
            'setuptools>=65.5.1'
        ], check=True)
        
        # Install numba first
        subprocess.run([
            sys.executable, '-m', 'pip', 'install',
            'numba==0.58.1'
        ], check=True)
        
        # Install core numerical packages
        subprocess.run([
            sys.executable, '-m', 'pip', 'install',
            'numpy==1.24.3',
            'pandas==2.0.3'
        ], check=True)
        
        # Install visualization packages
        subprocess.run([
            sys.executable, '-m', 'pip', 'install',
            'plotly==5.17.0',
            'seaborn==0.12.2',
            'matplotlib==3.7.1'
        ], check=True)
        
        # Install ML packages
        subprocess.run([
            sys.executable, '-m', 'pip', 'install',
            'scikit-learn==1.3.0',
            'xgboost==1.5.2',
            'lightgbm==4.1.0',
            'catboost==1.2.0'
        ], check=True)
        
        # Install ydata-profiling with dependencies
        subprocess.run([
            sys.executable, '-m', 'pip', 'install',
            'ydata-profiling==4.5.1'
        ], check=True)
        
        # Install remaining packages from requirements.txt
        subprocess.run([
            sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'
        ], check=True)
        
        print("All dependencies installed successfully")
        
    except subprocess.CalledProcessError as e:
        print(f"Error installing dependencies: {e}")
        sys.exit(1)

if __name__ == "__main__":
    install_dependencies() 