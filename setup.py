from pathlib import Path
import subprocess
import sys

def setup_project():
    """Initialize project structure and install dependencies"""
    try:
        # Create project directories
        directories = ['data', 'app', 'notebooks', 'tests', 'analysis', 'reports']
        for dir_name in directories:
            Path(dir_name).mkdir(exist_ok=True)
        
        # Install dependencies using the new script
        subprocess.run([
            sys.executable, 'install_dependencies.py'
        ], check=True)
        
        # Run Kaggle setup
        subprocess.run([
            sys.executable, 'setup_kaggle.py'
        ], check=True)
        
        # Create sample model and notebook
        subprocess.run([
            sys.executable, 'create_sample_model.py'
        ], check=True)
        
        subprocess.run([
            sys.executable, 'save_notebook.py'
        ], check=True)
        
        print("Project setup completed successfully")
        
    except Exception as e:
        print(f"Error during project setup: {e}")
        sys.exit(1)

if __name__ == "__main__":
    setup_project() 