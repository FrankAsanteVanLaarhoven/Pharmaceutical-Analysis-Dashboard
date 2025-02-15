from pathlib import Path
from dotenv import load_dotenv
import os
import json

class Config:
    """Configuration handler for the project"""
    
    def __init__(self):
        # Load environment variables from .env file
        env_path = Path('.env')
        load_dotenv(dotenv_path=env_path)
        
        # Kaggle credentials
        self.KAGGLE_USERNAME = os.getenv('KAGGLE_USERNAME')
        self.KAGGLE_KEY = os.getenv('KAGGLE_KEY')
        
        # Project paths
        self.PROJECT_ROOT = Path(__file__).parent
        self.DATA_DIR = self.PROJECT_ROOT / os.getenv('DATA_DIR', 'data')
        self.MODEL_DIR = self.PROJECT_ROOT / os.getenv('MODEL_DIR', 'models')
        self.NOTEBOOK_DIR = self.PROJECT_ROOT / os.getenv('NOTEBOOK_DIR', 'notebooks')
        
        # API configuration
        self.API_PORT = int(os.getenv('API_PORT', 8501))
        self.DEBUG_MODE = os.getenv('DEBUG_MODE', 'False').lower() == 'true'
        
        # MLflow configuration
        self.MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI', './mlruns')
        self.EXPERIMENT_NAME = os.getenv('EXPERIMENT_NAME', 'pharma_review_prediction')
    
    def setup_kaggle(self):
        """Setup Kaggle credentials"""
        kaggle_dir = Path.home() / '.kaggle'
        kaggle_dir.mkdir(exist_ok=True)
        
        credentials = {
            'username': self.KAGGLE_USERNAME,
            'key': self.KAGGLE_KEY
        }
        
        cred_path = kaggle_dir / 'kaggle.json'
        with open(cred_path, 'w') as f:
            json.dump(credentials, f)
        
        # Set proper permissions
        os.chmod(cred_path, 0o600)

# Create a global config instance
config = Config() 