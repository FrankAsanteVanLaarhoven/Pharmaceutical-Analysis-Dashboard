from pathlib import Path

class Config:
    """Application configuration"""
    PROJECT_ROOT = Path(__file__).parent.parent
    DATA_PATH = PROJECT_ROOT / 'data' / 'Medicine_Details.csv'
    MODELS_DIR = PROJECT_ROOT / 'models'
    REPORTS_DIR = PROJECT_ROOT / 'reports'
    # Use existing notebooks directory
    NOTEBOOKS_DIR = Path('/Users/frankvanlaarhoven/MEAP/notebooks')

config = Config() 