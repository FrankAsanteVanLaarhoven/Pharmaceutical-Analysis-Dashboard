from pathlib import Path
import sys
import os

def test_config():
    try:
        # Add project root to Python path
        project_root = Path(__file__).parent.parent
        sys.path.append(str(project_root))
        
        # Import config
        from config import config
        from config.plotting_config import PLOT_CONFIG
        
        print("✅ Config loaded successfully")
        print("\nProject structure:")
        print(f"ROOT: {config.PROJECT_ROOT}")
        print(f"DATA: {config.DATA_PATH}")
        print(f"MODELS: {config.MODELS_DIR}")
        print(f"REPORTS: {config.REPORTS_DIR}")
        
        print("\nPlotting config:")
        print(PLOT_CONFIG)
        
        return True
    except Exception as e:
        print(f"❌ Error loading config: {str(e)}")
        return False

if __name__ == "__main__":
    test_config() 