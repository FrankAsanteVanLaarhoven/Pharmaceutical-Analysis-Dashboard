import pandas as pd
from pathlib import Path

def test_data_loading():
    try:
        # Get project root
        project_root = Path(__file__).parent.parent
        data_path = project_root / "data" / "Medicine_Details.csv"
        
        # Load data
        df = pd.read_csv(data_path)
        
        print("✅ Data loaded successfully")
        print(f"Shape: {df.shape}")
        print("\nColumns:", df.columns.tolist())
        print("\nSample data:")
        print(df.head(2))
        
        return True
    except Exception as e:
        print(f"❌ Error loading data: {str(e)}")
        return False

if __name__ == "__main__":
    test_data_loading() 