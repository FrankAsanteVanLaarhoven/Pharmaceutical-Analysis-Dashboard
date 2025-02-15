import pandas as pd
from pathlib import Path

# Load the data file
data_path = Path('data') / 'Medicine_Details.csv'
df = pd.read_csv(data_path)

# Print column names
print("Available columns in data file:")
for col in df.columns:
    print(f"  - {col}") 