import pandas as pd
import numpy as np

# Create sample data
sample_data = {
    'Medicine_Name': [f'Medicine_{i}' for i in range(1, 101)],
    'Manufacturer': np.random.choice(['Pharma A', 'Pharma B', 'Pharma C'], 100),
    'Salt_Composition': ['Paracetamol 500mg' for _ in range(100)],
    'Uses': ['Fever, Pain relief' for _ in range(100)],
    'Side_Effects': ['Nausea, Headache' for _ in range(100)],
    'Review_Excellent': np.random.uniform(60, 95, 100),
    'Review_Average': np.random.uniform(10, 30, 100),
    'Review_Poor': np.random.uniform(0, 10, 100)
}

df = pd.DataFrame(sample_data)

# Save to data directory
df.to_csv('data/Medicine_Details.csv', index=False) 