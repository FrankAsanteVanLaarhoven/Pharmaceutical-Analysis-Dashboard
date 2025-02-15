import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class PharmaAugmenter(BaseEstimator, TransformerMixin):
    """Handles pharmaceutical data augmentation"""
    
    def __init__(self, augmentation_factor=2):
        self.augmentation_factor = augmentation_factor
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, df):
        """Apply pharmaceutical-specific augmentations"""
        try:
            augmented_dfs = [df]
            
            # Generate synthetic formulations
            if 'Composition' in df.columns:
                synthetic_df = df.copy()
                synthetic_df['Composition'] = df['Composition'].apply(
                    lambda x: f"{x} + Placebo 50mg" if pd.notna(x) else x
                )
                augmented_dfs.append(synthetic_df)
            
            # Add dosage variations
            if 'dosage_mg' in df.columns:
                dosage_df = df.copy()
                dosage_df['dosage_mg'] = df['dosage_mg'] * 0.8  # 80% dosage variant
                augmented_dfs.append(dosage_df)
            
            # Combine all augmentations
            result = pd.concat(augmented_dfs[:self.augmentation_factor], axis=0)
            result = result.reset_index(drop=True)
            
            return result
            
        except Exception as e:
            print(f"Error in data augmentation: {e}")
            return df 