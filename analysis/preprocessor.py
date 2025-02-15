import os
import json
from pathlib import Path
import pandas as pd
import re
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from config import config

# Update column mappings with possible variations
REQUIRED_COLUMNS = {
    'medicine_name': 'Medicine Name',
    'composition': 'Composition',
    'uses': 'Uses',
    'side_effects': 'Side_effects',
    'manufacturer': 'Manufacturer',
    'excellent_review': 'Excellent Review %',
    'average_review': 'Average Review %',
    'poor_review': 'Poor Review %'
}

class TextFeatureExtractor:
    """Simple text feature extractor to replace TfidfVectorizer"""
    
    def __init__(self, max_features=100):
        self.max_features = max_features
        self.vocabulary_ = None
        
    def fit(self, texts):
        # Create word frequency dictionary
        word_freq = {}
        for text in texts:
            if pd.isna(text):
                continue
            words = str(text).lower().split()
            for word in words:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Select top words as features
        self.vocabulary_ = dict(
            sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:self.max_features]
        )
        return self
        
    def transform(self, texts):
        if self.vocabulary_ is None:
            raise ValueError("Transformer must be fitted before transform")
        
        # Create feature matrix
        features = pd.DataFrame(index=range(len(texts)))
        for word in self.vocabulary_:
            features[f'contains_{word}'] = [
                1 if pd.notna(text) and word in str(text).lower() else 0 
                for text in texts
            ]
        return features

class PharmaPreprocessor:
    """Pharmaceutical data preprocessing pipeline"""
    
    def __init__(self):
        self.dose_pattern = re.compile(r'(\d+\.?\d*)\s*(mg|g|ml)')
        self.data_dir = config.DATA_DIR
        self.processed_dir = config.PROJECT_ROOT / 'processed'
        self.processed_dir.mkdir(exist_ok=True)
        
    def clean_columns(self, df):
        """Standardize column names based on dataset schema"""
        df = df.copy()
        
        # Print available columns for debugging
        print("Available columns:", df.columns.tolist())
        
        # Standardize column names
        df = df.rename(columns={v: k for k, v in REQUIRED_COLUMNS.items()})
        
        # Convert review percentages to float
        for col in ['excellent_review', 'average_review', 'poor_review']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col].str.rstrip('%'), errors='coerce')
        
        return df

    def extract_features(self, df):
        """Pharma-specific feature engineering"""
        df = df.copy()
        
        # Dosage extraction
        df['dosage_mg'] = df['medicine_name'].apply(self._extract_dosage)
        
        # Composition analysis
        df['num_ingredients'] = df['composition'].str.count('\+') + 1
        df['main_ingredient'] = (df['composition']
                               .str.split('+').str[0]
                               .str.split('(').str[0]
                               .str.strip())
        
        return df
    
    def _extract_dosage(self, text):
        """Extract and standardize dosage from text"""
        if pd.isna(text):
            return 0.0
            
        match = self.dose_pattern.search(str(text))
        if not match:
            return 0.0
            
        value = float(match.group(1))
        unit = match.group(2)
        
        # Convert to mg
        if unit == 'g':
            value *= 1000
        elif unit == 'ml':
            value *= 1  # Assuming 1ml = 1mg for simplicity
            
        return value

    def load_and_validate(self, data_file=None):
        """Load and validate the data"""
        try:
            data_file = data_file or self.data_dir / 'Medicine_Details.csv'
            
            if not data_file.exists():
                raise FileNotFoundError(f"Data file not found at {data_file}")
            
            # Load data
            df = pd.read_csv(data_file)
            
            # Debug info
            print("\nOriginal columns:", df.columns.tolist())
            
            # Verify required columns
            missing = [col for col, required in REQUIRED_COLUMNS.items() 
                      if required not in df.columns]
            
            if missing:
                raise KeyError(
                    f"Missing required columns: {missing}\n"
                    f"Available columns are: {df.columns.tolist()}"
                )
            
            # Rename columns to standard names
            df = df.rename(columns={v: k for k, v in REQUIRED_COLUMNS.items()})
            
            return df
            
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            raise

    def build_pipeline(self):
        """Create the preprocessing pipeline"""
        # Get available columns
        df = self.load_and_validate()
        available_columns = df.columns.tolist()
        
        # Create transformers based on available columns
        transformers = []
        
        if 'uses' in available_columns:
            transformers.append(
                ('uses_text', TextFeatureExtractor(max_features=50), 'uses')
            )
        
        if 'side_effects' in available_columns:
            transformers.append(
                ('side_effects_text', TextFeatureExtractor(max_features=30), 'side_effects')
            )
        
        if 'manufacturer' in available_columns:
            transformers.append(
                ('manufacturer', OneHotEncoder(handle_unknown='ignore'), ['manufacturer'])
            )
        
        if 'main_ingredient' in available_columns:
            transformers.append(
                ('ingredients', OneHotEncoder(handle_unknown='ignore'), ['main_ingredient'])
            )
        
        numeric_cols = ['dosage_mg', 'num_ingredients']
        numeric_cols = [col for col in numeric_cols if col in available_columns]
        if numeric_cols:
            transformers.append(
                ('numerical', StandardScaler(), numeric_cols)
            )
        
        return ColumnTransformer(
            transformers=transformers,
            remainder='passthrough'
        )

    def process_data(self, test_size=0.2, random_state=42):
        """Complete data processing workflow"""
        try:
            # Load data
            raw_df = self.load_and_validate()
            
            # Clean and engineer features
            df = self.clean_columns(raw_df)
            df = self.extract_features(df)
            
            # Split features and target
            X = df.drop(columns=['excellent_review', 'average_review', 'poor_review'])
            y = df[['excellent_review', 'average_review', 'poor_review']]
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
            
            # Build and fit pipeline
            preprocessor = self.build_pipeline()
            X_train_processed = preprocessor.fit_transform(X_train)
            X_test_processed = preprocessor.transform(X_test)
            
            # Save processed data
            processed_data = {
                'X_train': X_train_processed,
                'X_test': X_test_processed,
                'y_train': y_train,
                'y_test': y_test,
                'feature_names': preprocessor.get_feature_names_out(),
                'preprocessor': preprocessor
            }
            
            return processed_data
            
        except Exception as e:
            print(f"Error in data processing: {e}")
            return None

def run_preprocessing():
    """Run the complete preprocessing pipeline"""
    processor = PharmaPreprocessor()
    processed_data = processor.process_data()
    
    if processed_data is not None:
        print("\nPreprocessing Results:")
        print(f"Training data shape: {processed_data['X_train'].shape}")
        print(f"Test data shape: {processed_data['X_test'].shape}")
        print(f"Number of features: {len(processed_data['feature_names'])}")
    
    return processed_data

if __name__ == "__main__":
    run_preprocessing() 