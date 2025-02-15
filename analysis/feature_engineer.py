import re
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from config import config
from sklearn.pipeline import Pipeline
import difflib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

class PharmaFeatureEngineer:
    """Feature engineering for pharmaceutical data"""
    
    def __init__(self):
        self.dose_pattern = re.compile(r'(\d+\.?\d*)\s*(mg|g|ml)')
        self.expected_columns = {
            'mfr_total_medicines': 'Number of medicines per manufacturer',
            'mfr_avg_review': 'Average excellence rating',
            'mfr_market_share': 'Market share percentage',
            'mfr_market_tier': 'Market position (Small/Medium/Large/Leader)',
            'mfr_performance_tier': 'Performance level (Bronze/Silver/Gold/Platinum)',
            'mfr_rank': 'Overall manufacturer rank',
            'mfr_performance_diff': 'Performance vs average'
        }
        
    def _validate_column_name(self, column_name, available_columns):
        """Validate column name and suggest corrections"""
        if column_name not in available_columns:
            closest_match = difflib.get_close_matches(column_name, available_columns, n=1)
            if closest_match:
                print(f"Warning: '{column_name}' not found. Using '{closest_match[0]}' instead.")
                return closest_match[0]
            raise ValueError(f"Column '{column_name}' not found and no close match available.")
        return column_name
    
    def _validate_manufacturer_columns(self, df):
        """Validate all manufacturer-related columns"""
        missing_columns = []
        corrected_columns = {}
        
        for col in self.expected_columns:
            try:
                corrected_name = self._validate_column_name(col, df.columns)
                if corrected_name != col:
                    corrected_columns[col] = corrected_name
            except ValueError:
                missing_columns.append(col)
        
        if missing_columns:
            print(f"\nMissing manufacturer columns: {missing_columns}")
            print("Available columns:", df.columns.tolist())
        
        if corrected_columns:
            print("\nCorrected column names:", corrected_columns)
        
        return len(missing_columns) == 0
    
    def transform(self, df):
        """Apply feature engineering transformations"""
        df = df.copy()
        
        try:
            # First, convert all percentage columns to numeric
            percentage_columns = [col for col in df.columns if '%' in col]
            for col in percentage_columns:
                df[col] = pd.to_numeric(
                    df[col].astype(str).str.rstrip('%'),
                    errors='coerce'
                ).fillna(0)

            # Extract dosage from Medicine Name
            if 'Medicine Name' in df.columns:
                df['dosage_mg'] = df['Medicine Name'].apply(self._extract_dosage)
            
            # Count ingredients from Composition
            if 'Composition' in df.columns:
                df['num_ingredients'] = pd.to_numeric(
                    df['Composition'].str.count('\+') + 1,
                    errors='coerce'
                ).fillna(0)
                df['main_ingredient'] = (df['Composition']
                                       .str.split('+').str[0]
                                       .str.strip())
            
            # Enhanced Manufacturer Insights with validation
            if 'Manufacturer' in df.columns:
                print("\nGenerating manufacturer metrics...")
                mfr_stats = self._calculate_manufacturer_metrics(df)
                
                # Validate manufacturer columns before merge
                if not mfr_stats.empty and self._validate_manufacturer_columns(mfr_stats):
                    print("\nManufacturer columns before merge:", mfr_stats.columns.tolist())
                    df = df.merge(mfr_stats, on='Manufacturer', how='left')
                    print("\nFinal DataFrame columns:", df.columns.tolist())
                else:
                    print("Warning: Manufacturer metrics validation failed")
            
            # Text Analysis Features
            if 'Uses' in df.columns:
                df['uses_word_count'] = df['Uses'].str.split().str.len()
                df['uses_categories'] = df['Uses'].str.count(',') + 1
                
            if 'Side_effects' in df.columns:
                df['side_effects_count'] = df['Side_effects'].str.split().str.len()
                df['side_effects_severity'] = self._calculate_severity_score(df['Side_effects'])
            
            # Fill missing values
            numeric_cols = df.select_dtypes(include=['number']).columns
            df[numeric_cols] = df[numeric_cols].fillna(0)
            
            return df
            
        except Exception as e:
            print(f"Error in feature engineering: {str(e)}")
            print("\nAvailable columns:", df.columns.tolist())
            return df
    
    def _create_tiers(self, df, column, prefix, n_bins=4):
        """Create tiers for a given column with error handling"""
        try:
            # Remove any existing tiers to avoid duplicates
            tier_column = f'{prefix}_tier'
            if tier_column in df.columns:
                df = df.drop(columns=[tier_column])
            
            # Check if we have enough unique values
            unique_values = df[column].nunique()
            if unique_values < n_bins:
                print(f"Warning: Only {unique_values} unique values for {column}, using fewer bins")
                n_bins = max(2, unique_values)
            
            # Create tiers
            if prefix == 'mfr_market':
                labels = ['Small', 'Medium', 'Large', 'Leader'][:n_bins]
            else:  # performance tiers
                labels = ['Bronze', 'Silver', 'Gold', 'Platinum'][:n_bins]
            
            df[tier_column] = pd.qcut(
                df[column],
                q=n_bins,
                labels=labels,
                duplicates='drop'
            )
            
            return df
            
        except Exception as e:
            print(f"Warning: Could not create {prefix} tiers - {str(e)}")
            df[tier_column] = 'Unknown'
            return df

    def _calculate_manufacturer_metrics(self, df):
        """Calculate comprehensive manufacturer metrics"""
        try:
            # Debug info
            print("\nInput columns:", df.columns.tolist())
            
            # Basic statistics
            mfr_stats = df.groupby('Manufacturer').agg({
                'Medicine Name': 'count',
                'Excellent Review %': ['mean', 'std', 'count']
            }).round(2)
            
            # Flatten column names and add mfr_ prefix
            mfr_stats.columns = [
                'mfr_total_medicines', 'mfr_avg_review', 
                'mfr_review_std', 'mfr_review_count'
            ]
            mfr_stats = mfr_stats.reset_index()
            
            # Market metrics with consistent naming
            total_medicines = float(mfr_stats['mfr_total_medicines'].sum())
            mfr_stats['mfr_market_share'] = (
                mfr_stats['mfr_total_medicines'] / total_medicines * 100
            ).round(2)
            
            # Performance metrics
            mfr_stats['mfr_rank'] = mfr_stats['mfr_avg_review'].rank(
                method='dense',
                ascending=False
            ).astype(int)
            
            avg_review = df['Excellent Review %'].mean()
            mfr_stats['mfr_performance_diff'] = (
                mfr_stats['mfr_avg_review'] - avg_review
            ).round(2)
            
            # Create market and performance tiers
            mfr_stats = self._create_tiers(
                mfr_stats, 
                'mfr_market_share', 
                'mfr_market'
            )
            
            mfr_stats = self._create_tiers(
                mfr_stats, 
                'mfr_avg_review', 
                'mfr_performance'
            )
            
            # Validate required columns exist
            required_columns = [
                'mfr_market_tier', 'mfr_avg_review', 'mfr_market_share',
                'mfr_total_medicines', 'mfr_performance_tier'
            ]
            
            missing_columns = [col for col in required_columns if col not in mfr_stats.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns after calculation: {missing_columns}")
            
            # Debug info
            print("\nOutput columns:", mfr_stats.columns.tolist())
            print("\nTier distributions:")
            print(mfr_stats['mfr_market_tier'].value_counts())
            print(mfr_stats['mfr_performance_tier'].value_counts())
            
            return mfr_stats
            
        except Exception as e:
            print(f"Error calculating manufacturer metrics: {e}")
            print("\nAvailable columns:", df.columns.tolist())
            return pd.DataFrame()
    
    def _calculate_severity_score(self, side_effects_series):
        """Calculate severity score based on side effects"""
        severe_keywords = ['severe', 'dangerous', 'fatal', 'critical', 'emergency']
        moderate_keywords = ['moderate', 'significant', 'concerning']
        
        def get_severity(text):
            if pd.isna(text):
                return 0
            text = text.lower()
            if any(word in text for word in severe_keywords):
                return 3
            if any(word in text for word in moderate_keywords):
                return 2
            return 1
        
        return side_effects_series.apply(get_severity)
    
    def _extract_dosage(self, text):
        """Extract medication dosage in mg"""
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
            value *= 1  # Simplified conversion
            
        return float(value)

def create_feature_pipeline():
    """Create a pipeline with feature engineering"""
    return Pipeline([
        ('feature_engineer', PharmaFeatureEngineer())
    ])

# Example usage
if __name__ == "__main__":
    from data_analyzer import DataAnalyzer
    
    # Load and analyze data
    analyzer = DataAnalyzer()
    df = analyzer.analyze_data()
    
    if df is not None:
        # Create and fit feature pipeline
        pipeline = create_feature_pipeline()
        X_transformed = pipeline.fit_transform(df)
        
        print("\nFeature Engineering Results:")
        print(f"Original shape: {df.shape}")
        print(f"Transformed shape: {X_transformed.shape}")
        
        # Save feature names
        feature_names = pipeline.named_steps['feature_engineer'].get_feature_names()
        print("\nEngineered features:", feature_names)

def train_all_models(df):
    """Train models with advanced optimization"""
    try:
        # Prepare data with consistent handling
        df = df.copy()
        
        # Handle missing values first
        df = df.fillna({
            'Manufacturer': 'Unknown',
            'Composition': '',
            'Uses': '',
            'Side_effects': '',
            'Excellent Review %': df['Excellent Review %'].mean()
        })
        
        # Prepare features and target
        X = df[['Manufacturer', 'Composition', 'Uses', 'Side_effects']]
        y = (df['Excellent Review %'] > df['Excellent Review %'].median()).astype(int)
        
        # Ensure all arrays have the same length
        assert len(X) == len(y), "Feature and target arrays must have the same length"
        
        # Create preprocessor with consistent settings
        preprocessor = ColumnTransformer(
            transformers=[
                ('text_composition', TfidfVectorizer(
                    max_features=200,
                    ngram_range=(1, 2),
                    strip_accents='unicode',
                    lowercase=True
                ), 'Composition'),
                ('text_uses', TfidfVectorizer(
                    max_features=200,
                    ngram_range=(1, 2),
                    strip_accents='unicode',
                    lowercase=True
                ), 'Uses'),
                ('text_side_effects', TfidfVectorizer(
                    max_features=200,
                    ngram_range=(1, 2),
                    strip_accents='unicode',
                    lowercase=True
                ), 'Side_effects'),
                ('categorical', OneHotEncoder(
                    drop='first',
                    sparse=False,
                    handle_unknown='ignore'
                ), ['Manufacturer'])
            ],
            sparse_threshold=0,
            verbose_feature_names_out=False
        )
        
        # Print data shapes for debugging
        print(f"Input data shapes - X: {X.shape}, y: {y.shape}")
        
        # Rest of your model training code...
        return trained_models
        
    except Exception as e:
        print(f"Error in train_all_models: {str(e)}")
        print(f"Data shapes - X: {X.shape if 'X' in locals() else 'not created'}")
        print(f"Columns in df: {df.columns.tolist()}")
        raise e 