import pandas as pd
import numpy as np
import pytest
from analysis.feature_engineer import PharmaFeatureEngineer

def create_test_data():
    """Create sample data for testing"""
    return pd.DataFrame({
        'Manufacturer': ['A', 'B', 'C', 'A', 'B'],
        'Medicine Name': ['Drug1 500mg', 'Drug2', 'Drug3 1g', 'Drug4 250mg', 'Drug5'],
        'Excellent Review %': [90, 80, 70, 85, 75],
        'Composition': ['Paracetamol + Caffeine', 'Aspirin', 'Ibuprofen + Codeine', 
                       'Paracetamol', 'Aspirin + Codeine'],
        'Uses': ['Fever, Pain', 'Pain', 'Pain, Inflammation', 'Fever', 'Pain'],
        'Side_effects': ['Nausea', 'Moderate bleeding', 'Severe stomach pain', 
                        'None', 'Mild nausea']
    })

def test_basic_feature_engineering():
    """Test basic feature engineering functionality"""
    # Arrange
    df = create_test_data()
    engineer = PharmaFeatureEngineer()
    
    # Act
    result = engineer.transform(df)
    
    # Assert
    expected_base_columns = [
        'Manufacturer', 'Medicine Name', 'Excellent Review %',
        'Composition', 'Uses', 'Side_effects'
    ]
    expected_engineered_columns = [
        'dosage_mg', 'num_ingredients', 'main_ingredient',
        'mfr_total_medicines', 'mfr_avg_review', 'mfr_market_share',
        'mfr_market_tier', 'mfr_performance_tier'
    ]
    
    # Check all expected columns exist
    for col in expected_base_columns + expected_engineered_columns:
        assert col in result.columns, f"Missing column: {col}"

def test_manufacturer_metrics():
    """Test manufacturer metrics calculation"""
    # Arrange
    df = create_test_data()
    engineer = PharmaFeatureEngineer()
    
    # Act
    result = engineer.transform(df)
    
    # Assert
    # Check manufacturer A metrics (2 products)
    mfr_a = result[result['Manufacturer'] == 'A'].iloc[0]
    assert mfr_a['mfr_total_medicines'] == 2
    assert 87.0 <= mfr_a['mfr_avg_review'] <= 88.0  # (90 + 85) / 2
    assert 35.0 <= mfr_a['mfr_market_share'] <= 45.0  # 2/5 * 100

def test_dosage_extraction():
    """Test medication dosage extraction"""
    # Arrange
    df = create_test_data()
    engineer = PharmaFeatureEngineer()
    
    # Act
    result = engineer.transform(df)
    
    # Assert
    expected_dosages = {
        'Drug1 500mg': 500.0,
        'Drug2': 0.0,
        'Drug3 1g': 1000.0,
        'Drug4 250mg': 250.0,
        'Drug5': 0.0
    }
    
    for name, expected_dose in expected_dosages.items():
        actual_dose = result[result['Medicine Name'] == name]['dosage_mg'].iloc[0]
        assert actual_dose == expected_dose, f"Wrong dosage for {name}"

def test_tier_creation():
    """Test market and performance tier creation"""
    # Arrange
    df = create_test_data()
    engineer = PharmaFeatureEngineer()
    
    # Act
    result = engineer.transform(df)
    
    # Assert
    # Check tier columns exist
    assert 'mfr_market_tier' in result.columns
    assert 'mfr_performance_tier' in result.columns
    
    # Check tier values are valid
    valid_market_tiers = {'Small', 'Medium', 'Large', 'Leader', 'Unknown'}
    valid_performance_tiers = {'Bronze', 'Silver', 'Gold', 'Platinum', 'Unknown'}
    
    assert set(result['mfr_market_tier'].unique()).issubset(valid_market_tiers)
    assert set(result['mfr_performance_tier'].unique()).issubset(valid_performance_tiers)

def test_error_handling():
    """Test error handling with invalid data"""
    # Arrange
    invalid_df = pd.DataFrame({
        'Manufacturer': ['A'],
        'Invalid Column': [1]  # Missing required columns
    })
    engineer = PharmaFeatureEngineer()
    
    # Act
    result = engineer.transform(invalid_df)
    
    # Assert
    assert isinstance(result, pd.DataFrame)  # Should return DataFrame even on error
    assert len(result) == len(invalid_df)  # Should preserve original data

def test_text_feature_extraction():
    """Test text feature extraction from Uses and Side_effects"""
    # Arrange
    df = create_test_data()
    engineer = PharmaFeatureEngineer()
    
    # Act
    result = engineer.transform(df)
    
    # Assert
    # Check text features exist
    text_features = [
        'uses_word_count', 'uses_categories',
        'side_effects_count', 'side_effects_severity'
    ]
    for feature in text_features:
        assert feature in result.columns, f"Missing text feature: {feature}"
    
    # Check specific values
    first_row = result.iloc[0]
    assert first_row['uses_word_count'] == 2  # "Fever, Pain"
    assert first_row['uses_categories'] == 2  # Two categories
    assert first_row['side_effects_severity'] == 1  # "Nausea" - mild

def test_severity_scoring():
    """Test side effects severity scoring"""
    # Arrange
    test_data = pd.DataFrame({
        'Manufacturer': ['X'] * 4,
        'Medicine Name': ['Test'] * 4,
        'Excellent Review %': [80] * 4,
        'Side_effects': [
            'Severe headache, emergency',  # Should score 3
            'Moderate pain, concerning',   # Should score 2
            'Mild nausea',                # Should score 1
            np.nan                        # Should score 0
        ]
    })
    engineer = PharmaFeatureEngineer()
    
    # Act
    result = engineer.transform(test_data)
    
    # Assert
    expected_scores = [3, 2, 1, 0]
    actual_scores = result['side_effects_severity'].tolist()
    assert actual_scores == expected_scores, f"Expected {expected_scores}, got {actual_scores}"

def test_edge_cases():
    """Test edge cases in feature engineering"""
    # Arrange
    edge_case_data = pd.DataFrame({
        'Manufacturer': ['A', 'A', 'B'],
        'Medicine Name': ['Drug 0.5g', 'Drug NaN', ''],  # Test various dosage formats
        'Excellent Review %': [100, np.nan, 0],  # Test percentage extremes
        'Composition': [np.nan, '', 'Single'],   # Test missing compositions
        'Uses': ['', np.nan, 'Multiple, Uses, Here'],  # Test various text formats
        'Side_effects': ['SEVERE', 'moderate', '']  # Test case sensitivity
    })
    engineer = PharmaFeatureEngineer()
    
    # Act
    result = engineer.transform(edge_case_data)
    
    # Assert
    assert result['dosage_mg'].iloc[0] == 500.0  # Test 0.5g conversion
    assert result['dosage_mg'].iloc[1] == 0.0    # Test NaN handling
    assert result['num_ingredients'].iloc[2] == 1 # Test single ingredient
    assert result['uses_categories'].iloc[2] == 3 # Test multiple categories
    assert result['side_effects_severity'].iloc[0] == 3  # Test uppercase
    assert result['side_effects_severity'].iloc[1] == 2  # Test lowercase

def test_manufacturer_edge_cases():
    """Test manufacturer metrics with edge cases"""
    # Arrange
    edge_case_data = pd.DataFrame({
        'Manufacturer': ['Single'] * 5,  # Test single manufacturer
        'Medicine Name': [f'Drug{i}' for i in range(5)],
        'Excellent Review %': [100] * 5  # All perfect reviews
    })
    engineer = PharmaFeatureEngineer()
    
    # Act
    result = engineer.transform(edge_case_data)
    
    # Assert
    # Single manufacturer should have 100% market share
    assert result['mfr_market_share'].iloc[0] == 100.0
    # Perfect reviews should result in highest tier
    assert result['mfr_performance_tier'].iloc[0] == 'Platinum'
    # All metrics should be identical for single manufacturer
    assert len(result['mfr_avg_review'].unique()) == 1

def test_data_consistency():
    """Test data consistency after transformation"""
    # Arrange
    df = create_test_data()
    engineer = PharmaFeatureEngineer()
    
    # Act
    result = engineer.transform(df)
    
    # Assert
    # Row count should remain the same
    assert len(result) == len(df)
    # Original columns should be preserved
    for col in df.columns:
        assert col in result.columns
    # No NaN values in engineered features
    engineered_cols = [col for col in result.columns if col not in df.columns]
    assert not result[engineered_cols].isna().any().any()

if __name__ == "__main__":
    pytest.main([__file__]) 