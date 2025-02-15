import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import joblib
from pathlib import Path

def create_and_save_sample_model():
    """Create and save a sample model for the medicine review predictor"""
    try:
        # Create models directory if it doesn't exist
        models_dir = Path('models')
        models_dir.mkdir(exist_ok=True)
        
        # Load data
        data_path = Path('data/Medicine_Details.csv')
        df = pd.read_csv(data_path)
        
        # Create target variable (example: excellent review > 80%)
        df['high_rating'] = (df['Excellent Review %'] > 80).astype(int)
        
        # Split features and target
        X = df[['Manufacturer', 'Composition', 'Uses', 'Side_effects']]
        y = df['high_rating']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Create preprocessing pipeline
        preprocessor = ColumnTransformer(
            transformers=[
                ('text_composition', TfidfVectorizer(max_features=1000), 'Composition'),
                ('text_uses', TfidfVectorizer(max_features=1000), 'Uses'),
                ('text_side_effects', TfidfVectorizer(max_features=1000), 'Side_effects'),
                ('text_manufacturer', TfidfVectorizer(max_features=100), 'Manufacturer')
            ],
            remainder='drop'
        )
        
        # Create model pipeline
        model = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', XGBClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            ))
        ])
        
        # Fit model
        model.fit(X_train, y_train)
        
        # Save model and preprocessor
        joblib.dump(model, models_dir / 'xgb_review_predictor.joblib')
        
        # Create and save other models
        models = {
            'Random Forest': Pipeline([
                ('preprocessor', preprocessor),
                ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
            ]),
            'Gradient Boosting': Pipeline([
                ('preprocessor', preprocessor),
                ('classifier', GradientBoostingClassifier(n_estimators=100, random_state=42))
            ])
        }
        
        for name, model in models.items():
            model.fit(X_train, y_train)
            joblib.dump(model, models_dir / f'{name.lower().replace(" ", "_")}_review_predictor.joblib')
        
        print("✅ Models created and saved successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error creating models: {str(e)}")
        return False

if __name__ == "__main__":
    create_and_save_sample_model() 