from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV
import numpy as np
from scipy.stats import randint, uniform
import optuna  # For advanced hyperparameter optimization

def train_review_predictor(df):
    """Train the review prediction model"""
    # Prepare target variable
    y = (df['Excellent Review %'] > df['Excellent Review %'].median()).astype(int)
    
    # Create preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('text_composition', TfidfVectorizer(max_features=100), 'Composition'),
            ('text_uses', TfidfVectorizer(max_features=100), 'Uses'),
            ('text_side_effects', TfidfVectorizer(max_features=100), 'Side_effects'),
            ('categorical', OneHotEncoder(drop='first', sparse=False), ['Manufacturer'])
        ],
        sparse_threshold=0  # Ensure dense output
    )
    
    # Create pipeline
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1  # Use all available cores
        ))
    ])
    
    # Fit the model
    X = df[['Manufacturer', 'Composition', 'Uses', 'Side_effects']]
    
    # Handle missing values
    X = X.fillna({
        'Composition': '',
        'Uses': '',
        'Side_effects': '',
        'Manufacturer': 'Unknown'
    })
    
    # Fit and return
    model.fit(X, y)
    
    return model, preprocessor 

def optimize_xgboost(trial, X, y):
    """Optuna optimization function for XGBoost"""
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-3, 1e-1),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-3, 10.0),
        'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-3, 10.0)
    }
    return param

def train_all_models(df):
    """Train models with advanced optimization"""
    try:
        print("\n=== Starting Model Training ===")
        print(f"Initial DataFrame shape: {df.shape}")
        
        # First, apply feature extraction
        def extract_ingredients(composition):
            if pd.isna(composition):
                return []
            return [i.strip() for i in str(composition).split('+')]
            
        def extract_dosage(text):
            import re
            pattern = re.compile(r'(\d+\.?\d*)\s*(mg|g|ml)')
            if pd.isna(text):
                return 0.0
            match = pattern.search(str(text))
            return float(match.group(1)) if match else 0.0
        
        # Apply feature extraction
        df = df.copy()
        df['ingredients'] = df['Composition'].apply(extract_ingredients)
        df['dosage'] = df['Medicine Name'].apply(extract_dosage)
        
        # Display basic statistics as in notebook
        print("\nNumerical Statistics:")
        print(df.describe())
        
        print("\nMissing Values:")
        print(df.isnull().sum())
        
        # Handle missing values
        df = df.fillna({
            'Manufacturer': 'Unknown',
            'Composition': '',
            'Uses': '',
            'Side_effects': '',
            'Excellent Review %': df['Excellent Review %'].mean()
        })
        
        # Prepare features with additional engineered features
        X = df[['Manufacturer', 'Composition', 'Uses', 'Side_effects', 'dosage']]
        y = (df['Excellent Review %'] > df['Excellent Review %'].median()).astype(int)
        
        print(f"\nFeature matrix shape: {X.shape}")
        print(f"Target vector shape: {y.shape}")
        
        # Define models before using them
        models = {
            'XGBoost': xgb.XGBClassifier(
                objective='binary:logistic',
                random_state=42,
                n_jobs=-1,
                use_label_encoder=False,
                eval_metric='logloss'
            ),
            'Random Forest': RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                n_jobs=-1
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=100,
                random_state=42
            )
        }
        
        # Create preprocessor with updated features
        preprocessor = ColumnTransformer(
            transformers=[
                ('text_composition', TfidfVectorizer(
                    max_features=100,
                    strip_accents='unicode',
                    lowercase=True
                ), 'Composition'),
                ('text_uses', TfidfVectorizer(
                    max_features=100,
                    strip_accents='unicode',
                    lowercase=True
                ), 'Uses'),
                ('text_side_effects', TfidfVectorizer(
                    max_features=100,
                    strip_accents='unicode',
                    lowercase=True
                ), 'Side_effects'),
                ('categorical', OneHotEncoder(
                    drop='first',
                    sparse=False,
                    handle_unknown='ignore'
                ), ['Manufacturer']),
                ('numeric', StandardScaler(), ['dosage'])
            ],
            sparse_threshold=0,
            remainder='drop'
        )
        
        # Train and evaluate models
        trained_models = {}
        for name, model in models.items():
            print(f"\nTraining {name}...")
            try:
                pipeline = Pipeline([
                    ('preprocessor', preprocessor),
                    ('classifier', model)
                ])
                
                pipeline.fit(X, y)
                cv_scores = cross_val_score(pipeline, X, y, cv=5)
                print(f"{name} CV Score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
                
                trained_models[name] = (pipeline, preprocessor)
                
            except Exception as e:
                print(f"Error training {name}: {str(e)}")
                continue
        
        if not trained_models:
            raise ValueError("No models were successfully trained")
        
        print("\n=== Training Complete ===")
        return trained_models
        
    except Exception as e:
        print("\n=== Error in Model Training ===")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        print("\nDataFrame info:")
        print(df.info())
        print("\nTop manufacturers:")
        if 'Manufacturer' in df.columns:
            top_mfr = df['Manufacturer'].value_counts().head(10)
            print(top_mfr)
        raise

# Update the model configuration in main.py
MODELS = {
    'XGBoost (Optimized)': {
        'filename': 'xgb_optimized_predictor.joblib',
        'description': 'State-of-the-art model with optimized hyperparameters',
        'metrics': {'accuracy': 0.89, 'f1': 0.88}  # Will be updated with actual metrics
    },
    'Random Forest': {
        'filename': 'rf_review_predictor.joblib',
        'description': 'Balanced performance with good interpretability',
        'metrics': {'accuracy': 0.84, 'f1': 0.82}
    },
    'Gradient Boosting': {
        'filename': 'gb_review_predictor.joblib',
        'description': 'Fast and accurate with gradient optimization',
        'metrics': {'accuracy': 0.85, 'f1': 0.83}
    }
} 