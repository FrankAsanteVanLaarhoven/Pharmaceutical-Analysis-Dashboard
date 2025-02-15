try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    print("Warning: Optuna not available. Using default hyperparameters.")
    OPTUNA_AVAILABLE = False

import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

def create_optuna_study(X, y, custom_scorer, n_trials=100):
    """Create and run Optuna study for hyperparameter optimization"""
    if not OPTUNA_AVAILABLE:
        # Return default parameters if Optuna is not available
        return {
            'n_estimators': 1000,
            'learning_rate': 0.05,
            'max_depth': 6
        }
        
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 2000),
            'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 12)
        }
        model = XGBRegressor(**params)
        return cross_val_score(model, X, y, scoring=custom_scorer).mean()
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    return study.best_params

def select_features(X, y, estimator, custom_scorer):
    """Perform recursive feature elimination with cross-validation"""
    selector = RFECV(
        estimator=estimator,
        step=10,
        cv=3,
        scoring=custom_scorer
    )
    selector.fit(X, y)
    return selector

def train_and_evaluate_models(models, X_train, X_test, y_train, y_test):
    """Train and evaluate multiple regression models"""
    results = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            results[name] = {
                'model': model,
                'MSE': mse, 
                'R2': r2
            }
            
            print(f"{name}: MSE = {mse:.4f}, R2 = {r2:.4f}")
            
        except Exception as e:
            print(f"Error training {name}: {e}")
    
    best_model_name = min(results, key=lambda k: results[k]['MSE'])
    return results, best_model_name

def get_default_models(random_state=42):
    """Get dictionary of default model instances"""
    return {
        'RandomForest': RandomForestRegressor(random_state=random_state),
        'XGBoost': XGBRegressor(random_state=random_state),
        'LightGBM': LGBMRegressor(random_state=random_state),
        'CatBoost': CatBoostRegressor(random_state=random_state, verbose=0)
    } 