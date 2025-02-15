from sklearn.pipeline import make_pipeline, Pipeline
import joblib
from pathlib import Path
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import pandas as pd

# Local imports
from .feature_engineer import PharmaFeatureEngineer
from .preprocessor import PharmaPreprocessor
from .metrics import pharma_metrics
from .model_utils import (
    create_optuna_study, 
    select_features,
    train_and_evaluate_models,
    get_default_models
)
try:
    from .explain_utils import ModelExplainer
except ImportError:
    print("Warning: ModelExplainer not available. Some features will be limited.")
    ModelExplainer = None

try:
    from .augment_utils import PharmaAugmenter
except ImportError:
    print("Warning: PharmaAugmenter not available. Data augmentation will be skipped.")
    PharmaAugmenter = None

try:
    from .benchmark_utils import ModelBenchmarker
except ImportError:
    print("Warning: ModelBenchmarker not available. Benchmarking will be skipped.")
    ModelBenchmarker = None

class SimpleTextProcessor:
    def __init__(self, max_features=500):
        self.max_features = max_features
        self.vocabulary_ = None
        
    def fit(self, X, y=None):
        # Create word frequency dictionary
        word_freq = {}
        for text in X:
            if pd.isna(text):
                continue
            words = str(text).lower().split()
            for word in words:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Select top words as features
        self.vocabulary_ = dict(
            sorted(word_freq.items(), 
                  key=lambda x: x[1], 
                  reverse=True)[:self.max_features]
        )
        return self
        
    def transform(self, X):
        if self.vocabulary_ is None:
            raise ValueError("Transformer must be fitted before transform")
        
        # Create feature matrix
        features = np.zeros((len(X), len(self.vocabulary_)))
        for i, text in enumerate(X):
            if pd.isna(text):
                continue
            words = str(text).lower().split()
            for j, word in enumerate(self.vocabulary_):
                features[i, j] = words.count(word)
        return features

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)

class ModelTrainer:
    """Handles model training and pipeline management"""
    
    def __init__(self, config):
        self.config = config
        self.models_dir = config.PROJECT_ROOT / 'models'
        self.models_dir.mkdir(exist_ok=True)
        self.pipeline = None
        self.is_fitted = False
    
    def create_stacking_model(self):
        """Create an advanced stacking model"""
        from xgboost import XGBRegressor
        from lightgbm import LGBMRegressor
        from sklearn.ensemble import StackingRegressor
        from sklearn.model_selection import TimeSeriesSplit
        
        # Base estimators with GPU acceleration if available
        try:
            xgb_params = {
                'tree_method': 'gpu_hist',
                'objective': 'reg:squarederror',
                'n_estimators': 2000,
                'learning_rate': 0.02
            }
        except Exception:
            xgb_params = {
                'objective': 'reg:squarederror',
                'n_estimators': 1000,
                'learning_rate': 0.05
            }
        
        try:
            lgbm_params = {
                'device': 'gpu',
                'num_leaves': 127,
                'min_data_in_leaf': 50
            }
        except Exception:
            lgbm_params = {
                'num_leaves': 63,
                'min_data_in_leaf': 30
            }
        
        # Create stacking model
        stack = StackingRegressor(
            estimators=[
                ('xgb', XGBRegressor(**xgb_params)),
                ('lgbm', LGBMRegressor(**lgbm_params))
            ],
            final_estimator=XGBRegressor(
                n_estimators=500,
                learning_rate=0.05
            ),
            cv=TimeSeriesSplit(n_splits=5),
            passthrough=True
        )
        
        return stack
    
    def build_pipeline(self):
        """Create the complete training pipeline"""
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import StandardScaler, OneHotEncoder
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        # Feature engineering
        feature_engineer = PharmaFeatureEngineer()
        
        # Create column transformer
        preprocessor = ColumnTransformer([
            ('text', TfidfVectorizer(max_features=500), 'Uses'),
            ('cat', OneHotEncoder(handle_unknown='ignore'), 
             ['Manufacturer', 'primary_ingredient']),
            ('num', StandardScaler(), 
             ['dosage_mg', 'num_ingredients', 'mfr_mean_review'])
        ])
        
        # Create stacking model
        stack = self.create_stacking_model()
        
        # Build pipeline
        self.pipeline = make_pipeline(
            feature_engineer,
            preprocessor,
            stack
        )
        
        return self.pipeline
    
    def train(self, X_train, y_train):
        """Train the pipeline"""
        if self.pipeline is None:
            self.build_pipeline()
        
        print("Training pipeline...")
        self.pipeline.fit(X_train, y_train)
        print("Training completed!")
    
    def save_pipeline(self, filename='grandmaster_pipeline.pkl'):
        """Save the trained pipeline"""
        if self.pipeline is None:
            raise ValueError("Pipeline not trained yet!")
        
        save_path = self.models_dir / filename
        joblib.dump(self.pipeline, save_path)
        print(f"Pipeline saved to: {save_path}")
    
    def prepare_pipeline(self):
        """Prepare the preprocessing pipeline"""
        try:
            # Create feature engineering pipeline
            feature_engineer = PharmaFeatureEngineer()
            
            # Create preprocessing pipeline
            preprocessor = ColumnTransformer([
                ('text', SimpleTextProcessor(max_features=500), 'Uses'),
                ('cat', OneHotEncoder(handle_unknown='ignore'), 
                 ['Manufacturer', 'main_ingredient']),
                ('num', StandardScaler(), ['dosage_mg', 'num_ingredients'])
            ])
            
            # Create full pipeline
            self.pipeline = Pipeline([
                ('feature_engineer', feature_engineer),
                ('preprocessor', preprocessor),
                ('model', XGBRegressor(
                    n_estimators=1000,
                    learning_rate=0.05,
                    random_state=42
                ))
            ])
            
            return self.pipeline
            
        except Exception as e:
            print(f"Error preparing pipeline: {e}")
            return None

    def fit(self, X, y):
        """Fit the pipeline"""
        if self.pipeline is None:
            self.prepare_pipeline()
        
        self.pipeline.fit(X, y)
        self.is_fitted = True
        return self

    def predict(self, X):
        """Make predictions"""
        if not self.is_fitted:
            raise ValueError("Pipeline is not fitted. Call fit() first.")
        return self.pipeline.predict(X)

    def load_pipeline(self):
        """Load trained pipeline or create new one"""
        try:
            model_path = self.models_dir / 'trained_pipeline.pkl'
            if model_path.exists():
                self.pipeline = joblib.load(model_path)
                self.is_fitted = True
            else:
                print("No trained model found. Creating new pipeline...")
                self.prepare_pipeline()
                
            return self.pipeline
            
        except Exception as e:
            print(f"Error loading pipeline: {e}")
            return None
    
    def evaluate(self, X_test, y_test):
        """Evaluate the pipeline"""
        if self.pipeline is None:
            raise ValueError("Pipeline not trained or loaded!")
        
        try:
            # Make predictions
            y_pred = self.pipeline.predict(X_test)
            
            # Calculate all metrics
            metrics = pharma_metrics.evaluate_predictions(y_test, y_pred)
            
            # Perform cross-validation
            cv_results = pharma_metrics.cross_validate(
                self.pipeline, X_test, y_test
            )
            
            if cv_results:
                metrics.update({
                    'cv_score_mean': cv_results['mean_score'],
                    'cv_score_std': cv_results['std_score']
                })
            
            return metrics
            
        except Exception as e:
            print(f"Error in evaluation: {e}")
            return None

    def train_model(self, X_train, X_test, y_train, y_test):
        """Train models with optimization and analysis"""
        try:
            # Augment training data if available
            if PharmaAugmenter is not None:
                augmenter = PharmaAugmenter(augmentation_factor=2)
                X_train_aug = augmenter.transform(X_train)
                y_train_aug = np.repeat(y_train, 2, axis=0)
            else:
                X_train_aug, y_train_aug = X_train, y_train
            
            # Train models
            results, best_model_name = train_and_evaluate_models(
                get_default_models(),
                X_train_aug, X_test, 
                y_train_aug, y_test
            )
            
            if results is None:
                return None, None
            
            # Add cross-validation scores
            for name, result in results.items():
                model = result['model']
                cv_results = pharma_metrics.cross_validate(
                    model, X_test, y_test
                )
                if cv_results:
                    result['cv_scores'] = cv_results
            
            # Setup model explainer if available
            if ModelExplainer is not None:
                best_model = results[best_model_name]['model']
                explainer = ModelExplainer(best_model, self.feature_names)
                
                # Analyze errors
                y_pred = best_model.predict(X_test)
                error_analysis = explainer.analyze_errors(X_test, y_test, y_pred)
                results[best_model_name]['error_analysis'] = error_analysis
            
            return results, best_model_name
            
        except Exception as e:
            print(f"Error in model training: {e}")
            return None, None

def train_and_save_model(config):
    """Complete training workflow with benchmarking"""
    try:
        # Load and preprocess data
        preprocessor = PharmaPreprocessor()
        data = preprocessor.process_data()
        
        if data is None:
            raise ValueError("Failed to process data")
        
        X_train, X_test = data['X_train'], data['X_test']
        y_train, y_test = data['y_train'], data['y_test']
        
        # Run benchmarks if available
        if ModelBenchmarker is not None:
            print("\nRunning model benchmarks...")
            benchmarker = ModelBenchmarker(X_train, X_test, y_train, y_test)
            benchmarker.run_benchmarks()
            
            print("\nBenchmark Results:")
            print(benchmarker.get_results_table())
            
            benchmarker.save_results(config.PROJECT_ROOT / 'reports' / 'benchmark_results.csv')
            
            # Use best model from benchmarks
            trainer = ModelTrainer(config)
            best_model_name = min(
                benchmarker.results, 
                key=lambda k: benchmarker.results[k]['MAE']
            )
            trainer.pipeline = benchmarker.results[best_model_name]['model']
            
        else:
            # Fallback to basic training
            trainer = ModelTrainer(config)
            trainer.prepare_pipeline()
            trainer.train(X_train, y_train)
        
        # Save model
        trainer.save_pipeline('trained_model.pkl')
        return trainer
        
    except Exception as e:
        print(f"Error in training workflow: {e}")
        return None

if __name__ == "__main__":
    from config import config
    train_and_save_model(config) 