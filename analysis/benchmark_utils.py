import time
import pandas as pd
import numpy as np

try:
    from tabulate import tabulate
    TABULATE_AVAILABLE = True
except ImportError:
    print("Warning: tabulate not available. Using basic table formatting.")
    TABULATE_AVAILABLE = False

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import StackingRegressor
from .metrics import pharma_metrics

class ModelBenchmarker:
    """Utility for benchmarking different models"""
    
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.results = {}
        
    def benchmark_model(self, model, name):
        """Benchmark a single model"""
        try:
            # Training time
            train_start = time.time()
            model.fit(self.X_train, self.y_train)
            train_time = time.time() - train_start
            
            # Inference time (average over multiple runs)
            n_runs = 100
            inference_times = []
            for _ in range(n_runs):
                start = time.time()
                model.predict(self.X_test[:100])  # Test batch inference
                inference_times.append((time.time() - start) * 1000)  # Convert to ms
            
            inference_speed = np.mean(inference_times)
            
            # Predictions and metrics
            y_pred = model.predict(self.X_test)
            metrics = pharma_metrics.evaluate_predictions(self.y_test, y_pred)
            
            # Store results
            self.results[name] = {
                'model': model,
                'MAE': metrics['MAE'],
                'Safety Score': metrics['Pharma Safety Score'],
                'Review Accuracy': metrics['Review Accuracy'],
                'Training Time (s)': train_time,
                'Inference Speed (ms)': inference_speed
            }
            
        except Exception as e:
            print(f"Error benchmarking {name}: {e}")
    
    def run_benchmarks(self):
        """Run benchmarks on all models"""
        # Base XGBoost
        xgb = XGBRegressor(
            n_estimators=1000,
            learning_rate=0.05,
            random_state=42
        )
        self.benchmark_model(xgb, 'XGBoost')
        
        # Base LightGBM
        lgb = LGBMRegressor(
            n_estimators=1000,
            learning_rate=0.05,
            random_state=42
        )
        self.benchmark_model(lgb, 'LightGBM')
        
        # Stacking Model
        stack = StackingRegressor(
            estimators=[
                ('xgb', XGBRegressor(n_estimators=500)),
                ('lgb', LGBMRegressor(n_estimators=500))
            ],
            final_estimator=XGBRegressor(n_estimators=200),
            cv=5
        )
        self.benchmark_model(stack, 'Stacking')
    
    def get_results_table(self):
        """Generate formatted results table"""
        if not self.results:
            return "No benchmark results available"
            
        # Create DataFrame
        df = pd.DataFrame.from_dict(self.results, orient='index')
        df = df.drop('model', axis=1)  # Remove model object
        
        # Round numeric columns
        df = df.round({
            'MAE': 3,
            'Safety Score': 3,
            'Review Accuracy': 3,
            'Training Time (s)': 2,
            'Inference Speed (ms)': 2
        })
        
        if TABULATE_AVAILABLE:
            # Format as nice table with tabulate
            table = tabulate(
                df, 
                headers='keys', 
                tablefmt='pipe', 
                floatfmt='.3f'
            )
        else:
            # Basic string formatting
            table = "Model Benchmark Results:\n"
            table += "-" * 80 + "\n"
            table += df.to_string()
            
        return table
    
    def save_results(self, filename='benchmark_results.csv'):
        """Save benchmark results to file"""
        if self.results:
            df = pd.DataFrame.from_dict(self.results, orient='index')
            df = df.drop('model', axis=1)
            df.to_csv(filename)
            print(f"Benchmark results saved to {filename}") 