from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import mean_absolute_error, make_scorer
import numpy as np

class PharmaMetrics:
    """Custom metrics for pharmaceutical review prediction"""
    
    def __init__(self, n_splits=5, random_state=42):
        self.n_splits = n_splits
        self.random_state = random_state
        self.cv = StratifiedKFold(
            n_splits=n_splits, 
            shuffle=True, 
            random_state=random_state
        )
    
    @staticmethod
    def pharma_safety_score(y_true, y_pred):
        """
        Calculate safety-weighted score that penalizes:
        - Underpredicting low-reviewed drugs (< 35%)
        - Overpredicting high-reviewed drugs (> 70%)
        - Large errors in medium-range reviews
        """
        errors = np.abs(y_true - y_pred)
        base_mae = np.mean(errors)
        
        # Safety penalties
        low_mask = y_true < 35
        high_mask = y_true > 70
        
        under_prediction_penalty = np.mean(
            (y_true[low_mask] - y_pred[low_mask]) > 10
        ) if any(low_mask) else 0
        
        over_prediction_penalty = np.mean(
            (y_pred[high_mask] - y_true[high_mask]) > 10
        ) if any(high_mask) else 0
        
        # Combine into final score (0 to 1, higher is better)
        safety_score = 1 - (
            0.4 * base_mae/100 +  # Base accuracy
            0.3 * under_prediction_penalty +  # Low review penalty
            0.3 * over_prediction_penalty  # High review penalty
        )
        
        return max(0, min(1, safety_score))  # Clip to [0,1]
    
    @staticmethod
    def review_accuracy(y_true, y_pred):
        """Calculate review prediction accuracy (0 to 1)"""
        return 1 - np.mean(np.abs(y_true - y_pred)/100)
    
    def get_metrics(self):
        """Get dictionary of all available metrics"""
        return {
            'MAE': mean_absolute_error,
            'Pharma Safety Score': self.pharma_safety_score,
            'Review Accuracy': self.review_accuracy
        }
    
    def evaluate_predictions(self, y_true, y_pred):
        """Evaluate predictions using all metrics"""
        metrics = self.get_metrics()
        return {
            name: metric(y_true, y_pred) 
            for name, metric in metrics.items()
        }
    
    def cross_validate(self, model, X, y, cv=5):
        """Perform cross-validation with safety score"""
        try:
            cv_scores = cross_val_score(
                model,
                X,
                y,
                scoring=make_scorer(self.pharma_safety_score),
                cv=StratifiedKFold(cv)
            )
            
            return {
                'mean_score': np.mean(cv_scores),
                'std_score': np.std(cv_scores),
                'all_scores': cv_scores
            }
            
        except Exception as e:
            print(f"Error in cross-validation: {e}")
            return None

# Create default instance
pharma_metrics = PharmaMetrics() 