try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    print("Warning: SHAP not available. Model explanations will be limited.")
    SHAP_AVAILABLE = False

import pandas as pd
import numpy as np
import plotly.graph_objects as go

class ModelExplainer:
    """Handles model explanation and error analysis"""
    
    def __init__(self, model, feature_names=None):
        self.model = model
        self.feature_names = feature_names
        self.explainer = None
        
    def setup_explainer(self):
        """Initialize the SHAP explainer"""
        if not SHAP_AVAILABLE:
            print("SHAP not available. Cannot create explainer.")
            return False
            
        try:
            self.explainer = shap.TreeExplainer(self.model)
            return True
        except Exception as e:
            print(f"Error setting up explainer: {e}")
            return False
            
    def explain_prediction(self, sample):
        """Generate SHAP explanation for a single prediction"""
        if not SHAP_AVAILABLE:
            return {
                'warning': 'SHAP not available. Install SHAP for detailed explanations.',
                'feature_importance': self._basic_feature_importance(sample)
            }
            
        if self.explainer is None:
            self.setup_explainer()
            
        try:
            shap_values = self.explainer.shap_values(sample)
            
            if isinstance(shap_values, list):
                shap_values = shap_values[0]  # For multi-output models
                
            return {
                'shap_values': shap_values,
                'base_value': self.explainer.expected_value
            }
        except Exception as e:
            print(f"Error generating explanation: {e}")
            return None
            
    def _basic_feature_importance(self, sample):
        """Provide basic feature importance when SHAP is not available"""
        try:
            if hasattr(self.model, 'feature_importances_'):
                return {
                    'feature_importance': self.model.feature_importances_,
                    'feature_names': self.feature_names
                }
            return None
        except Exception as e:
            print(f"Error calculating basic feature importance: {e}")
            return None
            
    def analyze_errors(self, X_test, y_true, y_pred):
        """Analyze prediction errors"""
        try:
            error_df = pd.DataFrame({
                'True_Value': y_true,
                'Predicted': y_pred,
                'Error': np.abs(y_true - y_pred)
            })
            
            if self.feature_names is not None:
                for i, name in enumerate(self.feature_names):
                    error_df[name] = X_test[:, i]
            
            return error_df.sort_values('Error', ascending=False)
            
        except Exception as e:
            print(f"Error analyzing predictions: {e}")
            return None 