import os
import json
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ydata_profiling import ProfileReport
from config import config

class DataAnalyzer:
    """Class for handling data analysis and visualization"""
    
    def __init__(self):
        self.data_dir = config.DATA_DIR
        self.reports_dir = config.PROJECT_ROOT / 'reports'
        self.reports_dir.mkdir(exist_ok=True)
    
    def load_data(self):
        """Load data from the project's data directory"""
        data_file = self.data_dir / 'Medicine_Details.csv'
        
        if not data_file.exists():
            raise FileNotFoundError(
                f"Medicine_Details.csv not found in {self.data_dir}. "
                "Please ensure the data file is present in the data directory."
            )
        
        return pd.read_csv(data_file)
    
    def generate_profile_report(self, df, title="Pharmaceutical Analysis Report"):
        """Generate a pandas profiling report"""
        try:
            from ydata_profiling import ProfileReport
            
            # Create profile report with custom config
            profile = ProfileReport(
                df,
                title=title,
                explorative=True,
                minimal=True,  # Faster generation
                correlations={
                    "pearson": {"calculate": True},
                    "spearman": {"calculate": True},
                    "kendall": {"calculate": False},
                    "phi_k": {"calculate": False},
                    "cramers": {"calculate": False},
                }
            )
            
            # Save report
            report_path = self.reports_dir / 'profile_report.html'
            profile.to_file(report_path)
            
            return True
            
        except Exception as e:
            print(f"Error generating profile report: {e}")
            raise
    
    def visualize_data(self, df):
        """Create and save data visualizations"""
        # Distribution of numerical features
        numeric_columns = df.select_dtypes(include=["number"])
        
        if not numeric_columns.empty:
            # Create distributions plot
            fig_dist = plt.figure(figsize=(12, 10))
            numeric_columns.hist(bins=30)
            plt.suptitle("Distribution of Numerical Features", fontsize=16)
            plt.tight_layout()
            
            # Save distributions plot
            dist_path = self.reports_dir / 'distributions.png'
            fig_dist.savefig(dist_path)
            plt.close(fig_dist)
            
            # Create correlation matrix
            fig_corr = plt.figure(figsize=(10, 8))
            sns.heatmap(numeric_columns.corr(), annot=True, 
                       cmap="coolwarm", fmt=".2f", cbar=True)
            plt.title("Correlation Matrix", fontsize=16)
            plt.tight_layout()
            
            # Save correlation matrix
            corr_path = self.reports_dir / 'correlation_matrix.png'
            fig_corr.savefig(corr_path)
            plt.close(fig_corr)
            
            print(f"Visualizations saved to: {self.reports_dir}")
        else:
            print("No numerical columns found for visualization.")
    
    def analyze_data(self):
        """Main analysis workflow"""
        try:
            # Load the dataset
            df = self.load_data()
            
            # Basic analysis
            print("\nDataset Overview:")
            print(df.head())
            
            print("\nBasic Statistics:")
            print(df.describe())
            
            print("\nMissing Values:")
            print(df.isnull().sum())
            
            # Generate profiling report
            self.generate_profile_report(df, "Medicine Details Analysis")
            
            # Create visualizations
            self.visualize_data(df)
            
            return df
            
        except Exception as e:
            print(f"An error occurred during analysis: {e}")
            return None

def run_analysis():
    """Run the complete analysis pipeline"""
    analyzer = DataAnalyzer()
    return analyzer.analyze_data()

if __name__ == "__main__":
    run_analysis() 