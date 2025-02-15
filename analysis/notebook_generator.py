import nbformat as nbf
import pandas as pd
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go

class NotebookGenerator:
    """Generate analysis notebook from data"""
    
    def __init__(self, df, config):
        self.df = df
        self.config = config
        self.nb = nbf.v4.new_notebook()
        
    def generate(self):
        """Generate complete analysis notebook"""
        self._add_header()
        self._add_data_overview()
        self._add_feature_analysis()
        self._add_manufacturer_analysis()
        self._add_review_analysis()
        return self.nb
    
    def _add_header(self):
        """Add notebook header and imports"""
        self.nb.cells.append(nbf.v4.new_markdown_cell("""
        # 🏥 Pharmaceutical Data Analysis
        
        This notebook provides a comprehensive analysis of the pharmaceutical dataset.
        """))
        
        self.nb.cells.append(nbf.v4.new_code_cell("""
        import pandas as pd
        import numpy as np
        import plotly.express as px
        import plotly.graph_objects as go
        import seaborn as sns
        import matplotlib.pyplot as plt
        
        # Set plotting style
        plt.style.use('seaborn')
        """))
        
    def _add_data_overview(self):
        """Add data overview section"""
        # Data loading
        self.nb.cells.append(nbf.v4.new_markdown_cell("## Data Overview"))
        self.nb.cells.append(nbf.v4.new_code_cell(
            f"df = pd.read_csv('{self.config.DATA_PATH}')\n"
            "df.head()"
        ))
        
        # Basic statistics
        self.nb.cells.append(nbf.v4.new_code_cell("""
        print("Dataset Shape:", df.shape)
        print("\nColumn Types:")
        print(df.dtypes)
        print("\nMissing Values:")
        print(df.isnull().sum())
        """))
        
    def _add_feature_analysis(self):
        """Add feature engineering analysis"""
        self.nb.cells.append(nbf.v4.new_markdown_cell("""
        ## Feature Analysis
        
        This section explores the engineered features and their relationships.
        """))
        
        # Add manufacturer insights code
        self.nb.cells.append(nbf.v4.new_code_cell("""
        # Calculate manufacturer insights
        def calculate_mfr_insights(df):
            # Convert review percentage to numeric
            df['Excellent Review %'] = df['Excellent Review %'].apply(
                lambda x: float(str(x).rstrip('%')) if pd.notnull(x) else 0
            )
            
            # Calculate manufacturer statistics
            mfr_stats = df.groupby('Manufacturer').agg({
                'Excellent Review %': {
                    'mfr_avg_review': 'mean',
                    'mfr_review_std': 'std',
                    'mfr_review_count': 'count'
                },
                'Medicine Name': {
                    'mfr_total_medicines': 'count'
                }
            })
            
            # Flatten column names
            mfr_stats.columns = mfr_stats.columns.map('_'.join)
            mfr_stats = mfr_stats.reset_index()
            
            # Calculate market share
            total_medicines = float(mfr_stats['Medicine Name_mfr_total_medicines'].sum())
            mfr_stats['mfr_market_share'] = (
                mfr_stats['Medicine Name_mfr_total_medicines'].astype(float) / 
                total_medicines * 100
            )
            
            # Calculate ranking and performance
            mfr_stats['mfr_rank'] = (
                mfr_stats['Excellent Review %_mfr_avg_review']
                .rank(method='dense', ascending=False)
            )
            
            avg_review = float(df['Excellent Review %'].mean())
            mfr_stats['mfr_relative_performance'] = (
                mfr_stats['Excellent Review %_mfr_avg_review'] - avg_review
            )
            
            return mfr_stats
        
        # Calculate and display insights
        mfr_insights = calculate_mfr_insights(df)
        print("Manufacturer Insights Preview:")
        display(mfr_insights.head())
        
        # Visualize manufacturer performance
        fig = px.scatter(
            mfr_insights,
            x='mfr_market_share',
            y='Excellent Review %_mfr_avg_review',
            size='Medicine Name_mfr_total_medicines',
            color='mfr_relative_performance',
            hover_data=['Manufacturer'],
            title='Manufacturer Performance Matrix',
            labels={
                'mfr_market_share': 'Market Share (%)',
                'Excellent Review %_mfr_avg_review': 'Average Review Score',
                'Medicine Name_mfr_total_medicines': 'Number of Products',
                'mfr_relative_performance': 'Relative Performance'
            }
        )
        fig.show()
        """))
        
    def _add_manufacturer_analysis(self):
        """Add manufacturer analysis"""
        self.nb.cells.append(nbf.v4.new_markdown_cell("""
        ## Manufacturer Analysis
        
        This section analyzes manufacturer performance and market presence.
        """))
        
        # Manufacturer statistics
        self.nb.cells.append(nbf.v4.new_code_cell("""
        # Calculate manufacturer metrics
        mfr_stats = df.groupby('Manufacturer').agg({
            'Medicine Name': 'count',
            'Excellent Review %': ['mean', 'std', 'count']
        }).round(2)
        
        # Flatten column names
        mfr_stats.columns = ['total_medicines', 'avg_review', 'review_std', 'review_count']
        mfr_stats = mfr_stats.reset_index()
        
        # Calculate market share
        mfr_stats['market_share'] = (mfr_stats['total_medicines'] / mfr_stats['total_medicines'].sum() * 100).round(2)
        
        # Calculate manufacturer ranking
        mfr_stats['rank'] = mfr_stats['avg_review'].rank(method='dense', ascending=False)
        
        # Display top manufacturers
        print("Top 10 Manufacturers by Market Share:")
        display(mfr_stats.nlargest(10, 'market_share'))
        """))
        
        # Market share visualization
        self.nb.cells.append(nbf.v4.new_code_cell("""
        # Create market share treemap
        fig = px.treemap(
            mfr_stats.nlargest(15, 'market_share'),
            path=['Manufacturer'],
            values='market_share',
            color='avg_review',
            title='Top 15 Manufacturers Market Share and Average Reviews',
            color_continuous_scale='RdYlGn'
        )
        fig.show()
        """))
        
        # Performance analysis
        self.nb.cells.append(nbf.v4.new_code_cell("""
        # Create performance scatter plot
        fig = px.scatter(
            mfr_stats[mfr_stats['total_medicines'] > 5],  # Filter out small manufacturers
            x='market_share',
            y='avg_review',
            size='total_medicines',
            color='review_std',
            hover_data=['Manufacturer', 'review_count'],
            title='Manufacturer Performance Analysis',
            labels={
                'market_share': 'Market Share (%)',
                'avg_review': 'Average Excellence Rating (%)',
                'review_std': 'Review Consistency',
                'total_medicines': 'Number of Medicines'
            }
        )
        fig.show()
        """))
        
        # Review consistency
        self.nb.cells.append(nbf.v4.new_code_cell("""
        # Analyze review consistency
        top_mfrs = mfr_stats.nlargest(10, 'total_medicines')['Manufacturer'].tolist()
        mfr_reviews = df[df['Manufacturer'].isin(top_mfrs)]
        
        fig = px.box(
            mfr_reviews,
            x='Manufacturer',
            y='Excellent Review %',
            title='Review Distribution for Top Manufacturers',
            points='outliers'
        )
        fig.update_xaxes(tickangle=45)
        fig.show()
        """))
    
    def _add_review_analysis(self):
        """Add review analysis"""
        self.nb.cells.append(nbf.v4.new_markdown_cell("## Review Analysis"))
        
        # Review distribution
        self.nb.cells.append(nbf.v4.new_code_cell("""
        review_cols = ['Excellent Review %', 'Average Review %', 'Poor Review %']
        
        fig = go.Figure()
        for col in review_cols:
            fig.add_trace(go.Box(y=df[col], name=col))
            
        fig.update_layout(title='Distribution of Review Scores',
                         yaxis_title='Percentage')
        fig.show()
        """))

def save_notebook(df, config):
    """Generate and save analysis notebook"""
    try:
        # Create notebook
        generator = NotebookGenerator(df, config)
        nb = generator.generate()
        
        # Create notebooks directory if it doesn't exist
        notebook_dir = config.PROJECT_ROOT / 'notebooks'
        notebook_dir.mkdir(exist_ok=True)
        
        # Save notebook
        notebook_path = notebook_dir / 'Pharma_Analysis_EDA.ipynb'
        with open(notebook_path, 'w', encoding='utf-8') as f:
            nbf.write(nb, f)
            
        return notebook_path
        
    except Exception as e:
        print(f"Error generating notebook: {e}")
        return None 