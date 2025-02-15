import nbformat as nbf
import pandas as pd
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go

def create_notebook(df, config):
    """Create a Jupyter notebook with pharmaceutical data analysis"""
    nb = nbf.v4.new_notebook()
    
    # Add title and imports
    nb.cells.append(nbf.v4.new_markdown_cell("""
    # 🏥 Pharmaceutical Data Analysis
    
    This notebook provides a comprehensive analysis of the pharmaceutical dataset.
    """))
    
    nb.cells.append(nbf.v4.new_code_cell("""
    import pandas as pd
    import numpy as np
    import plotly.express as px
    import plotly.graph_objects as go
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    # Set plotting style
    plt.style.use('seaborn')
    """))
    
    # Data loading
    nb.cells.append(nbf.v4.new_markdown_cell("## Data Loading and Overview"))
    nb.cells.append(nbf.v4.new_code_cell(
        f"df = pd.read_csv('{config.DATA_PATH}')\n"
        "print('Dataset Shape:', df.shape)\n"
        "df.head()"
    ))
    
    # Basic statistics
    nb.cells.append(nbf.v4.new_markdown_cell("## Basic Statistics"))
    nb.cells.append(nbf.v4.new_code_cell("""
    # Display basic statistics
    print("\\nNumerical Statistics:")
    display(df.describe())
    
    print("\\nMissing Values:")
    display(df.isnull().sum())
    """))
    
    # Manufacturer Analysis
    nb.cells.append(nbf.v4.new_markdown_cell("## Manufacturer Analysis"))
    nb.cells.append(nbf.v4.new_code_cell("""
    # Top manufacturers
    top_mfr = df['Manufacturer'].value_counts().head(10)
    
    fig = px.bar(
        x=top_mfr.index,
        y=top_mfr.values,
        title='Top 10 Manufacturers by Number of Medicines',
        labels={'x': 'Manufacturer', 'y': 'Number of Medicines'}
    )
    fig.show()
    """))
    
    # Review Analysis
    nb.cells.append(nbf.v4.new_markdown_cell("## Review Analysis"))
    nb.cells.append(nbf.v4.new_code_cell("""
    # Review distribution
    review_cols = ['Excellent Review %', 'Average Review %', 'Poor Review %']
    
    fig = go.Figure()
    for col in review_cols:
        fig.add_trace(go.Box(y=df[col], name=col))
    
    fig.update_layout(
        title='Distribution of Review Scores',
        yaxis_title='Percentage'
    )
    fig.show()
    
    # Top rated medicines
    print("\\nTop 10 Medicines by Excellent Reviews:")
    display(df.nlargest(10, 'Excellent Review %')[
        ['Medicine Name', 'Manufacturer', 'Excellent Review %']
    ])
    """))
    
    # Composition Analysis
    nb.cells.append(nbf.v4.new_markdown_cell("## Composition Analysis"))
    nb.cells.append(nbf.v4.new_code_cell("""
    def extract_ingredients(composition):
        if pd.isna(composition):
            return []
        return [i.strip() for i in str(composition).split('+')]
    
    # Get all ingredients
    all_ingredients = []
    for comp in df['Composition']:
        all_ingredients.extend(extract_ingredients(comp))
    
    # Count ingredients
    ingredient_counts = pd.Series(all_ingredients).value_counts()
    
    # Plot top ingredients
    fig = px.bar(
        x=ingredient_counts.head(15).values,
        y=ingredient_counts.head(15).index,
        orientation='h',
        title='Top 15 Most Common Ingredients'
    )
    fig.show()
    """))
    
    # Feature Engineering
    nb.cells.append(nbf.v4.new_markdown_cell("## Feature Engineering"))
    nb.cells.append(nbf.v4.new_code_cell("""
    # Extract dosage from medicine names
    def extract_dosage(text):
        import re
        pattern = re.compile(r'(\d+\.?\d*)\s*(mg|g|ml)')
        if pd.isna(text):
            return 0.0
        match = pattern.search(str(text))
        return float(match.group(1)) if match else 0.0
    
    # Create engineered features
    df['dosage_mg'] = df['Medicine Name'].apply(extract_dosage)
    df['num_ingredients'] = df['Composition'].str.count('\+') + 1
    df['uses_word_count'] = df['Uses'].str.split().str.len()
    df['side_effects_count'] = df['Side_effects'].str.split().str.len()
    
    # Show distributions
    numeric_features = ['dosage_mg', 'num_ingredients', 'uses_word_count', 'side_effects_count']
    
    fig = px.box(
        df[numeric_features],
        title='Distribution of Engineered Features'
    )
    fig.show()
    """))
    
    return nb

def save_notebook(df, config):
    """Generate and save the analysis notebook"""
    try:
        # Create notebook
        nb = create_notebook(df, config)
        
        # Create notebooks directory if it doesn't exist
        notebook_dir = config.PROJECT_ROOT / 'notebooks'
        notebook_dir.mkdir(parents=True, exist_ok=True)
        
        # Save notebook
        notebook_path = notebook_dir / 'Pharma_Analysis_EDA.ipynb'
        with open(notebook_path, 'w', encoding='utf-8') as f:
            nbf.write(nb, f)
        
        print(f"Notebook saved to: {notebook_path}")
        return notebook_path
        
    except Exception as e:
        print(f"Error generating notebook: {e}")
        return None

if __name__ == "__main__":
    from config import config
    import pandas as pd
    
    # Test notebook generation
    df = pd.read_csv(config.DATA_PATH)
    save_notebook(df, config)