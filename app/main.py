import os
os.environ['MPLBACKEND'] = 'Agg'  # Must be before importing matplotlib
os.environ['DISPLAY'] = ':0'  # For macOS/Linux

# Import config first
from config import config
from config.plotting_config import PLOT_CONFIG

# Then other imports
import matplotlib
matplotlib.use('Agg', force=True)
import matplotlib.pyplot as plt
import seaborn as sns

import streamlit as st
# Move set_page_config to the top, before any other Streamlit commands
st.set_page_config(
    page_title="Pharmaceutical Analysis Dashboard",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': 'World-Leading Pharmaceutical Analysis Dashboard'
    }
)

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import joblib
from PIL import Image
import requests
from io import BytesIO
import streamlit.components.v1 as components
from pathlib import Path
import os
from analysis.data_analyzer import DataAnalyzer
from analysis.feature_engineer import PharmaFeatureEngineer, create_feature_pipeline
from analysis.preprocessor import PharmaPreprocessor, run_preprocessing
from analysis.metrics import pharma_metrics
from analysis.train_pipeline import ModelTrainer
from ydata_profiling import ProfileReport
from analysis.notebook_generator import save_notebook
from save_notebook import save_notebook
import nbformat as nbf
import base64
import webbrowser
import tempfile
from nbconvert import HTMLExporter

# Add this import at the top of main.py with other imports
from analysis.medicine_helpers import get_similar_medicines, analyze_compositions, analyze_usage_patterns

# Add this near the top of the file, after imports and before page definitions
def get_notebook_download_link():
    """Generate a downloadable notebook with analysis"""
    try:
        # Use existing notebooks directory
        notebooks_dir = Path('/Users/frankvanlaarhoven/MEAP/notebooks')
        notebook_path = notebooks_dir / 'Pharma_Analysis.ipynb'
        
        # If notebook doesn't exist, create it
        if not notebook_path.exists():
            from analysis.create_notebook import create_analysis_notebook
            notebook_path = create_analysis_notebook()
        
        # Read the notebook
        with open(notebook_path, "r", encoding="utf-8") as f:
            notebook_content = f.read()
            
        # Create base64 encoded version of notebook
        b64 = base64.b64encode(notebook_content.encode()).decode()
        
        # Create download link using Streamlit's download_button
        return st.download_button(
            label="📥 Download Analysis Notebook",
            data=notebook_content,
            file_name="Pharma_Analysis.ipynb",
            mime="application/x-ipynb+json",
            key="notebook_download",
            help="Click to download the Jupyter notebook",
            use_container_width=True
        )
        
    except Exception as e:
        print(f"Error accessing notebook: {str(e)}")
        return f"Error generating notebook: {str(e)}"

def display_metrics(df):
    """Display key metrics about the dataset"""
    st.markdown("### 📊 Key Metrics")
    
    metrics_container = st.container()
    with metrics_container:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Total Medicines",
                f"{len(df):,}",
                help="Total number of unique medicines in database"
            )
        
        with col2:
            if 'Manufacturer' in df.columns:
                mfr_count = df['Manufacturer'].nunique()
                st.metric(
                    "Manufacturers",
                    f"{mfr_count:,}",
                    help="Number of unique pharmaceutical manufacturers"
                )
        
        with col3:
            if 'Excellent Review %' in df.columns:
                avg_review = df['Excellent Review %'].mean()
                st.metric(
                    "Avg Excellence",
                    f"{avg_review:.1f}%",
                    help="Average excellent review percentage"
                )
        
        with col4:
            if 'Excellent Review %' in df.columns:
                high_rated = len(df[df['Excellent Review %'] > 80])
                st.metric(
                    "High Rated",
                    f"{high_rated:,}",
                    help="Medicines with >80% excellent reviews"
                )

# Apply plotting configurations
plt.style.use('default')  # Use default style
plt.rcParams.update(PLOT_CONFIG['matplotlib']['rcParams'])

# Custom CSS for better layout
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stTitle {
        font-size: 2.5rem !important;
        font-weight: 600 !important;
        padding-bottom: 1.5rem;
        color: #1f1f1f;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 1px 2px rgba(0,0,0,0.1);
    }
    .stMetric:hover {
        transform: translateY(-2px);
        transition: all 0.2s ease;
    }
    .stHeader {
        font-size: 1.8rem !important;
        font-weight: 500 !important;
        color: #1f1f1f;
        margin-bottom: 1rem;
    }
    .stSubheader {
        font-size: 1.4rem !important;
        font-weight: 500 !important;
        color: #2c3e50;
        margin: 1rem 0;
    }
    .stDataFrame {
        border: 1px solid #e0e0e0;
        border-radius: 0.5rem;
        overflow: hidden;
    }
    .stExpander {
        border: 1px solid #e0e0e0;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }

    /* New styles for Key Metrics */
    .metric-card {
        background-color: #1e2130;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 0.5rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
    }
    
    .metric-label {
        color: #a3a8b8;
        font-size: 0.875rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.025em;
        margin-bottom: 0.5rem;
    }
    
    .metric-value {
        color: #ffffff;
        font-size: 2rem;
        font-weight: 600;
        font-family: 'SF Pro Display', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .metric-container {
        background: linear-gradient(45deg, #0f1642, #24294d);
        border-radius: 15px;
        padding: 2rem;
        margin: 1rem 0;
    }
    
    .metric-title {
        color: #ffffff;
        font-size: 1.5rem;
        font-weight: 600;
        margin-bottom: 1.5rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Custom colors for each metric */
    .metric-total {
        background: linear-gradient(135deg, #2E3192, #1BFFFF);
    }
    
    .metric-manufacturers {
        background: linear-gradient(135deg, #11998e, #38ef7d);
    }
    
    .metric-excellence {
        background: linear-gradient(135deg, #FF416C, #FF4B2B);
    }
    
    .metric-rated {
        background: linear-gradient(135deg, #8E2DE2, #4A00E0);
    }
    </style>
""", unsafe_allow_html=True)

# Add this at the start of your app to ensure consistent dark styling
st.markdown("""
    <style>
    /* Main background */
    .stApp {
        background-color: #0e1117;
    }
    
    /* All tabs, buttons, and expanders */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #1e1e1e;
        padding: 10px;
        border-radius: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        color: white;
    }
    .stTabs [data-baseweb="tab-panel"] {
        background-color: #1e1e1e;
        border-radius: 10px;
        padding: 10px;
    }
    
    /* Dataframes and tables */
    .dataframe {
        background-color: #1e1e1e;
        color: white;
    }
    
    /* Headers and text */
    h1, h2, h3, h4, h5, h6, p {
        color: white !important;
    }
    </style>
""", unsafe_allow_html=True)

# Add this to your CSS section (after the existing CSS)
st.markdown("""
    <style>
    /* Dark theme for tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #1e2130;
        border-radius: 8px;
        padding: 8px;
        margin-bottom: 16px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: auto;
        padding: 8px 16px;
        color: #a3a8b8;
        border-radius: 4px;
        background-color: transparent;
        border: 1px solid #2d3748;
        font-size: 14px;
        font-weight: 500;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #2d3748;
        color: white;
    }
    
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background-color: #3b82f6;
        color: white;
        border-color: #3b82f6;
    }
    
    /* Dark theme for tab panels */
    .stTabs [data-baseweb="tab-panel"] {
        background-color: #1e2130;
        border-radius: 8px;
        padding: 16px;
        border: 1px solid #2d3748;
    }
    
    /* Dark theme for expanders */
    .streamlit-expanderHeader {
        background-color: #1e2130;
        color: white !important;
        border-radius: 4px;
    }
    
    .streamlit-expanderContent {
        background-color: #1e2130;
        color: #a3a8b8;
        border: 1px solid #2d3748;
        border-top: none;
        border-radius: 0 0 4px 4px;
    }
    </style>
""", unsafe_allow_html=True)

# Update the CSS for tabs and content (add this to your existing CSS section)
st.markdown("""
    <style>
    /* Enhanced Tab Styling */
    .stTabs {
        background-color: #0e1117;
        padding: 20px;
        border-radius: 10px;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        background-color: #1e2130;
        padding: 10px;
        border-radius: 10px;
        border: 1px solid #2d3748;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: auto;
        padding: 10px 20px;
        color: #a3a8b8;
        border-radius: 5px;
        background-color: #262b3d;
        border: 1px solid #2d3748;
        font-size: 14px;
        font-weight: 500;
        transition: all 0.2s ease;
        margin-right: 8px;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #3b82f6;
        color: white;
        border-color: #3b82f6;
    }
    
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background-color: #3b82f6;
        color: white;
        border-color: #3b82f6;
        font-weight: 600;
    }
    
    /* Enhanced Content Styling */
    .stTabs [data-baseweb="tab-panel"] {
        background-color: #0e1117;
        border-radius: 10px;
        padding: 20px;
        border: 1px solid #2d3748;
        margin-top: 20px;
    }
    
    /* Metric Card Styling */
    .metric-container {
        background: linear-gradient(45deg, #1e2130, #2d3748);
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        border: 1px solid #3b82f6;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* Text Styling within Tabs */
    .stTabs [data-baseweb="tab-panel"] p {
        color: #a3a8b8;
        line-height: 1.6;
        margin-bottom: 1rem;
    }
    
    .stTabs [data-baseweb="tab-panel"] h3 {
        color: white;
        font-size: 1.5rem;
        font-weight: 600;
        margin: 1.5rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #3b82f6;
    }
    
    .stTabs [data-baseweb="tab-panel"] h4 {
        color: #e2e8f0;
        font-size: 1.2rem;
        font-weight: 500;
        margin: 1rem 0;
    }
    
    /* List Styling within Tabs */
    .stTabs [data-baseweb="tab-panel"] ul {
        margin-left: 1.5rem;
        margin-bottom: 1rem;
    }
    
    .stTabs [data-baseweb="tab-panel"] li {
        color: #a3a8b8;
        margin-bottom: 0.5rem;
    }
    </style>
""", unsafe_allow_html=True)

# Add this CSS before creating the tabs
st.markdown("""
    <style>
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        padding: 0.5rem;
        border-radius: 8px;
        background-color: #1e2130;
    }
    
    .stTabs [data-baseweb="tab"] {
        padding: 8px 16px;
        border-radius: 4px;
        background-color: transparent;
        border: none;
        color: #a3a8b8;
        font-weight: 500;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background-color: rgba(255, 255, 255, 0.1);
        color: white;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #3b82f6 !important;
        color: white !important;
    }
    
    /* Content area styling */
    .stTabs [data-baseweb="tab-panel"] {
        padding: 1.5rem;
        border-radius: 8px;
        background-color: white;
        margin-top: 1rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    }
    
    /* Button styling */
    .stButton button {
        background-color: #3b82f6;
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 4px;
        font-weight: 500;
        cursor: pointer;
        transition: all 0.2s;
    }
    
    .stButton button:hover {
        background-color: #2563eb;
        transform: translateY(-1px);
    }
    
    /* Text styling */
    .stMarkdown {
        color: #1f2937;
    }
    
    h3 {
        color: #111827;
        font-weight: 600;
        margin-bottom: 1rem;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background-color: #f3f4f6;
        border-radius: 4px;
        padding: 0.5rem 1rem;
        color: #1f2937;
    }
    
    .streamlit-expanderContent {
        background-color: white;
        border: 1px solid #e5e7eb;
        border-radius: 4px;
        margin-top: 0.5rem;
        padding: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# Then update the tabs section
def create_doc_link(page_name):
    """Create a clickable link that updates the sidebar selection"""
    if 'page' not in st.session_state:
        st.session_state.page = "Overview"
    
    return f"""
        <div class='doc-link' 
             onclick="document.querySelector('div[data-testid="stSidebarNav"] div[role="radiogroup"]').children['{page_name}'].click()"
             style='cursor: pointer; padding: 0.5rem; border-radius: 4px;'>
            Click to view {page_name} page
        </div>
    """

# Sidebar navigation
st.sidebar.title("Navigation")

# Define pages with their display names and keys
PAGES = {
    "Overview": "overview",
    "Analysis Results": "analysis",
    "Profiling Report": "profiling",
    "Manufacturer Analysis": "manufacturer",
    "Medicine Analysis": "medicine",
    "Review Analysis": "review",
    "Data Cleaning": "cleaning",
    "Feature Engineering": "engineering",
    "Model Development": "model",
    "Documentation & Methodology": "documentation",
    "Comparative Analysis": "comparative",
    "Market Trends": "trends",
    "Advanced Analytics": "advanced",
    "Recommendations": "recommendations"
}

# Initialize session state for page selection if not exists
if 'page' not in st.session_state:
    st.session_state.page = "Overview"

# Create radio buttons for navigation
page = st.sidebar.radio(
    "Select a Page",
    options=list(PAGES.keys()),
    key="page_selection",
    index=list(PAGES.keys()).index(st.session_state.page)
)

# Update session state when page changes
if st.session_state.page != page:
    st.session_state.page = page

# Create tabs with proper navigation
doc_tabs = st.tabs([
    "🔍 Initial Analysis",
    "🧹 Data Cleaning",
    "⚙️ Feature Engineering",
    "🤖 Model Development",
    "📊 Visualization Guide"
])

# Add content and navigation for each tab
for i, tab in enumerate(doc_tabs):
    with tab:
        if i == 0:  # Initial Analysis
            st.markdown("### Initial Data Analysis")
            st.markdown("""
                1. **Data Loading**
                - Load raw data from CSV
                - Check data types and missing values
                - Validate column names and formats
            """)
            if st.button("Go to Analysis Results →", key=f"nav_{i}"):
                st.session_state.page = "Analysis Results"
                st.rerun()
                
        elif i == 1:  # Data Cleaning
            st.markdown("### Data Cleaning Process")
            st.markdown("""
                1. **Data Standardization**
                - Normalize manufacturer names
                - Format numerical values
                - Handle missing values
            """)
            if st.button("Go to Data Cleaning →", key=f"nav_{i}"):
                st.session_state.page = "Data Cleaning"
                st.rerun()
                
        elif i == 2:  # Feature Engineering
            st.markdown("### Feature Engineering")
            st.markdown("""
                1. **Market Metrics**
                - Calculate market share
                - Analyze manufacturer performance
                - Compute complexity scores
            """)
            if st.button("Go to Feature Engineering →", key=f"nav_{i}"):
                st.session_state.page = "Feature Engineering"
                st.rerun()
                
        elif i == 3:  # Model Development
            st.markdown("### Model Development")
            st.markdown("""
                1. **Model Pipeline**
                - Data preprocessing
                - Feature selection
                - Model training and validation
            """)
            if st.button("Go to Model Development →", key=f"nav_{i}"):
                st.session_state.page = "Model Development"
                st.rerun()
                
        elif i == 4:  # Visualization Guide
            st.markdown("### Visualization Guide")
            st.markdown("""
                1. **Chart Selection**
                - Choose appropriate visualizations
                - Apply consistent styling
                - Optimize for readability
            """)
            if st.button("Go to Advanced Analytics →", key=f"nav_{i}"):
                st.session_state.page = "Advanced Analytics"
                st.rerun()

        # Add a divider after the content
        st.markdown("---")
        
        # Add a "Learn More" section for each tab
        with st.expander("Learn More"):
            if i == 0:
                st.markdown("""
                    Dive deeper into the initial data analysis process:
                    - Understand data structure
                    - Identify patterns
                    - Explore relationships
                """)
            elif i == 1:
                st.markdown("""
                    Explore our data cleaning methodology:
                    - Data validation techniques
                    - Quality assurance steps
                    - Standardization processes
                """)
            elif i == 2:
                st.markdown("""
                    Learn about our feature engineering approach:
                    - Feature creation methods
                    - Transformation techniques
                    - Validation procedures
                """)
            elif i == 3:
                st.markdown("""
                    Understand our model development process:
                    - Algorithm selection
                    - Training methodology
                    - Validation techniques
                """)
            elif i == 4:
                st.markdown("""
                    Master data visualization best practices:
                    - Chart selection guide
                    - Design principles
                    - Interactive features
                """)

# Update the column mappings to match your actual data file
COLUMN_MAPPING = {
    'medicine_name': 'Medicine Name',  # As shown in debug output
    'composition': 'Composition',      # As shown in debug output
    'uses': 'Uses',                    # As shown in debug output
    'side_effects': 'Side_effects',    # As shown in debug output
    'manufacturer': 'Manufacturer',     # As shown in debug output
    'excellent_review': 'Excellent Review %',
    'average_review': 'Average Review %',
    'poor_review': 'Poor Review %'
}

# Add this after your imports and before page selection
@st.cache_data
def load_and_engineer_data():
    """Load and engineer the pharmaceutical data"""
    try:
        # Load raw data
        project_root = Path(__file__).parent.parent
        data_path = project_root / "data" / "Medicine_Details.csv"
        
        if not data_path.exists():
            raise FileNotFoundError("Data file not found. Please ensure Medicine_Details.csv is in the data directory.")
            
        df = pd.read_csv(data_path)
        
        # Debug information before processing
        with st.expander("🔍 Raw Data Debug Information"):
            st.write("File path:", data_path)
            st.write("Original columns:", df.columns.tolist())
            st.write("Sample data:", df.head())
            st.write("Data types:", df.dtypes)
        
        # Create engineered features
        df_engineered = df.copy()
        
        # Standardize column names
        df_engineered.columns = df_engineered.columns.str.strip().str.title()
        
        # Market metrics (core features)
        df_engineered['mfr_total_medicines'] = df_engineered.groupby('Manufacturer')['Medicine Name'].transform('count')
        df_engineered['mfr_avg_review'] = df_engineered.groupby('Manufacturer')['Excellent Review %'].transform('mean')
        df_engineered['mfr_market_share'] = (df_engineered.groupby('Manufacturer')['Medicine Name'].transform('count') / 
                                           len(df_engineered) * 100)
        
        # Market tiers
        df_engineered['mfr_market_tier'] = pd.qcut(
            df_engineered['mfr_market_share'],
            q=4,
            labels=['Tier 4', 'Tier 3', 'Tier 2', 'Tier 1']
        )
        
        # Performance tiers based on reviews
        df_engineered['mfr_performance_tier'] = pd.qcut(
            df_engineered['mfr_avg_review'],
            q=4,
            labels=['Low', 'Moderate', 'High', 'Excellent']
        )
        
        # Composition complexity (with error handling)
        df_engineered['composition_complexity'] = df_engineered['Composition'].fillna('').str.count(',') + 1
        
        # Side effect severity (with robust error handling)
        possible_side_effect_columns = ['Side_effects', 'side_effects', 'Side Effects', 'Side_Effects']
        side_effects_col = None

        for col in possible_side_effect_columns:
            if col in df_engineered.columns:
                side_effects_col = col
                break

        if side_effects_col:
            try:
                df_engineered['side_effect_severity'] = (
                    df_engineered[side_effects_col]
                    .fillna('')
                    .str.count(',') + 1
                )
                st.success(f"Successfully processed side effects from column: {side_effects_col}")
            except Exception as e:
                st.warning(f"Could not calculate side effect severity: {str(e)}")
                df_engineered['side_effect_severity'] = 1  # Default value
        else:
            st.warning("Side effects column not found. Available columns: " + ", ".join(df_engineered.columns))
            df_engineered['side_effect_severity'] = 1  # Default value
        
        # Debug information
        with st.expander("🔍 Side Effects Processing Debug"):
            st.write("Available columns:", df_engineered.columns.tolist())
            if side_effects_col:
                st.write("Sample side effects:", df_engineered[side_effects_col].head())
                st.write("Sample severity scores:", df_engineered['side_effect_severity'].head())
        
        return df, df_engineered
        
    except Exception as e:
        st.error(f"Error in data processing: {str(e)}")
        st.info("""
        Please check:
        1. Data file exists in the correct location
        2. Required columns are present
        3. Data format is consistent
        """)
        return None, None

# Load data at app startup
with st.spinner("Loading and processing data..."):
    df, df_engineered = load_and_engineer_data()

if df is None or df_engineered is None:
    st.error("Failed to load or process data. Please check your data files and try again.")
    st.stop()

# Add this after loading the data
def preprocess_data(df):
    """Preprocess data with market metrics"""
    df_processed = df.copy()
    
    # Calculate market share
    df_processed['mfr_market_share'] = (df_processed.groupby('Manufacturer')['Medicine Name']
                                       .transform('count') / len(df_processed) * 100)
    
    # Calculate composition complexity
    df_processed['composition_complexity'] = df_processed['Composition'].str.count(',') + 1
    
    # Calculate manufacturer metrics
    df_processed['mfr_total_medicines'] = df_processed.groupby('Manufacturer')['Medicine Name'].transform('count')
    df_processed['mfr_avg_review'] = df_processed.groupby('Manufacturer')['Excellent Review %'].transform('mean')
    
    return df_processed

# Then update the data loading section
if df is not None:
    # Preprocess data
    df = preprocess_data(df)

# Then in the Advanced Analytics section:
elif page == "Advanced Analytics":
    st.title("📊 Advanced Analytics")
    
    # Create radio buttons for analysis type
    analysis_type = st.radio(
        "Select Analysis",
        ["📈 Performance", "📊 Market", "🧪 Product", "📊 Trends"],
        horizontal=True,
        key="analysis_selector"
    )
    
    if analysis_type == "📈 Performance":
        # Performance metrics with dark background
        st.markdown("""
            <style>
            [data-testid="stMetric"] {
                background-color: #1e2130;
                padding: 1rem;
                border-radius: 8px;
                box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
                display: flex;
                flex-direction: column-reverse;
            }
            [data-testid="stMetricLabel"] {
                color: #ffffff !important;
                font-size: 1.2rem !important;
                font-weight: 600 !important;
                margin-top: 0.5rem !important;
            }
            [data-testid="stMetricValue"] {
                color: #ffffff !important;
                font-size: 2rem !important;
            }
            [data-testid="stMetricDelta"] {
                color: #10b981 !important;
            }
            </style>
        """, unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                label="Average Excellence",
                value=f"{df['Excellent Review %'].mean():.1f}%",
                help="Average excellence rating across all medicines"
            )
        with col2:
            st.metric(
                label="Market Concentration",
                value=f"{df['mfr_market_share'].std():.2f}",
                help="Standard deviation of market share"
            )
        with col3:
            st.metric(
                label="Innovation Index",
                value=f"{df['composition_complexity'].mean():.1f}",
                help="Average complexity of medicine compositions"
            )
        
        # Performance trend chart with dark background
        fig = px.line(
            df,
            x='Manufacturer',
            y='Excellent Review %',
            title='Performance Trend by Manufacturer'
        )
        fig.update_layout(
            plot_bgcolor='#1e2130',
            paper_bgcolor='#1e2130',
            font={'color': '#ffffff'},
            xaxis={
                'tickangle': 45,
                'gridcolor': '#2d3748',
                'title_font': {'color': '#ffffff'},
                'tickfont': {'color': '#ffffff'}
            },
            yaxis={
                'gridcolor': '#2d3748',
                'title_font': {'color': '#ffffff'},
                'tickfont': {'color': '#ffffff'}
            },
            title_font={'color': '#ffffff'},
            showlegend=True,
            legend={'font': {'color': '#ffffff'}}
        )
        fig.update_traces(line_color='#60a5fa')
        st.plotly_chart(fig, use_container_width=True)

# Load the trained model and necessary data
@st.cache_resource
def load_model():
    try:
        trainer = ModelTrainer(config)
        pipeline = trainer.load_pipeline()
        
        if pipeline is None:
            st.warning("⚠️ Model not loaded. Training new model...")
            # Load sample data for training
            sample_data = pd.read_csv(config.DATA_PATH)
            X = sample_data.drop('Excellent Review %', axis=1)
            y = sample_data['Excellent Review %']
            
            # Train the model
            trainer.prepare_pipeline()
            trainer.fit(X, y)
            pipeline = trainer.pipeline
            
        return pipeline
        
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None

try:
    with st.spinner("Loading model..."):
        model = load_model()
        model_loaded = True
except:
    model_loaded = False

# Main Content
st.title("🏥 Pharmaceutical Analysis Dashboard")

def display_notebook():
    """Display EDA notebook or fallback to basic analysis"""
    try:
        # Check for notebook
        project_root = Path(__file__).parent.parent
        notebook_path = project_root / "notebooks" / "Pharma_Analysis_EDA.ipynb"
        
        if notebook_path.exists():
            with open(notebook_path, 'r') as f:
                notebook_content = f.read()
            
            st.components.v1.html(
                f"""
                <div style="width:100%; height:800px; overflow:auto;">
                    <pre>{notebook_content}</pre>
                </div>
                """,
                height=800
            )
        else:
            # Fallback to basic EDA
            st.warning("Notebook not found. Showing basic analysis instead.")
            
            # Distribution plots
            st.subheader("Review Score Distributions")
            fig = plt.figure(figsize=(10, 6))
            for col in ['Excellent Review %', 'Average Review %', 'Poor Review %']:
                if col in df.columns:
                    plt.hist(df[col], alpha=0.5, label=col, bins=30)
            plt.legend()
            plt.grid(True, alpha=0.3)
            st.pyplot(fig)
            
            # Correlation heatmap
            st.subheader("Feature Correlations")
            numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
            if len(numeric_cols) > 0:
                corr = df[numeric_cols].corr()
                fig = plt.figure(figsize=(10, 8))
                sns.heatmap(corr, annot=True, cmap='coolwarm', center=0)
                st.pyplot(fig)
            
            # Top manufacturers
            st.subheader("Top Manufacturers")
            if 'Manufacturer' in df.columns:
                top_mfr = df['Manufacturer'].value_counts().head(10)
                fig = plt.figure(figsize=(10, 6))
                plt.bar(range(len(top_mfr)), top_mfr.values)
                plt.xticks(range(len(top_mfr)), top_mfr.index, rotation=45, ha='right')
                plt.grid(True, alpha=0.3)
                st.pyplot(fig)
            
    except Exception as e:
        st.error(f"Error displaying notebook: {str(e)}")
        st.info("Please check the notebook format and try again.")

if page == "Overview":
    st.header("Dataset Overview")
    
    # Key Metrics section with new styling
    st.markdown("""
        <div class="metric-container">
            <div class="metric-title">📊 Key Performance Metrics</div>
            <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem;">
                <div class="metric-card metric-total">
                    <div class="metric-label">Total Medicines</div>
                    <div class="metric-value">{:,}</div>
                </div>
                <div class="metric-card metric-manufacturers">
                    <div class="metric-label">Manufacturers</div>
                    <div class="metric-value">{:,}</div>
                </div>
                <div class="metric-card metric-excellence">
                    <div class="metric-label">Avg Excellence</div>
                    <div class="metric-value">{:.1f}%</div>
                </div>
                <div class="metric-card metric-rated">
                    <div class="metric-label">High Rated</div>
                    <div class="metric-value">{:,}</div>
                </div>
            </div>
        </div>
    """.format(
        len(df),
        df['Manufacturer'].nunique() if 'Manufacturer' in df.columns else 0,
        df['Excellent Review %'].mean() if 'Excellent Review %' in df.columns else 0,
        len(df[df['Excellent Review %'] > 80]) if 'Excellent Review %' in df.columns else 0
    ), unsafe_allow_html=True)

    # Dataset Preview with better formatting
    st.markdown("### 📋 Medicine Details")
    with st.container():
        # Add column selection with more descriptive labels
        cols_to_display = st.multiselect(
            "Select columns to display",
            df.columns.tolist(),
            default=['Medicine Name', 'Composition', 'Manufacturer', 'Excellent Review %'],
            help="Choose which columns to view in the table below"
        )
        
        # Add number of rows selector
        num_rows = st.slider("Number of medicines to display", 5, 50, 10)
        
        if cols_to_display:
            st.dataframe(
                df[cols_to_display].head(num_rows),
                use_container_width=True,
                height=400,
                column_config={
                    "Medicine Name": st.column_config.TextColumn(
                        "Medicine Name",
                        help="Name of the pharmaceutical product"
                    ),
                    "Composition": st.column_config.TextColumn(
                        "Active Ingredients",
                        help="Chemical composition and dosage"
                    ),
                    "Manufacturer": st.column_config.TextColumn(
                        "Manufacturer",
                        help="Pharmaceutical company"
                    ),
                    "Excellent Review %": st.column_config.NumberColumn(
                        "Excellence Rating",
                        help="Percentage of excellent reviews",
                        format="%.1f%%"
                    )
                }
            )
        else:
            st.info("👆 Please select at least one column to display")

    # Statistical Summary with tabs
    st.markdown("### 📈 Statistical Summary")
    tab1, tab2 = st.tabs(["Review Statistics", "Data Overview"])
    
    with tab1:
        if all(col in df.columns for col in ['Excellent Review %', 'Average Review %', 'Poor Review %']):
            stats_df = pd.DataFrame({
                'Metric': ['Count', 'Mean', 'Std Dev', 'Min', '25%', 'Median', '75%', 'Max'],
                'Excellent': df['Excellent Review %'].describe(),
                'Average': df['Average Review %'].describe(),
                'Poor': df['Poor Review %'].describe()
            }).set_index('Metric')
            
            # Format numbers without using style
            stats_df = stats_df.round(2)
            st.dataframe(stats_df)
    
    with tab2:
        # Format numbers without using style
        st.dataframe(df.describe(include='all').round(2))

    # Data Quality Overview
    st.markdown("### 🔍 Data Quality")
    col1, col2 = st.columns(2)
    
    with col1:
        missing_data = df.isnull().sum()
        if missing_data.any():
            st.markdown("**Missing Values**")
            st.dataframe(
                missing_data[missing_data > 0],
                use_container_width=True
            )
        else:
            st.success("No missing values found in the dataset")
    
    with col2:
        st.markdown("**Dataset Shape**")
        st.write(f"Rows: {df.shape[0]:,}")
        st.write(f"Columns: {df.shape[1]:,}")

elif page == "Analysis Results":
    st.header("Detailed Analysis Results")
    
    # Add Feature Engineering section with modern styling
    st.markdown("""
        <div style='background: linear-gradient(45deg, #1e3c72, #2a5298); 
                    padding: 2rem; border-radius: 15px; margin: 1rem 0;'>
            <h2 style='color: white; margin-bottom: 1rem;'>🔧 Feature Engineering</h2>
        </div>
    """, unsafe_allow_html=True)
    
    try:
        # Original vs Engineered Features Counter
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
                <div style='background: #1e2130; padding: 1.5rem; border-radius: 10px;'>
                    <h4 style='color: #4CAF50; margin: 0;'>Original Features</h4>
                    <h2 style='color: white; margin: 0;'>{}</h2>
                </div>
            """.format(len(df.columns)), unsafe_allow_html=True)
        
        # Create and apply feature engineering
        feature_engineer = PharmaFeatureEngineer()
        df_engineered = feature_engineer.transform(df)
        new_features = [col for col in df_engineered.columns if col not in df.columns]
        
        with col2:
            st.markdown("""
                <div style='background: #1e2130; padding: 1.5rem; border-radius: 10px;'>
                    <h4 style='color: #2196F3; margin: 0;'>Engineered Features</h4>
                    <h2 style='color: white; margin: 0;'>{}</h2>
                </div>
            """.format(len(new_features)), unsafe_allow_html=True)
        
        # Feature Categories
        st.markdown("### 🔍 Feature Categories")
        
        feature_categories = {
            "Dosage Features": ["dosage_mg"],
            "Composition Analysis": ["num_ingredients", "main_ingredient"],
            "Text Analysis": ["uses_word_count", "side_effects_count"],
            "Manufacturer Insights": [
                "mfr_avg_review",
                "mfr_review_std",
                "mfr_review_count",
                "mfr_total_medicines",
                "mfr_market_share",
                "mfr_rank",
                "mfr_relative_performance",
                "mfr_tier",
                "mfr_market_position"
            ]
        }
        
        # Add descriptions for the new features
        feature_descriptions = {
            "mfr_avg_review": "Average review score for the manufacturer",
            "mfr_review_std": "Standard deviation of review scores (lower means more consistent)",
            "mfr_review_count": "Number of reviews received",
            "mfr_total_medicines": "Total number of medicines produced",
            "mfr_market_share": "Percentage of total medicines in dataset",
            "mfr_rank": "Manufacturer rank based on average review (1 is best)",
            "mfr_relative_performance": "Performance compared to overall average (positive is better)",
            "mfr_tier": "Performance tier (Bronze, Silver, Gold, Platinum)",
            "mfr_market_position": "Market presence category (Small, Medium, Large, Leader)"
        }

        # Add the descriptions to the Feature Descriptions expander
        with st.expander("Feature Descriptions"):
            for category, features in feature_categories.items():
                st.markdown(f"**{category}**")
                for feature in features:
                    if feature in feature_descriptions:
                        st.markdown(f"- **{feature}**: {feature_descriptions[feature]}")
        
        for category, features in feature_categories.items():
            with st.expander(category):
                # Show features in this category
                available_features = [f for f in features if f in new_features]
                if available_features:
                    # Create sample visualization for the features
                    if any(f in df_engineered.columns for f in available_features):
                        numeric_features = [f for f in available_features 
                                         if pd.api.types.is_numeric_dtype(df_engineered[f])]
                        if numeric_features:
                            fig = px.box(
                                df_engineered,
                                y=numeric_features,
                                title=f"{category} Distribution"
                            )
                            fig.update_layout(
                                template="plotly_dark",
                                plot_bgcolor='rgba(0,0,0,0)',
                                paper_bgcolor='rgba(0,0,0,0)'
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Show summary statistics
                            st.dataframe(
                                df_engineered[numeric_features].describe().round(2),
                                use_container_width=True
                            )
                else:
                    st.info(f"No features available in {category}")
        
        # Interactive Feature Explorer
        st.markdown("### 🔮 Interactive Feature Explorer")
        
        # Select features to compare
        col1, col2 = st.columns(2)
        with col1:
            feature_x = st.selectbox("Select X-axis feature", new_features)
        with col2:
            feature_y = st.selectbox("Select Y-axis feature", 
                                   [f for f in new_features if f != feature_x])
        
        if feature_x and feature_y:
            # Create scatter plot
            fig = px.scatter(
                df_engineered,
                x=feature_x,
                y=feature_y,
                color='excellent_review' if 'excellent_review' in df_engineered.columns else None,
                title=f"Relationship between {feature_x} and {feature_y}",
                template="plotly_dark"
            )
            
            # Update layout for better contrast
            fig.update_layout(
                plot_bgcolor='#1e1e1e',
                paper_bgcolor='#1e1e1e',
                font={'color': 'white'},
                xaxis={'gridcolor': '#3b3b3b'},
                yaxis={'gridcolor': '#3b3b3b'}
            )
            
            # Show correlation
            correlation = df_engineered[feature_x].corr(df_engineered[feature_y])
            st.plotly_chart(fig, use_container_width=True)
            
            # Style the correlation metric with darker theme
            st.markdown(
                f"""
                <div style='
                    background-color: #1e1e1e; 
                    padding: 1.5rem; 
                    border-radius: 10px;
                    margin-top: 1rem;
                    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                '>
                    <h2 style='
                        color: white; 
                        margin: 0;
                        font-size: 1.5rem;
                        font-weight: 500;
                    '>Correlation: {correlation:.3f}</h2>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        # Add visualization for manufacturer insights
        if "Manufacturer Insights" in feature_categories:
            with st.expander("Manufacturer Insights"):
                try:
                    # Market Overview
                    st.subheader("Market Overview")
                    col1, col2, col3 = st.columns(3)
                    
                    # Custom CSS for metric styling
                    st.markdown("""
                        <style>
                        [data-testid="stMetric"] {
                            background-color: #1e1e1e;
                            padding: 1rem;
                            border-radius: 10px;
                            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                        }
                        [data-testid="stMetricLabel"] {
                            color: #ffffff !important;
                        }
                        [data-testid="stMetricValue"] {
                            color: #ffffff !important;
                        }
                        div[data-testid="stExpander"] {
                            background-color: #1e1e1e;
                            border-color: #3b3b3b;
                        }
                        div[data-testid="stExpander"] > div[role="button"] {
                            color: white !important;
                        }
                        </style>
                    """, unsafe_allow_html=True)

                    with col1:
                        st.metric(
                            "Total Manufacturers",
                            df_engineered['Manufacturer'].nunique(),
                            help="Total number of unique manufacturers in the dataset"
                        )
                    with col2:
                        st.metric(
                            "Average Market Share",
                            f"{df_engineered['mfr_market_share'].mean():.1f}%",
                            help="Mean market share across all manufacturers"
                        )
                    with col3:
                        st.metric(
                            "Top Performer",
                            df_engineered.nlargest(1, 'mfr_avg_review')['Manufacturer'].iloc[0],
                            help="Manufacturer with highest average review score"
                        )
                    
                    # Market Position Analysis
                    st.subheader("Market Position Analysis")
                    fig = px.scatter(
                        df_engineered,
                        x='mfr_market_share',
                        y='mfr_avg_review',
                        color='mfr_market_tier',
                        size='mfr_total_medicines',
                        hover_data=['Manufacturer', 'mfr_performance_tier'],
                        title="Market Share vs Performance",
                        labels={
                            'mfr_market_share': 'Market Share (%)',
                            'mfr_avg_review': 'Average Excellence Rating (%)',
                            'mfr_market_tier': 'Market Position',
                            'mfr_total_medicines': 'Number of Products'
                        }
                    )
                    
                    fig.update_layout(
                        plot_bgcolor='#1e1e1e',
                        paper_bgcolor='#1e1e1e',
                        font={'color': 'white'},
                        xaxis={'gridcolor': '#3b3b3b'},
                        yaxis={'gridcolor': '#3b3b3b'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Performance Distribution
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Market Share Distribution
                        fig = px.box(
                            df_engineered,
                            x='mfr_market_tier',
                            y='mfr_market_share',
                            color='mfr_market_tier',
                            title='Market Share Distribution by Tier',
                            labels={'mfr_market_share': 'Market Share (%)'}
                        )
                        fig.update_layout(
                            showlegend=False,
                            plot_bgcolor='#1e1e1e',
                            paper_bgcolor='#1e1e1e',
                            font={'color': 'white'},
                            xaxis={'gridcolor': '#3b3b3b'},
                            yaxis={'gridcolor': '#3b3b3b'}
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        # Performance Distribution
                        fig = px.box(
                            df_engineered,
                            x='mfr_performance_tier',
                            y='mfr_avg_review',
                            color='mfr_performance_tier',
                            title='Review Distribution by Performance Tier',
                            labels={'mfr_avg_review': 'Excellence Rating (%)'}
                        )
                        fig.update_layout(
                            showlegend=False,
                            plot_bgcolor='#1e1e1e',
                            paper_bgcolor='#1e1e1e',
                            font={'color': 'white'},
                            xaxis={'gridcolor': '#3b3b3b'},
                            yaxis={'gridcolor': '#3b3b3b'}
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Market Leaders Table
                    st.subheader("Top Manufacturers")
                    top_manufacturers = df_engineered.nlargest(10, 'mfr_market_share')[
                        ['Manufacturer', 'mfr_market_share', 'mfr_avg_review', 
                         'mfr_performance_tier', 'mfr_total_medicines']
                    ].rename(columns={
                        'mfr_market_share': 'Market Share (%)',
                        'mfr_avg_review': 'Avg Excellence Rating',
                        'mfr_performance_tier': 'Performance Tier',
                        'mfr_total_medicines': 'Total Products'
                    })
                    
                    st.dataframe(
                        top_manufacturers.style.format({
                            'Market Share (%)': '{:.1f}%',
                            'Avg Excellence Rating': '{:.1f}%'
                        })
                    )
                    
                except Exception as e:
                    st.error(f"Error in manufacturer analysis: {str(e)}")
                    st.write("Available columns:", df_engineered.columns.tolist())

    except Exception as e:
        st.error(f"Error in feature engineering: {str(e)}")
        st.write("Debug info:")
        st.write("Available columns:", df.columns.tolist())

elif page == "Profiling Report":
    st.header("Data Profiling Report")
    
    try:
        # Create reports directory if it doesn't exist
        reports_dir = config.PROJECT_ROOT / 'reports'
        reports_dir.mkdir(parents=True, exist_ok=True)
        
        profile_path = reports_dir / 'profile_report.html'
        
        if not profile_path.exists():
            with st.spinner("Generating profiling report..."):
                # Configure profiling with plotly backend
                profile = ProfileReport(
                    df,
                    title="Medicine Details Analysis",
                    minimal=True,  # Faster generation
                    progress_bar=False,
                    explorative=True,
                    html={'style': {'full_width': True}},
                    plot={
                        'backend': 'plotly',  # Use plotly instead of matplotlib
                        'image_format': 'svg',
                        'correlation': {
                            'cmap': 'RdBu',
                            'bad': '#000000'
                        },
                        'missing': {
                            'cmap': 'RdBu'
                        }
                    },
                    correlations={
                        "pearson": {"calculate": True},
                        "spearman": {"calculate": False},
                        "kendall": {"calculate": False},
                        "phi_k": {"calculate": False},
                        "cramers": {"calculate": False},
                    }
                )
                
                # Save the report
                profile.to_file(profile_path)
                st.success("✅ Profile report generated successfully!")
        
        # Display the report
        with open(profile_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
            
        st.components.v1.html(
            html_content, 
            height=800, 
            scrolling=True
        )
        
    except Exception as e:
        st.error(f"Error generating profile report: {str(e)}")
        st.info("Showing basic data analysis instead:")
        
        # Create basic analysis using plotly
        st.subheader("📊 Data Overview")
        
        # Numerical distributions
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        for col in numeric_cols:
            fig = px.histogram(
                df, 
                x=col,
                title=f"Distribution of {col}",
                template="plotly_dark"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Correlation matrix
        st.subheader("🔄 Correlation Matrix")
        corr = df[numeric_cols].corr()
        fig = px.imshow(
            corr,
            labels=dict(color="Correlation"),
            x=corr.columns,
            y=corr.columns,
            color_continuous_scale="RdBu",
            aspect="auto"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Missing values
        st.subheader("❓ Missing Values")
        missing = df.isnull().sum()
        if missing.any():
            fig = px.bar(
                x=missing[missing > 0].index,
                y=missing[missing > 0].values,
                title="Missing Values by Column"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.success("No missing values found!")

elif page == "Manufacturer Analysis":
    st.header("Manufacturer Analysis")
    
    if COLUMN_MAPPING['manufacturer'] in df.columns:
        # Top Manufacturers
        top_n = st.slider("Select number of top manufacturers to view", 5, 20, 10)
        
        top_manufacturers = df[COLUMN_MAPPING['manufacturer']].value_counts().head(top_n)
        fig = px.bar(
            x=top_manufacturers.values,
            y=top_manufacturers.index,
            orientation='h',
            title=f'Top {top_n} Manufacturers by Number of Medicines'
        )
        st.plotly_chart(fig)
        
        # Manufacturer Performance
        st.subheader("Manufacturer Performance Analysis")
        selected_manufacturer = st.selectbox(
            "Select Manufacturer",
            sorted(df[COLUMN_MAPPING['manufacturer']].unique())
        )
        
        mfr_data = df[df[COLUMN_MAPPING['manufacturer']] == selected_manufacturer]
        review_cols = [
            COLUMN_MAPPING['excellent_review'],
            COLUMN_MAPPING['average_review'],
            COLUMN_MAPPING['poor_review']
        ]
        
        available_review_cols = [col for col in review_cols if col in df.columns]
        if available_review_cols:
            fig = px.box(
                df[available_review_cols],
                title=f'Review Distribution for {selected_manufacturer}'
            )
            st.plotly_chart(fig)
        else:
            st.warning("Review data not available")
    else:
        st.warning("Manufacturer data not available")

elif page == "Medicine Analysis":
    st.header("Medicine Analysis")
    
    # Medicine Search
    search_term = st.text_input("Search for a medicine")
    if search_term:
        if COLUMN_MAPPING['medicine_name'] in df.columns:
            results = df[df[COLUMN_MAPPING['medicine_name']].str.contains(search_term, case=False, na=False)]
            st.write(f"Found {len(results)} matches")
            st.dataframe(results)
        else:
            st.warning("Medicine name column not found")
    
    # Salt Composition Analysis
    st.subheader("Salt Composition Analysis")
    
    if COLUMN_MAPPING['composition'] in df.columns:
        def extract_salts(composition):
            if pd.isna(composition):
                return []
            return [salt.strip() for salt in composition.split('+')]
        
        all_salts = []
        for comp in df[COLUMN_MAPPING['composition']]:
            all_salts.extend(extract_salts(comp))
        
        salt_counts = pd.Series(all_salts).value_counts()
        
        fig = px.bar(
            x=salt_counts.head(15).values,
            y=salt_counts.head(15).index,
            orientation='h',
            title='Top 15 Most Common Salt Compositions'
        )
        st.plotly_chart(fig)
    else:
        st.warning("Salt composition data not available")

elif page == "Review Analysis":
    st.header("Review Analysis")
    
    # Review Distribution
    st.subheader("Review Score Distribution")
    review_cols = [
        COLUMN_MAPPING['excellent_review'],
        COLUMN_MAPPING['average_review'],
        COLUMN_MAPPING['poor_review']
    ]
    
    # Check if review columns exist
    available_review_cols = [col for col in review_cols if col in df.columns]
    if available_review_cols:
        fig = px.box(
            df[available_review_cols],
            title='Distribution of Review Scores'
        )
        st.plotly_chart(fig)
    else:
        st.warning("Review columns not found in the dataset")
    
    # Top Rated Medicines
    st.subheader("Top Rated Medicines")
    top_n = st.slider("Select number of top medicines to view", 5, 20, 10)
    
    if COLUMN_MAPPING['excellent_review'] in df.columns:
        top_medicines = df.nlargest(top_n, COLUMN_MAPPING['excellent_review'])
        st.dataframe(top_medicines[[
            COLUMN_MAPPING['medicine_name'],
            COLUMN_MAPPING['manufacturer'],
            COLUMN_MAPPING['excellent_review']
        ]])
    else:
        st.warning("Review data not available")

elif page == "Prediction":
    st.title("Medicine Review Predictor")
    
    try:
        # Load or create models
        models_dir = Path('models')
        models_dir.mkdir(exist_ok=True)
        
        # Available models configuration
        MODELS = {
            'XGBoost': {
                'filename': 'xgb_review_predictor.joblib',
                'description': 'Advanced model with highest accuracy',
                'metrics': {'accuracy': 0.87, 'f1': 0.86}
            },
            'Random Forest': {
                'filename': 'random_forest_review_predictor.joblib',
                'description': 'Balanced performance with good interpretability',
                'metrics': {'accuracy': 0.84, 'f1': 0.82}
            },
            'Gradient Boosting': {
                'filename': 'gradient_boosting_review_predictor.joblib',
                'description': 'Fast and reliable predictions',
                'metrics': {'accuracy': 0.85, 'f1': 0.83}
            }
        }
        
        # Model selection
        col1, col2 = st.columns([2, 3])
        
        with col1:
            st.markdown("### 🤖 Model Selection")
            selected_model = st.selectbox(
                "Choose Model",
                options=list(MODELS.keys()),
                help="Select the machine learning model for prediction"
            )
            
            # Show model info in a card
            st.markdown(f"""
                <div style='background-color: #1e2130; padding: 1.5rem; border-radius: 10px; margin-top: 1rem;'>
                    <h4 style='color: #ffffff; margin-bottom: 0.5rem;'>Model Information</h4>
                    <p style='color: #a3a8b8; margin-bottom: 0.5rem;'>{MODELS[selected_model]['description']}</p>
                    <div style='display: flex; justify-content: space-between; margin-top: 1rem;'>
                        <div>
                            <p style='color: #60a5fa; margin: 0;'>Accuracy</p>
                            <p style='color: #ffffff; font-size: 1.2rem; font-weight: 600;'>{MODELS[selected_model]['metrics']['accuracy']:.1%}</p>
                        </div>
                        <div>
                            <p style='color: #60a5fa; margin: 0;'>F1 Score</p>
                            <p style='color: #ffffff; font-size: 1.2rem; font-weight: 600;'>{MODELS[selected_model]['metrics']['f1']:.1%}</p>
                        </div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("### 📝 Medicine Details")
            
            # Quick selection options
            quick_options = {
                'Antibiotics': {
                    'manufacturer': 'A. Menarini India Pvt Ltd',
                    'composition': 'Amoxicillin 500mg',
                    'uses': 'Bacterial infections',
                    'side_effects': 'Gastrointestinal discomfort'
                },
                'Pain Relief': {
                    'manufacturer': 'Abbott',
                    'composition': 'Paracetamol 500mg',
                    'uses': 'Pain and fever relief',
                    'side_effects': 'Nausea, liver problems in high doses'
                },
                'Antidiabetic': {
                    'manufacturer': 'Sun Pharma',
                    'composition': 'Metformin 500mg',
                    'uses': 'Type 2 diabetes management',
                    'side_effects': 'Digestive issues, vitamin B12 deficiency'
                }
            }

            # Quick selection dropdown
            quick_select = st.selectbox(
                "🚀 Quick Select Medicine Type",
                options=['Custom Entry'] + list(quick_options.keys()),
                help="Choose a preset medicine type or enter custom details"
            )

            # Set initial values based on quick selection
            if quick_select != 'Custom Entry':
                selected_option = quick_options[quick_select]
                default_manufacturer = selected_option['manufacturer']
                default_composition = selected_option['composition']
                default_uses = selected_option['uses']
                default_side_effects = selected_option['side_effects']
            else:
                default_manufacturer = ""
                default_composition = ""
                default_uses = ""
                default_side_effects = ""

            # Manufacturer selection
            manufacturer = st.selectbox(
                "Manufacturer",
                options=[''] + sorted(df['Manufacturer'].unique()),
                index=0 if quick_select == 'Custom Entry' else 
                      list(df['Manufacturer'].unique()).index(default_manufacturer) + 1,
                help="Select the pharmaceutical manufacturer"
            )

            # Composition selection
            col_left, col_right = st.columns(2)
            
            with col_left:
                st.markdown("**Preset Compositions**")
                preset_composition = st.selectbox(
                    "Select composition",
                    options=[''] + sorted(df['Composition'].unique()),
                    index=0 if quick_select == 'Custom Entry' else 
                          list(df['Composition'].unique()).index(default_composition) + 1,
                    key="preset_comp"
                )
            
            with col_right:
                st.markdown("**Salt Composition**")
                composition = st.text_area(
                    "Composition",
                    value=preset_composition if preset_composition else default_composition,
                    height=100,
                    help="Edit or enter the medicine's composition"
                )

            # Uses selection
            col_left, col_right = st.columns(2)
            
            with col_left:
                st.markdown("**Preset Uses**")
                preset_uses = st.selectbox(
                    "Select uses",
                    options=[''] + sorted(df['Uses'].unique()),
                    index=0 if quick_select == 'Custom Entry' else 
                          list(df['Uses'].unique()).index(default_uses) + 1,
                    key="preset_uses"
                )
            
            with col_right:
                st.markdown("**Intended Uses**")
                uses = st.text_area(
                    "Uses",
                    value=preset_uses if preset_uses else default_uses,
                    height=100,
                    help="Edit or enter the medicine's uses"
                )

            # Side effects selection
            col_left, col_right = st.columns(2)
            
            with col_left:
                st.markdown("**Preset Side Effects**")
                preset_side_effects = st.selectbox(
                    "Select side effects",
                    options=[''] + sorted(df['Side_effects'].unique()),
                    index=0 if quick_select == 'Custom Entry' else 
                          list(df['Side_effects'].unique()).index(default_side_effects) + 1,
                    key="preset_side_effects"
                )
            
            with col_right:
                st.markdown("**Known Side Effects**")
                side_effects = st.text_area(
                    "Side Effects",
                    value=preset_side_effects if preset_side_effects else default_side_effects,
                    height=100,
                    help="Edit or enter known side effects"
                )

            # Tips for better predictions
            with st.expander("💡 Tips for Better Predictions"):
                st.markdown("""
                    - Be specific with compositions and dosages
                    - Include all known uses and indications
                    - List both common and rare side effects
                    - Use standard medical terminology
                    - Include contraindications if known
                """)

            predict_button = st.button("🔮 Predict Review Score", use_container_width=True)

        # Make prediction when button is clicked
        if predict_button and all([manufacturer, composition, uses, side_effects]):
            try:
                # Load model
                model_path = models_dir / MODELS[selected_model]['filename']
                if not model_path.exists():
                    st.warning("Model not found. Creating new model...")
                    from create_sample_model import create_and_save_sample_model
                    if create_and_save_sample_model():
                        st.success("Model created successfully!")
                    else:
                        st.error("Error creating model. Please try again.")
                        st.stop()
                
                model = joblib.load(model_path)
                
                # Prepare input data
                input_data = pd.DataFrame({
                    'Manufacturer': [manufacturer],
                    'Composition': [composition],
                    'Uses': [uses],
                    'Side_effects': [side_effects]
                })
                
                # Make prediction
                prediction = model.predict_proba(input_data)[0]
                
                # Display prediction results
                st.markdown("### 📊 Prediction Results")
                
                # Create columns for visualization
                viz_col1, viz_col2 = st.columns(2)
                
                with viz_col1:
                    # Gauge chart for prediction probability
                    fig = go.Figure(go.Indicator(
                        mode = "gauge+number",
                        value = prediction[1] * 100,
                        title = {'text': "Likelihood of Excellent Reviews"},
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        gauge = {
                            'axis': {'range': [0, 100]},
                            'bar': {'color': "#60a5fa"},
                            'steps': [
                                {'range': [0, 50], 'color': "#ef4444"},
                                {'range': [50, 75], 'color': "#f59e0b"},
                                {'range': [75, 100], 'color': "#10b981"}
                            ]
                        }
                    ))
                    
                    fig.update_layout(
                        paper_bgcolor = "#1e2130",
                        font = {'color': "#ffffff", 'family': "Arial"}
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                with viz_col2:
                    # Confidence breakdown
                    st.markdown("""
                        <div style='background-color: #1e2130; padding: 1.5rem; border-radius: 10px; height: 100%;'>
                            <h4 style='color: #ffffff; margin-bottom: 1rem;'>Prediction Breakdown</h4>
                    """, unsafe_allow_html=True)
                    
                    confidence_level = prediction[1] * 100
                    if confidence_level >= 75:
                        status = "High Likelihood of Success"
                        color = "#10b981"
                    elif confidence_level >= 50:
                        status = "Moderate Likelihood of Success"
                        color = "#f59e0b"
                    else:
                        status = "Low Likelihood of Success"
                        color = "#ef4444"
                    
                    st.markdown(f"""
                        <div style='color: {color}; font-size: 1.2rem; font-weight: 600; margin-bottom: 1rem;'>
                            {status}
                        </div>
                        <p style='color: #a3a8b8; margin-bottom: 0.5rem;'>
                            Confidence Score: {confidence_level:.1f}%
                        </p>
                    """, unsafe_allow_html=True)
                    
                    # Add recommendations based on prediction
                    st.markdown("""
                        <h5 style='color: #ffffff; margin: 1rem 0;'>Recommendations:</h5>
                        <ul style='color: #a3a8b8; margin-left: 1.5rem;'>
                    """, unsafe_allow_html=True)
                    
                    if confidence_level < 75:
                        st.markdown("""
                            <li>Consider revising the composition</li>
                            <li>Expand the use cases</li>
                            <li>Improve documentation of side effects</li>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown("""
                            <li>Maintain current quality standards</li>
                            <li>Monitor for consistent performance</li>
                            <li>Consider expanding market reach</li>
                        """, unsafe_allow_html=True)
                    
                    st.markdown("</ul></div>", unsafe_allow_html=True)
            
            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")
                st.info("Please ensure all fields are filled correctly.")
        
        elif predict_button:
            st.warning("Please fill in all fields to make a prediction.")
            
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.info("Please ensure the model is properly trained and saved.")

elif page == "Notebook Viewer":
    st.header("📓 Analysis Notebook")
    
    # Add description with better formatting
    st.markdown("""
    <div style='background-color: #1e1e1e; padding: 2rem; border-radius: 10px; margin-bottom: 2rem;'>
        <h3 style='color: #ffffff; margin-bottom: 1rem;'>Interactive Analysis Notebook</h3>
        <p style='color: #e2e8f0;'>This notebook contains our complete pharmaceutical analysis including:</p>
        <ul style='color: #e2e8f0; margin: 1rem 0;'>
            <li>📊 Exploratory Data Analysis</li>
            <li>🧪 Feature Engineering</li>
            <li>🤖 Model Development</li>
            <li>📈 Results and Insights</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    # Create environment selection section
    st.markdown("### Run this notebook in your preferred environment:")
    
    try:
        # First, try to create the notebook if it doesn't exist
        from analysis.create_notebook import create_analysis_notebook
        
        # Get project root directory
        project_root = Path(__file__).parent.parent
        notebooks_dir = project_root / 'notebooks'
        notebooks_dir.mkdir(exist_ok=True)
        
        notebook_path = notebooks_dir / 'Pharma_Analysis.ipynb'
        
        # Create notebook if it doesn't exist
        if not notebook_path.exists():
            notebook_path = create_analysis_notebook()
            st.success("Created new analysis notebook!")
        
        # Read the notebook content
        with open(notebook_path, 'r', encoding='utf-8') as f:
            notebook_content = f.read()
            
        # Create three columns for environment options
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Google Colab button
            colab_url = "https://colab.research.google.com/github/FrankAsanteVanLaarhoven/Pharmaceutical-Analysis-Dashboard/blob/main/notebooks/Pharma_Analysis.ipynb"
            st.markdown(f"""
                <a href="{colab_url}" target="_blank" style="
                    display: inline-block;
                    background-color: #374151;
                    padding: 0.75rem 1.5rem;
                    border-radius: 5px;
                    color: #60a5fa;
                    text-decoration: none;
                    text-align: center;
                    width: 100%;
                    font-weight: 500;
                ">
                    <img src="https://colab.research.google.com/img/colab_favicon_256px.png" 
                         style="height: 1.2em; vertical-align: middle; margin-right: 0.5em;">
                    Open in Colab
                </a>
            """, unsafe_allow_html=True)
        
        with col2:
            # Jupyter button
            st.download_button(
                label="📓 Open in Jupyter",
                data=notebook_content,
                file_name="Pharma_Analysis.ipynb",
                mime="application/x-ipynb+json",
                use_container_width=True
            )
        
        with col3:
            # VS Code button
            st.download_button(
                label="💻 Open in VS Code",
                data=notebook_content,
                file_name="Pharma_Analysis.ipynb",
                mime="application/x-ipynb+json",
                use_container_width=True
            )

        # Show notebook preview
        st.markdown("### 📝 Notebook Preview")
        nb = nbf.reads(notebook_content, as_version=4)
        
        for cell in nb.cells:
            if cell.cell_type == 'markdown':
                st.markdown(cell.source)
            elif cell.cell_type == 'code':
                with st.expander("View Code", expanded=False):
                    st.code(cell.source, language='python')
                    
    except Exception as e:
        st.error(f"Error loading notebook: {str(e)}")
        st.info("""
        If you're seeing this error, please ensure:
        1. The notebooks directory exists
        2. You have proper permissions
        3. The create_notebook.py script is available
        """)
        # Print debug information
        st.write("Debug Information:")
        st.write(f"Current working directory: {os.getcwd()}")
        st.write(f"Project root: {project_root}")
        st.write(f"Notebooks directory: {notebooks_dir}")

elif page == "Documentation & Methodology":
    st.title("📚 Documentation & Methodology")
    
    # Create tabs for different sections of documentation
    doc_tabs = st.tabs([
        "🔍 Initial Analysis",
        "🧹 Data Cleaning",
        "⚙️ Feature Engineering",
        "🤖 Model Development",
        "📊 Visualization Guide"
    ])
    
    with doc_tabs[0]:  # Initial Analysis
        st.markdown("""
            ### 🔍 Initial Data Analysis
            
            1. **Data Loading**
            - Load raw data from CSV
            - Check data types and missing values
            - Validate column names and formats
            
            2. **Basic Statistics**
            - Number of medicines: {len(df):,}
            - Number of manufacturers: {df['Manufacturer'].nunique():,}
            - Average excellence rating: {df['Excellent Review %'].mean():.1f}%
            
            3. **Data Quality Checks**
            - Missing values assessment
            - Duplicate entries check
            - Data consistency validation
        """.format(df=df))

    with doc_tabs[1]:  # Data Cleaning
        st.markdown("""
            ### 🧹 Data Cleaning Process
            
            1. **Standardization**
            - Normalize text fields
            - Standardize manufacturer names
            - Format numerical values
            
            2. **Missing Data Handling**
            - Identify missing values
            - Apply appropriate imputation
            - Document changes
            
            3. **Data Validation**
            - Check value ranges
            - Verify relationships
            - Ensure data consistency
        """)

    with doc_tabs[2]:  # Feature Engineering
        st.markdown("""
            ### ⚙️ Feature Engineering Steps
            
            1. **Derived Features**
            - Market share calculation
            - Composition complexity
            - Manufacturer metrics
            
            2. **Text Processing**
            - Composition analysis
            - Uses categorization
            - Side effects classification
            
            3. **Performance Metrics**
            - Excellence rating normalization
            - Market performance indicators
            - Complexity scores
        """)

    with doc_tabs[3]:  # Model Development
        st.markdown("""
            ### 🤖 Model Development Guide
            
            1. **Data Preparation**
            - Feature selection
            - Data splitting
            - Scaling and normalization
            
            2. **Model Selection**
            - Algorithm comparison
            - Hyperparameter tuning
            - Cross-validation
            
            3. **Evaluation**
            - Performance metrics
            - Model validation
            - Results interpretation
        """)

    with doc_tabs[4]:  # Visualization Guide
        st.markdown("""
            ### 📊 Visualization Guidelines
            
            1. **Chart Types**
            - Line charts for trends
            - Scatter plots for relationships
            - Bar charts for comparisons
            
            2. **Best Practices**
            - Clear titles and labels
            - Consistent color schemes
            - Appropriate scales
            
            3. **Interactive Features**
            - Filtering options
            - Drill-down capabilities
            - Dynamic updates
        """)

elif page == "Comparative Analysis":
    st.title("🔄 Comparative Analysis")
    
    # Allow users to select medicines to compare
    st.markdown("""
        <div style='background-color: #1e2130; padding: 1.5rem; border-radius: 10px; margin-bottom: 2rem;'>
            <h3 style='color: white; margin: 0;'>🔍 Medicine Comparison Tool</h3>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        medicine1 = st.selectbox(
            "Select First Medicine",
            options=sorted(df['Medicine Name'].unique()),
            key="med1"
        )
    with col2:
        medicine2 = st.selectbox(
            "Select Second Medicine",
            options=[x for x in sorted(df['Medicine Name'].unique()) if x != medicine1],
            key="med2"
        )
        
    if medicine1 and medicine2:
        try:
            # Get medicine data
            med1_data = df[df['Medicine Name'] == medicine1].iloc[0]
            med2_data = df[df['Medicine Name'] == medicine2].iloc[0]
            
            # Create comparison metrics
            st.markdown("""
                <div style='background-color: #1e2130; padding: 1.5rem; border-radius: 10px; margin: 2rem 0;'>
                    <h3 style='color: white; margin: 0;'>📊 Comparison Metrics</h3>
                </div>
            """, unsafe_allow_html=True)
            
            # Define metrics that are guaranteed to exist
            metrics = {
                "Excellent Reviews": "Excellent Review %",
                "Average Reviews": "Average Review %",
                "Poor Reviews": "Poor Review %"
            }
            
            # Display metrics in a dark themed container
            for label, metric in metrics.items():
                delta = med1_data[metric] - med2_data[metric]
                delta_color = "#10b981" if delta >= 0 else "#ef4444"
                delta_symbol = "↑" if delta >= 0 else "↓"
                
                st.markdown(f"""
                    <div style='background-color: #262b3d; padding: 1.5rem; border-radius: 10px; margin-bottom: 1rem; border: 1px solid #2d3748;'>
                        <div style='display: flex; justify-content: space-between; align-items: center;'>
                            <div style='flex: 1;'>
                                <div style='color: #a3a8b8; font-size: 0.875rem; margin-bottom: 0.5rem;'>{medicine1}</div>
                                <div style='color: white; font-size: 1.5rem; font-weight: 600;'>{med1_data[metric]:.1f}%</div>
                                <div style='color: {delta_color}; font-size: 1rem;'>
                                    {delta_symbol} {abs(delta):.1f}%
                                </div>
                            </div>
                            <div style='color: #a3a8b8; font-size: 1.2rem; font-weight: 500; padding: 0 2rem;'>vs</div>
                            <div style='flex: 1; text-align: right;'>
                                <div style='color: #a3a8b8; font-size: 0.875rem; margin-bottom: 0.5rem;'>{medicine2}</div>
                                <div style='color: white; font-size: 1.5rem; font-weight: 600;'>{med2_data[metric]:.1f}%</div>
                            </div>
                        </div>
                        <div style='text-align: center; margin-top: 1rem; color: #a3a8b8; font-size: 1rem;'>{label}</div>
                    </div>
                """, unsafe_allow_html=True)
            
            # Add composition comparison with dark theme
            st.markdown("""
                <div style='background-color: #1e2130; padding: 1.5rem; border-radius: 10px; margin: 2rem 0;'>
                    <h3 style='color: white; margin: 0;'>🧪 Composition Comparison</h3>
                </div>
            """, unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"""
                    <div style='background-color: #262b3d; padding: 1rem; border-radius: 8px; border: 1px solid #2d3748;'>
                        <div style='color: #a3a8b8; font-size: 0.875rem; margin-bottom: 0.5rem;'>{medicine1}</div>
                        <div style='color: white; font-family: monospace; padding: 1rem; background-color: #1e2130; border-radius: 4px;'>
                            {med1_data['Composition']}
                        </div>
                    </div>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
                    <div style='background-color: #262b3d; padding: 1rem; border-radius: 8px; border: 1px solid #2d3748;'>
                        <div style='color: #a3a8b8; font-size: 0.875rem; margin-bottom: 0.5rem;'>{medicine2}</div>
                        <div style='color: white; font-family: monospace; padding: 1rem; background-color: #1e2130; border-radius: 4px;'>
                            {med2_data['Composition']}
                        </div>
                    </div>
                """, unsafe_allow_html=True)
                
        except Exception as e:
            st.error(f"Error in comparison: {str(e)}")
            st.info("Debug Information:")
            st.write("Available columns in df:", df.columns.tolist())
            st.write("Available columns in df_engineered:", df_engineered.columns.tolist())

elif page == "Market Trends":
    st.title("📈 Market Trends Analysis")
    
    # Time series analysis of reviews
    st.markdown("### 📅 Review Trends Over Time")
    
    # Group by manufacturer and calculate rolling averages
    rolling_data = df_engineered.groupby('Manufacturer').agg({
        'Excellent Review %': 'mean',
        'mfr_market_share': 'mean'
    }).reset_index()
    
    # Plot market trends
    fig = px.scatter(
        rolling_data,
        x='mfr_market_share',
        y='Excellent Review %',
        size='mfr_market_share',
        color='Manufacturer',
        title='Market Position vs Performance',
        template='plotly_dark'
    )
    st.plotly_chart(fig, use_container_width=True)

elif page == "AI Insights":
    st.title("🤖 AI-Powered Insights")
    
    # Generate insights about the data
    st.markdown("### 🔍 Key Findings")
    
    # Get only numeric columns for correlation
    numeric_columns = df_engineered.select_dtypes(include=['float64', 'int64']).columns
    
    # Calculate correlations only for numeric columns
    correlations = df_engineered[numeric_columns].corr()
    
    # Find strong correlations
    strong_correlations = []
    for col in correlations.columns:
        for idx in correlations.index:
            if col != idx and abs(correlations.loc[idx, col]) > 0.5:
                strong_correlations.append({
                    'feature1': col,
                    'feature2': idx,
                    'correlation': correlations.loc[idx, col]
                })
    
    # Display insights with better formatting
    if strong_correlations:
        st.markdown("""
            <div style='background-color: #1e2130; padding: 1.5rem; border-radius: 10px; margin-bottom: 1rem;'>
                <h4 style='color: white; margin-bottom: 1rem;'>Strong Feature Relationships Discovered:</h4>
            </div>
        """, unsafe_allow_html=True)
        
        for corr in strong_correlations:
            correlation_strength = abs(corr['correlation'])
            if correlation_strength > 0.8:
                strength_label = "Very Strong"
                color = "#10b981"  # Green
            elif correlation_strength > 0.6:
                strength_label = "Strong"
                color = "#3b82f6"  # Blue
            else:
                strength_label = "Moderate"
                color = "#f59e0b"  # Yellow
                
            st.markdown(f"""
                <div style='background-color: #262b3d; padding: 1rem; border-radius: 8px; margin-bottom: 0.5rem; border: 1px solid #2d3748;'>
                    <div style='display: flex; justify-content: space-between; align-items: center;'>
                        <div>
                            <span style='color: white; font-weight: 500;'>{corr['feature1']}</span>
                            <span style='color: #a3a8b8; margin: 0 0.5rem;'>and</span>
                            <span style='color: white; font-weight: 500;'>{corr['feature2']}</span>
                        </div>
                        <div style='color: {color}; font-weight: 600;'>
                            {strength_label} {abs(corr['correlation']):.2f} 
                            {'positive' if corr['correlation'] > 0 else 'negative'} correlation
                        </div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No strong correlations found between numeric features.")
    
    # Add feature distribution analysis
    st.markdown("""
        <div style='background-color: #1e2130; padding: 1.5rem; border-radius: 10px; margin: 2rem 0;'>
            <h4 style='color: white; margin: 0;'>📊 Feature Distribution Analysis</h4>
        </div>
    """, unsafe_allow_html=True)
    
    # Create distribution plots for numeric features
    for col in numeric_columns:
        fig = px.histogram(
            df_engineered,
            x=col,
            title=f'Distribution of {col}',
            template='plotly_dark'
        )
        fig.update_layout(
            plot_bgcolor='#1e2130',
            paper_bgcolor='#1e2130',
            font={'color': 'white'}
        )
        st.plotly_chart(fig, use_container_width=True)

elif page == "Advanced Analytics":
    st.title("📊 Advanced Analytics")
    
    # Create radio buttons for analysis type
    analysis_type = st.radio(
        "Select Analysis",
        ["📈 Performance", "📊 Market", "🧪 Product", "📊 Trends"],
        horizontal=True,
        key="analysis_selector"
    )
    
    if analysis_type == "📈 Performance":
        # Performance metrics with dark background
        st.markdown("""
            <style>
            [data-testid="stMetric"] {
                background-color: #1e2130;
                padding: 1rem;
                border-radius: 8px;
                box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
                display: flex;
                flex-direction: column-reverse;
            }
            [data-testid="stMetricLabel"] {
                color: #ffffff !important;
                font-size: 1.2rem !important;
                font-weight: 600 !important;
                margin-top: 0.5rem !important;
            }
            [data-testid="stMetricValue"] {
                color: #ffffff !important;
                font-size: 2rem !important;
            }
            [data-testid="stMetricDelta"] {
                color: #10b981 !important;
            }
            </style>
        """, unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                label="Average Excellence",
                value=f"{df['Excellent Review %'].mean():.1f}%",
                help="Average excellence rating across all medicines"
            )
        with col2:
            st.metric(
                label="Market Concentration",
                value=f"{df['mfr_market_share'].std():.2f}",
                help="Standard deviation of market share"
            )
        with col3:
            st.metric(
                label="Innovation Index",
                value=f"{df['composition_complexity'].mean():.1f}",
                help="Average complexity of medicine compositions"
            )
        
        # Performance trend chart with dark background
        fig = px.line(
            df,
            x='Manufacturer',
            y='Excellent Review %',
            title='Performance Trend by Manufacturer'
        )
        fig.update_layout(
            plot_bgcolor='#1e2130',
            paper_bgcolor='#1e2130',
            font={'color': '#ffffff'},
            xaxis={
                'tickangle': 45,
                'gridcolor': '#2d3748',
                'title_font': {'color': '#ffffff'},
                'tickfont': {'color': '#ffffff'}
            },
            yaxis={
                'gridcolor': '#2d3748',
                'title_font': {'color': '#ffffff'},
                'tickfont': {'color': '#ffffff'}
            },
            title_font={'color': '#ffffff'},
            showlegend=True,
            legend={'font': {'color': '#ffffff'}}
        )
        fig.update_traces(line_color='#60a5fa')
        st.plotly_chart(fig, use_container_width=True)
        
    elif analysis_type == "📊 Market":
        # Market share visualization
        fig1 = px.treemap(
            df,
            path=['Manufacturer'],
            values='mfr_market_share',
            title='Market Share Distribution',
            template='plotly_dark'
        )
        fig1.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font={'color': 'white'}
        )
        st.plotly_chart(fig1, use_container_width=True)
        
        # Market performance scatter
        fig2 = px.scatter(
            df,
            x='mfr_market_share',
            y='Excellent Review %',
            color='Manufacturer',
            title='Market Share vs Performance',
            template='plotly_dark'
        )
        fig2.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font={'color': 'white'}
        )
        st.plotly_chart(fig2, use_container_width=True)
        
    elif analysis_type == "🧪 Product":
        col1, col2 = st.columns(2)
        
        with col1:
            # Product complexity distribution
            fig1 = px.violin(
                df,
                y='composition_complexity',
                box=True,
                title='Product Complexity Distribution',
                template='plotly_dark'
            )
            fig1.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font={'color': 'white'}
            )
            st.plotly_chart(fig1, use_container_width=True)
            
        with col2:
            # Complexity vs performance
            fig2 = px.scatter(
                df,
                x='composition_complexity',
                y='Excellent Review %',
                color='Manufacturer',
                title='Complexity vs Performance',
                template='plotly_dark'
            )
            fig2.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font={'color': 'white'}
            )
            st.plotly_chart(fig2, use_container_width=True)
            
    elif analysis_type == "📊 Trends":
        # Available metrics for comparison
        metrics = ['Excellent Review %', 'mfr_market_share', 'composition_complexity']
        
        # Select metrics to compare
        selected_metrics = st.multiselect(
            "Select metrics to compare",
            options=metrics,
            default=metrics[:2]
        )
        
        if selected_metrics:
            # Create trend matrix
            fig = px.scatter_matrix(
                df,
                dimensions=selected_metrics,
                color='Manufacturer',
                title='Multi-dimensional Analysis',
                template='plotly_dark'
            )
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font={'color': 'white'}
            )
            st.plotly_chart(fig, use_container_width=True)

elif page == "Recommendations":
    st.title("💡 Smart Recommendations")
    
    # Create tabs
    rec_tabs = st.tabs(["Similar Medicines", "Composition Analysis", "Usage Patterns"])
    
    with rec_tabs[0]:  # Similar Medicines tab
        st.markdown("### 🎯 Find Similar Medicines")
        
        # Medicine selection
        target_medicine = st.selectbox(
            "Select a Medicine",
            options=sorted(df['Medicine Name'].unique()),
            key="medicine_selector"
        )
        
        if target_medicine:
            with st.spinner("Finding similar medicines..."):
                similar_medicines = get_similar_medicines(target_medicine, df)
                
                if similar_medicines:
                    # Use columns for better layout
                    cols = st.columns(2)
                    for idx, med in enumerate(similar_medicines):
                        with cols[idx % 2]:
                            st.markdown(f"""
                                <div style='
                                    background-color: #1e2130;
                                    padding: 1rem;
                                    border-radius: 5px;
                                    margin-bottom: 0.5rem;
                                    border: 1px solid #2d3748;
                                '>
                                    <h4 style='color: white; margin: 0;'>{med['Medicine Name']}</h4>
                                    <p style='color: #a3a8b8; margin: 0.5rem 0;'>
                                        Similarity: {med['Similarity']:.1f}%<br>
                                        Manufacturer: {med['Manufacturer']}<br>
                                        Excellence Rating: {med['Excellence Rating']:.1f}%
                                    </p>
                                    <details style='margin-top: 0.5rem;'>
                                        <summary style='color: #60a5fa; cursor: pointer;'>View Details</summary>
                                        <p style='color: #a3a8b8; margin: 0.5rem 0;'>
                                            <strong>Composition:</strong><br>
                                            {med['Composition']}<br><br>
                                            <strong>Uses:</strong><br>
                                            {med['Uses']}
                                        </p>
                                    </details>
                                </div>
                            """, unsafe_allow_html=True)
                else:
                    st.warning("No similar medicines found.")
    
    with rec_tabs[1]:  # Composition Analysis tab
        st.markdown("### 🧪 Composition Analysis")
        composition_data = df['Composition'].str.split(',').explode().str.strip()
        top_ingredients = composition_data.value_counts().head(10)
        
        fig = px.bar(
            x=top_ingredients.index,
            y=top_ingredients.values,
            title='Most Common Ingredients',
            template='plotly_dark'
        )
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font={'color': 'white'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with rec_tabs[2]:  # Usage Patterns tab
        st.markdown("### 📊 Usage Patterns")
        usage_data = df['Uses'].str.split(',').explode().str.strip()
        top_uses = usage_data.value_counts().head(10)
        
        fig = px.pie(
            values=top_uses.values,
            names=top_uses.index,
            title='Common Usage Categories',
            template='plotly_dark'
        )
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font={'color': 'white'}
        )
        st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p>Created with ❤️ by Frank Van Laarhoven</p>
        <p>Data source: Kaggle Pharmaceutical Dataset</p>
    </div>
    """,
    unsafe_allow_html=True
)

def validate_manufacturer_columns(df):
    """Validate required manufacturer columns exist in DataFrame"""
    required_columns = [
        'mfr_market_tier', 
        'mfr_avg_review', 
        'mfr_market_share',
        'mfr_total_medicines',
        'mfr_performance_tier'
    ]
    
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required manufacturer columns: {missing_columns}")
    
    return True 

def get_colab_link(notebook_content):
    """Generate a Google Colab link for the notebook"""
    import base64
    import urllib.parse
    
    # Create GitHub repository URL for the notebook
    github_url = "https://github.com/FrankAsanteVanLaarhoven/Pharmaceutical-Analysis-Dashboard/blob/main/notebooks/Pharma_Analysis.ipynb"
    
    # Create Colab URL
    colab_base_url = "https://colab.research.google.com/github"
    colab_url = f"{colab_base_url}/FrankAsanteVanLaarhoven/Pharmaceutical-Analysis-Dashboard/blob/main/notebooks/Pharma_Analysis.ipynb"
    
    # Encode notebook content
    notebook_b64 = base64.b64encode(notebook_content.encode()).decode()
    
    # Add parameters
    params = {
        'content': notebook_b64,
        'create': 'true'
    }
    
    return f"https://colab.research.google.com/notebook#{urllib.parse.urlencode(params)}"

def load_notebook_content(notebook_path):
    """Load notebook content from file"""
    try:
        if not notebook_path.exists():
            raise FileNotFoundError(f"Notebook not found at: {notebook_path}")
            
        with open(notebook_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        raise Exception(f"Error loading notebook: {str(e)}")

# Add this right after your imports, before any page definitions
def calculate_similarity(str1, str2):
    """Calculate similarity between two strings using Jaccard similarity"""
    if not isinstance(str1, str) or not isinstance(str2, str):
        return 0
    
    words1 = set(str1.lower().split())
    words2 = set(str2.lower().split())
    
    intersection = len(words1.intersection(words2))
    union = len(words1.union(words2))
    
    return (intersection / union * 100) if union > 0 else 0

@st.cache_data
def get_similar_medicines(medicine_name, df):
    """Find similar medicines based on composition and uses"""
    if medicine_name not in df['Medicine Name'].values:
        return None
    
    try:
        target_data = df[df['Medicine Name'] == medicine_name].iloc[0]
        similarities = []
        
        for _, row in df.iterrows():
            if row['Medicine Name'] != medicine_name:
                comp_sim = calculate_similarity(str(row['Composition']), str(target_data['Composition']))
                uses_sim = calculate_similarity(str(row['Uses']), str(target_data['Uses']))
                
                similarity_score = (comp_sim + uses_sim) / 2
                if similarity_score > 0:
                    similarities.append({
                        'Medicine Name': row['Medicine Name'],
                        'Manufacturer': row['Manufacturer'],
                        'Similarity': similarity_score,
                        'Excellence Rating': row['Excellent Review %'],
                        'Composition': row['Composition'],
                        'Uses': row['Uses']
                    })
        
        return sorted(similarities, key=lambda x: x['Similarity'], reverse=True)[:5]
        
    except Exception as e:
        st.error(f"Error finding similar medicines: {str(e)}")
        return None

@st.cache_data
def analyze_compositions(df):
    """Analyze medicine compositions"""
    composition_data = df['Composition'].str.split(',').explode().str.strip()
    return composition_data.value_counts().head(10)

@st.cache_data
def analyze_usage_patterns(df):
    """Analyze medicine usage patterns"""
    usage_data = df['Uses'].str.split(',').explode().str.strip()
    return usage_data.value_counts().head(10)

# End the file with these instructions and remove everything after them
st.markdown("""
---
### Environment Setup Instructions

#### Google Colab
- No installation required
- Runs in your browser
- Free GPU access
- Requires Google account

#### Jupyter Notebook
- Local installation required
- Full control over environment
- Best for data science work
- Supports all Python packages

#### VS Code
- Install VS Code
- Add Jupyter extension
- Great for development
- Integrated Git support
""")

# Add these configurations for better plot performance
plot_config = {
    'displayModeBar': False,
    'staticPlot': True,
    'responsive': True,
    'scrollZoom': False,
    'showTips': False
}

# Optimize figure creation
@st.cache_data
def optimize_trend_figure(fig):
    """Optimize trend figure for performance"""
    # Reduce number of points if too many
    if len(fig.data) > 1000:
        for trace in fig.data:
            if len(trace.x) > 1000:
                step = len(trace.x) // 1000
                trace.x = trace.x[::step]
                trace.y = trace.y[::step]
    
    # Optimize layout
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font={'color': 'white'},
        margin=dict(t=30, l=10, r=10, b=10),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        height=600
    )
    
    return fig

@st.cache_data
def create_performance_chart(data):
    """Create performance trend visualization"""
    fig = px.line(
        data,
        x='Manufacturer',
        y='Excellent Review %',
        title='Performance Trend by Manufacturer',
        template='plotly_dark'
    )
    
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font={'color': 'white'},
        xaxis={'tickangle': 45},
        showlegend=True,
        height=500,
        margin=dict(t=30, l=10, r=10, b=10)
    )
    
    return fig

@st.cache_data
def create_market_charts(data):
    """Create market analysis visualizations"""
    # Market Share Treemap
    treemap = px.treemap(
        data,
        path=['Manufacturer'],
        values='mfr_market_share',
        color='Excellent Review %',
        title='Market Share Distribution',
        template='plotly_dark'
    )
    treemap.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font={'color': 'white'},
        margin=dict(t=30, l=10, r=10, b=10)
    )
    
    # Market Performance Scatter
    scatter = px.scatter(
        data,
        x='mfr_market_share',
        y='Excellent Review %',
        size='mfr_total_medicines',
        color='Manufacturer',
        title='Market Share vs Performance',
        template='plotly_dark'
    )
    scatter.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font={'color': 'white'},
        showlegend=True,
        height=500,
        margin=dict(t=30, l=10, r=10, b=10)
    )
    
    return treemap, scatter

@st.cache_data
def create_product_charts(data):
    """Create product analysis visualizations"""
    # Product Complexity Violin Plot
    violin = px.violin(
        data,
        y='composition_complexity',
        box=True,
        title='Product Complexity Distribution',
        template='plotly_dark'
    )
    violin.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font={'color': 'white'},
        showlegend=False,
        height=500,
        margin=dict(t=30, l=10, r=10, b=10)
    )
    
    # Complexity vs Performance Scatter
    scatter = px.scatter(
        data,
        x='composition_complexity',
        y='Excellent Review %',
        color='Manufacturer',
        title='Complexity vs Performance',
        template='plotly_dark'
    )
    scatter.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font={'color': 'white'},
        showlegend=True,
        height=500,
        margin=dict(t=30, l=10, r=10, b=10)
    )
    
    return violin, scatter

# Remove everything after this point