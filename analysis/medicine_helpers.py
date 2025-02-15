import streamlit as st
import pandas as pd

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