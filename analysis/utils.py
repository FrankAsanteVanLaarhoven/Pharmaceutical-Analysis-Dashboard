def check_column_names(df):
    """Print actual column names in the data file"""
    print("\nActual columns in data file:")
    for col in df.columns:
        print(f"  - {col}")
    return df.columns.tolist() 