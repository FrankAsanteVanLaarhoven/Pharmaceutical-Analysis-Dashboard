import sys
import importlib

def test_imports():
    required_packages = [
        'numpy',
        'pandas',
        'streamlit',
        'plotly',
        'matplotlib',
        'scikit-learn',
        'xgboost',
        'pandas_profiling',
        'nbformat'
    ]
    
    results = []
    for package in required_packages:
        try:
            module = importlib.import_module(package)
            version = getattr(module, '__version__', 'unknown')
            results.append(f"✅ {package} (version {version})")
        except ImportError as e:
            results.append(f"❌ {package} (ERROR: {str(e)})")
    
    return results

def main():
    print("Python version:", sys.version)
    print("\nChecking required packages:")
    for result in test_imports():
        print(result)

if __name__ == "__main__":
    main() 