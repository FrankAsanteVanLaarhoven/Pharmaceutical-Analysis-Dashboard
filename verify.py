
import sys
print("Python:", sys.version.split()[0])

try:
    import numpy as np
    print("NumPy:", np.__version__)
except ImportError as e:
    print("❌ NumPy not installed correctly")
    sys.exit(1)

try:
    import pandas as pd
    print("Pandas:", pd.__version__)
except ImportError as e:
    print("❌ Pandas not installed correctly")
    sys.exit(1)

try:
    import matplotlib
    matplotlib.use('Agg')
    print("Matplotlib:", matplotlib.__version__)
except ImportError as e:
    print("❌ Matplotlib not installed correctly")
    sys.exit(1)

try:
    import seaborn as sns
    print("Seaborn:", sns.__version__)
except ImportError as e:
    print("❌ Seaborn not installed correctly")
    sys.exit(1)

try:
    import plotly
    print("Plotly:", plotly.__version__)
except ImportError as e:
    print("❌ Plotly not installed correctly")
    sys.exit(1)

try:
    import streamlit as st
    print("Streamlit:", st.__version__)
except ImportError as e:
    print("❌ Streamlit not installed correctly")
    sys.exit(1)

print("✨ All packages verified successfully!")
