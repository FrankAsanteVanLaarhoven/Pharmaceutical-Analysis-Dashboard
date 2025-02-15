"""Plotting configuration for the application"""
import matplotlib
matplotlib.use('Agg', force=True)

import seaborn as sns
import plotly.io as pio
import matplotlib.pyplot as plt

# Export settings
PLOT_CONFIG = {
    'matplotlib': {
        'backend': 'Agg',
        'style': 'default',
        'rcParams': {
            'figure.figsize': (10, 6),
            'axes.grid': True,
            'grid.alpha': 0.3,
            'axes.facecolor': '#1e1e1e',  # Dark background
            'figure.facecolor': '#1e1e1e',  # Dark background
            'text.color': 'white',
            'axes.labelcolor': 'white',
            'axes.edgecolor': 'white',
            'xtick.color': 'white',
            'ytick.color': 'white',
            'savefig.dpi': 100,
            'savefig.bbox': 'tight',
            'savefig.pad_inches': 0.2,
            'font.size': 10,
            'axes.labelsize': 12,
            'axes.titlesize': 14
        }
    },
    'plotly': {
        'template': 'plotly_dark',
        'renderer': 'browser',
        'config': {
            'displayModeBar': True,
            'showTips': True,
            'responsive': True,
            'plot_bgcolor': '#1e1e1e',
            'paper_bgcolor': '#1e1e1e',
            'font': {'color': 'white'}
        }
    },
    'seaborn': {
        'style': 'darkgrid',
        'context': 'notebook',
        'palette': 'deep'
    }
}

# Configure matplotlib
plt.style.use('default')
matplotlib.rcParams.update(PLOT_CONFIG['matplotlib']['rcParams'])

# Configure seaborn with dark theme
sns.set_theme(
    style='darkgrid',
    context='notebook',
    palette='deep',
    rc={
        'axes.facecolor': '#1e1e1e',
        'figure.facecolor': '#1e1e1e',
        'text.color': 'white',
        'axes.labelcolor': 'white',
        'axes.edgecolor': 'white',
        'xtick.color': 'white',
        'ytick.color': 'white'
    }
)

# Configure plotly
pio.templates.default = PLOT_CONFIG['plotly']['template']

# Custom color scales for correlations
CORRELATION_COLORSCALE = [
    [0.0, '#2d0c4e'],
    [0.2, '#45278d'],
    [0.4, '#3b518b'],
    [0.6, '#2c718e'],
    [0.8, '#21908d'],
    [1.0, '#27ae60']
] 