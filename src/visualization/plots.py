import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import joblib

def plot_feature_importance(top_n=10):
    """
    CLIENT SIDE: Creates an interactive, hoverable Plotly bar chart for feature importance.
    """
    try:
        model = joblib.load('model/best_xgb_model.pkl')
        features = joblib.load('model/feature_columns.pkl')
    except FileNotFoundError:
        return None

    importance = model.feature_importances_
    
    importance_df = pd.DataFrame({
        'Feature': features,
        'Importance': importance
    }).sort_values(by='Importance', ascending=True).tail(top_n)

    # Interactive Plotly Bar Chart
    fig = px.bar(
        importance_df, 
        x='Importance', 
        y='Feature', 
        orientation='h',
        title='Top Risk Factors (Interactive)',
        color='Importance',
        color_continuous_scale='Reds'
    )
    
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=0, t=40, b=0),
        coloraxis_showscale=False
    )
    
    return fig

def plot_risk_gauge(probability):
    """
    CLIENT SIDE: Creates a sleek, interactive 3-tier speedometer gauge.
    """
    # Determine color based on 3 risk tiers (20% and 40% thresholds)
    if probability < 0.20:
        bar_color = "#00cc96"  # Green
    elif probability < 0.40:
        bar_color = "#ffa15a"  # Orange/Yellow
    else:
        bar_color = "#ff4b4b"  # Red
        
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = probability * 100,
        number = {'suffix': "%", 'font': {'size': 40}},
        title = {'text': "Default Probability", 'font': {'size': 20}},
        gauge = {
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': bar_color},
            'bgcolor': "rgba(0,0,0,0)",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 20], 'color': "rgba(0, 204, 150, 0.15)"},   # Safe Zone
                {'range': [20, 40], 'color': "rgba(255, 161, 90, 0.15)"},  # Warning Zone
                {'range': [40, 100], 'color': "rgba(255, 75, 75, 0.15)"}   # Danger Zone
            ],
            'threshold': {
                'line': {'color': "white", 'width': 3},
                'thickness': 0.75,
                'value': probability * 100
            }
        }
    ))
    
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", 
        font={'color': "white", 'family': "Arial"},
        height=300,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    
    return fig