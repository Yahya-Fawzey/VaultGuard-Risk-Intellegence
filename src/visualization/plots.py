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

def plot_risk_gauge(probability, threshold=0.30):
    """
    CLIENT SIDE: Creates a sleek, interactive speedometer gauge for the risk score.
    """
    # Determine color based on risk
    bar_color = "#ff4b4b" if probability >= threshold else "#00cc96"
    
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
                {'range': [0, threshold * 100], 'color': "rgba(0, 204, 150, 0.2)"},
                {'range': [threshold * 100, 100], 'color': "rgba(255, 75, 75, 0.2)"}
            ],
            'threshold': {
                'line': {'color': "white", 'width': 4},
                'thickness': 0.75,
                'value': threshold * 100
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