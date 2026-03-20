import pandas as pd

def load_training_data(filepath="data/raw/cs-training.csv"):
    """
    Loads the raw training CSV and returns the DataFrame.
    All splitting and preprocessing will be handled downstream.
    """
    df = pd.read_csv(filepath)
    return df

def load_uploaded_data(uploaded_file):
    """
    Loads the demo/uploaded CSV for Streamlit inference.
    """
    df = pd.read_csv(uploaded_file)
    return df