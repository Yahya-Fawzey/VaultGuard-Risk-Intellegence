import pandas as pd
import joblib

def predict_risk(preprocessed_df):
    """
    CLIENT SIDE: Takes fully cleaned and engineered data, loads the saved XGBoost model,
    and returns the model's default predictions and raw risk probabilities.
    """
    # Loading the model and the exact feature blueprint
    try:
        xgb_model = joblib.load('model/best_xgb_model.pkl')
        feature_columns = joblib.load('model/feature_columns.pkl')
    except FileNotFoundError:
        raise FileNotFoundError(
            "Model files missing! The developer side needs to run train_model.py first."
        )
    
    # Aligning columns 
    
    aligned_df = preprocessed_df.reindex(columns=feature_columns, fill_value=0)
    
    # Getting the exact probability of defaulting (Class 1)
    probabilities = xgb_model.predict_proba(aligned_df)[:, 1]
    
    # Getting the model's standard default prediction (0 or 1)
    predictions = xgb_model.predict(aligned_df)
    
    return predictions, probabilities