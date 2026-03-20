import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

def run_training():
    # loading the preprocessed training data
    input_path = "data/processed/training_data/processed_train.csv"
    print(f"Loading preprocessed data from {input_path}...")
    
    try:
        df = pd.read_csv(input_path)
    except FileNotFoundError:
        print(f"Error: {input_path} not found. Make sure to run the preprocessing script first!")
        return

    # separating features and target
    target_col = 'SeriousDlqin2yrs'
    X = df.drop(columns=[target_col])
    y = df[target_col]

    print("Splitting data into 80% train and 20% test...")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # saving the test split for later evaluation in the Streamlit app
    test_df = X_test.copy()
    test_df[target_col] = y_test
    test_df.to_csv("data/processed/training_data/test_split.csv", index=False)
    print("Saved test split to data/processed/training_data/test_split.csv")

    # calculating scale_pos_weight for XGBoost to handle class imbalance
    scale_weight = (y_train == 0).sum() / (y_train == 1).sum()
    
    print("Training XGBoost model on the 80% training data...")
    # setting up with the specified hyperparameters from previous researching..
    xgb_model = XGBClassifier(
        subsample=0.7, 
        n_estimators=200, 
        max_depth=3, 
        learning_rate=0.05, 
        colsample_bytree=0.7,
        scale_pos_weight=scale_weight, 
        eval_metric='auc', 
        random_state=42
    )
    
    # training the model
    xgb_model.fit(X_train, y_train)

    # saving .pkl files for the model and feature columns to the 'model/' folder
    os.makedirs("model", exist_ok=True)
    joblib.dump(xgb_model, 'model/best_xgb_model.pkl')
    joblib.dump(list(X_train.columns), 'model/feature_columns.pkl')
    
    print("Success! XGBoost model and feature columns saved to the 'model/' folder.")

if __name__ == "__main__":
    run_training()