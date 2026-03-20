import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay

def run_evaluation():
    # Loading the 20% unseen test split saved during training
    test_data_path = "data/processed/training_data/test_split.csv"
    print(f"Loading unseen test data from {test_data_path}...\n")
    
    try:
        test_df = pd.read_csv(test_data_path)
    except FileNotFoundError:
        print(f"Error: {test_data_path} not found. Run train_model.py first!")
        return

    # Separating features and target
    target_col = 'SeriousDlqin2yrs'
    X_test = test_df.drop(columns=[target_col])
    y_test = test_df[target_col]

    # Loading the saved model and feature columns 
    print("Loading saved XGBoost model from the 'model/' folder...")
    try:
        xgb_model = joblib.load('model/best_xgb_model.pkl')
        feature_columns = joblib.load('model/feature_columns.pkl')
    except FileNotFoundError:
        print("Error: Model files not found. Run train_model.py first!")
        return

    # Ensuring test data columns exactly match the model's blueprint
    X_test = X_test.reindex(columns=feature_columns, fill_value=0)

    # Predictions and probabilities for evaluation
    print("Scoring the model...\n")
    predictions = xgb_model.predict(X_test)
    probabilities = xgb_model.predict_proba(X_test)[:, 1]

    # Printing the Final Text Report 
    roc_auc = roc_auc_score(y_test, probabilities)
    
    print("="*50)
    print("🏆 FINAL MODEL EVALUATION REPORT 🏆")
    print("="*50)
    print(f"ROC-AUC Score: {roc_auc:.4f}\n")
    
    print("Classification Report:")
    print(classification_report(y_test, predictions))
    print("="*50)
    
    # Visual Confusion Matrix
    print("Opening visual Confusion Matrix... (Close the popup window to finish the script)")
    cm = confusion_matrix(y_test, predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Safe (0)', 'Risky (1)'])
    
    
    disp.plot(cmap=plt.cm.Blues, values_format='d')
    plt.title("XGBoost Confusion Matrix")
    
    
    plt.show()

if __name__ == "__main__":
    run_evaluation()