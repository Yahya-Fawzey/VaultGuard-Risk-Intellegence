import pandas as pd
import os

def preprocess_training_data(df):
    """
    DEVELOPER SIDE: Cleans data dynamically, applies feature engineering, 
    and SAVES the cleaned CSV to data/processed/training_data/
    """
    df = df.copy()
    
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
        
    # DYNAMIC MEDIANS Calculated straight from the training data
    df['MonthlyIncome'] = df['MonthlyIncome'].fillna(df['MonthlyIncome'].median())
    df['NumberOfDependents'] = df['NumberOfDependents'].fillna(df['NumberOfDependents'].median())
    
    # FEATURE ENGINEERING
    df['total_late_payments'] = (
        df['NumberOfTimes90DaysLate'] + 
        df['NumberOfTime30-59DaysPastDueNotWorse'] + 
        df['NumberOfTime60-89DaysPastDueNotWorse']
    )
    df['debt_income_ratio'] = df['DebtRatio'] / (df['MonthlyIncome'] + 1)
    df['credit_pressure'] = df['RevolvingUtilizationOfUnsecuredLines'] * df['DebtRatio']
    
    # SAVING PHYSICAL CSV TO PROCESSED FOLDER
    output_dir = "data/processed/training_data"
    os.makedirs(output_dir, exist_ok=True)
    output_path = f"{output_dir}/processed_train.csv"
    
    df.to_csv(output_path, index=False)
    print(f"Success! Cleaned training data saved to {output_path}")
    
    return df


def preprocess_new_data(df, filename="processed_uploaded.csv"):
    """
    CLIENT SIDE: Cleans uploaded data dynamically, applies feature engineering,
    and SAVES the cleaned CSV to data/processed/uploaded_data/
    """
    df = df.copy()
    
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
        
    # DYNAMIC MEDIANS 
    
    income_median = df['MonthlyIncome'].median()
    df['MonthlyIncome'] = df['MonthlyIncome'].fillna(income_median if not pd.isna(income_median) else 0)
    
    dep_median = df['NumberOfDependents'].median()
    df['NumberOfDependents'] = df['NumberOfDependents'].fillna(dep_median if not pd.isna(dep_median) else 0)
    
    # FEATURE ENGINEERING
    df['total_late_payments'] = (
        df['NumberOfTimes90DaysLate'] + 
        df['NumberOfTime30-59DaysPastDueNotWorse'] + 
        df['NumberOfTime60-89DaysPastDueNotWorse']
    )
    df['debt_income_ratio'] = df['DebtRatio'] / (df['MonthlyIncome'] + 1)
    df['credit_pressure'] = df['RevolvingUtilizationOfUnsecuredLines'] * df['DebtRatio']
    
    # SAVING PHYSICAL CSV TO PROCESSED FOLDER
    output_dir = "data/processed/uploaded_data"
    os.makedirs(output_dir, exist_ok=True)
    output_path = f"{output_dir}/{filename}"
    
    df.to_csv(output_path, index=False)
    print(f"Success! Cleaned uploaded data saved to {output_path}")
    
    return df