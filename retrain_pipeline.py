import os
from data.load_data import load_training_data
from src.preprocessing import preprocess_training_data
from src.modeling.train_model import run_training
from src.modeling.evaluate import run_evaluation

def setup_developer_backend():
    print("🚀 STARTING DEVELOPER PIPELINE 🚀\n")
    print("="*50)

    print("STEP 1: Loading raw training data...")
    raw_df = load_training_data("data/raw/cs-training.csv")
    print(f"Loaded {len(raw_df)} rows of raw data.\n")

    print("STEP 2: Cleaning data and engineering features...")
    preprocess_training_data(raw_df)
    print("\n")

    print("STEP 3: Splitting data and training XGBoost Model...")
    run_training()
    print("\n")

    print("STEP 4: Running final evaluation on the 20% holdout set...")
    run_evaluation()

    print("\n✅ PIPELINE COMPLETE! The backend is fully set up.")

if __name__ == "__main__":
    setup_developer_backend()