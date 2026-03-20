# 🛡️ VaultGuard Risk Intelligence

![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B)
![Machine Learning](https://img.shields.io/badge/Model-XGBoost-orange)
![License](https://img.shields.io/badge/license-MIT-green)

## 📌 Project Overview
**VaultGuard Risk Intelligence** is an end-to-end machine learning solution designed to evaluate, predict, and visualize risk metrics. Whether deployed for credit scoring, fraud detection, or operational risk assessment, VaultGuard leverages advanced predictive modeling to transform raw data into actionable intelligence.

This repository contains the complete pipeline: from data ingestion and preprocessing to model training with XGBoost, and finally, a dynamic user-facing deployment utilizing Streamlit.

---

## 🎯 The Problem & The Approach

### The Problem
In modern financial and security ecosystems, manual risk assessment is slow, prone to human error, and unable to scale with massive datasets. Clients need a way to instantly evaluate risk profiles based on historical data, pinpointing high-risk entities before they impact the bottom line.

### The Approach
VaultGuard solves this by utilizing a supervised machine learning approach:
1. **Predictive Engine:** An optimized XGBoost classifier/regressor trained on historical risk data.
2. **Automated Pipeline:** Modularized Python scripts that handle data cleaning, feature engineering, and model retraining without manual intervention.
3. **Interactive UI:** A Streamlit dashboard that abstracts the complex ML backend, allowing non-technical clients to input data and receive instant risk scores and visualizations.

---

## 📂 Project Architecture & Developer Pipeline

The repository follows a strict separation of concerns, ensuring maintainability and scalability.

    VaultGuard Risk Intelligence/
    │
    ├── app/                      # Client-facing UI
    │   └── streamlit_app.py      # Streamlit dashboard application
    │
    ├── data/                     # Data storage (ignored in version control)
    │   ├── raw/                  # Original, immutable datasets
    │   ├── processed/            # Cleaned data ready for modeling
    │   ├── demo/                 # Sample data for testing the UI
    │   └── load_data.py          # Used for loading data both demo and training
    │
    ├── model/                    # Serialized model artifacts
    │   ├── best_xgb_model.pkl    # Trained XGBoost model
    │   └── feature_columns.pkl   # Saved feature names to ensure input alignment
    │
    ├── notebooks/                # Jupyter notebooks for EDA and prototyping
    │
    ├── src/                      # Core backend source code
    │   ├── modeling/             
    │   │   ├── train_model.py    # Logic for hyperparameter tuning and model training
    │   │   ├── predict.py        # Inference logic for new data
    │   │   └── evaluate.py       # Model performance metrics generation
    │   ├── visualization/        
    │   │   └── plots.py          # Functions for generating UI charts
    │   └── preprocessing.py      # Data cleaning, scaling, and feature engineering
    │
    ├── requirements.txt          # Project dependencies
    ├── retrain_pipeline.py       # End-to-end script to process data and output a new model
    └── Run_VaultGuard.bat        # Batch script for one-click local deployment

### 🛠️ The Developer Pipeline (Backend)
1. **Data Preprocessing (`src/preprocessing.py`):** Raw data is ingested, missing values are handled, categorical variables are encoded, and numerical features are scaled.
2. **Model Training (`src/modeling/train_model.py`):** The processed data is fed into an XGBoost model. Hyperparameters are tuned to maximize accuracy and reliability.
3. **Serialization:** The winning model and the exact feature columns used are pickled into the `model/` directory.
4. **Continuous Learning (`retrain_pipeline.py`):** When new data arrives, this script can be executed to automatically run the preprocessing and training scripts, updating the `.pkl` files dynamically.

---

## 💻 The Client Pipeline (Frontend)

Clients interact with VaultGuard entirely through the **Streamlit Dashboard** (`app/streamlit_app.py`).

1. **User Input:** The client enters parameters either manually via sidebar sliders/inputs or by uploading a batch CSV.
2. **Inference:** The app passes the inputs through the saved `feature_columns.pkl` structure to ensure the data matches the model's expectations, then calls `src/modeling/predict.py`.
3. **Actionable Output:** The dashboard displays:
   - The predicted Risk Score / Classification.
   - Confidence intervals or probability scores.
   - Interactive visualizations (generated via `src/visualization/plots.py`) explaining *why* the model made that decision (e.g., feature importance).

---

## 📊 The Data

*Note: Due to privacy and security, raw operational data is not pushed to this repository.*

- **Data Sources:** Structured operational and historical logs.
- **Features:** The model evaluates key risk indicators extracted during the preprocessing phase.
- **Target Variable:** The model predicts the likelihood of the defined risk event occurring.

---

## 🚀 Installation & Usage

### Prerequisites
- Python 3.8+
- Git

### Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/yourusername/VaultGuard-Risk-Intelligence.git](https://github.com/yourusername/VaultGuard-Risk-Intelligence.git)
   cd "VaultGuard Risk Intelligence"

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt

---

## Running the Application

**For Clients / End-Users:**
Simply run the batch file to launch the dashboard locally:
```dos
Run_VaultGuard.bat
```
(Alternatively, run `streamlit run app/streamlit_app.py` in your terminal).


**For Developers (Model Retraining):**
If you have dropped new raw data into data/raw/ and wish to update the model:
```bash
python retrain_pipeline.py
```


---
## Video Showcase
https://github.com/user-attachments/assets/e30d6141-6028-4f9d-b7dd-60d97b8d4653
