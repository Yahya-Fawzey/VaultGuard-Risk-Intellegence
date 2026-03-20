import streamlit as st
import pandas as pd
import sys
import os

# Add the project root to the path so we can import our backend modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.preprocessing import preprocess_new_data
from src.modeling.predict import predict_risk
from src.visualization.plots import plot_risk_gauge, plot_feature_importance

# =================================================================
# 1. PAGE CONFIGURATION & STYLING
# =================================================================
st.set_page_config(
    page_title="VaultGuard Risk Intelligence", 
    page_icon="🏦", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a sleek banking vibe
st.markdown("""
    <style>
    .main-title { font-size: 3rem; font-weight: 800; color: #F8FAFC; margin-bottom: 0; }
    .sub-title { font-size: 1.2rem; color: #94A3B8; margin-bottom: 2rem; }
    .risk-alert { padding: 1rem; border-radius: 0.5rem; background-color: rgba(255, 75, 75, 0.1); border-left: 5px solid #ff4b4b; }
    .safe-alert { padding: 1rem; border-radius: 0.5rem; background-color: rgba(0, 204, 150, 0.1); border-left: 5px solid #00cc96; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-title">🏦 VaultGuard Risk Intelligence</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Advanced AI Credit Risk Assessment Engine</p>', unsafe_allow_html=True)

# =================================================================
# 2. HELPER FUNCTION: APPLICANT DEEP DIVE
# =================================================================
def display_deep_dive(probability, raw_data_row):
    """Renders the detailed charts and risk factors for a specific applicant."""
    st.divider()
    st.subheader("🔍 Applicant Deep Dive")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # 1. The Interactive Plotly Gauge
        st.plotly_chart(plot_risk_gauge(probability), use_container_width=True)
        
    with col2:
        # 2. Heuristic Reason Generation
        st.markdown("### Key Risk Indicators")
        reasons = []
        
        # Check specific values from the raw data to give realistic reasons
        if raw_data_row['NumberOfTimes90DaysLate'].values[0] > 0:
            reasons.append("🚩 **Severe Delinquency:** Applicant has a history of being 90+ days late.")
        if raw_data_row['NumberOfTime30-59DaysPastDueNotWorse'].values[0] > 1:
            reasons.append("⚠️ **Recent Lateness:** Multiple recent 30-59 day late payments.")
        if raw_data_row['DebtRatio'].values[0] > 0.5:
            reasons.append("⚠️ **High Debt Burden:** Debt-to-Income ratio exceeds 50%.")
        if raw_data_row['RevolvingUtilizationOfUnsecuredLines'].values[0] > 0.8:
            reasons.append("🚩 **Maxed Credit:** High utilization of available credit lines.")
        if raw_data_row['NumberRealEstateLoansOrLines'].values[0] == 0:
            reasons.append("ℹ️ **Lack of Secured Assets:** No active real estate loans/mortgages.")
            
        if not reasons and probability < 0.30:
            st.success("✅ **Applicant profile is exceptionally clean. No major risk factors detected.**")
        elif not reasons and probability >= 0.30:
            st.warning("⚠️ **Model detected complex risk patterns based on global feature combinations.**")
            
        for r in reasons:
            st.write(r)

    # 3. Show Global Feature Importance underneath to explain how the AI thinks
    with st.expander("📊 View AI Decision Drivers (Global Model Insights)"):
        st.plotly_chart(plot_feature_importance(), use_container_width=True)


# =================================================================
# 3. SIDEBAR NAVIGATION
# =================================================================
mode = st.sidebar.radio("Select Operation Mode:", ["Batch CSV Analysis", "Single Individual Entry"])
st.sidebar.divider()
st.sidebar.info("VaultGuard relies on an XGBoost model trained on historical default data to estimate 2-year credit delinquency probability.")

# =================================================================
# MODE A: BATCH CSV ANALYSIS
# =================================================================
if mode == "Batch CSV Analysis":
    st.header("📂 Batch Portfolio Analysis")
    st.write("Upload a CSV dataset. The system will flag high-risk individuals and prioritize them at the top of the queue.")
    
    uploaded_file = st.file_uploader("Upload customer data (CSV)", type="csv")
    
    if uploaded_file is not None:
        raw_batch_df = pd.read_csv(uploaded_file)
        
        with st.spinner("VaultGuard AI is evaluating portfolio risk..."):
            try:
                # 1. Preprocess & Predict using the backend
                cleaned_batch_df = preprocess_new_data(raw_batch_df, filename="batch_processed.csv")
                predictions, probabilities = predict_risk(cleaned_batch_df)
                
                # 2. Attach results to the original raw dataframe for display
                display_df = raw_batch_df.copy()
                display_df['Risk_Probability'] = probabilities
                display_df['Status'] = ["🚨 HIGH RISK" if p >= 0.30 else "✅ SAFE" for p in probabilities]
                
                # 3. SORTING: Push all Risky people to the very top!
                display_df = display_df.sort_values(by='Risk_Probability', ascending=False).reset_index(drop=True)
                
                # 4. Display high-level metrics
                col_a, col_b, col_c = st.columns(3)
                col_a.metric("Total Applicants Evaluated", len(display_df))
                col_b.metric("High Risk Identified", sum(predictions >= 0.30))
                safe_count = len(display_df) - sum(predictions >= 0.30)
                col_c.metric("Safe Applicants", safe_count)
                
                # 5. Show the beautifully sorted table
                st.write("### AI Assessment Results Queue")
                
                # Format the probability to look like a percentage in the table
                styled_df = display_df.head(100).style.format({'Risk_Probability': "{:.1%}"}).applymap(
                    lambda x: 'background-color: rgba(255, 75, 75, 0.2)' if '🚨 HIGH RISK' in str(x) else '',
                    subset=['Status']
                )
                st.dataframe(styled_df, use_container_width=True, height=300)
                
                # 6. INTERACTIVE DEEP DIVE DROPDOWN
                st.write("### Inspect Specific Applicant")
                applicant_indices = display_df.index.tolist()
                
                # Create a readable label for the dropdown
                dropdown_options = {
                    idx: f"Index {idx} | Status: {display_df.loc[idx, 'Status']} | Probability: {display_df.loc[idx, 'Risk_Probability']*100:.1f}%" 
                    for idx in applicant_indices
                }
                
                selected_idx = st.selectbox(
                    "Select an applicant from the queue to view their specific risk factors:", 
                    options=applicant_indices,
                    format_func=lambda x: dropdown_options[x]
                )
                
                if selected_idx is not None:
                    # Extract the single row of raw data and its probability
                    selected_raw_row = display_df.loc[[selected_idx]]
                    selected_prob = display_df.loc[selected_idx, 'Risk_Probability']
                    display_deep_dive(selected_prob, selected_raw_row)

            except Exception as e:
                st.error(f"An error occurred in the backend pipeline: {e}")

# =================================================================
# MODE B: SINGLE INDIVIDUAL ENTRY
# =================================================================
elif mode == "Single Individual Entry":
    st.header("👤 Manual Applicant Assessment")
    st.write("Enter financial details below. All fields are required to generate an assessment.")
    
    with st.form("single_customer_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.number_input("Age", min_value=18, max_value=120, step=1, value=None)
            income = st.number_input("Monthly Income ($)", min_value=0.0, step=100.0, value=None)
            
            # UPDATED: Now asks for percentage (0 to 100)
            debt_ratio_pct = st.number_input("Debt Ratio (%)", min_value=0.0, max_value=100.0, step=1.0, value=None)
            
            dependents = st.number_input("Number of Dependents", min_value=0, step=1, value=None)
            
            # UPDATED: Now asks for percentage (0 to 100)
            revol_util_pct = st.number_input("Revolving Utilization of Unsecured Lines (%)", min_value=0.0, max_value=100.0, step=1.0, value=None)
            
        with col2:
            open_lines = st.number_input("Open Credit Lines & Loans", min_value=0, step=1, value=None)
            real_estate = st.number_input("Real Estate Loans", min_value=0, step=1, value=None)
            late_30_59 = st.number_input("Times 30-59 Days Late", min_value=0, step=1, value=None)
            late_60_89 = st.number_input("Times 60-89 Days Late", min_value=0, step=1, value=None)
            late_90 = st.number_input("Times 90+ Days Late", min_value=0, step=1, value=None)

        submit_clicked = st.form_submit_button("Run AI Assessment", type="primary")

    if submit_clicked:
        inputs = [age, income, debt_ratio_pct, dependents, revol_util_pct, open_lines, real_estate, late_30_59, late_60_89, late_90]
        
        if None in inputs:
            st.error("⚠️ **Validation Error:** You must fill out ALL fields before generating an assessment.")
        else:
            with st.spinner("Processing applicant profile..."):
                try:
                    # CONVERT PERCENTAGES BACK TO DECIMALS FOR THE AI MODEL
                    debt_ratio_model = debt_ratio_pct / 100.0
                    revol_util_model = revol_util_pct / 100.0

                    # 1. Create raw dataframe using the converted decimals
                    raw_data = pd.DataFrame([{
                        'RevolvingUtilizationOfUnsecuredLines': revol_util_model,
                        'age': age,
                        'NumberOfTime30-59DaysPastDueNotWorse': late_30_59,
                        'DebtRatio': debt_ratio_model,
                        'MonthlyIncome': income,
                        'NumberOfOpenCreditLinesAndLoans': open_lines,
                        'NumberOfTimes90DaysLate': late_90,
                        'NumberRealEstateLoansOrLines': real_estate,
                        'NumberOfTime60-89DaysPastDueNotWorse': late_60_89,
                        'NumberOfDependents': dependents
                    }])
                    
                    # 2. Process & Predict
                    cleaned_df = preprocess_new_data(raw_data, filename="manual_entry_processed.csv")
                    predictions, probabilities = predict_risk(cleaned_df)
                    probability = probabilities[0]
                    
                    # 3. High-level result banner
                    if probability >= 0.30:
                        st.markdown('<div class="risk-alert">🚨 <b>HIGH RISK STATUS DETECTED</b><br>Applicant exceeds the acceptable default probability threshold.</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="safe-alert">✅ <b>SAFE STATUS</b><br>Applicant falls within acceptable risk parameters.</div>', unsafe_allow_html=True)
                        st.balloons()
                        
                    # 4. Show the interactive deep dive!
                    display_deep_dive(probability, raw_data)
                    
                except Exception as e:
                    st.error(f"An error occurred: {e}")