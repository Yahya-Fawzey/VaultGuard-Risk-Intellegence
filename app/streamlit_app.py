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

# Custom CSS for 3-tier alerts
st.markdown("""
    <style>
    .main-title { font-size: 3rem; font-weight: 800; color: #F8FAFC; margin-bottom: 0; }
    .sub-title { font-size: 1.2rem; color: #94A3B8; margin-bottom: 2rem; }
    .risk-alert { padding: 1rem; border-radius: 0.5rem; background-color: rgba(255, 75, 75, 0.1); border-left: 5px solid #ff4b4b; margin-bottom: 1rem;}
    .warning-alert { padding: 1rem; border-radius: 0.5rem; background-color: rgba(255, 161, 90, 0.1); border-left: 5px solid #ffa15a; margin-bottom: 1rem;}
    .safe-alert { padding: 1rem; border-radius: 0.5rem; background-color: rgba(0, 204, 150, 0.1); border-left: 5px solid #00cc96; margin-bottom: 1rem;}
    </style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-title">🏦 VaultGuard Risk Intelligence</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Advanced AI Credit Risk Assessment Engine</p>', unsafe_allow_html=True)

# =================================================================
# 2. HELPER FUNCTIONS
# =================================================================
def get_risk_assessment(probability):
    """Returns the Tier, Recommendation, and CSS class based on 3-tier logic."""
    if probability < 0.20:
        return "🟢 LOW RISK", "<b>Action Required:</b> Auto-Approve. Standard prime interest rates apply.", "safe-alert"
    elif probability < 0.40:
        return "🟡 MEDIUM RISK", "<b>Action Required:</b> Route to Manual Underwriting. Consider lowering credit limit, adjusting interest rate, or requiring a co-signer.", "warning-alert"
    else:
        return "🚨 HIGH RISK", "<b>Action Required:</b> High probability of default. Decline application, or require a highly qualified co-signer and secured collateral.", "risk-alert"

def display_deep_dive(probability, raw_data_row):
    """Renders the detailed charts and risk factors for a specific applicant."""
    st.divider()
    
    # Get the specific recommendation
    tier, recommendation, alert_class = get_risk_assessment(probability)
    
    st.subheader(f"🔍 Applicant Deep Dive: {tier}")
    
    # Display the strategic recommendation box (using HTML for bolding to avoid markdown conflict)
    st.markdown(f'<div class="{alert_class}">{recommendation}</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.plotly_chart(plot_risk_gauge(probability), use_container_width=True)
        
    with col2:
        st.markdown("### Key Risk Indicators")
        reasons = []
        
        # LOWERED THRESHOLDS: So it catches more specific reasons instead of the generic fallback
        if raw_data_row['NumberOfTimes90DaysLate'].values[0] > 0:
            reasons.append("🚩 **Severe Delinquency:** Applicant has a history of being 90+ days late.")
        if raw_data_row['NumberOfTime30-59DaysPastDueNotWorse'].values[0] > 0:
            reasons.append("⚠️ **Recent Lateness:** Applicant has recent 30-59 day late payments.")
        if raw_data_row['DebtRatio'].values[0] > 0.35:
            reasons.append("⚠️ **High Debt Burden:** Debt-to-Income ratio is concerningly high (>35%).")
        if raw_data_row['RevolvingUtilizationOfUnsecuredLines'].values[0] > 0.50:
            reasons.append("🚩 **High Credit Utilization:** Applicant is using over 50% of their available credit.")
        if raw_data_row['NumberRealEstateLoansOrLines'].values[0] == 0:
            reasons.append("ℹ️ **Lack of Secured Assets:** No active real estate loans/mortgages on file.")
            
        if not reasons and probability < 0.20:
            st.success("✅ **Applicant profile is exceptionally clean. No major risk factors detected.**")
        elif not reasons and probability >= 0.20:
            st.warning("⚠️ **Model detected complex risk patterns based on global feature combinations.**")
            
        for r in reasons:
            st.write(r)

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
    st.write("Upload a CSV dataset. The system will categorize individuals into Low, Medium, and High risk tiers.")
    
    uploaded_file = st.file_uploader("Upload customer data (CSV)", type="csv")
    
    if uploaded_file is not None:
        raw_batch_df = pd.read_csv(uploaded_file)
        
        with st.spinner("VaultGuard AI is evaluating portfolio risk..."):
            try:
                cleaned_batch_df = preprocess_new_data(raw_batch_df, filename="batch_processed.csv")
                predictions, probabilities = predict_risk(cleaned_batch_df)
                
                display_df = raw_batch_df.copy()
                display_df['Risk_Probability'] = probabilities
                
                display_df['Status'] = [get_risk_assessment(p)[0] for p in probabilities]
                
                display_df = display_df.sort_values(by='Risk_Probability', ascending=False).reset_index(drop=True)
                
                col_a, col_b, col_c, col_d = st.columns(4)
                col_a.metric("Total Evaluated", len(display_df))
                col_b.metric("High Risk", sum(display_df['Risk_Probability'] >= 0.40))
                col_c.metric("Medium Risk", sum((display_df['Risk_Probability'] >= 0.20) & (display_df['Risk_Probability'] < 0.40)))
                col_d.metric("Low Risk", sum(display_df['Risk_Probability'] < 0.20))
                
                st.write("### AI Assessment Results Queue (Top 100)")
                
                def color_status(val):
                    if '🚨' in str(val): return 'background-color: rgba(255, 75, 75, 0.2)'
                    elif '🟡' in str(val): return 'background-color: rgba(255, 161, 90, 0.2)'
                    return ''
                
                styled_df = display_df.head(100).style.format({'Risk_Probability': "{:.1%}"}).map(color_status, subset=['Status'])
                st.dataframe(styled_df, use_container_width=True, height=400)
                
                st.write("### Inspect Specific Applicant")
                
                # FIX: Gives the entire list of applicants to the dropdown, not just the top 100
                applicant_indices = display_df.index.tolist()
                
                dropdown_options = {
                    idx: f"Index {idx} | {display_df.loc[idx, 'Status']} | Prob: {display_df.loc[idx, 'Risk_Probability']*100:.1f}%" 
                    for idx in applicant_indices
                }
                
                selected_idx = st.selectbox(
                    "Select an applicant from the queue to view their specific risk factors:", 
                    options=applicant_indices,
                    format_func=lambda x: dropdown_options[x]
                )
                
                if selected_idx is not None:
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
            debt_ratio_pct = st.number_input("Debt Ratio (%)", min_value=0.0, max_value=100.0, step=1.0, value=None)
            dependents = st.number_input("Number of Dependents", min_value=0, step=1, value=None)
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
                    debt_ratio_model = debt_ratio_pct / 100.0
                    revol_util_model = revol_util_pct / 100.0

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
                    
                    cleaned_df = preprocess_new_data(raw_data, filename="manual_entry_processed.csv")
                    predictions, probabilities = predict_risk(cleaned_df)
                    probability = probabilities[0]
                    
                    display_deep_dive(probability, raw_data)
                    
                except Exception as e:
                    st.error(f"An error occurred: {e}")