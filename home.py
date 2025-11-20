# Save this file as Home.py
import streamlit as st
import pandas as pd
# Ensure prototype_logic.py is in the same directory
from prototype_logic import MINIMAL_FEATURES, predict_n_status_and_recommend

# --- 0. UI Configuration and Setup ---
st.set_page_config(
    page_title="Oil Palm Nutrient Management Prototype", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Input Configuration (Unchanged)
INPUT_CONFIG = {
    'P': {'min': 0.12, 'max': 0.42, 'mean': 0.1661, 'step': 0.0001, 'format': "%.4f", 'help': 'Total Phosphorus content in soil.'},
    'Available_P': {'min': 1.0, 'max': 152.0, 'mean': 25.3014, 'step': 0.0001, 'format': "%.4f", 'help': 'Phosphorus available for plant uptake.'},
    'Exchangeable_Ca': {'min': 0.13, 'max': 5.88, 'mean': 0.6505, 'step': 0.0001, 'format': "%.4f", 'help': 'Exchangeable Calcium levels (key for soil pH and structure).'},
    'Ca': {'min': 0.42, 'max': 1.6, 'mean': 0.9521, 'step': 0.0001, 'format': "%.4f", 'help': 'Total Calcium content.'},
    'Bulk_Density': {'min': 0.185, 'max': 1.49, 'mean': 1.105, 'step': 0.0001, 'format': "%.4f", 'help': 'A physical measure of soil compaction (high value means dense soil).'},
    'Exchangeable_K': {'min': 0.015, 'max': 1.505, 'mean': 0.0991, 'step': 0.0001, 'format': "%.4f", 'help': 'Exchangeable Potassium (major cation).'},
    'Mg': {'min': 0.05, 'max': 0.32, 'mean': 0.1501, 'step': 0.0001, 'format': "%.4f", 'help': 'Total Magnesium content.'}
}


# Initialize session state variables
if 'prediction_made' not in st.session_state:
    st.session_state['prediction_made'] = False
if 'status' not in st.session_state:
    st.session_state['status'] = 'N/A'
if 'recommendation' not in st.session_state:
    st.session_state['recommendation'] = 'Please input parameters and generate a recommendation.'


# --- Sidebar ---
with st.sidebar:
    st.title("Palm Nutrient Predictor 🌴")
    st.markdown("---")
    st.header("Project Overview")
    st.info("Navigate between the Prediction Tool and the Model Dashboard using the sidebar.")
    st.caption("Developed using Streamlit and Random Forest Classifier.")

st.title("🌱 Functional Predictive Prototype: Oil Palm Nutrient Management")
st.markdown("A focused tool for real-time Nitrogen status prediction and actionable fertilizer recommendations.")


# --- 1. Input Parameters Section ---
is_expanded = not st.session_state['prediction_made']

with st.expander("1️⃣ Input Critical Soil Parameters", expanded=is_expanded):
    st.caption("Please input values exactly from your soil test report for accurate prediction.")
    
    input_data = {}
    cols = st.columns(3)
    
    # Input fields here (Unchanged)
    for i, feature in enumerate(MINIMAL_FEATURES):
        config = INPUT_CONFIG.get(feature)
        with cols[i % 3]:
            input_data[feature] = st.number_input(
                f"**{feature}**", 
                min_value=config['min'], 
                max_value=config['max'], 
                value=config['mean'],
                step=config['step'],
                format=config['format'],
                help=config['help']
            )

    st.markdown("---")
    st.markdown("<p style='text-align: center;'>", unsafe_allow_html=True)
    
    if st.button("Generate Recommendation 🚀", type="primary", use_container_width=True):
        
        # Run prediction and store results in session state
        status, recommendation = predict_n_status_and_recommend(input_data)
        st.session_state['status'] = status
        st.session_state['recommendation'] = recommendation
        st.session_state['prediction_made'] = True
        
        st.rerun() 
        
    st.markdown("</p>", unsafe_allow_html=True)


# --- 2. Prediction & Action Section ---
if st.session_state['prediction_made']:
    st.header("2️⃣ Prediction & Action: Results")
    
    status = st.session_state['status']
    recommendation = st.session_state['recommendation']
    
    # Define Metric style based on status
    if status == 'Deficient':
        metric_label = "⚠️ Nitrogen Status: DEFICIENT"
        color_box = st.error
    elif status == 'Sufficient':
        metric_label = "✅ Nitrogen Status: SUFFICIENT"
        color_box = st.success
    elif status == 'Excess':
        metric_label = "🛑 Nitrogen Status: EXCESS"
        color_box = st.warning
    else:
        metric_label = "❓ Nitrogen Status: UNKNOWN"
        color_box = st.info
        
    
    col_status, col_nav = st.columns([1, 3])
    
    with col_status:
        st.metric(label="Predicted Fertility Class", value=status, delta=metric_label)
    
    with col_nav:
        st.markdown("") # Add space
        st.info("View detailed model performance by navigating to the **'Dashboard'** page in the sidebar.")

    st.markdown("---")
    st.subheader("Nutrient Management Recommendation")
    color_box(recommendation)