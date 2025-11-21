# Save this file as pages/dashboard.py
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# --- 1. Cached Data Generation Function ---

# @st.cache_resource is used to cache the heavy analysis and model objects.
# Change the 'cache_version' number (e.g., 2 to 3) when the data is updated.
@st.cache_resource(show_spinner="Running comprehensive model analysis... this may take a moment.")
def generate_dashboard_data(cache_version): 
    """Performs full ML analysis, returns metrics DF, visualization dataframes, and key scores."""
    
    # --- A. Data Setup (Must replicate training environment) ---
    try:
        df = pd.read_csv('merged_soil_palm_data_preprocessed.csv')
        model_path = 'best_n_status_classifier.pkl'
    except FileNotFoundError:
        st.error("Error: Required data or model file not found in the root directory.")
        return None, None, None, None, None

    # Data preprocessing
    cols_to_impute = ['Bulk_Density', 'Clay', 'Silt', 'Sand']
    for col in cols_to_impute:
        df[col].fillna(df[col].mean(), inplace=True)

    # Feature/Target Definition
    FEATURES = [
        'P', 'K', 'Mg', 'Ca', 'Bulk_Density', 'Total_N', 'Total_organic_C',
        'Available_P', 'Exchangeable_K', 'Exchangeable_Ca', 'Exchangeable_Mg',
        'Soil_pH', 'Clay', 'Silt', 'Sand', 'Soil_Mineral', 'Soil_Organic'
    ]
    X = df[FEATURES].copy()
    y = df['N_Status']

    X['Soil_Mineral'] = X['Soil_Mineral'].astype(int)
    X['Soil_Organic'] = X['Soil_Organic'].astype(int)

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    LABEL_CLASSES = le.classes_.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
    )

    # --- B. Model Training, Metric Collection, and AUC Calculation ---
    models = {
        "Random Forest": joblib.load(model_path), 
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42)
    }

    comparison_metrics = []
    rf_model = models["Random Forest"]
    rf_roc_auc_score = 0.0

    # One-hot encode the true labels for multi-class ROC AUC calculation
    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    y_test_ohe = ohe.fit_transform(y_test.reshape(-1, 1))

    for name, model in models.items():
        if name != "Random Forest": 
            model.fit(X_train, y_train)
            
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)
        
        # Calculate Micro-Averaged ROC AUC Score
        if y_test_ohe.shape[1] > 1: 
             auc_score = roc_auc_score(y_test_ohe, y_proba, average="micro")
        else:
            auc_score = roc_auc_score(y_test, y_proba[:, 1])

        report = classification_report(y_test, y_pred, target_names=LABEL_CLASSES, output_dict=True)
        
        comparison_metrics.append({
            "Model": name,
            "Accuracy": round(report['accuracy'], 4),
            "F1 (Deficient)": round(report['Deficient']['f1-score'], 4),
            "F1 (Sufficient)": round(report['Sufficient']['f1-score'], 4),
            "F1 (Excess)": round(report['Excess']['f1-score'], 4)
        })
        
        if name == "Random Forest":
            y_pred_rf = y_pred
            rf_roc_auc_score = round(auc_score, 4)

    metrics_df = pd.DataFrame(comparison_metrics)

    # --- C. Generate DataFrames for Streamlit Visuals ---

    # Confusion Matrix Dataframe
    cm = confusion_matrix(y_test, y_pred_rf)
    cm_df = pd.DataFrame(cm, index=LABEL_CLASSES, columns=LABEL_CLASSES)
    cm_df.index.name = 'True Status'
    cm_df.columns.name = 'Predicted Status'
    cm_df = cm_df.stack().reset_index(name='Count')
    
    # Feature Importance Dataframe
    importances = rf_model.feature_importances_
    fi_df = pd.DataFrame({
        'Feature': FEATURES,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False).head(10)

    # Training Class Distribution Dataframe
    class_counts = pd.Series(y_train).map(dict(enumerate(LABEL_CLASSES))).value_counts().reset_index()
    class_counts.columns = ['N_Status', 'Count']
    
    return metrics_df, cm_df, fi_df, rf_roc_auc_score, class_counts


# --- 2. Streamlit Dashboard Rendering ---

st.set_page_config(layout="wide")
st.title("📊 Model Performance Dashboard")
st.markdown("---")

# Execute the analysis function. MUST use the cache_version control to trigger re-run.
metrics_df, cm_df, fi_df, rf_roc_auc_score, class_counts = generate_dashboard_data(cache_version=2) 

if metrics_df is not None:
    st.header("Academic Validation & Model Reliability")

    # --- Key Metrics ---
    st.subheader("Key Performance Indicators")
    col_acc, col_auc, col_empty = st.columns([1, 1, 2])
    
    with col_acc:
        rf_accuracy = metrics_df[metrics_df['Model'] == 'Random Forest']['Accuracy'].iloc[0]
        st.metric(label="Random Forest Accuracy", value=f"{rf_accuracy * 100:.2f}%")
        
    with col_auc:
        st.metric(
            label="Micro-Averaged ROC AUC Score", 
            value=f"{rf_roc_auc_score:.4f}",
            help="Measures the model's ability to distinguish between all fertility classes (1.0 is perfect)."
        )
    
    st.markdown("---")

    # --- 1. Model Comparison (Metrics Table) ---
    st.subheader("1. Algorithm Comparison by Fertility Level (F1-Score)")
    
    def highlight_max(s):
        is_max = s == s.max()
        return ['background-color: #e0f2ff; font-weight: bold' if v else '' for v in is_max]

    st.dataframe(
        metrics_df.style.apply(highlight_max, subset=['Accuracy', 'F1 (Deficient)', 'F1 (Sufficient)', 'F1 (Excess)']),
        use_container_width=True
    )

    st.markdown("---")

    # --- 2. Data Balance and Model Reliability ---
    st.subheader("2. Data Balance and Model Reliability")

    col_cm, col_dist = st.columns(2)

    with col_cm:
        st.markdown("### Confusion Matrix (Test Set)")
        
        # Altair Heatmap for Confusion Matrix
        base = alt.Chart(cm_df).encode(
            x=alt.X('Predicted Status:N'),
            y=alt.Y('True Status:N'),
        ).properties(title="Confusion Matrix (Test Set)")

        heatmap = base.mark_rect().encode(
            color=alt.Color('Count:Q', scale=alt.Scale(range='heatmap')),
            tooltip=['True Status', 'Predicted Status', 'Count']
        )
        text = base.mark_text().encode(text=alt.Text('Count:Q'), color=alt.value('black'))
        
        chart_cm = (heatmap + text).interactive().properties(height=350)
        st.altair_chart(chart_cm, use_container_width=True)
        st.caption("The diagonal shows correct predictions for each fertility level.")

    with col_dist:
        st.markdown("### Training Data Class Distribution")
        
        # Altair Bar Chart for Class Distribution
        chart_dist = alt.Chart(class_counts).mark_bar().encode(
            x=alt.X('Count:Q', axis=alt.Axis(title='Number of Samples')),
            y=alt.Y('N_Status:N', sort='-x', title='Fertility Status'),
            color=alt.Color('N_Status:N', legend=None),
            tooltip=['N_Status', 'Count']
        ).properties(title="Training Set Class Counts", height=350).interactive()
        
        st.altair_chart(chart_dist, use_container_width=True)
        st.caption("This chart confirms the data balance used for training.")


    st.markdown("---")
    
    # --- 3. Feature Influence ---
    st.subheader("3. Feature Influence")
    
    chart_fi = alt.Chart(fi_df).mark_bar().encode(
        x=alt.X('Importance:Q', axis=alt.Axis(title='Gini Importance')),
        y=alt.Y('Feature:N', sort='-x'), 
        color=alt.Color('Feature:N', legend=None),
        tooltip=['Feature', alt.Tooltip('Importance', format=".4f")]
    ).properties(title="Top 10 Feature Importance").interactive()
    
    st.altair_chart(chart_fi, use_container_width=True)
    st.caption("This chart validates the 7 minimal inputs are the most predictive factors.")
