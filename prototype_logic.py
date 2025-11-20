import joblib
import pandas as pd
import numpy as np

# --- Configuration ---
MODEL_PATH = 'best_n_status_classifier.pkl' # Ensure this file is in the directory.
LABEL_CLASSES = ['Deficient', 'Excess', 'Sufficient']

# 7 MOST IMPORTANT FEATURES (User Input)
MINIMAL_FEATURES = [
    'P', 'Available_P', 'Exchangeable_Ca', 'Ca', 'Bulk_Density', 
    'Exchangeable_K', 'Mg'
]

# 10 EXCLUDED FEATURES (Auto-Filled with Mean Values from Training Data)
# These mean values ensure the model's performance is not compromised.
DEFAULT_EXCLUDED_VALUES = {
    'K': 0.9469, 'Total_N': 0.3099, 'Total_organic_C': 3.5565, 
    'Exchangeable_Mg': 0.14, 'Soil_pH': 4.5675, 'Clay': 8.9062, 
    'Silt': 30.4021, 'Sand': 60.6913, 'Soil_Mineral': 0.84, 'Soil_Organic': 0.16
}

# The FULL feature list, in the exact order the model expects
FULL_FEATURE_LIST = MINIMAL_FEATURES + list(DEFAULT_EXCLUDED_VALUES.keys())
FULL_FEATURE_LIST_ORDERED = [
    'P', 'K', 'Mg', 'Ca', 'Bulk_Density', 'Total_N', 'Total_organic_C', 
    'Available_P', 'Exchangeable_K', 'Exchangeable_Ca', 'Exchangeable_Mg', 
    'Soil_pH', 'Clay', 'Silt', 'Sand', 'Soil_Mineral', 'Soil_Organic'
]

# Load the trained model
try:
    model = joblib.load(MODEL_PATH)
    print(f"Successfully loaded model from {MODEL_PATH}")
except FileNotFoundError:
    print(f"ERROR: Model file {MODEL_PATH} not found. Ensure it is in the current directory.")
    model = None

# --- Core Functions ---

def get_nutrient_recommendation(n_status: str) -> str:
    """
    Provides a clear, actionable nutrient management recommendation
    based on the predicted Nitrogen status.
    """
    if n_status == 'Deficient':
        return (
            "**Action: Apply 110 kg/ha of Urea (Nitrogen source).** "
            "The soil is deficient in Nitrogen. This application is crucial to restore optimal growth."
        )
    elif n_status == 'Sufficient':
        return (
            "**Action: No Nitrogen fertilizer is required at this time.** "
            "The current Nitrogen status is sufficient. Maintain current management practices and monitor via next soil test."
        )
    elif n_status == 'Excess':
        return (
            "**Action: Cease all Nitrogen fertilizer application immediately.** "
            "Excessive Nitrogen is wasteful, may lead to environmental run-off, and can cause nutrient imbalances."
        )
    else:
        return "Prediction status unknown. Review input data."


def predict_n_status_and_recommend(input_data: dict) -> tuple:
    """
    Takes minimal user input, combines it with default values, 
    predicts N_Status, and generates a recommendation.

    :param input_data: A dictionary of the 7 minimal features and their values.
    :return: A tuple of (predicted_status, recommendation_text)
    """
    if model is None:
        return "Model Error", "Cannot make a prediction as the model could not be loaded."

    # 1. Combine user input with default values
    full_input = {**DEFAULT_EXCLUDED_VALUES, **input_data}

    # 2. Prepare data for prediction
    try:
        # Create a DataFrame from the combined dictionary, ensuring correct feature order
        input_df = pd.DataFrame([full_input])
        input_df = input_df[FULL_FEATURE_LIST_ORDERED]

        # Convert boolean-like features to integers as used in training
        # Since Soil_Mineral and Soil_Organic are now float means (e.g., 0.84),
        # we ensure they are integers if the model expects them that way,
        # but for compatibility, we'll keep them as floats which is what the mean calculation yielded.
        
    except KeyError as e:
        return "Input Error", f"Missing required feature: {e}. Ensure all minimal features are provided."

    # 3. Make Prediction
    prediction_encoded = model.predict(input_df)[0]
    predicted_status = LABEL_CLASSES[prediction_encoded] # Map the encoded result back to the label

    # 4. Generate Recommendation
    recommendation = get_nutrient_recommendation(predicted_status)

    return predicted_status, recommendation

# --- Demonstration ---
if __name__ == "__main__":
    # Sample input using only the 7 minimal features
    sample_minimal_input = {
        'P': 0.12, 'Available_P': 42.135, 'Exchangeable_Ca': 0.485, 
        'Ca': 1.23, 'Bulk_Density': 1.155, 'Exchangeable_K': 0.045, 
        'Mg': 0.21
    }

    print("\n--- Demonstration with Minimal Input (7 Features) ---")
    predicted_status, recommendation = predict_n_status_and_recommend(sample_minimal_input)
    print(f"Prediction Output: Predicted N_Status is **{predicted_status}**")
    print(f"Nutrient Management Recommendation: {recommendation}")