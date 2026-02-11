import streamlit as st
import pandas as pd
import pickle
import numpy as np

# --- Streamlit Page Configuration ---
st.set_page_config(layout="wide")

st.title("Second Hand Mobile Resale Price Predictor")
st.write("Enter the mobile phone details to predict its resale price.")

# --- Data and Model Loading ---
@st.cache_resource # Cache resource loading for performance
def load_resources():
    # Load the original dataset to get unique categorical values for selectboxes
    df_original = pd.read_csv('second_hand_mobile_dat.csv')

    # Load the pre-trained model, label encoders, and feature list
    with open('dt_model_and_encoders.pkl', 'rb') as file:
        model_pipeline = pickle.load(file)

    dt_model = model_pipeline['model']
    label_encoders = model_pipeline['label_encoders']
    features = model_pipeline['features']
    
    # Store unique values for categorical features for display in selectboxes
    categorical_options = {}
    for col in label_encoders.keys():
        categorical_options[col] = sorted(df_original[col].unique())

    return dt_model, label_encoders, features, categorical_options

# Load resources once
dt_model, label_encoders, features, categorical_options = load_resources()

# --- Streamlit UI for User Input ---
st.sidebar.header("Input Features")

# Numerical Inputs
original_price = st.sidebar.number_input("Original Price (INR)", min_value=0, max_value=150000, value=60000, step=1000)
age_months = st.sidebar.slider("Age (Months)", min_value=1, max_value=72, value=24, step=1)
ram_gb = st.sidebar.selectbox("RAM (GB)", sorted(categorical_options['RAM_GB'] if 'RAM_GB' in categorical_options else [3,4,6,8,12,16]), index=1)
storage_gb = st.sidebar.selectbox("Storage (GB)", sorted(categorical_options['Storage_GB'] if 'Storage_GB' in categorical_options else [32,64,128,256,512]), index=2)
battery_health = st.sidebar.slider("Battery Health (%)", min_value=65, max_value=100, value=85, step=1)
camera_mp = st.sidebar.selectbox("Camera (MP)", sorted(categorical_options['Camera_MP'] if 'Camera_MP' in categorical_options else [8,12,16,32,48,64,108]), index=3)
screen_size = st.sidebar.number_input("Screen Size (inches)", min_value=5.0, max_value=7.5, value=6.1, step=0.1, format="%.1f")

# Categorical Inputs (using original values for display)
brand = st.sidebar.selectbox("Brand", categorical_options['Brand'])

# Filter models based on selected brand
# Note: This requires the original_df or similar logic to filter models
# For simplicity, assuming 'Model' selectbox gets all unique models for now
# A more complex app would need to load the full df_original for filtering
model_options_for_brand = sorted(df_original[df_original['Brand'] == brand]['Model'].unique()) if 'df_original' in locals() else categorical_options['Model']
model = st.sidebar.selectbox("Model", model_options_for_brand)

condition = st.sidebar.selectbox("Condition", categorical_options['Condition'])
sg_support = st.sidebar.selectbox("5G Support", categorical_options['5G_Support'])
warranty_remaining = st.sidebar.selectbox("Warranty Remaining", categorical_options['Warranty_Remaining'])

# Prediction Button
if st.sidebar.button("Predict Resale Price"):
    try:
        # Encode user inputs using the fitted LabelEncoders
        encoded_brand = label_encoders['Brand'].transform([brand])[0]
        encoded_model = label_encoders['Model'].transform([model])[0]
        encoded_condition = label_encoders['Condition'].transform([condition])[0]
        encoded_5g_support = label_encoders['5G_Support'].transform([sg_support])[0]
        encoded_warranty_remaining = label_encoders['Warranty_Remaining'].transform([warranty_remaining])[0]

        # Create a DataFrame for the user's input, ensuring correct column order
        user_input_data = {
            'Brand': encoded_brand,
            'Model': encoded_model,
            'Original_Price': original_price,
            'Age_Months': age_months,
            'RAM_GB': ram_gb,
            'Storage_GB': storage_gb,
            'Battery_Health': battery_health,
            'Camera_MP': camera_mp,
            'Screen_Size': screen_size,
            'Condition': encoded_condition,
            '5G_Support': encoded_5g_support,
            'Warranty_Remaining': encoded_warranty_remaining
        }

        user_input_df = pd.DataFrame([user_input_data], columns=features)

        # Make prediction
        prediction = dt_model.predict(user_input_df)

        st.success(f"Predicted Resale Price: ₹{prediction[0]:,.2f}")

    except ValueError as e:
        st.error(f"Error during prediction: {e}. Please ensure all selections are valid and match the model's training data.")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
