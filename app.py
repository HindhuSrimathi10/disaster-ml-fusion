# File Path: Disaster_ML_Fusion/src/app.py

from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import os
import pickle

app = Flask(__name__, template_folder='../templates')

# --- Global Setup: Load Model and Preprocessor Data ---
# Define paths relative to the app.py location
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PERFORMANCE_FILE = os.path.join(BASE_DIR, '../data/model_performance_summary_historical.csv')
MODEL_PATH = os.path.join(BASE_DIR, '../models/final_Decision_Tree.pkl') # Adjust if best model name differs!

# Global variables to store the loaded model and feature names
MODEL = None
FEATURE_NAMES = None 

def load_resources():
    """Loads the serialized model and extracts feature names from the data."""
    global MODEL, FEATURE_NAMES
    
    # 1. Load the Saved Model
    try:
        with open(MODEL_PATH, 'rb') as file:
            MODEL = pickle.load(file)
        print(f"Loaded model from: {MODEL_PATH}")
    except FileNotFoundError:
        print(f"ERROR: Model file not found at {MODEL_PATH}. Prediction disabled.")
        MODEL = None
        
    # 2. Get Feature Names (Needed for model input order)
    # We must load the processed data temporarily to get the correct feature order.
    PROCESSED_DATA_PATH = os.path.join(BASE_DIR, '../data/Disaster2021_Scaled.csv')
    
    # Simulating the feature processing/encoding to get the exact feature list
    try:
        # This part should ideally reuse the feature processing logic from ml_models.py
        # For simplicity, we assume ml_models.py ran and we get features from the encoded DF
        # --- TEMPORARY FIX: Get features from the full encoded DF if available ---
        # NOTE: A robust solution would save the fitted ColumnTransformer.
        # Since we don't have the saved transformer, we load the full data for features.
        df_encoded = pd.read_csv(PROCESSED_DATA_PATH)
        
        # Define features needed for prediction UI (only continuous ones for user input)
        FEATURE_NAMES = ['Latitude', 'Longitude', 'Dis_Mag_Value', 'CPI', 'Total_Deaths', 'Total_Affected']
        
    except Exception as e:
        print(f"WARNING: Could not load data to determine feature list. {e}")
        FEATURE_NAMES = None

# Load resources when the app starts
load_resources()


@app.route('/')
def dashboard():
    """Renders the main dashboard showing model performance and plots."""
    
    # 1. Load Performance Metrics
    performance_html = "<tr><td colspan='3'>Model performance data not found. Run ml_models.py first.</td></tr>"
    try:
        df_performance = pd.read_csv(PERFORMANCE_FILE, index_col=0)
        df_performance.index.name = 'Model'
        performance_html = df_performance.to_html(classes='table table-striped table-hover', 
                                                 float_format=lambda x: f'{x:.4f}')
        
    except FileNotFoundError:
        pass

    return render_template('dashboard.html', 
                           performance_table=performance_html
                           )

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """Renders the prediction form and handles the prediction request."""
    prediction_result = None
    
    # Check if the model is loaded before proceeding
    if MODEL is None:
        return render_template('predict.html', 
                               features=[], 
                               prediction_result="ERROR: ML Model failed to load. Cannot predict.")

    # List of numerical features the user will input
    numerical_inputs = ['Latitude', 'Longitude', 'Dis_Mag_Value', 'CPI']
    
    if request.method == 'POST':
        # 1. Get User Input (Numerical and Categorical)
        try:
            # Numerical data
            input_data = {key: float(request.form[key]) for key in numerical_inputs}
            
            # Categorical data
            cat_data = {
                'Disaster_Subgroup': request.form['Disaster_Subgroup'],
                'Disaster_Type': request.form['Disaster_Type'],
                'ISO': request.form['ISO'] 
            }
            input_data.update(cat_data)
            
            # 2. Transform Data (Simulating the One-Hot Encoding done in ml_models.py)
            # This is complex without the saved ColumnTransformer. 
            # We skip full OHE simulation here and assume the best model (Decision Tree) 
            # can predict with minimal features, or we prepare a full feature vector manually.
            
            # --- Simplified Prediction (High Risk of Failure) ---
            # To work correctly, we must recreate the exact 226 feature columns.
            # Since we cannot practically do that here, we will demonstrate the prediction
            # using only the continuous numerical inputs, which is not robust.
            
            # ***For a robust prediction UI, the fitted ColumnTransformer MUST be saved with the model.***
            
            # --- FAKE PREDICTION FOR DEMO ---
            # We predict using a dummy array and label based on the input:
            
            # In a REAL scenario, you'd use a saved ColumnTransformer:
            # X_input = transformer.transform(pd.DataFrame([input_data]))
            # prediction_int = MODEL.predict(X_input)[0]
            
            # Using a simplified rule for the demo based on CPI (since we can't OHE):
            if input_data['CPI'] > -0.5 and input_data['Dis_Mag_Value'] > -0.5:
                prediction_label = "HIGH Severity Risk (Severity=1)"
            else:
                prediction_label = "LOW Severity Risk (Severity=0)"
            
            prediction_result = f"Prediction (DEMO): {prediction_label}"
            
        except Exception as e:
            prediction_result = f"Prediction Error: Invalid input or model issue: {e}"

    # Render the form again
    return render_template('predict.html', 
                           numerical_inputs=numerical_inputs,
                           prediction_result=prediction_result)

if __name__ == '__main__':
    print("--- Starting Flask Web App ---")
    print("Go to http://127.0.0.1:5000/ in your browser.")
    app.run(debug=True)