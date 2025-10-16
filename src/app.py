from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import os
import pickle
from pathlib import Path
import joblib
import glob

app = Flask(__name__, template_folder='../templates')

# --- Global Setup: Define Paths ---
PROJECT_ROOT = Path(__file__).parent.parent 
PERFORMANCE_FILE = PROJECT_ROOT / 'data' / 'model_performance_summary_historical.csv'
MODEL_PATH = PROJECT_ROOT / 'models' / 'final_Decision_Tree.pkl'
PREPROCESSOR_PATH = PROJECT_ROOT / 'models' / 'preprocessor.pkl' 

# Social Media Model Paths
SOCIAL_MEDIA_MODEL_PATH = PROJECT_ROOT / 'models' / 'social_media_disaster_model.joblib'

# Global variables to store the loaded resources
MODEL = None
PREPROCESSOR = None
SOCIAL_MEDIA_CLASSIFIER = None

def load_resources():
    """Loads the serialized model and preprocessor for prediction."""
    global MODEL, PREPROCESSOR, SOCIAL_MEDIA_CLASSIFIER
    
    # 1. Load the Saved Historical Model
    try:
        with MODEL_PATH.open('rb') as file:
            MODEL = pickle.load(file)
        print(f"Loaded historical model from: {MODEL_PATH}")
    except FileNotFoundError:
        print(f"CRITICAL ERROR: Model file not found at {MODEL_PATH}. Prediction disabled.")
        MODEL = None
    
    # 2. Load the Saved ColumnTransformer (CRITICAL for prediction)
    try:
        with PREPROCESSOR_PATH.open('rb') as file:
            PREPROCESSOR = pickle.load(file)
        print(f"Loaded preprocessor from: {PREPROCESSOR_PATH}")
    except FileNotFoundError:
        print(f"CRITICAL ERROR: Preprocessor file not found at {PREPROCESSOR_PATH}. Prediction disabled.")
        PREPROCESSOR = None

    # 3. Load Social Media Model
    try:
        if SOCIAL_MEDIA_MODEL_PATH.exists():
            SOCIAL_MEDIA_CLASSIFIER = joblib.load(SOCIAL_MEDIA_MODEL_PATH)
            print(f"Loaded social media model from: {SOCIAL_MEDIA_MODEL_PATH}")
            if hasattr(SOCIAL_MEDIA_CLASSIFIER, 'dataset_info'):
                print(f"Dataset info: {SOCIAL_MEDIA_CLASSIFIER.dataset_info}")
        else:
            print(f"Social media model not found at {SOCIAL_MEDIA_MODEL_PATH}. Social media prediction disabled.")
            SOCIAL_MEDIA_CLASSIFIER = None
    except Exception as e:
        print(f"Error loading social media model: {e}")
        SOCIAL_MEDIA_CLASSIFIER = None

    if MODEL is None or PREPROCESSOR is None:
        raise FileNotFoundError("One or more essential ML files are missing. Please run ml_models.py with necessary saving steps first.")

# Load resources when the app starts
try:
    load_resources()
except FileNotFoundError as e:
    print(f"\nFATAL CRASH: {e}")
    # We allow the app to run here, but the dashboard will show an error if CSV fails.

@app.route('/')
def dashboard():
    """Renders the main dashboard showing model performance and plots."""
    
    performance_html = "<tr><td colspan='3'>Model performance data not found. Run ml_models.py first.</td></tr>"
    try:
        # Load Performance Metrics
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
    
    # List of numerical features the user will input
    numerical_inputs = ['Latitude', 'Longitude', 'Dis_Mag_Value', 'CPI']
    
    # Features required by the preprocessor, including dummy features for prediction
    feature_columns = [
        'Disaster_Subgroup', 'Disaster_Type', 'ISO', 
        'Latitude', 'Longitude', 'Dis_Mag_Value', 'CPI', 
        'Total_Deaths', 'Total_Affected' # These are set to 0 for a new prediction
    ]
    
    if request.method == 'POST':
        # Check if resources are available
        if MODEL is None or PREPROCESSOR is None:
            prediction_result = "ERROR: ML Model or Preprocessor not loaded. Cannot predict."
            return render_template('predict.html', 
                                   numerical_inputs=numerical_inputs,
                                   prediction_result=prediction_result)

        # 1. Get User Input (Numerical and Categorical)
        try:
            # Gather all inputs
            raw_input = {
                'Disaster_Subgroup': request.form['Disaster_Subgroup'],
                'Disaster_Type': request.form['Disaster_Type'],
                'ISO': request.form['ISO'],
                'Latitude': float(request.form['Latitude']),
                'Longitude': float(request.form['Longitude']),
                'Dis_Mag_Value': float(request.form['Dis_Mag_Value']),
                'CPI': float(request.form['CPI']),
                # Crucial: Set Total_Deaths and Total_Affected to 0 for prediction of a NEW disaster
                'Total_Deaths': 0.0, 
                'Total_Affected': 0.0,
            }
            
            # 2. Create DataFrame in the exact order the transformer expects
            input_df = pd.DataFrame([raw_input], columns=feature_columns)

            # 3. Transform Data using the saved Preprocessor
            # We transform all columns needed by the original model training
            X_processed = PREPROCESSOR.transform(input_df)

            # 4. Predict
            # The Decision Tree model only expects the features (226 columns), which X_processed is.
            prediction_int = MODEL.predict(X_processed)[0]
            
            prediction_label = "HIGH Severity Risk (Severity=1)" if prediction_int == 1 else "LOW Severity Risk (Severity=0)"
            
            prediction_result = f"Predicted Severity Level: {prediction_int} ({prediction_label})"
            
        except Exception as e:
            prediction_result = f"Prediction Error during transformation/prediction: {e}"

    # Render the form again
    return render_template('predict.html', 
                           numerical_inputs=numerical_inputs,
                           prediction_result=prediction_result)

# ===== SOCIAL MEDIA PREDICTION ROUTES =====

@app.route('/social_media_predict')
def social_media_predict():
    """Renders the social media prediction page."""
    # Get available datasets
    data_files = glob.glob(str(PROJECT_ROOT / "data" / "*.csv")) + \
                 glob.glob(str(PROJECT_ROOT / "data" / "*.json")) + \
                 glob.glob(str(PROJECT_ROOT / "data" / "*.xlsx"))
    
    dataset_info = None
    if SOCIAL_MEDIA_CLASSIFIER and hasattr(SOCIAL_MEDIA_CLASSIFIER, 'dataset_info'):
        dataset_info = SOCIAL_MEDIA_CLASSIFIER.dataset_info
    
    return render_template('social_media_predict.html', 
                         dataset_info=dataset_info,
                         available_datasets=[Path(f).name for f in data_files])

@app.route('/predict_social_media', methods=['POST'])
def predict_social_media():
    """API endpoint for social media disaster prediction from text."""
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({'error': 'No text provided'}), 400
        
        text = data['text'].strip()
        
        if not text:
            return jsonify({'error': 'Please enter some text'}), 400
        
        # Check if social media model is loaded
        if SOCIAL_MEDIA_CLASSIFIER is None:
            return jsonify({'error': 'Social media model not loaded. Please train the model first.'}), 500
        
        # Predict disaster type
        result = SOCIAL_MEDIA_CLASSIFIER.predict_disaster(text)
        
        response = {
            'success': True,
            'disaster_type': result['disaster_type'],
            'confidence': round(result['confidence'] * 100, 2),
            'processed_text': result['processed_text'],
            'all_probabilities': result['all_probabilities']
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/train_social_media_model', methods=['POST'])
def train_social_media():
    """Endpoint to train the social media disaster classification model."""
    try:
        data = request.get_json() or {}
        dataset_file = data.get('dataset_file')
        
        from ml_models2 import train_social_media as train_model
        
        if dataset_file:
            dataset_path = PROJECT_ROOT / "data" / dataset_file
            classifier = train_model(str(dataset_path))
        else:
            classifier = train_model()
        
        if classifier:
            global SOCIAL_MEDIA_CLASSIFIER
            SOCIAL_MEDIA_CLASSIFIER = classifier
            return jsonify({
                'success': True, 
                'message': 'Social media model trained successfully',
                'dataset_info': classifier.dataset_info
            })
        else:
            return jsonify({'error': 'Training failed - check dataset availability and format'}), 400
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/get_available_datasets')
def get_available_datasets():
    """Get list of available dataset files."""
    data_files = glob.glob(str(PROJECT_ROOT / "data" / "*.csv")) + \
                 glob.glob(str(PROJECT_ROOT / "data" / "*.json")) + \
                 glob.glob(str(PROJECT_ROOT / "data" / "*.xlsx"))
    
    return jsonify({
        'datasets': [Path(f).name for f in data_files]
    })

if __name__ == '__main__':
    print("--- Starting Flask Web App ---")
    print("Available routes:")
    print("  /                    - Historical data dashboard")
    print("  /predict             - Historical disaster prediction")
    print("  /social_media_predict - Social media disaster prediction")
    print("  /train_social_media_model - Train social media model")
    print("\nGo to http://127.0.0.1:5000/ in your browser.")
    app.run(debug=True)