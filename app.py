"""
Flask API for ML Model Deployment
Author: Your Name
Version: 1.1.0
"""

from datetime import datetime
import numpy as np
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib
from config import MODEL_CONFIG

app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests

# Load trained model and metadata
model = joblib.load(MODEL_CONFIG['MODEL_PATH'])
feature_columns = MODEL_CONFIG['FEATURE_COLUMNS']
state_mapping = MODEL_CONFIG['STATE_ENCODING']

def validate_input(data: dict) -> tuple:
    """
    Validate and sanitize input data
    Returns: (cleaned_data, error_message)
    """
    required_fields = ['R&D', 'Administration', 'Marketing', 'State']
    
    # Check missing fields
    if any(field not in data for field in required_fields):
        return None, f"Missing required fields: {required_fields}"
    
    try:
        cleaned = {
            'R&D': float(data['R&D']),
            'Administration': float(data['Administration']),
            'Marketing': float(data['Marketing']),
            'State': data['State'].title()
        }
    except ValueError:
        return None, "Numerical fields must contain valid numbers"
    
    # Validate state
    if cleaned['State'] not in state_mapping:
        valid_states = list(state_mapping.keys())
        return None, f"Invalid state. Valid options: {valid_states}"
    
    return cleaned, None

def prepare_features(input_data: dict) -> np.ndarray:
    """
    Convert input data to model-ready format
    """
    # Create base array with numerical features
    numerical = np.array([
        input_data['R&D'],
        input_data['Administration'],
        input_data['Marketing']
    ])
    
    # Add one-hot encoded state features
    state_vector = np.zeros(len(state_mapping))
    state_index = state_mapping[input_data['State']]
    state_vector[state_index] = 1
    
    return np.concatenate([numerical, state_vector]).reshape(1, -1)

@app.route('/', methods=['GET'])
def dashboard():
    """Render prediction dashboard"""
    return render_template('index.html', states=list(state_mapping.keys()))

@app.route('/predict', methods=['POST'])
def predict():
    """
    Handle prediction requests from both form and API
    Accepts: application/x-www-form-urlencoded or application/json
    """
    # Get input data based on content type
    if request.content_type == 'application/json':
        input_data = request.get_json()
    else:
        input_data = request.form.to_dict()
    
    # Validate input
    cleaned_data, error = validate_input(input_data)
    if error:
        return jsonify({
            'status': 'error',
            'message': error,
            'timestamp': datetime.utcnow().isoformat()
        }), 400
    
    try:
        # Prepare features and predict
        features = prepare_features(cleaned_data)
        prediction = model.predict(features)
        result = round(float(prediction[0]), 2)
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Prediction failed: {str(e)}',
            'timestamp': datetime.utcnow().isoformat()
        }), 500

    # Format response
    response = {
        'status': 'success',
        'prediction': result,
        'currency': 'USD',
        'timestamp': datetime.utcnow().isoformat(),
        'model_version': MODEL_CONFIG['VERSION']
    }
    
    # Return appropriate response format
    if request.content_type == 'application/json':
        return jsonify(response)
    return render_template(
        'index.html',
        prediction_text=f'Predicted Profit: ${result}',
        states=list(state_mapping.keys())
    )

if __name__ == '__main__':
    app.run(host=MODEL_CONFIG['HOST'], 
            port=MODEL_CONFIG['PORT'], 
            debug=MODEL_CONFIG['DEBUG'])
