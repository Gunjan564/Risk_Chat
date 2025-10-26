# C:\Users\hp\OneDrive\Desktop\Risk_Chat\api_server.py

import sys
import os
from flask import Flask, request, jsonify
from flask_cors import CORS
# Import the model loading function from your utility file
from model_utils import load_ensemble_models 

# --- FLASK SETUP ---
app = Flask(__name__)
# Allow cross-origin requests from your Streamlit app
CORS(app) 

# Global variable to hold the initialized ensemble model
GLOBAL_ENSEMBLE_MODEL = None

def initialize_ensemble_model():
    """Initializes the model once at startup."""
    global GLOBAL_ENSEMBLE_MODEL
    try:
        print("Starting ensemble model initialization...")
        # Use your provided loading function
        GLOBAL_ENSEMBLE_MODEL = load_ensemble_models() 
        print("✅ GLOBAL_ENSEMBLE_MODEL initialized and ready.")
        return True
    except Exception as e:
        print(f"❌ FATAL: Failed to initialize ensemble model: {e}", file=sys.stderr)
        return False

@app.route('/predict_sentiment', methods=['POST'])
def predict_sentiment():
    """API endpoint to receive text and return ensemble prediction."""
    global GLOBAL_ENSEMBLE_MODEL

    if GLOBAL_ENSEMBLE_MODEL is None:
        return jsonify({'error': 'Model not initialized. Server is unavailable.'}), 503

    try:
        data = request.get_json()
        text = data.get('text', '')

        if not text:
            return jsonify({'error': 'No text provided'}), 400

        # Run prediction using the loaded model instance
        risk_label, confidence = GLOBAL_ENSEMBLE_MODEL.predict(text)
        
        # Ensure output format matches what app.py expects
        return jsonify({
            'sentiment': risk_label,
            'confidence': float(confidence) 
        })

    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({'error': f'Prediction failed due to internal model error: {str(e)}'}), 500


if __name__ == '__main__':
    # Initialize the model at startup
    if initialize_ensemble_model():
        print("--- Starting Flask Server ---")
        # Run on the port the Streamlit app is looking for
        app.run(host='0.0.0.0', port=5001)
    else:
        print("Server NOT started due to model initialization failure.")