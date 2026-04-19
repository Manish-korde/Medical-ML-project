"""
Medical ML Flask Application
Optimized for deployment with lazy model loading.
"""

import os
import sys

# Configure for CPU-only (reduces memory)
os.environ['CUDA_VISIBLE_DEVICES'] = ''

# =============================================================================
# PATH FIX FOR DEPLOYMENT
# =============================================================================
# Get the project root directory (parent of app/)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Also add current directory just in case
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

print(f"[INFO] Project root: {PROJECT_ROOT}")
print(f"[INFO] Python path: {sys.path[:3]}")

# =============================================================================
# FLASK APP SETUP
# =============================================================================
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = 'app/static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# =============================================================================
# LAZY MODEL LOADING
# =============================================================================
# Models are loaded only when first needed (first API call)
# This significantly reduces memory at startup for Render free tier

_models_loaded = False

def _load_models():
    """Lazy load ML models on first request."""
    global _models_loaded
    if not _models_loaded:
        print("[INFO] Loading ML models...")
        from src.api.predict import predict_image, predict_tabular, generate_medical_report
        globals()['predict_image_fn'] = predict_image
        globals()['predict_tabular_fn'] = predict_tabular
        globals()['generate_report_fn'] = generate_medical_report
        _models_loaded = True
        print("[INFO] Models loaded successfully")
    return True

# =============================================================================
# ROUTES
# =============================================================================

@app.route('/')
def index():
    """Render the main HTML page."""
    return render_template('index.html')

@app.route('/predict/image', methods=['POST'])
def predict_image_api():
    """Predict chest X-ray image."""
    # Load models on first request
    _load_models()
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)
    
    try:
        result = predict_image(filepath)
        return jsonify(result)
    except Exception as e:
        print(f"[ERROR] Image prediction: {str(e)}")
        return jsonify({'error': str(e), 'prediction': None}), 500

@app.route('/predict/tabular', methods=['POST'])
def predict_tabular_api():
    """Predict heart disease from clinical data."""
    # Load models on first request
    _load_models()
    
    data = request.json
    
    # Required fields validation
    required_fields = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
                    'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
    for field in required_fields:
        if field not in data:
            return jsonify({'error': f'Missing required field: {field}'}), 400
    
    try:
        result = predict_tabular(data)
        return jsonify(result)
    except Exception as e:
        print(f"[ERROR] Tabular prediction: {str(e)}")
        return jsonify({'error': str(e), 'prediction': None}), 500

@app.route('/predict/complete', methods=['POST'])
def predict_complete_api():
    """Complete analysis: Image + Tabular + LLM Report."""
    # Load models on first request
    _load_models()
    
    data = request.json
    image_result = None
    tabular_result = None
    llm_analysis = None
    
    print("[INFO] Processing complete prediction")
    
    # Process chest X-ray
    if data.get('image_data'):
        try:
            import base64
            from PIL import Image
            import io
            
            img_data = base64.b64decode(data['image_data'])
            img = Image.open(io.BytesIO(img_data))
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'temp_complete.png')
            img.save(filepath)
            
            image_result = predict_image(filepath)
            print(f"[INFO] Image: {image_result.get('prediction')} ({image_result.get('confidence')}%)")
        except Exception as e:
            image_result = {'error': str(e)}
            print(f"[ERROR] Image: {str(e)}")
    
    # Process tabular
    if data.get('tabular_data'):
        try:
            tabular_result = predict_tabular(data['tabular_data'])
            print(f"[INFO] Heart: {tabular_result.get('prediction')} ({tabular_result.get('confidence')}%)")
        except Exception as e:
            tabular_result = {'error': str(e)}
            print(f"[ERROR] Heart: {str(e)}")
    
    # Get patient data for context
    patient_data = data.get('tabular_data', None)
    
    # Validate before LLM
    image_valid = image_result and not image_result.get('error')
    tabular_valid = tabular_result and not tabular_result.get('error')
    
    if image_valid or tabular_valid:
        try:
            print("[INFO] Generating LLM report...")
            llm_analysis = generate_medical_report(image_result, tabular_result, patient_data)
            print("[INFO] LLM report done")
        except Exception as e:
            llm_analysis = f"Error: {str(e)}"
            print(f"[ERROR] LLM: {str(e)}")
    else:
        llm_analysis = "Unable to generate report. Check model inputs."
        print("[WARNING] No valid predictions for LLM")
    
    return jsonify({
        'image_prediction': image_result,
        'heart_risk': tabular_result,
        'llm_analysis': llm_analysis,
        'validation': {'image_valid': image_valid, 'tabular_valid': tabular_valid}
    })

# =============================================================================
# MAIN ENTRY POINT
# =============================================================================
if __name__ == '__main__':
    print("="*50)
    print("Medical ML App - Starting...")
    print("Models will load on first API request (lazy loading)")
    print("="*50)
    app.run(debug=True, host='0.0.0.0', port=5000)