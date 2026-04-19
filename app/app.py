"""
Medical ML - Ultra Memory Optimized for Render Free Tier
Models load only when first API call happens
"""

import os
import sys
os.environ['CUDA_VISIBLE_DEVICES'] = ''

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _root not in sys.path:
    sys.path.insert(0, _root)

# DON'T import torch here - will load models lazily
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from werkzeug.exceptions import HTTPException

app = Flask(__name__)
CORS(app)

@app.errorhandler(Exception)
def handle_exception(e):
    """Return JSON instead of HTML for HTTP errors."""
    if isinstance(e, HTTPException):
        return jsonify(error=str(e.description)), e.code
    return jsonify(error=str(e)), 500

app.config['UPLOAD_FOLDER'] = 'app/static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Lazy load functions
_predict_image = None
_predict_tabular = None
_generate_report = None

def _load_ml():
    """Load ML functions only when first needed"""
    global _predict_image, _predict_tabular, _generate_report
    if _predict_image is None:
        print("[INFO] Loading models...")
        from src.api.predict import predict_image, predict_tabular, generate_medical_report
        _predict_image = predict_image
        _predict_tabular = predict_tabular
        _generate_report = generate_medical_report
        print("[INFO] Models loaded")
    return True

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict/image', methods=['POST'])
def predict_image_api():
    _load_ml()
    if 'file' not in request.files:
        return jsonify({'error': 'No file'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file'}), 400
    fp = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(fp)
    try:
        return jsonify(_predict_image(fp))
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict/tabular', methods=['POST'])
def predict_tabular_api():
    _load_ml()
    data = request.json
    required = ['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal']
    for f in required:
        if f not in data:
            return jsonify({'error': f'Missing: {f}'}), 400
    try:
        return jsonify(_predict_tabular(data))
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict/complete', methods=['POST'])
def complete_analysis():
    try:
        _load_ml()
        data = request.get_json(silent=True) or {}
        print("[DEBUG] /predict/complete received:", list(data.keys()) if data else "empty")
        
        pneu_result, heart_result, report_text = None, None, None
        
        if data.get('image_data'):
            try:
                import base64
                from PIL import Image
                import io
                img = Image.open(io.BytesIO(base64.b64decode(data['image_data'])))
                fp = os.path.join(app.config['UPLOAD_FOLDER'], 'tmp.png')
                img.save(fp)
                pneu_result = _predict_image(fp)
            except Exception as e:
                pneu_result = {'error': str(e)}
        
        if data.get('tabular_data'):
            try:
                heart_result = _predict_tabular(data['tabular_data'])
            except Exception as e:
                heart_result = {'error': str(e)}
        
        if (pneu_result and not pneu_result.get('error')) or (heart_result and not heart_result.get('error')):
            try:
                report_text = _generate_report(pneu_result, heart_result, data.get('tabular_data'))
            except Exception as e:
                report_text = f"Error: {str(e)}"
        
        print("Response:", {
            "pneumonia": pneu_result,
            "heart": heart_result
        })

        return jsonify({
            "pneumonia": pneu_result,
            "heart": heart_result,
            "report": report_text
        })

    except Exception as e:
        print("ERROR:", str(e))
        return jsonify({
            "error": str(e)
        }), 500

if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 5000))
    print(f"Starting on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)