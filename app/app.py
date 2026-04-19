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

app = Flask(__name__)
CORS(app)
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
def predict_complete_api():
    _load_ml()
    data = request.json
    img_res, tab_res, report = None, None, None
    
    if data.get('image_data'):
        try:
            import base64
            from PIL import Image
            import io
            img = Image.open(io.BytesIO(base64.b64decode(data['image_data'])))
            fp = os.path.join(app.config['UPLOAD_FOLDER'], 'tmp.png')
            img.save(fp)
            img_res = _predict_image(fp)
        except Exception as e:
            img_res = {'error': str(e)}
    
    if data.get('tabular_data'):
        try:
            tab_res = _predict_tabular(data['tabular_data'])
        except Exception as e:
            tab_res = {'error': str(e)}
    
    if (img_res and not img_res.get('error')) or (tab_res and not tab_res.get('error')):
        try:
            report = _generate_report(img_res, tab_res, data.get('tabular_data'))
        except Exception as e:
            report = f"Error: {str(e)}"
    
    return jsonify({'image_prediction': img_res, 'heart_risk': tab_res, 'llm_analysis': report})

if __name__ == '__main__':
    print("Starting - models load on first request")
    app.run(host='0.0.0.0', port=5000, debug=False)