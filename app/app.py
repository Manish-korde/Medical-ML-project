"""
Medical ML Flask Application
Simplified for deployment reliability.
"""

import os
import sys

# Force CPU only
os.environ['CUDA_VISIBLE_DEVICES'] = ''

# Fix path for deployment
_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _root not in sys.path:
    sys.path.insert(0, _root)

print("[INFO] Starting Medical ML app...")

# Flask setup
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

app.config['UPLOAD_FOLDER'] = 'app/static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Import ML functions at module level (works after path is fixed)
from src.api.predict import predict_image, predict_tabular, generate_medical_report

print("[INFO] ML functions imported successfully")

# =============================================================================
# ROUTES
# =============================================================================

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict/image', methods=['POST'])
def predict_image_api():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        
        result = predict_image(filepath)
        return jsonify(result)
    except Exception as e:
        print(f"[ERROR] Image: {str(e)}")
        return jsonify({'error': str(e), 'prediction': None}), 500

@app.route('/predict/tabular', methods=['POST'])
def predict_tabular_api():
    try:
        data = request.json
        required = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
                 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
        
        for field in required:
            if field not in data:
                return jsonify({'error': f'Missing: {field}'}), 400
        
        result = predict_tabular(data)
        return jsonify(result)
    except Exception as e:
        print(f"[ERROR] Tabular: {str(e)}")
        return jsonify({'error': str(e), 'prediction': None}), 500

@app.route('/predict/complete', methods=['POST'])
def predict_complete_api():
    try:
        data = request.json
        image_result = None
        tabular_result = None
        llm_analysis = None
        
        # Process image
        if data.get('image_data'):
            try:
                import base64
                from PIL import Image
                import io
                
                img_data = base64.b64decode(data['image_data'])
                img = Image.open(io.BytesIO(img_data))
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'temp.png')
                img.save(filepath)
                
                image_result = predict_image(filepath)
                print(f"[INFO] Image: {image_result.get('prediction')}")
            except Exception as e:
                image_result = {'error': str(e)}
                print(f"[ERROR] Image: {e}")
        
        # Process tabular
        if data.get('tabular_data'):
            try:
                tabular_result = predict_tabular(data['tabular_data'])
                print(f"[INFO] Heart: {tabular_result.get('prediction')}")
            except Exception as e:
                tabular_result = {'error': str(e)}
                print(f"[ERROR] Heart: {e}")
        
        # Generate LLM report
        image_valid = image_result and not image_result.get('error')
        tabular_valid = tabular_result and not tabular_result.get('error')
        
        if image_valid or tabular_valid:
            try:
                patient_data = data.get('tabular_data')
                llm_analysis = generate_medical_report(image_result, tabular_result, patient_data)
            except Exception as e:
                llm_analysis = f"Report error: {str(e)}"
        
        return jsonify({
            'image_prediction': image_result,
            'heart_risk': tabular_result,
            'llm_analysis': llm_analysis,
            'validation': {'image_valid': image_valid, 'tabular_valid': tabular_valid}
        })
    except Exception as e:
        print(f"[ERROR] Complete: {str(e)}")
        return jsonify({'error': str(e)}), 500

# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == '__main__':
    print("=" * 50)
    print("Medical ML App running on http://localhost:5000")
    print("=" * 50)
    app.run(debug=True, host='0.0.0.0', port=5000)