from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.api.predict import predict_image, predict_tabular, generate_medical_report

app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = 'app/static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict/image', methods=['POST'])
def predict_image_api():
    """Process chest X-ray image for pneumonia prediction."""
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
        print(f"[ERROR] Image prediction failed: {str(e)}")
        return jsonify({'error': str(e), 'prediction': None}), 500

@app.route('/predict/tabular', methods=['POST'])
def predict_tabular_api():
    """Process heart disease risk from clinical data."""
    data = request.json
    
    # IMPROVEMENT #5: Strict validation
    required_fields = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
    for field in required_fields:
        if field not in data:
            return jsonify({'error': f'Missing required field: {field}'}), 400
    
    try:
        result = predict_tabular(data)
        return jsonify(result)
    except Exception as e:
        print(f"[ERROR] Tabular prediction failed: {str(e)}")
        return jsonify({'error': str(e), 'prediction': None}), 500

@app.route('/predict/complete', methods=['POST'])
def predict_complete_api():
    """
    Complete analysis: Image + Tabular + LLM Report
    
    IMPROVEMENT #5: Strict pipeline validation
    - Validates each model output before passing to LLM
    - Passes patient data for context-rich reports
    """
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
            print(f"[ERROR] Image failed: {str(e)}")
    
    # Process tabular data
    if data.get('tabular_data'):
        try:
            tabular_result = predict_tabular(data['tabular_data'])
            print(f"[INFO] Heart: {tabular_result.get('prediction')} ({tabular_result.get('confidence')}%)")
        except Exception as e:
            tabular_result = {'error': str(e)}
            print(f"[ERROR] Heart failed: {str(e)}")
    
    # Get patient data for context (age, sex, etc)
    patient_data = data.get('tabular_data', None)
    
    # Validate both predictions before LLM
    # IMPROVEMENT #5: Pipeline validation layer
    image_valid = image_result and not image_result.get('error')
    tabular_valid = tabular_result and not tabular_result.get('error')
    
    if image_valid or tabular_valid:
        try:
            print("[INFO] Generating LLM report...")
            # Pass patient data for context-rich report
            llm_analysis = generate_medical_report(
                image_result, 
                tabular_result,
                patient_data=patient_data
            )
            print(f"[INFO] LLM report generated")
        except Exception as e:
            llm_analysis = f"Error generating report: {str(e)}"
            print(f"[ERROR] LLM failed: {str(e)}")
    else:
        # No valid predictions - don't call LLM
        llm_analysis = "Unable to generate report. Please check model inputs and try again."
        print("[WARNING] No valid predictions for LLM")
    
    # Return complete response with validation info
    return jsonify({
        'image_prediction': image_result,
        'heart_risk': tabular_result,
        'llm_analysis': llm_analysis,
        'validation': {
            'image_valid': image_valid,
            'tabular_valid': tabular_valid
        }
    })

if __name__ == '__main__':
    print("[INFO] Starting Medical ML App on http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)