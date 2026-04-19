import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import joblib
import pandas as pd

# =============================================================================
# Model Configuration & Metadata
# =============================================================================
# 
# HEART DISEASE MODEL (Random Forest)
# - Training Dataset: UCI Cleveland Heart Disease (297 samples)
# - Features: 13 clinical parameters
# - Trained: 80/20 split, stratified
# - Performance: Accuracy=88.33%, Precision=0.9565, Recall=0.7857, F1=0.8627
#
# CAVEATS:
# - Lower recall (0.78) means model may miss some positive cases (false negatives)
# - High precision (0.95) means when it predicts disease, it's usually correct
# - Use clinically with caution - consult cardiologists for definitive diagnosis
#
# CHEST X-RAY MODEL (ResNet18 CNN)
# - Training Dataset: Kaggle Chest X-Ray (5216 images, imbalanced)
# - Class Distribution: Normal=1341, Pneumonia=3875 (imbalanced ~3:1 ratio)
# - Performance: Accuracy=98.08%, Precision=0.9869, Recall=0.9869, F1=0.9869
#
# CAVEATS:
# - Dataset imbalance may favor pneumonia predictions
# - High confidence doesn't guarantee accuracy
# - Use alongside clinical correlation
#
# =============================================================================

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

device = torch.device('cpu')

# Image preprocessing - Resize to 224x224 and normalize with ImageNet stats
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Model cache (lazy loading)
_cnn_model = None
_ml_model = None
_scaler = None
_client = None

# Model performance thresholds for reporting
HEART_MODEL_INFO = {
    'name': 'Random Forest Classifier',
    'precision': 0.9565,
    'recall': 0.7857,
    'accuracy': 0.8833,
    'dataset': 'UCI Cleveland (297 samples)'
}

CNN_MODEL_INFO = {
    'name': 'ResNet18 (Fine-tuned)',
    'precision': 0.9869,
    'recall': 0.9869,
    'accuracy': 0.9808,
    'dataset': 'Kaggle Chest X-Ray (5216 images)'
}

def _get_cnn_model():
    """Load CNN model for chest X-ray classification."""
    global _cnn_model
    if _cnn_model is None:
        _cnn_model = models.resnet18(weights=None)
        _cnn_model.fc = nn.Linear(_cnn_model.fc.in_features, 2)  # 2 classes: Normal, Pneumonia
        _cnn_model.load_state_dict(torch.load('models/cnn_model.pth', map_location=device))
        _cnn_model.to(device)
        _cnn_model.eval()
    return _cnn_model

def _get_ml_models():
    """Load heart disease model and scaler."""
    global _ml_model, _scaler
    if _ml_model is None:
        _ml_model = joblib.load('models/ml_model.pkl')
        _scaler = joblib.load('models/scaler.pkl')
    return _ml_model, _scaler

def _get_groq_client():
    """Initialize Groq API client."""
    global _client
    if _client is None:
        api_key = os.environ.get('GROQ_API_KEY') or os.environ.get('GROQ_KEY')
        if not api_key:
            try:
                from dotenv import load_dotenv
                load_dotenv()
                api_key = os.environ.get('GROQ_API_KEY')
            except Exception:
                pass
        
        if api_key:
            _client = Groq(api_key=api_key)
    return _client

def predict_image(image_path):
    """
    Predict chest X-ray image for pneumonia.
    
    Returns:
        dict: prediction, confidence, probabilities, model_info
    """
    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)
    model = _get_cnn_model()
    
    with torch.no_grad():
        output = model(img_tensor)
        probs = torch.softmax(output, dim=1)[0]
        pred_idx = output.argmax(1).item()
    
    classes = ['NORMAL', 'PNEUMONIA']
    primary_prob = probs[pred_idx].item() * 100
    
    # Note: Dataset is imbalanced (Pneumonia ~3x Normal), so high confidence
    # should be interpreted cautiously
    result = {
        'prediction': classes[pred_idx],
        'confidence': round(primary_prob, 2),
        'probabilities': {
            'NORMAL': round(probs[0].item() * 100, 2),
            'PNEUMONIA': round(probs[1].item() * 100, 2)
        },
        'model_info': CNN_MODEL_INFO,
        'caution': 'Dataset is imbalanced - high confidence should be verified clinically'
    }
    return result

def predict_tabular(data, threshold=0.5):
    """
    Predict heart disease risk from clinical data.
    
    Args:
        data: dict with 13 clinical features
        threshold: probability threshold for risk classification (default 0.5)
    
    Returns:
        dict: prediction (High/Low Risk), confidence, probabilities, model_info
    """
    print("[DEBUG] predict_tabular() called")
    print(f"[DEBUG] Input data: {data}")
    
    ml_model, scaler = _get_ml_models()
    print(f"[DEBUG] Model loaded: {HEART_MODEL_INFO['name']}")
    
    df = pd.DataFrame([data])
    print(f"[DEBUG] Input shape: {df.shape}, columns: {list(df.columns)}")
    
    df_scaled = scaler.transform(df)
    
    pred = ml_model.predict(df_scaled)[0]
    prob = ml_model.predict_proba(df_scaled)[0]
    
    # Probability of heart disease (class 1)
    risk_prob = prob[1]
    risk_percent = round(risk_prob * 100, 2)
    
    # IMPROVEMENT #1: Meaningful risk labels
    # Instead of "Risk (55%)" -> "High Risk (55%)" or "Low Risk (45%)"
    if risk_prob >= threshold:
        risk_label = "High Risk"
    else:
        risk_label = "Low Risk"
    
    print(f"[DEBUG] Risk probability: {risk_percent}%, Label: {risk_label}")
    
    result = {
        'prediction': risk_label,
        'confidence': risk_percent,
        'probabilities': {
            'No Risk': round(prob[0] * 100, 2),
            'Risk': round(prob[1] * 100, 2)
        },
        'model_info': HEART_MODEL_INFO,
        'caution': 'High precision (0.95) but lower recall (0.78) - may miss some positive cases'
    }
    print(f"[DEBUG] Returning: {result}")
    return result

def generate_fallback_report(image_result=None, tabular_result=None):
    """Generate fallback report when LLM is unavailable."""
    report_lines = ["MEDICAL DIAGNOSIS SUMMARY", "=" * 40]
    
    if image_result and not image_result.get('error'):
        pred = image_result.get('prediction', 'Unknown')
        conf = image_result.get('confidence', 0)
        report_lines.append(f"\nChest X-Ray Analysis:")
        report_lines.append(f"  - Diagnosis: {pred} ({conf}%)")
        
        if pred == 'PNEUMONIA':
            report_lines.append("  - Note: Follow-up clinical correlation recommended")
        else:
            report_lines.append("  - Note: No acute cardiopulmonary abnormality detected")
    
    if tabular_result and not tabular_result.get('error'):
        pred = tabular_result.get('prediction', 'Unknown')
        conf = tabular_result.get('confidence', 0)
        report_lines.append(f"\nHeart Disease Risk Assessment:")
        report_lines.append(f"  - Risk Level: {pred} ({conf}%)")
        
        if 'High' in pred:
            report_lines.append("  - Note: Consult cardiologist for further evaluation")
        else:
            report_lines.append("  - Note: Maintain healthy lifestyle with regular check-ups")
    
    report_lines.append("\n" + "=" * 40)
    report_lines.append("DISCLAIMER: This is an automated analysis.")
    report_lines.append("Please consult a healthcare professional for definitive diagnosis.")
    return "\n".join(report_lines)

def _filter_llm_output(report, image_valid=False, image_conf=0, tabular_valid=False, tabular_prob=0):
    """
    CRITICAL: Filter LLM output to ensure safety and model-LLM consistency.
    
    - Remove banned medical terms
    - Force consistency with model predictions
    """
    # Banned words that indicate medical advice/treatment
    banned_words = [
        "angiography", "catheterization", "stent", "bypass", 
        "antibiotic", "medication", "prescription", "treatment", 
        "therapy", "surgery", "procedure", "diagnosis of",
        "take medication", "start treatment", "undergo", "recommend taking"
    ]
    
    report_lower = report.lower()
    for word in banned_words:
        if word in report_lower:
            print(f"[WARNING] Banned word detected: {word}")
            # Replace with safe fallback
            return "Output restricted. Please consult a healthcare professional for medical advice."
    
    # Force model-LLM consistency
    # If model predicts Low Risk, LLM must NOT say "high risk"
    if tabular_valid:
        if tabular_prob < 0.5:
            # Model says Low Risk - override any "high risk" mentions
            report_safe = report.replace("high cardiovascular risk", "low cardiovascular risk")
            report_safe = report_safe.replace("elevated cardiovascular risk", "low cardiovascular risk")
            if "high risk" in report_lower and "high cardiovascular" not in report_lower:
                report_safe = report_safe.replace("high risk", "potential risk")
            return report_safe
        else:
            # Model says High Risk - override any "low risk" mentions  
            report_safe = report.replace("low cardiovascular risk", "elevated cardiovascular risk")
            report_safe = report_safe.replace("minimal risk", "elevated risk")
            return report_safe
    
    return report

def generate_medical_report(image_result=None, tabular_result=None, patient_data=None):
    """
    Generate STRUCTURED and RESTRICTED medical report using Groq API.
    
    STRICT RULES:
    - NO treatments, medications, or medical procedures
    - NO disease speculation beyond predictions
    - MAX 120 words
    - General precautions ONLY
    """
    print(f"[DEBUG] generate_medical_report called")
    
    # Strict validation
    image_valid = image_result and not image_result.get('error') and 'prediction' in image_result
    tabular_valid = tabular_result and not tabular_result.get('error') and 'prediction' in tabular_result
    
    if not image_valid and not tabular_valid:
        print("[ERROR] No valid predictions - skipping LLM")
        return "Error: No valid predictions available."
    
    if not GROQ_AVAILABLE:
        print("[DEBUG] Groq unavailable - using fallback")
        return generate_fallback_report(image_result, tabular_result)
    
    client = _get_groq_client()
    if client is None:
        print("[ERROR] Groq client not initialized")
        return generate_fallback_report(image_result, tabular_result)
    
    # Get prediction values for consistency check
    image_conf = image_result['confidence'] if image_valid else 0
    tabular_prob = tabular_result['probabilities']['Risk'] / 100 if tabular_valid else 0
    
    # Build STRICT prompt
    prompt_lines = []
    prompt_lines.append("PATIENT DATA:")
    if patient_data:
        age = patient_data.get('age', 'N/A')
        sex = 'Male' if patient_data.get('sex') == 1 else 'Female' if patient_data.get('sex') == 0 else 'N/A'
        prompt_lines.append(f"Age: {age}, Sex: {sex}")
    
    prompt_lines.append("\nMODEL PREDICTIONS:")
    if image_valid:
        prompt_lines.append(f"Pneumonia: {image_result['prediction']} ({image_conf}%)")
    if tabular_valid:
        heart_label = "High Risk" if tabular_result['probabilities']['Risk'] >= 50 else "Low Risk"
        prompt_lines.append(f"Heart Risk: {heart_label} ({tabular_result['confidence']}%)")
    
    prompt_lines.append("\nCLINICAL INPUTS:")
    if patient_data:
        prompt_lines.append(f"Chest pain type: {patient_data.get('cp', 'N/A')}")
        prompt_lines.append(f"Exercise angina: {patient_data.get('exang', 'N/A')}")
        prompt_lines.append(f"ST depression: {patient_data.get('oldpeak', 'N/A')}")
    
    prompt_lines.append("\n" + "="*40)
    prompt_lines.append("STRICT TASK - Generate a SHORT report (MAX 100 words):")
    prompt_lines.append("1. SUMMARY (1-2 lines): What the models predicted")
    prompt_lines.append("2. RISK FACTORS: Based ONLY on given inputs (age, cp, exang, etc)")
    prompt_lines.append("3. PRECAUTIONS: General lifestyle only (diet, exercise, rest)")
    prompt_lines.append("4. DISCLAIMER: Must consult professional")
    prompt_lines.append("")
    prompt_lines.append("CRITICAL RULES:")
    prompt_lines.append("- NO medication suggestions")
    prompt_lines.append("- NO treatment plans")
    prompt_lines.append("- NO medical procedures")
    prompt_lines.append("- Match output to prediction probabilities")
    prompt_lines.append("- MAX 100 words")
    
    prompt = "\n".join(prompt_lines)
    print(f"[DEBUG] Strict prompt: {len(prompt)} chars")
    
    # Send to LLM with strict system prompt
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system", 
                    "content": "You are a STRICT medical summary assistant. Generate ONLY summaries of model predictions. NO medical advice, treatments, or procedures. Max 100 words. Match your output to the prediction probabilities given."
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=200  # Limit output length
        )
        raw_report = response.choices[0].message.content
        print(f"[DEBUG] Raw LLM: {len(raw_report)} chars")
        
        # CRITICAL: Filter output
        report = _filter_llm_output(
            raw_report, 
            image_valid, 
            image_conf,
            tabular_valid,
            tabular_prob
        )
        
        # Final word check
        word_count = len(report.split())
        if word_count > 120:
            print(f"[WARNING] Report too long ({word_count} words), truncating")
            words = report.split()
            report = " ".join(words[:120]) + "..."
            word_count = 120
        
        print(f"[DEBUG] Filtered report: {len(report)} chars, {word_count} words")
        return report
        
    except Exception as e:
        print(f"[ERROR] LLM API error: {str(e)}")
        return generate_fallback_report(image_result, tabular_result)

# =============================================================================
# Entry Point for Testing
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Medical ML Prediction Module")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Groq Available: {GROQ_AVAILABLE}")
    print(f"Heart Model: {HEART_MODEL_INFO['name']}")
    print(f"  - Accuracy: {HEART_MODEL_INFO['accuracy']:.2%}")
    print(f"  - Precision: {HEART_MODEL_INFO['precision']:.2%}")
    print(f"  - Recall: {HEART_MODEL_INFO['recall']:.2%}")
    print(f"CNN Model: {CNN_MODEL_INFO['name']}")
    print(f"  - Accuracy: {CNN_MODEL_INFO['accuracy']:.2%}")
    print("=" * 60)