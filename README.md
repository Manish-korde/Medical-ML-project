# Medical ML Project

AI-powered medical diagnosis system with chest X-ray classification and heart disease prediction.

## Quick Start

```powershell
cd C:\Users\manis\OneDrive\Desktop\medical-ML project
.\venv\Scripts\Activate.ps1
python app/app.py
```

Open **http://localhost:5000** in your browser.

---

## Features

| Feature | Model | Dataset | Accuracy |
|---------|-------|---------|----------|
| Chest X-Ray Analysis | ResNet18 (CNN) | Kaggle Chest X-Ray | 98.08% |
| Heart Disease Prediction | Random Forest | UCI Cleveland | 88.33% |
| Medical Report Generation | Groq API (Llama 3.3) | - | - |

---

## Project Structure

```
medical-ML project/
├── app/
│   ├── app.py              # Flask application
│   ├── templates/
│   │   └── index.html      # Web UI
│   └── static/uploads/     # Uploaded images
├── src/
│   ├── api/
│   │   └── predict.py      # Prediction utilities
│   ├── models/
│   │   ├── train_ml.py     # Train heart disease model
│   │   └── train_cnn.py    # Train chest X-ray model
│   └── data/
├── data/
│   ├── chest_xray/chest_xray/train/   # X-ray images
│   └── heart_cleveland_upload.csv       # Heart dataset
├── models/
│   ├── cnn_model.pth       # Trained ResNet18
│   ├── ml_model.pkl        # Trained Random Forest
│   └── scaler.pkl          # StandardScaler
├── .env                   # Environment variables
├── requirements.txt       # Python dependencies
├── Procfile               # Render deployment
├── runtime.txt           # Python version
├── DEPLOYMENT.md         # Deployment guide
├── MODEL_WORKFLOW.md     # Model selection guide
└── MODELS_SUMMARY.md    # Training results
```

---

## Requirements

- Python 3.11
- Groq API key (get from [console.groq.com](https://console.groq.com/))

### Install Dependencies

```powershell
pip install -r requirements.txt
```

---

## Running the App

### Development

```powershell
python app/app.py
```

Or:

```powershell
$env:FLASK_APP = "app:app"
flask run
```

### Access

Open **http://localhost:5000**

---

## Testing the App

### 1. Chest X-Ray Analysis

1. Go to "Chest X-Ray Analysis" tab
2. Upload a chest X-ray image (JPEG/PNG)
3. Click "Analyze Image"
4. View: NORMAL or PNEUMONIA prediction with confidence

### 2. Heart Disease Risk

1. Go to "Heart Disease Risk" tab
2. Enter 12 clinical values:
   - age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, slope, ca, thal
3. Click "Predict Risk"
4. View: Risk or No Risk prediction with confidence

### 3. Complete Analysis

1. Go to "Complete Analysis" tab
2. Optionally upload chest X-ray
3. Enter patient data
4. Click "Run Complete Analysis"
5. View: Image prediction + Heart risk + AI Medical Report

---

## Training Models (Optional)

Only needed if you want to retrain:

```powershell
python src/models/train_ml.py    # Train heart disease model
python src/models/train_cnn.py  # Train chest X-ray model
```

Note: CNN training takes ~1-2 hours on CPU.

---

## Deployment

See `DEPLOYMENT.md` for cloud deployment instructions (Render, Railway, Heroku).

---

## Environment Variables

Create `.env` file:

```
GROQ_API_KEY=your_groq_api_key_here
```

Get free API key at [console.groq.com](https://console.groq.com/)

---

## Technology Stack

| Component | Technology |
|-----------|------------|
| Frontend | HTML, JavaScript, CSS |
| Backend | Flask |
| Image Model | PyTorch, ResNet18 |
| ML Model | scikit-learn, Random Forest |
| LLM | Groq API (Llama 3.3) |
| Deployment | Cloud-agnostic (CPU) |

---

## Model Performance

| Model | Accuracy | Precision | Recall | F1 Score |
|-------|----------|-----------|--------|---------|
| Chest X-Ray | 98.08% | 0.9869 | 0.9869 | 0.9869 |
| Heart Disease | 88.33% | 0.9565 | 0.7857 | 0.8627 |

---

## License

MIT License

---

## Authors

Medical ML Project - AI Diagnosis Assistant