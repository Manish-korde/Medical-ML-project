# AGENTS.md

## Setup

```powershell
cd C:\Users\manis\OneDrive\Desktop\medical-ML project
.\venv\Scripts\Activate.ps1
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121
pip install scikit-learn pandas matplotlib seaborn flask jupyter ipykernel joblib
```

## Running the App

Flask app is in `app/`. Run with:
```powershell
python app/app.py
```

Or:
```powershell
$env:FLASK_APP = "app:app"
flask run
```

## Project Structure

- `app/` - Flask application (app.py, templates/index.html, static/uploads/)
- `src/data/` - Dataset generation scripts
- `src/models/` - Model training scripts (train_cnn.py, train_ml.py)
- `src/api/` - Prediction utilities (predict.py)
- `data/chest_xray/chest_xray/train/` - Chest X-ray images (NORMAL, PNEUMONIA) [Kaggle]
- `data/heart_cleveland_upload.csv` - Heart disease dataset (UCI Cleveland)
- `models/` - Trained models (cnn_model.pth, ml_model.pkl, scaler.pkl)

## Training Models

```powershell
python src/models/train_ml.py    # Train Random Forest for heart disease
python src/models/train_cnn.py  # Train ResNet18 for chest X-ray
python src/data/generate_dataset.py  # Generate synthetic X-ray images
```

## Requirements

- Groq API key for LLM report generation