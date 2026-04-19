# Medical ML Models - Training Summary

## Datasets Used

### 1. Heart Disease Dataset
- **Source**: UCI Cleveland Heart Disease Dataset
- **File**: `data/heart_cleveland_upload.csv`
- **Records**: 297
- **Features**: 13 (age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal)
- **Target**: condition (0 = No Disease, 1 = Disease)

### 2. Chest X-Ray Dataset
- **Source**: Kaggle Chest X-Ray Pneumonia Dataset
- **Folder**: `data/chest_xray/chest_xray/train/`
- **Total Images**: 5,216
  - NORMAL: 1,341
  - PNEUMONIA: 3,875

---

## Model Architectures

### Heart Disease Model
- **Algorithm**: Random Forest Classifier
- **Parameters**: n_estimators=100, max_depth=10, random_state=42
- **Preprocessing**: StandardScaler (fit on train, transform on test)

### Chest X-Ray Model
- **Architecture**: ResNet18 (pretrained on ImageNet)
- **Modified**: fc layer changed to output 2 classes
- **Preprocessing**: Resize to 224x224, normalize with ImageNet stats

---

## Training Configuration

### Data Split
- **Train/Test Split**: 80% / 20%
- **Stratification**: Yes (maintains class distribution)
- **Random State**: 42

### Training Parameters
- **Optimizer**: Adam (lr=0.001)
- **Scheduler**: StepLR (step_size=3, gamma=0.1)
- **Epochs**: 10
- **Batch Size**: 16
- **Loss Function**: CrossEntropyLoss

---

## Results

### Heart Disease Model (ML)

```
============================================================
STEP 1: Loading Data
============================================================
Data loaded from: data/heart_cleveland_upload.csv
Total records: 297

============================================================
STEP 2: Data Verification
============================================================
Dataset Shape: (297, 14)
Columns: ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'condition']

Missing Values per Column:
No missing values detected [OK]

Target Distribution (condition):
condition
0    160
1    137
Name: count, dtype: int64
  - No Disease (0): 160
  - Disease (1): 137

============================================================
STEP 3: Preprocessing
============================================================
Missing values handled [OK]

Features: ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
Target: 'condition' (0=No Disease, 1=Disease)

============================================================
STEP 4: Train-Test Split (80/20, Stratified)
============================================================
Training set size: 237
Test set size: 60
Features scaled with StandardScaler [OK]

============================================================
STEP 5: Training Random Forest Model
============================================================
Training model...
Model training complete [OK]

============================================================
STEP 6: Model Evaluation
============================================================

Metrics:
  Accuracy:  0.8833 (88.33%)
  Precision: 0.9565
  Recall:    0.7857
  F1 Score: 0.8627

Confusion Matrix:
                Predicted
              No Risk  Risk
Actual No Risk    31     1
       Risk        6    22

Classification Report:
              precision    recall  f1-score   support

     No Risk       0.84      0.97      0.90        32
        Risk       0.96      0.79      0.86        28

    accuracy                           0.88        60
   macro avg       0.90      0.88      0.88        60
weighted avg       0.89      0.88      0.88        60

============================================================
STEP 7: Saving Model
============================================================
Model saved to: models/ml_model.pkl
Scaler saved to: models/scaler.pkl

Top 5 Important Features:
  thalach: 0.1613
  cp: 0.1245
  oldpeak: 0.1126
  ca: 0.1119
  age: 0.0950
```

---

### Chest X-Ray Model (CNN)

```
============================================================
STEP 1: Loading Data
============================================================
Data loaded from: data/chest_xray/chest_xray/train

============================================================
STEP 2: Data Verification & Preprocessing
============================================================
Dataset loaded successfully [OK]
Total images: 5216
  - NORMAL: 1341
  - PNEUMONIA: 3875

============================================================
STEP 3: Train-Validation Split (80/20)
============================================================
Training set size: 4172
Validation set size: 1044
Data loaders created [OK]

============================================================
STEP 4: Model Setup (ResNet18)
============================================================
Model: ResNet18 (pretrained on ImageNet)
Modified final layer for 2 classes: NORMAL, PNEUMONIA
Optimizer: Adam (lr=0.001)
Scheduler: StepLR (step_size=3, gamma=0.1)

============================================================
STEP 5: Training (10 Epochs)
============================================================
Starting training...

Epoch 1/10 - Loss: 0.1681, Train Acc: 93.60%
Validation Accuracy: 96.46%
[OK] Best model saved (val acc: 96.46%)

Epoch 2/10 - Loss: 0.1298, Train Acc: 96.05%
Validation Accuracy: 96.84%
[OK] Best model saved (val acc: 96.84%)

Epoch 3/10 - Loss: 0.0965, Train Acc: 97.17%
Validation Accuracy: 97.22%
[OK] Best model saved (val acc: 97.22%)

Epoch 4/10 - Loss: 0.0675, Train Acc: 97.88%
Validation Accuracy: 97.51%
[OK] Best model saved (val acc: 97.51%)

Epoch 5/10 - Loss: 0.0482, Train Acc: 98.24%
Validation Accuracy: 97.80%

Epoch 6/10 - Loss: 0.0399, Train Acc: 98.62%
Validation Accuracy: 97.51%

Epoch 7/10 - Loss: 0.0335, Train Acc: 98.83%
Validation Accuracy: 97.89%

Epoch 8/10 - Loss: 0.0310, Train Acc: 98.85%
Validation Accuracy: 98.37%
[OK] Best model saved (val acc: 98.37%)  ← BEST

Epoch 9/10 - Loss: 0.0316, Train Acc: 98.99%
Validation Accuracy: 97.89%

Epoch 10/10 - Loss: 0.0293, Train Acc: 99.02%
Validation Accuracy: 97.80%

Training complete. Best validation accuracy: 98.37%

============================================================
STEP 6: Final Evaluation
============================================================

Metrics on Validation Set:
  Accuracy:  0.9808 (98.08%)
  Precision: 0.9869
  Recall:    0.9869
  F1 Score: 0.9869

Confusion Matrix:
                Predicted
              Normal  Pneumonia
Actual Normal     272       10
       Pneumonia   10      752

Classification Report:
              precision    recall  f1-score   support

      NORMAL       0.96      0.96      0.96       282
   PNEUMONIA       0.99      0.99      0.99       762

    accuracy                           0.98      1044
   macro avg       0.98      0.98      0.98      1044
weighted avg       0.98      0.98      0.98      1044
```

---

## Model Limitations & Caveats

### Heart Disease Model
- **Precision (0.95)**: High - when model predicts disease, it's usually correct
- **Recall (0.78)**: Lower - model may MISS some positive cases (false negatives)
- **Implication**: Patients with heart disease may be incorrectly classified as "Low Risk"
- **Clinical Use**: Should be used with cardiologist consultation

### Chest X-Ray Model (CNN)
- **Dataset Imbalance**: Pneumonia=3875, Normal=1341 (ratio ~3:1)
- **Implication**: Model may be biased toward predicting Pneumonia
- **High Confidence Warning**: Confidence >95% on Pneumonia should be verified clinically

---

## Summary Comparison

| Model | Dataset | Accuracy | Precision | Recall | F1 Score |
|-------|---------|----------|-----------|--------|---------|
| **Heart Disease (ML)** | UCI Cleveland | 88.33% | 0.9565 | 0.7857 | 0.8627 |
| **Chest X-Ray (CNN)** | Kaggle | **98.08%** | 0.9869 | 0.9869 | 0.9869 |

---

## Output Format Examples

### Heart Disease
- Low Risk (22.0%) - Healthy patient
- High Risk (81.25%) - At-risk patient

### Chest X-Ray
- NORMAL (95.2%) - Normal chest
- PNEUMONIA (87.3%) - Pneumonia detected

### LLM Report
- Structured medical report with specific precautions
- Model limitations mentioned
- Disclaimer to consult professional

---

## Files Generated

| File | Description |
|------|-------------|
| `models/ml_model.pkl` | Trained Random Forest for heart disease |
| `models/scaler.pkl` | StandardScaler for feature scaling |
| `models/cnn_model.pth` | Trained ResNet18 for chest X-ray |

---

## Notes

- Both models are trained on CPU
- No GPU required for inference
- Models are deployment-ready (CPU-compatible)
- Groq API used for LLM medical report generation
- All preprocessing steps are visible in training scripts
- Evaluation metrics include confusion matrix for both models

Generated: April 2026