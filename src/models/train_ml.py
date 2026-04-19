import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
import joblib

def train_ml_model():
    """
    Train heart disease prediction model using UCI Cleveland dataset.
    
    Pipeline:
    1. Load data from heart_cleveland_upload.csv
    2. Data verification (shape, missing values)
    3. Preprocessing (handle missing, scale features)
    4. Train-test split (80/20, stratified)
    5. Train Random Forest model
    6. Evaluate with multiple metrics
    7. Save model and scaler
    """
    
    # =============================================================================
    # Step 1: Load Data
    # =============================================================================
    print("=" * 60)
    print("STEP 1: Loading Data")
    print("=" * 60)
    
    df = pd.read_csv('data/heart_cleveland_upload.csv')
    print(f"Data loaded from: data/heart_cleveland_upload.csv")
    print(f"Total records: {len(df)}")
    
    # =============================================================================
    # Step 2: Data Verification
    # =============================================================================
    print("\n" + "=" * 60)
    print("STEP 2: Data Verification")
    print("=" * 60)
    
    print(f"\nDataset Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    print(f"\nMissing Values per Column:")
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print(missing[missing > 0])
        print("\n[!] Missing values detected - will be handled")
    else:
        print("No missing values detected [OK]")
    
    print(f"\nTarget Distribution (condition):")
    print(df['condition'].value_counts())
    print(f"  - No Disease (0): {(df['condition'] == 0).sum()}")
    print(f"  - Disease (1): {(df['condition'] == 1).sum()}")
    
    # =============================================================================
    # Step 3: Preprocessing
    # =============================================================================
    print("\n" + "=" * 60)
    print("STEP 3: Preprocessing")
    print("=" * 60)
    
    # Handle missing values by filling with median for numerical columns
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    for col in numerical_cols:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].median(), inplace=True)
            print(f"Filled missing values in '{col}' with median")
    
    print("Missing values handled [OK]")
    
    # Separate features and target
    # Target column is 'condition' (0 = no disease, 1 = disease)
    X = df.drop('condition', axis=1)
    y = df['condition']
    
    print(f"\nFeatures: {list(X.columns)}")
    print(f"Target: 'condition' (0=No Disease, 1=Disease)")
    
    # =============================================================================
    # Step 4: Train-Test Split
    # =============================================================================
    print("\n" + "=" * 60)
    print("STEP 4: Train-Test Split (80/20, Stratified)")
    print("=" * 60)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2, 
        random_state=42, 
        stratify=y
    )
    
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    # Scale features using StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("Features scaled with StandardScaler [OK]")
    
    # =============================================================================
    # Step 5: Train Model
    # =============================================================================
    print("\n" + "=" * 60)
    print("STEP 5: Training Random Forest Model")
    print("=" * 60)
    
    model = RandomForestClassifier(
        n_estimators=100, 
        max_depth=10, 
        random_state=42, 
        n_jobs=-1
    )
    
    print("Training model...")
    model.fit(X_train_scaled, y_train)
    print("Model training complete [OK]")
    
    # =============================================================================
    # Step 6: Evaluation
    # =============================================================================
    print("\n" + "=" * 60)
    print("STEP 6: Model Evaluation")
    print("=" * 60)
    
    y_pred = model.predict(X_test_scaled)
    
    # Calculate metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"\nMetrics:")
    print(f"  Accuracy:  {acc:.4f} ({acc*100:.2f}%)")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:   {rec:.4f}")
    print(f"  F1 Score: {f1:.4f}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"\nConfusion Matrix:")
    print(f"                Predicted")
    print(f"              No Risk  Risk")
    print(f"Actual No Risk  {cm[0][0]:4d}  {cm[0][1]:4d}")
    print(f"       Risk    {cm[1][0]:4d}  {cm[1][1]:4d}")
    
    # Classification Report
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['No Risk', 'Risk']))
    
    # =============================================================================
    # Step 7: Save Model
    # =============================================================================
    print("\n" + "=" * 60)
    print("STEP 7: Saving Model")
    print("=" * 60)
    
    joblib.dump(model, 'models/ml_model.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')
    print("Model saved to: models/ml_model.pkl")
    print("Scaler saved to: models/scaler.pkl")
    
    # Feature Importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\nTop 5 Important Features:")
    for idx, row in feature_importance.head().iterrows():
        print(f"  {row['feature']}: {row['importance']:.4f}")
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    train_ml_model()