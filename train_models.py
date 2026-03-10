
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
import xgboost as xgb

# Set up paths
MODELS_DIR = "models"
if not os.path.exists(MODELS_DIR):
    os.makedirs(MODELS_DIR)

print("Loading dataset...")
try:
    df = pd.read_csv("Students Social Media Addiction.csv")
except FileNotFoundError:
    print("Dataset not found!")
    exit(1)

# Preprocessing
if 'Student_ID' in df.columns:
    df.drop("Student_ID", axis=1, inplace=True)

# Correct Country
df.loc[df['Country'] == 'Israel', 'Country'] = 'Palestine'

# Encode Categorical Features
categorical_cols = df.select_dtypes(include='object').columns
encoders = {}

print("Encoding categorical features...")
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# Save Encoders
joblib.dump(encoders, os.path.join(MODELS_DIR, "encoders.joblib"))
print("Encoders saved.")

# --- Classification Task (Addiction) ---
print("Training Addiction Classification Model...")
df["Target"] = df["Addicted_Score"].apply(lambda x: 1 if x >= 7 else 0)

# Features for classification
X_cls = df.drop(["Addicted_Score", "Target"], axis=1)
y_cls = df["Target"]

# Scale Features
scaler_cls = StandardScaler()
X_cls_scaled = scaler_cls.fit_transform(X_cls)

# Train Logistic Regression
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_cls_scaled, y_cls)

# Save Classification Artifacts
joblib.dump(lr_model, os.path.join(MODELS_DIR, "model_addiction.joblib"))
joblib.dump(scaler_cls, os.path.join(MODELS_DIR, "scaler_addiction.joblib"))
print("Addiction model and scaler saved.")

# --- Regression Task (Mental Health) ---
print("Training Mental Health Regression Model...")
# Features for regression (remove target and potential leakage like addicted score if desired, 
# but notebook used all features except target)
X_reg = df.drop(["Mental_Health_Score", "Target"], axis=1)
y_reg = df["Mental_Health_Score"]

# Scale Features
scaler_reg = StandardScaler()
X_reg_scaled = scaler_reg.fit_transform(X_reg)

# Train XGBoost
xgb_model = xgb.XGBRegressor(random_state=42)
xgb_model.fit(X_reg_scaled, y_reg)

# Save Regression Artifacts
joblib.dump(xgb_model, os.path.join(MODELS_DIR, "model_mental_health.joblib"))
joblib.dump(scaler_reg, os.path.join(MODELS_DIR, "scaler_mental_health.joblib"))
print("Mental Health model and scaler saved.")

print("All models trained and saved successfully!")
