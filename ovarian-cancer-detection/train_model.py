# train_model.py
# FINAL VERSION: Feature Selection + Training + Save Model

import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

# -----------------------------
# 1. LOAD DATA
# -----------------------------
df = pd.read_csv("dataset/ovarian.csv")
print("Dataset loaded:", df.shape)

# -----------------------------
# 2. CLEAN AFP COLUMN
# -----------------------------
def clean_numeric(value):
    if pd.isna(value):
        return np.nan
    value = str(value).replace("\t", "").strip()
    if value.lower() == "unknown":
        return np.nan
    if value.startswith(">"):
        value = value.replace(">", "")
    try:
        return float(value)
    except:
        return np.nan

df["AFP"] = df["AFP"].apply(clean_numeric)

# -----------------------------
# 3. HANDLE MISSING VALUES
# -----------------------------
num_cols = df.select_dtypes(include=[np.number]).columns
df[num_cols] = df[num_cols].fillna(df[num_cols].median())

print("Missing values handled")

# -----------------------------
# 4. CREATE TARGET (Outcome)
# -----------------------------
afp_threshold = df["AFP"].median()
df["Outcome"] = df["AFP"].apply(lambda x: 1 if x > afp_threshold else 0)

print("Outcome distribution:")
print(df["Outcome"].value_counts())

# -----------------------------
# 5. FEATURE SELECTION (IMPORTANT)
# -----------------------------
FEATURES = ["Age", "AFP", "ALB", "ALP", "ALT", "AST"]

X = df[FEATURES]
y = df["Outcome"]

# -----------------------------
# 6. TRAIN–TEST SPLIT
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -----------------------------
# 7. TRAIN MODEL
# -----------------------------
model = RandomForestClassifier(n_estimators=300, random_state=42)
model.fit(X_train, y_train)

preds = model.predict(X_test)
acc = accuracy_score(y_test, preds)

print("Model Accuracy:", acc)

# -----------------------------
# 8. SAVE MODEL & FEATURES
# -----------------------------
joblib.dump(model, "model/ovarian_model.pkl")
joblib.dump(FEATURES, "model/features.pkl")

print("Model and features saved successfully")
