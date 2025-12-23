# explain.py
# FINAL CORRECT SHAP SUMMARY PLOT (NO INTERACTIONS)

import shap
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Load model and features
# -----------------------------
model = joblib.load("model/ovarian_model.pkl")
features = joblib.load("model/features.pkl")

# -----------------------------
# Load dataset
# -----------------------------
df = pd.read_csv("dataset/ovarian.csv")

# -----------------------------
# Clean numeric values
# -----------------------------
def clean_value(x):
    try:
        x = str(x).replace("\t", "").strip()
        if x.lower() == "unknown":
            return np.nan
        if x.startswith(">"):
            x = x.replace(">", "")
        return float(x)
    except:
        return np.nan

for col in features:
    df[col] = df[col].apply(clean_value)

df[features] = df[features].fillna(df[features].median())

# -----------------------------
# Use a sample (best practice)
# -----------------------------
X = df[features].sample(200, random_state=42)

# -----------------------------
# Create SHAP explainer
# -----------------------------
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

# -----------------------------
# STANDARD SUMMARY PLOT (FIX)
# -----------------------------
plt.figure(figsize=(10, 6))
shap.summary_plot(
    shap_values[1],   # class 1 = high risk
    X,
    feature_names=features,
    plot_type="dot",
    show=False
)

plt.title("SHAP Feature Importance – Ovarian Cancer Risk")
plt.tight_layout()
plt.savefig("static/shap_summary.png", dpi=300)
plt.close()

print("✅ SHAP summary plot generated correctly!")
