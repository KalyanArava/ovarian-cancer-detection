# app.py
# FINAL ABSOLUTE SAFE VERSION – NO SHAP INDEXING AT ALL

from flask import Flask, render_template, request
import joblib
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

app = Flask(__name__)

# Load model
model = joblib.load("model/ovarian_model.pkl")

MODEL_FEATURES = ["Age", "AFP", "ALB", "ALP", "ALT", "AST"]
ALL_FEATURES = [
    "Age", "AFP", "ALB", "ALP", "ALT", "AST",
    "PLT", "RBC", "TBIL", "TP"
]

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # -----------------------------
        # Read input
        # -----------------------------
        form_data = {f: float(request.form[f]) for f in ALL_FEATURES}
        X = np.array([[form_data[f] for f in MODEL_FEATURES]])

        # -----------------------------
        # Prediction
        # -----------------------------
        prob = model.predict_proba(X)[0][1]
        risk_percent = round(prob * 100, 2)

        if risk_percent < 40:
            result = "✅ Low Risk of Ovarian Cancer"
        elif risk_percent <= 70:
            result = "⚠️ Medium Risk of Ovarian Cancer"
        else:
            result = "🚨 High Risk of Ovarian Cancer"

        # -----------------------------
        # INDIVIDUAL SHAP (FINAL SAFE METHOD)
        # -----------------------------
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)

        # 🔥 UNIVERSAL EXTRACTION (NO INDEX ASSUMPTIONS)
        if isinstance(shap_values, list):
            patient_shap = shap_values[0]
        else:
            patient_shap = shap_values

        # Flatten completely
        patient_shap = np.asarray(patient_shap).reshape(-1)

        # Align size with features (extra safety)
        patient_shap = patient_shap[:len(MODEL_FEATURES)]

        # Create DataFrame
        shap_df = pd.DataFrame({
            "Feature": MODEL_FEATURES,
            "Contribution": patient_shap
        })

        # Sort by impact
        shap_df = shap_df.reindex(
            shap_df["Contribution"].abs().sort_values().index
        )

        # Plot bar chart
        plt.figure(figsize=(7, 4))
        plt.barh(
            shap_df["Feature"],
            shap_df["Contribution"],
            color=["red" if v > 0 else "blue" for v in shap_df["Contribution"]]
        )

        plt.axvline(0, color="black", linewidth=0.8)
        plt.xlabel("Impact on Risk")
        plt.title("Individual Patient Feature Contributions (SHAP)")
        plt.tight_layout()
        plt.savefig("static/patient_shap.png", dpi=300)
        plt.close()

        return render_template(
            "result.html",
            prediction=result,
            probability=risk_percent
        )

    except Exception as e:
        return f"Error: {e}"
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)

