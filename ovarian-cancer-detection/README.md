# 🧬 Optimizing Machine Learning-Based Ovarian Cancer Detection

This project is an AI-powered web application designed to predict the **risk of ovarian cancer** using clinical biomarker data.  
It combines **Machine Learning**, **Explainable AI (SHAP)**, and a **Flask-based web interface** to provide transparent, interpretable, and user-friendly predictions.

---

## 🚀 Key Features

- Machine Learning–based ovarian cancer risk prediction
- Uses clinical biomarkers such as Age, AFP, ALB, ALP, ALT, and AST
- Explainable AI using SHAP (Global and Individual explanations)
- Doctor Dashboard for model insights
- Accuracy and evaluation graphs
- Flask-based web application
- Ready for cloud deployment (Render)

---

## 🛠️ Technologies Used

- Python  
- Flask  
- Scikit-learn  
- Pandas & NumPy  
- Matplotlib  
- SHAP (Explainable AI)  
- HTML, CSS (Bootstrap)  
- Git & GitHub  

---

## 📂 Project Structure

ovarian-cancer-detection/
│
├── app.py
├── requirements.txt
├── README.md
│
├── model/
│ └── ovarian_model.pkl
│
├── dataset/
│ └── ovarian.csv
│
├── templates/
│ ├── index.html
│ ├── result.html
│ ├── dashboard.html
│ └── metrics.html
│
├── static/
│ ├── shap_summary.png
│ └── patient_shap.png
│
└── evaluate.py


---

## ⚙️ How to Run the Project Locally

### Step 1: Clone the Repository
```bash
git clone https://github.com/KalyanArava/ovarian-cancer-detection.git
cd ovarian-cancer-detection

Step 2: Install Dependencies
pip install -r requirements.txt

Step 3: Run the Application
python app.py

Step 4: Open in Browser
http://127.0.0.1:5000

🧠 Explainable AI (SHAP)

Global SHAP shows overall feature importance across the dataset

Individual SHAP explains how each biomarker influenced a specific patient’s prediction

Improves transparency, trust, and interpretability

📊 Prediction Output

The model predicts ovarian cancer risk as:

Low Risk

Medium Risk

High Risk

Along with probability percentage and SHAP-based explanations.

🔮 Future Work

Integration of CNN-based medical image analysis (Ultrasound, MRI, CT scans)

Multimodal learning combining biomarkers and medical images

Real-time clinical decision support system

Cloud scalability and mobile-friendly deployment

⚠️ Disclaimer

This project is intended only for academic and research purposes.
It should not be used as a medical diagnostic tool.

👨‍💻 Author

Kalyan Arava
MCA Final Year Student
GitHub: https://github.com/KalyanArava


---

If you want next (also one-click ready):
- **Project Abstract**
- **Deployment section**
- **Viva Questions & Answers**
- **Resume project description**

Just say the word 👍
