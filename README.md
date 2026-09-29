# ❤️ Heart Disease Predictor & Diagnostic Assistant

An intelligent Clinical Decision Support System (CDSS) powered by Machine Learning and Streamlit, trained on the comprehensive **UCI Heart Disease Benchmark Dataset** (919 clinical records across Cleveland, Hungary, Switzerland, and VA Long Beach).

---

## 🚀 Features

- **High-Accuracy Classification**: Employs an optimized **Random Forest Classifier Pipeline** with median imputation, standard scaling, and one-hot encoding (~83.3% test accuracy, 0.923 ROC-AUC).
- **Dual Diagnosis & Risk Scoring**:
  - Binary Diagnosis: Absence (Low Risk / Healthy) vs Presence (High Risk / Disease Detected).
  - Risk Probability: Exact percentage likelihood of cardiovascular disease.
  - Severity Staging: Multiclass estimation (Healthy, Stage 1 Mild, Stage 2 Moderate, Stage 3 Severe, Stage 4 Critical).
- **Personalized Risk Factor Breakdown**: Automated analysis of high blood pressure, cholesterol, exercise angina, ST depression, and fluoroscopy findings.
- **Actionable Health Recommendations**: Evidence-based cardiovascular wellness advice and clinical follow-up guidance.
- **One-Click Patient Presets**: Instantly load healthy, borderline, and high-risk patient profiles for rapid testing.
- **Modern Responsive Web UI**: Crafted with intuitive tabs, real-time metrics, risk badges, and probability visualization.

---

## 🛠️ Project Structure

```text
├── Models/
│   ├── best_classification_model.pkl   # Fitted production Scikit-Learn Pipeline
│   └── model_metadata.json             # Model metrics, feature definitions, and stages
├── Training/
│   ├── Heart_Detection.ipynb           # Original exploration & training notebook
│   └── heart_disease_uci.csv           # Cleaned UCI benchmark dataset (919 rows)
├── app.py                              # Streamlit web application
├── README.md                           # Documentation
```

---

## 📊 Input Clinical Attributes

| Feature | Name | Description / Options |
|---|---|---|
| `age` | Age | Patient age in years (18 - 100) |
| `sex` | Biological Sex | `Male` or `Female` |
| `cp` | Chest Pain Type | `typical angina`, `atypical angina`, `non-anginal`, `asymptomatic` |
| `trestbps` | Resting Blood Pressure | mm Hg upon admission (Normal: < 120) |
| `chol` | Serum Cholesterol | mg/dl (Desirable: < 200, Elevated: >= 240) |
| `fbs` | Fasting Blood Sugar | `True` (> 120 mg/dl) or `False` (<= 120 mg/dl) |
| `restecg` | Resting ECG | `normal`, `st-t abnormality`, `lv hypertrophy` |
| `thalch` | Max Heart Rate Achieved | bpm during exercise stress test (60 - 220) |
| `exang` | Exercise-Induced Angina | `True` (Yes) or `False` (No) |
| `oldpeak` | ST Depression | mm induced by exercise relative to rest |
| `slope` | ST Segment Slope | `upsloping`, `flat`, `downsloping` |
| `ca` | Major Vessels (Fluoroscopy) | Number of colored vessels (0 to 3) |
| `thal` | Thallium Scintigraphy | `normal`, `fixed defect`, `reversable defect` |

---

## 🏃 Running the Application

### 1. Ensure Dependencies are Installed
```bash
pip install streamlit pandas scikit-learn joblib xgboost
```

### 2. Launch the Streamlit Web Application
```bash
streamlit run app.py
```

### 3. Open in Browser
Visit **[http://localhost:8501](http://localhost:8501)** in your web browser.

---

## ⚠️ Medical Disclaimer
This application is developed strictly for research, educational, and clinical demonstration purposes. It should not be utilized as a substitute for professional medical advice, clinical diagnosis, or patient management by qualified healthcare providers.
