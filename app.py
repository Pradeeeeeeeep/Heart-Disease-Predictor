import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import json

# Set Streamlit page configuration
st.set_page_config(
    page_title="CardioCare AI - Heart Disease Predictor",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 50%, #06b6d4 100%);
        padding: 2.2rem 2rem;
        border-radius: 18px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 10px 25px -5px rgba(59, 130, 246, 0.3);
    }
    
    .main-header h1 {
        color: white;
        font-weight: 800;
        font-size: 2.2rem;
        margin-bottom: 0.4rem;
        display: flex;
        align-items: center;
        gap: 0.6rem;
    }
    
    .main-header p {
        color: #e0f2fe;
        font-size: 1.05rem;
        margin-bottom: 0;
    }

    .metric-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 1.2rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.08);
    }

    .risk-card-high {
        background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%);
        border: 2px solid #ef4444;
        border-radius: 16px;
        padding: 1.8rem;
        color: #991b1b;
        margin-top: 1rem;
        box-shadow: 0 10px 20px -5px rgba(239, 68, 68, 0.2);
    }

    .risk-card-low {
        background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
        border: 2px solid #22c55e;
        border-radius: 16px;
        padding: 1.8rem;
        color: #166534;
        margin-top: 1rem;
        box-shadow: 0 10px 20px -5px rgba(34, 197, 94, 0.2);
    }

    .stButton>button {
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
        color: white;
        font-weight: 600;
        border-radius: 10px;
        padding: 0.65rem 1.5rem;
        border: none;
        transition: all 0.2s ease;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.25);
    }

    .stButton>button:hover {
        background: linear-gradient(135deg, #1d4ed8 0%, #1e40af 100%);
        transform: translateY(-1px);
        box-shadow: 0 6px 16px rgba(37, 99, 235, 0.35);
    }

    .badge {
        display: inline-block;
        padding: 0.25rem 0.65rem;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to load model
@st.cache_resource
def load_heart_disease_model():
    model_path = os.path.join(os.path.dirname(__file__), 'Models', 'best_classification_model.pkl')
    metadata_path = os.path.join(os.path.dirname(__file__), 'Models', 'model_metadata.json')
    
    if not os.path.exists(model_path):
        st.error(f"Model file not found at: {model_path}")
        return None, None
        
    model = joblib.load(model_path)
    
    metadata = {}
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
            
    return model, metadata

model, metadata = load_heart_disease_model()

# Header Section
st.markdown("""
<div class="main-header">
    <h1>❤️ CardioCare AI Diagnostic Assistant</h1>
    <p>Predict heart disease risk using patient clinical metrics evaluated by Machine Learning trained on the UCI Heart Disease benchmark.</p>
</div>
""", unsafe_allow_html=True)

# Sidebar with Model Details & Presets
with st.sidebar:
    st.image("https://images.unsplash.com/photo-1628348068343-c6a848d2b6dd?w=600&auto=format&fit=crop&q=80")
    st.subheader("📋 Quick Test Profiles")
    st.caption("Select a sample profile to pre-fill the form instantly:")
    
    col_pre1, col_pre2 = st.columns(2)
    with col_pre1:
        load_healthy = st.button("🟢 Healthy", use_container_width=True)
    with col_pre2:
        load_risk = st.button("🔴 High Risk", use_container_width=True)
        
    load_moderate = st.button("🟡 Borderline Patient", use_container_width=True)

    st.markdown("---")
    st.subheader("🔬 Model Overview")
    if metadata:
        st.markdown(f"**Algorithm:** {metadata.get('model_type', 'Random Forest')}")
        st.markdown(f"**Benchmark Accuracy:** `{metadata.get('accuracy', 0.833):.1%}`")
        st.markdown(f"**Benchmark ROC-AUC:** `{metadata.get('roc_auc', 0.923):.3f}`")
        st.markdown("**Dataset:** UCI Heart Disease (919 patients)")
    else:
        st.markdown("**Algorithm:** Random Forest Classifier Pipeline")
        st.markdown("**Dataset:** UCI Heart Disease (Cleveland, Hungary, Switzerland, VA)")

    st.markdown("---")
    st.info("💡 **Disclaimer:** This tool is designed for educational & clinical demonstration purposes. Always consult licensed medical professionals for diagnoses.")

# Preset State Management
if 'preset' not in st.session_state:
    st.session_state.preset = 'default'

if load_healthy:
    st.session_state.preset = 'healthy'
elif load_risk:
    st.session_state.preset = 'risk'
elif load_moderate:
    st.session_state.preset = 'moderate'

# Preset Values
presets = {
    'healthy': {
        'age': 38, 'sex': 'Female', 'cp': 'typical angina', 'trestbps': 118, 'chol': 185,
        'fbs': 'False', 'restecg': 'normal', 'thalch': 175, 'exang': 'False',
        'oldpeak': 0.0, 'slope': 'upsloping', 'ca': 0, 'thal': 'normal'
    },
    'risk': {
        'age': 65, 'sex': 'Male', 'cp': 'asymptomatic', 'trestbps': 160, 'chol': 285,
        'fbs': 'True', 'restecg': 'lv hypertrophy', 'thalch': 112, 'exang': 'True',
        'oldpeak': 2.8, 'slope': 'flat', 'ca': 2, 'thal': 'reversable defect'
    },
    'moderate': {
        'age': 54, 'sex': 'Male', 'cp': 'non-anginal', 'trestbps': 135, 'chol': 238,
        'fbs': 'False', 'restecg': 'st-t abnormality', 'thalch': 142, 'exang': 'False',
        'oldpeak': 1.2, 'slope': 'flat', 'ca': 1, 'thal': 'normal'
    },
    'default': {
        'age': 52, 'sex': 'Male', 'cp': 'atypical angina', 'trestbps': 125, 'chol': 212,
        'fbs': 'False', 'restecg': 'normal', 'thalch': 155, 'exang': 'False',
        'oldpeak': 0.6, 'slope': 'upsloping', 'ca': 0, 'thal': 'normal'
    }
}

active_preset = presets.get(st.session_state.preset, presets['default'])

# Input Form organized in intuitive clinical sections
st.subheader("Patient Clinical Data")
st.caption("Fill in the patient's diagnostic and laboratory test details below.")

with st.form("prediction_form"):
    tab1, tab2, tab3 = st.tabs(["👤 Demographics & Basic Vitals", "🏃 Cardiac & Exercise Metrics", "🔬 ECG & Advanced Tests"])
    
    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            age = st.slider("Patient Age (Years)", min_value=18, max_value=100, value=active_preset['age'], help="Patient age in years")
            sex = st.radio("Biological Sex", options=["Male", "Female"], index=0 if active_preset['sex'] == "Male" else 1, horizontal=True)
            fbs = st.radio("Fasting Blood Sugar > 120 mg/dl (FBS)", options=["False", "True"], 
                           index=0 if active_preset['fbs'] == "False" else 1,
                           format_func=lambda x: "No (<= 120 mg/dl - Normal)" if x == "False" else "Yes (> 120 mg/dl - Elevated)",
                           horizontal=True)
        with col2:
            trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=80, max_value=240, value=active_preset['trestbps'], step=1,
                                       help="Systolic blood pressure measured upon admission (Normal: < 120 mm Hg)")
            chol = st.number_input("Serum Cholesterol (mg/dl)", min_value=100, max_value=600, value=active_preset['chol'], step=1,
                                   help="Serum cholesterol in mg/dl (Desirable: < 200 mg/dl)")
    
    with tab2:
        col3, col4 = st.columns(2)
        with col3:
            cp_options = ["typical angina", "atypical angina", "non-anginal", "asymptomatic"]
            cp_labels = {
                "typical angina": "Typical Angina (Substernal chest pressure triggered by exertion)",
                "atypical angina": "Atypical Angina (Discomfort not meeting classic angina criteria)",
                "non-anginal": "Non-Anginal Pain (Non-cardiac chest sensations)",
                "asymptomatic": "Asymptomatic (No overt chest pain sensation)"
            }
            cp = st.selectbox("Chest Pain Type (CP)", options=cp_options, 
                              index=cp_options.index(active_preset['cp']),
                              format_func=lambda x: cp_labels.get(x, x))
            
            thalch = st.slider("Maximum Heart Rate Achieved (bpm)", min_value=60, max_value=220, value=active_preset['thalch'],
                               help="Peak heart rate achieved during physical treadmill or stress test")
            
        with col4:
            exang = st.radio("Exercise-Induced Angina (ExAng)", options=["False", "True"],
                             index=0 if active_preset['exang'] == "False" else 1,
                             format_func=lambda x: "No (No chest discomfort during exercise)" if x == "False" else "Yes (Chest pain induced by exertion)",
                             horizontal=True)
            
            oldpeak = st.slider("ST Depression (Oldpeak in mm)", min_value=0.0, max_value=6.5, value=float(active_preset['oldpeak']), step=0.1,
                                help="ST depression induced by exercise relative to resting state on ECG")
            
    with tab3:
        col5, col6 = st.columns(2)
        with col5:
            restecg_options = ["normal", "st-t abnormality", "lv hypertrophy"]
            restecg_labels = {
                "normal": "Normal (No significant ECG anomalies)",
                "st-t abnormality": "ST-T Wave Abnormality (T wave inversions or ST shift)",
                "lv hypertrophy": "LV Hypertrophy (Probable or definite left ventricular enlargement)"
            }
            restecg = st.selectbox("Resting Electrocardiogram (RestECG)", options=restecg_options,
                                   index=restecg_options.index(active_preset['restecg']),
                                   format_func=lambda x: restecg_labels.get(x, x))
            
            slope_options = ["upsloping", "flat", "downsloping"]
            slope_labels = {
                "upsloping": "Upsloping (Favorable ST segment recovery during peak exercise)",
                "flat": "Flat (Horizontal ST slope, common ischemic indicator)",
                "downsloping": "Downsloping (Downward ST deviation, prominent risk factor)"
            }
            slope = st.selectbox("Peak Exercise ST Segment Slope", options=slope_options,
                                 index=slope_options.index(active_preset['slope']),
                                 format_func=lambda x: slope_labels.get(x, x))
            
        with col6:
            ca = st.selectbox("Major Vessels Colored by Fluoroscopy (0-3)", options=[0, 1, 2, 3],
                              index=active_preset['ca'],
                              help="Number of major coronary vessels with visualized blood flow via fluoroscopy")
            
            thal_options = ["normal", "fixed defect", "reversable defect"]
            thal_labels = {
                "normal": "Normal (Normal blood flow uptake)",
                "fixed defect": "Fixed Defect (Non-reversible cold spot / previous myocardial infarction)",
                "reversable defect": "Reversible Defect (Temporary ischemia observed under stress)"
            }
            thal = st.selectbox("Thalassemia Thallium Stress Result", options=thal_options,
                                index=thal_options.index(active_preset['thal']),
                                format_func=lambda x: thal_labels.get(x, x))

    submit_button = st.form_submit_button("🔍 Run Heart Disease Risk Assessment", use_container_width=True)

# Process Prediction
if submit_button:
    if model is None:
        st.error("Model is not loaded. Please verify 'Models/best_classification_model.pkl'.")
    else:
        # Build patient input DataFrame with exact schema
        input_data = pd.DataFrame([{
            'age': float(age),
            'sex': str(sex),
            'cp': str(cp),
            'trestbps': float(trestbps),
            'chol': float(chol),
            'fbs': str(fbs),
            'restecg': str(restecg),
            'thalch': float(thalch),
            'exang': str(exang),
            'oldpeak': float(oldpeak),
            'slope': str(slope),
            'ca': float(ca),
            'thal': str(thal)
        }])

        try:
            # Predict binary classification and probability
            prediction = model.predict(input_data)[0]
            probabilities = model.predict_proba(input_data)[0]
            risk_probability = probabilities[1] * 100
            
            # Predict clinical stage if available
            predicted_stage = 0
            if hasattr(model, 'stage_model_'):
                try:
                    predicted_stage = model.stage_model_.predict(input_data)[0]
                except Exception:
                    predicted_stage = 1 if prediction == 1 else 0

            st.markdown("---")
            st.subheader("Diagnostic Assessment Results")
            
            # Key Metric Summary Row
            m_col1, m_col2, m_col3, m_col4 = st.columns(4)
            with m_col1:
                st.metric("Risk Probability", f"{risk_probability:.1f}%")
            with m_col2:
                status_label = "Disease Detected" if prediction == 1 else "Normal / Low Risk"
                st.metric("Primary Diagnosis", status_label)
            with m_col3:
                stage_names = {
                    0: "Healthy (Stage 0)",
                    1: "Mild (Stage 1)",
                    2: "Moderate (Stage 2)",
                    3: "Severe (Stage 3)",
                    4: "Critical (Stage 4)"
                }
                st.metric("Estimated Stage", stage_names.get(predicted_stage, f"Stage {predicted_stage}"))
            with m_col4:
                confidence = max(probabilities) * 100
                st.metric("Model Confidence", f"{confidence:.1f}%")

            # Risk Card
            if prediction == 1:
                st.markdown(f"""
                <div class="risk-card-high">
                    <h3 style="margin-top:0;">⚠️ High Risk: Heart Disease Indication Detected</h3>
                    <p>The predictive model estimated a <strong>{risk_probability:.1f}% probability</strong> of coronary artery disease presence based on the entered clinical profile.</p>
                    <p style="margin-bottom:0;"><strong>Clinical Recommendation:</strong> Immediate medical follow-up with a cardiologist is strongly advised for confirmatory angiography, stress echocardiogram, and diagnostic workup.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="risk-card-low">
                    <h3 style="margin-top:0;">✅ Low Risk: No Heart Disease Detected</h3>
                    <p>The predictive model estimated a low risk of coronary artery disease (<strong>{100 - risk_probability:.1f}% probability of healthy cardiac profile</strong>).</p>
                    <p style="margin-bottom:0;"><strong>Clinical Recommendation:</strong> Maintain heart-healthy habits including routine aerobic exercise, balanced nutrition, and annual health screenings.</p>
                </div>
                """, unsafe_allow_html=True)

            # Risk Factors Analysis & Recommendations
            st.markdown("<br>", unsafe_allow_html=True)
            col_factors, col_recs = st.columns(2)
            
            with col_factors:
                st.markdown("#### 🩺 Clinical Factors Observed")
                factors = []
                if trestbps >= 140:
                    factors.append(f"**Stage 2 Hypertension:** Resting blood pressure is high ({trestbps} mm Hg).")
                elif trestbps >= 130:
                    factors.append(f"**Stage 1 Hypertension:** Elevated resting blood pressure ({trestbps} mm Hg).")
                else:
                    factors.append(f"**Normal Blood Pressure:** {trestbps} mm Hg is within healthy parameters.")
                    
                if chol >= 240:
                    factors.append(f"**Hypercholesterolemia:** Serum cholesterol is elevated ({chol} mg/dl).")
                elif chol >= 200:
                    factors.append(f"**Borderline Cholesterol:** {chol} mg/dl.")
                else:
                    factors.append(f"**Desirable Cholesterol:** {chol} mg/dl is in healthy range.")
                    
                if exang == "True":
                    factors.append("**Exercise-Induced Angina:** Symptomatic chest tightness during exertion detected.")
                    
                if oldpeak >= 1.5:
                    factors.append(f"**Significant ST Depression:** Oldpeak is {oldpeak:.1f} mm, indicating myocardial stress.")
                    
                if thal in ["reversable defect", "fixed defect"]:
                    factors.append(f"**Abnormal Thallium Perfusion:** {thal.capitalize()} identified.")
                    
                if ca > 0:
                    factors.append(f"**Coronary Calcification / Blockage:** {int(ca)} major vessel(s) affected.")

                for f in factors:
                    st.markdown(f"- {f}")

            with col_recs:
                st.markdown("#### 🥗 Personalized Action Plan")
                if prediction == 1 or risk_probability > 40:
                    st.markdown("""
                    1. **Comprehensive Cardiac Evaluation:** Schedule consultation with a board-certified cardiologist.
                    2. **Diagnostic Confirmation:** Consider 12-lead ECG, 2D Echocardiography, or Coronary CT Angiography.
                    3. **Medication Management:** Discuss blood pressure and lipid-lowering therapies (e.g., statins, ACE inhibitors).
                    4. **Dietary Changes:** Adopt the Mediterranean or DASH diet (low sodium, high omega-3 fatty acids).
                    5. **Monitored Physical Activity:** Engage in doctor-supervised cardiac rehabilitation exercises.
                    """)
                else:
                    st.markdown("""
                    1. **Routine Checkups:** Continue standard periodic cardiovascular wellness evaluations.
                    2. **Aerobic Exercise:** Aim for at least 150 minutes of moderate aerobic activity weekly.
                    3. **Heart-Smart Nutrition:** Emphasize whole grains, fiber, lean proteins, and antioxidant-rich foods.
                    4. **Stress & Sleep Hygiene:** Maintain 7-8 hours of quality sleep and manage stress.
                    5. **Biomarker Monitoring:** Check lipid profile and blood pressure at least once every 12 months.
                    """)

            # Detailed Probability Breakdown
            with st.expander("📊 View Detailed Probability & Stage Distribution"):
                st.write("**Binary Probability Distribution:**")
                prob_df = pd.DataFrame({
                    "Outcome": ["No Heart Disease (Healthy)", "Heart Disease Detected"],
                    "Probability": [probabilities[0], probabilities[1]]
                })
                st.bar_chart(prob_df.set_index("Outcome"))
                
                st.write("**Patient Input Feature Vector:**")
                st.dataframe(input_data, width="stretch")

        except Exception as e:
            st.error(f"Prediction error occurred: {str(e)}")
            st.exception(e)
