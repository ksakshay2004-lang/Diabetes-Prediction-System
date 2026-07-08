"""
Diabetes Prediction System - Streamlit Web Application
Verveox Technologies AI & ML Internship - Week 3 Task
-------------------------------------------------------
Loads the trained Decision Tree model and predicts whether a patient
is Diabetic or Non-Diabetic based on user-entered medical parameters.
"""

import streamlit as st
import pandas as pd
import joblib

# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------
st.set_page_config(
    page_title="Diabetes Prediction System",
    page_icon="🩺",
    layout="centered"
)

# ---------------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------------
@st.cache_resource
def load_model():
    model = joblib.load("diabetes_model.pkl")
    feature_names = joblib.load("feature_names.pkl")
    return model, feature_names

model, feature_names = load_model()

# ---------------------------------------------------------
# HEADER
# ---------------------------------------------------------
st.title("🩺 Diabetes Prediction System")
st.write(
    "This application uses a **Decision Tree Classifier** trained on the "
    "**Pima Indians Diabetes Dataset** to predict whether a patient is "
    "likely to be diabetic based on key medical parameters."
)
st.divider()

# ---------------------------------------------------------
# USER INPUT FORM
# ---------------------------------------------------------
st.subheader("Enter Patient Medical Details")

col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1, step=1)
    glucose = st.number_input("Glucose Level (mg/dL)", min_value=0, max_value=300, value=120)
    blood_pressure = st.number_input("Blood Pressure (mm Hg)", min_value=0, max_value=200, value=70)
    skin_thickness = st.number_input("Skin Thickness (mm)", min_value=0, max_value=100, value=20)

with col2:
    insulin = st.number_input("Insulin Level (mu U/ml)", min_value=0, max_value=900, value=79)
    bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0, step=0.1)
    dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5, step=0.01)
    age = st.number_input("Age (years)", min_value=1, max_value=120, value=30)

st.divider()

# ---------------------------------------------------------
# PREDICTION
# ---------------------------------------------------------
if st.button("🔍 Predict", use_container_width=True):
    input_data = pd.DataFrame(
        [[pregnancies, glucose, blood_pressure, skin_thickness,
          insulin, bmi, dpf, age]],
        columns=feature_names
    )

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error(f"⚠️ The patient is likely **Diabetic** "
                  f"(confidence: {probability[1]*100:.1f}%)")
    else:
        st.success(f"✅ The patient is likely **Non-Diabetic** "
                    f"(confidence: {probability[0]*100:.1f}%)")

    with st.expander("View Prediction Probabilities"):
        st.write(f"Non-Diabetic probability: {probability[0]*100:.2f}%")
        st.write(f"Diabetic probability: {probability[1]*100:.2f}%")

st.divider()
st.caption("Built with Streamlit · Decision Tree Classifier · Scikit-learn")
