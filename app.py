import pickle

import pandas as pd
import streamlit as st
import joblib

st.set_page_config(page_title="Diabetes Risk Screening", page_icon="🩺", layout="centered")

@st.cache_resource
def load_bundle(path: str = "model.pkl"):
    # Try joblib first because the notebook saves model.pkl with joblib.dump.
    try:
        return joblib.load(path)
    except Exception:
        with open(path, "rb") as f:
            return pickle.load(f)

try:
    bundle = load_bundle()
except Exception as exc:
    st.error("Failed to load model artifact. Please check that model.pkl and dependencies are correct.")
    st.caption(f"Error type: {type(exc).__name__}")
    st.code(str(exc))
    st.stop()

model = bundle["model"]
features = bundle["features"]
model_name = bundle.get("model_name", "Trained Model")

st.title("Diabetes Risk Screening")
st.write("Enter patient values below to estimate diabetes risk.")
st.caption("Educational screening support only. This is not a medical diagnosis.")

col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=2)
    glucose = st.number_input("Glucose", min_value=0, max_value=250, value=120)
    blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=150, value=70)
    skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20)

with col2:
    insulin = st.number_input("Insulin", min_value=0, max_value=900, value=79)
    bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=32.0, step=0.1)
    dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.47, step=0.01)
    age = st.number_input("Age", min_value=10, max_value=100, value=33)

if st.button("Predict Risk"):
    row = {
        "Pregnancies": pregnancies,
        "Glucose": glucose,
        "BloodPressure": blood_pressure,
        "SkinThickness": skin_thickness,
        "Insulin": insulin,
        "BMI": bmi,
        "DiabetesPedigreeFunction": dpf,
        "Age": age,
    }
    x = pd.DataFrame([row])[features]
    pred = int(model.predict(x)[0])
    prob = float(model.predict_proba(x)[0][1])

    label = "At Risk" if pred == 1 else "Not At Risk"
    st.subheader(f"Prediction: {label}")
    st.write(f"Risk probability: {prob:.2%}")
    st.caption(f"Model in use: {model_name}")
