import streamlit as st
import pandas as pd
import joblib

# 1. Load the model
model = joblib.load('heart_model.pkl')

st.title("🏥 MediLens AI: Heart Risk Predictor")
st.write("Enter patient clinical data for a real-time diagnosis.")

# 2. Layout with 3 columns to fit all 13 features
col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Age", 1, 120, 50)
    sex = st.selectbox("Sex (1=M, 0=F)", [1, 0])
    cp = st.selectbox("Chest Pain (0-3)", [0, 1, 2, 3])
    trestbps = st.number_input("Resting BP", 80, 200, 120)
    chol = st.number_input("Cholesterol", 100, 600, 200)

with col2:
    fbs = st.selectbox("Fasting Sugar > 120", [0, 1])
    restecg = st.selectbox("Resting ECG (0-2)", [0, 1, 2])
    thalach = st.number_input("Max Heart Rate", 60, 220, 150)
    exang = st.selectbox("Exercise Angina", [0, 1])
    oldpeak = st.number_input("ST Depression", 0.0, 6.0, 1.0)

with col3:
    # THE MISSING 3 FEATURES:
    slope = st.selectbox("ST Slope (0-2)", [0, 1, 2])
    ca = st.selectbox("Major Vessels (0-4)", [0, 1, 2, 3, 4])
    thal = st.selectbox("Thalassemia (0-3)", [0, 1, 2, 3])

# 3. Prediction Logic
if st.button("Get Diagnosis"):
    # Must be in the exact order the model was trained on
    data = [[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]]
    
    # Matching columns to your training data
    cols = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
    input_df = pd.DataFrame(data, columns=cols)
    
    prediction = model.predict(input_df)
    
    if prediction[0] == 1:
        st.error("🚨 Prediction: HIGH RISK")
    else:
        st.success("✅ Prediction: LOW RISK")

st.info("XAI Note: This model identifies Chest Pain and Sex as high-impact features.")
