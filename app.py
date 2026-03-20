import streamlit as st
import pandas as pd
import joblib

# 1. Load the "Brain" we just saved
model = joblib.load('heart_model.pkl')

# 2. Set up the Website Look
st.title("🏥 Heart Disease Risk Predictor")
st.subheader("Real-time AI Diagnosis Tool")
st.write("Fill in the patient details below to get an instant risk assessment.")

# 3. Create the Input Fields (Matching your model's features)
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=1, max_value=120, value=50)
    sex = st.selectbox("Sex (1=Male, 0=Female)", [1, 0])
    cp = st.selectbox("Chest Pain Type (0-3)", [0, 1, 2, 3])
    trestbps = st.number_input("Resting Blood Pressure", value=120)
    chol = st.number_input("Cholesterol", value=200)

with col2:
    fbs = st.selectbox("Fasting Blood Sugar > 120 (1=True, 0=False)", [0, 1])
    restecg = st.selectbox("Resting ECG results (0-2)", [0, 1, 2])
    thalach = st.number_input("Max Heart Rate Achieved", value=150)
    exang = st.selectbox("Exercise Induced Angina (1=Yes, 0=No)", [0, 1])
    oldpeak = st.number_input("ST Depression (Oldpeak)", value=1.0)

# 4. The "Predict" Button
if st.button("Get Diagnosis"):
    # Arrange inputs into a format the model understands
    input_data = pd.DataFrame([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak]], 
                              columns=['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak'])
    
    prediction = model.predict(input_data)
    
    if prediction[0] == 1:
        st.error("⚠️ Prediction: HIGH RISK of Heart Disease")
    else:
        st.success("✅ Prediction: LOW RISK of Heart Disease")

st.info("Note: This is an AI Ethics demo and should not replace professional medical advice.")
