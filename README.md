Heart Disease Risk Prediction & Ethics Audit
📊 Project Overview
This project uses Machine Learning to predict the risk of heart disease based on clinical and demographic data. Beyond just achieving high accuracy, this project includes an AI Ethics Check to interpret why the model makes specific decisions.

🚀 Key Results
Model Accuracy: 85%

Primary Driver: Chest Pain Type (cp)

Ethics Finding: The model shows high sensitivity to demographic factors like sex, suggesting a need for clinical oversight to ensure fairness.
🛠️ Step-by-Step Methodology
Data Preprocessing: * Cleaned the dataset and handled missing values.

Encoded categorical features (like sex and cp) for the model.

Model Training: * Trained a classification model to distinguish between "High Risk" and "Low Risk."

Achieved an 85% success rate on unseen test data.

Ethics & Interpretability (The "Why"):

Generated a Feature Importance Plot (SHAP/Bar Chart) to audit the model's logic.

Finding: Medical markers like cp (Chest Pain) and ca (Number of Vessels) are strong predictors, but demographic data (sex) also plays a major role, which requires careful ethical consideration in a healthcare setting.
