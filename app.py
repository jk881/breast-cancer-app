import streamlit as st
import joblib

# Load the trained model and scaler
model = joblib.load("Logistic_Regression_model.pkl")


st.title("Breast Cancer Prediction App")

# Feature names
feature_names = [
    'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean',
    'compactness_mean', 'concavity_mean', 'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
    'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
    'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se', 'fractal_dimension_se',
    'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', 'smoothness_worst',
    'compactness_worst', 'concavity_worst', 'concave points_worst', 'symmetry_worst', 'fractal_dimension_worst'
]

# Manual Input
st.subheader("Enter Feature Values Manually")
features = []
for name in feature_names:
    value = st.number_input(name, value=0.0)
    features.append(value)

if st.button("Predict"):
    scaled_features = scaler.transform([features])
    prediction = model.predict(scaled_features)
    st.success(f"Prediction: {'Malignant' if prediction[0] == 1 else 'Benign'}")
