import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load("random_forest_model.pkl")

st.title("Breast Cancer Prediction App")

# Input method selection
input_method = st.radio("Select input method", ["Manual Input", "Upload CSV"])

# ========== Manual Input ==========
if input_method == "Manual Input":
    st.subheader("Enter Feature Values Manually")

    radius_mean = st.number_input("radius_mean", value=0.0)
    texture_mean = st.number_input("texture_mean", value=0.0)
    perimeter_mean = st.number_input("perimeter_mean", value=0.0)
    area_mean = st.number_input("area_mean", value=0.0)
    smoothness_mean = st.number_input("smoothness_mean", value=0.0)
    compactness_mean = st.number_input("compactness_mean", value=0.0)
    concavity_mean = st.number_input("concavity_mean", value=0.0)
    concave_points_mean = st.number_input("concave points_mean", value=0.0)
    symmetry_mean = st.number_input("symmetry_mean", value=0.0)
    fractal_dimension_mean = st.number_input("fractal_dimension_mean", value=0.0)

    radius_se = st.number_input("radius_se", value=0.0)
    texture_se = st.number_input("texture_se", value=0.0)
    perimeter_se = st.number_input("perimeter_se", value=0.0)
    area_se = st.number_input("area_se", value=0.0)
    smoothness_se = st.number_input("smoothness_se", value=0.0)
    compactness_se = st.number_input("compactness_se", value=0.0)
    concavity_se = st.number_input("concavity_se", value=0.0)
    concave_points_se = st.number_input("concave points_se", value=0.0)
    symmetry_se = st.number_input("symmetry_se", value=0.0)
    fractal_dimension_se = st.number_input("fractal_dimension_se", value=0.0)

    radius_worst = st.number_input("radius_worst", value=0.0)
    texture_worst = st.number_input("texture_worst", value=0.0)
    perimeter_worst = st.number_input("perimeter_worst", value=0.0)
    area_worst = st.number_input("area_worst", value=0.0)
    smoothness_worst = st.number_input("smoothness_worst", value=0.0)
    compactness_worst = st.number_input("compactness_worst", value=0.0)
    concavity_worst = st.number_input("concavity_worst", value=0.0)
    concave_points_worst = st.number_input("concave points_worst", value=0.0)
    symmetry_worst = st.number_input("symmetry_worst", value=0.0)
    fractal_dimension_worst = st.number_input("fractal_dimension_worst", value=0.0)

    features = [[
        radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean,
        compactness_mean, concavity_mean, concave_points_mean, symmetry_mean, fractal_dimension_mean,
        radius_se, texture_se, perimeter_se, area_se, smoothness_se,
        compactness_se, concavity_se, concave_points_se, symmetry_se, fractal_dimension_se,
        radius_worst, texture_worst, perimeter_worst, area_worst, smoothness_worst,
        compactness_worst, concavity_worst, concave_points_worst, symmetry_worst, fractal_dimension_worst
    ]]

    if st.button("Predict"):
        prediction = model.predict(features)
        st.success(f"Prediction: {'Malignant' if prediction[0] == 1 else 'Benign'}")

# ========== CSV Upload ==========
elif input_method == "/content/drive/MyDrive/dataset/sample_input.csv":
    file = st.file_uploader("/content/drive/MyDrive/dataset/sample_input.csv", type=["csv"])
    if file is not None:
        data = pd.read_csv(file)
        prediction = model.predict(data)
        st.write("Predictions:", prediction)
