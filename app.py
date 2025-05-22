import streamlit as st
import pandas as pd
import joblib

# Load model and scaler
model = joblib.load("logistic_regression_model.pkl")  # Your best model
scaler = joblib.load("scaler.pkl")  # Ensure this is saved during training

st.set_page_config(layout="centered")
st.title("Breast Cancer Prediction App")

# Input method selection
input_method = st.radio("Select input method", ["Manual Input", "Upload CSV"])

# ========== Manual Input ==========
if input_method == "Manual Input":
    st.subheader("Enter Feature Values Manually")

    inputs = []
    feature_names = [
        "radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean",
        "compactness_mean", "concavity_mean", "concave points_mean", "symmetry_mean", "fractal_dimension_mean",
        "radius_se", "texture_se", "perimeter_se", "area_se", "smoothness_se",
        "compactness_se", "concavity_se", "concave points_se", "symmetry_se", "fractal_dimension_se",
        "radius_worst", "texture_worst", "perimeter_worst", "area_worst", "smoothness_worst",
        "compactness_worst", "concavity_worst", "concave points_worst", "symmetry_worst", "fractal_dimension_worst"
    ]

    for feature in feature_names:
        value = st.number_input(feature, value=0.0)
        inputs.append(value)

    if st.button("Predict"):
        X = scaler.transform([inputs])
        prediction = model.predict(X)
        result = "Benign" if prediction[0] == 1 else "Malignant"
        st.success(f"Prediction: {result}")

# ========== CSV Upload ==========
elif input_method == "Upload CSV":
    st.subheader("Upload CSV File for Prediction")

    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.write("CSV Preview:")
            st.dataframe(df.head())

            X = scaler.transform(df)
            predictions = model.predict(X)
            df['Prediction'] = ['Benign' if p == 1 else 'Malignant' for p in predictions]

            st.write("Prediction Results:")
            st.dataframe(df)

            csv_download = df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Predictions as CSV", csv_download, "predictions.csv", "text/csv")

        except Exception as e:
            st.error(f"Error: {e}")
