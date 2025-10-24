import streamlit as st
import pandas as pd
from tensorflow.keras.models import load_model
import joblib

# Load model, scaler, label encoder
model = load_model("binary_classifier_model.keras")
scaler = joblib.load("scaler.save")
le = joblib.load("label_encoder.save")

st.title("Binary Response Predictor")

age = st.number_input("Age", min_value=18, max_value=100, value=30)
income = st.number_input("Annual Income (INR)", value=2000000)
credit = st.number_input("Credit Score", value=650)
price = st.number_input("Product Price (INR)", value=50000)

if st.button("Predict"):
    df = pd.DataFrame({
        'Age':[age],
        'Annual_Income_INR':[income],
        'Credit_Score':[credit],
        'Product_Price_INR':[price]
    })
    df_scaled = scaler.transform(df)
    pred = model.predict(df_scaled)
    label = le.inverse_transform((pred>0.5).astype(int).flatten())
    st.write(f"Prediction: {label[0]}")
