import streamlit as st
import pandas as pd
import numpy as np
import joblib

# -----------------------
# Load trained model and label encoder
# -----------------------
model = joblib.load("marketing_model_rf.pkl")
le = joblib.load("label_encoder.save")

st.title("Marketing Response Predictor (Random Forest)")
st.write("Enter customer details to predict if they will respond to the marketing campaign.")

# -----------------------
# Input fields
# -----------------------
age = st.number_input("Age", 18, 100, 30)
income = st.number_input("Annual Income (INR)", 10000, 10000000, 500000)
credit_score = st.number_input("Credit Score", 300, 850, 700)
product_price = st.number_input("Product Price (INR)", 1000, 1000000, 50000)

# -----------------------
# Predict button
# -----------------------
if st.button("Predict"):
    # Engineered features (must match training)
    affordability = income / product_price
    credit_age_ratio = credit_score / age
    log_income = np.log1p(income)
    log_product_price = np.log1p(product_price)
    
    # Prepare input data
    input_data = pd.DataFrame({
        'Age': [age],
        'Annual_Income_INR': [income],
        'Credit_Score': [credit_score],
        'Product_Price_INR': [product_price],
        'Affordability': [affordability],
        'Credit_Age_Ratio': [credit_age_ratio],
        'Log_Income': [log_income],
        'Log_ProductPrice': [log_product_price]
    })
    
    # Make prediction
    pred = model.predict(input_data)
    label = le.inverse_transform(pred)
    
    # Predict probabilities
    prob = model.predict_proba(input_data)[0]
    
    st.success(f"Predicted Response: {label[0]}")
    st.info(f"Yes Probability: {prob[1]*100:.1f}% | No Probability: {prob[0]*100:.1f}%")
