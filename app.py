import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("marketing_model_rf.pkl")
le = joblib.load("label_encoder.save")

st.title("Marketing Response Predictor (Random Forest)")
st.write("Enter customer details:")

age = st.number_input("Age", 18, 100, 30)
income = st.number_input("Annual Income (INR)", 10000, 10000000, 500000)
credit_score = st.number_input("Credit Score", 300, 850, 700)
product_price = st.number_input("Product Price (INR)", 1000, 1000000, 50000)

if st.button("Predict"):
    # Optional engineered features
    affordability = income / product_price
    credit_age_ratio = credit_score / age
    
    input_data = pd.DataFrame({
        'Age': [age],
        'Annual_Income_INR': [income],
        'Credit_Score': [credit_score],
        'Product_Price_INR': [product_price],
        'Affordability': [affordability],
        'Credit_Age_Ratio': [credit_age_ratio]
    })
    
    pred = model.predict(input_data)
    label = le.inverse_transform(pred)
    
    # Optional probability
    prob = model.predict_proba(input_data)[0]
    st.success(f"Predicted Response: {label[0]} (Yes: {prob[1]*100:.1f}%, No: {prob[0]*100:.1f}%)")
