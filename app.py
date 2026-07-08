import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the trained model
# Make sure 'rf_pipeline.pkl' is in the same directory as this app.py or provide the full path
with open('rf_pipeline.pkl', 'rb') as f:
    model = pickle.load(f)

st.title('Car Selling Price Predictor')
st.write('Enter car details to predict its selling price.')

# Define the input options based on your preprocessing setup
fuel_options = ['Petrol', 'Diesel', 'CNG', 'LPG', 'Electric']
seller_type_options = ['Individual', 'Dealer', 'Trustmark Dealer']
transmission_options = ['Manual', 'Automatic']
owner_options = ['Test Drive Car', 'First Owner', 'Second Owner', 'Third Owner', 'Fourth & Above Owner']
# Assuming unique_brands was extracted and is available here, or hardcode if needed.
# For demonstration, I'll use a sample list or you can replace with your 'unique_brands'
brand_options = sorted(["Maruti", "Hyundai", "Datsun", "Honda", "Fiat", "Chevrolet", "Renault", "Volkswagen", "Tata", "Land", "Mahindra", "Ford", "Nissan", "Audi", "Skoda", "Toyota", "Jeep", "Volvo", "Mitsubishi", "Mercedes-Benz", "BMW", "Ambassador", "Ashok", "Force", "Isuzu", "MG", "Daewoo", "Kia"])

# Input widgets
brand = st.selectbox('Car Brand', brand_options)
km_driven = st.number_input('Kilometers Driven', min_value=0, value=50000)
year = st.number_input('Manufacturing Year (e.g., 2017)', min_value=1990, max_value=2026, value=2017)
fuel = st.selectbox('Fuel Type', fuel_options)
seller_type = st.selectbox('Seller Type', seller_type_options)
transmission = st.selectbox('Transmission Type', transmission_options)
owner = st.selectbox('Owner Type', owner_options)

# Calculate car_age
car_age = 2026 - year

# Create a DataFrame for prediction
# The column order and names must match the training data before preprocessing
input_data = pd.DataFrame([{
    'km_driven': km_driven,
    'fuel': fuel,
    'seller_type': seller_type,
    'transmission': transmission,
    'owner': owner,
    'brand': brand,
    'car_age': car_age
}])

if st.button('Predict Selling Price'):
    try:
        prediction = model.predict(input_data)
        st.success(f'Predicted Selling Price: ₹{prediction[0]:,.2f}')
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        st.warning("Please ensure all inputs are valid and the model file is correctly loaded.")


