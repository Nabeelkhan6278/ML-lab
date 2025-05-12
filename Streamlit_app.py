import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime

# Load model and scaler
model = joblib.load('random_forest_model.pkl')
scaler = joblib.load('scaler.pkl')

# Prediction logging
log_file = 'predictions_log.csv'

# Initialize log file if it doesn't exist
if not os.path.exists(log_file):
    pd.DataFrame(columns=['Timestamp', 'Inputs', 'Prediction']).to_csv(log_file, index=False)

# App Title
st.title("🚴‍♂️ Seoul Bike Rental Prediction App")

# Sidebar Inputs
st.sidebar.header("Input Features")

def user_input_features():
    # Sidebar Inputs for numerical features
    hour = st.sidebar.slider('Hour', 0, 23, 12)
    temperature = st.sidebar.number_input('Temperature (°C)', min_value=-20.0, max_value=40.0, value=15.0)
    humidity = st.sidebar.slider('Humidity (%)', 0, 100, 50)
    wind_speed = st.sidebar.number_input('Wind speed (m/s)', 0.0, 10.0, 1.5)
    visibility = st.sidebar.number_input('Visibility (10m)', 0, 2000, 1000)
    solar_radiation = st.sidebar.number_input('Solar Radiation (MJ/m2)', 0.0, 3.0, 0.5)
    rainfall = st.sidebar.number_input('Rainfall (mm)', 0.0, 50.0, 0.0)
    snowfall = st.sidebar.number_input('Snowfall (cm)', 0.0, 10.0, 0.0)

    # Sidebar Inputs for categorical features
    year = st.sidebar.selectbox('Year', [2017, 2018])
    month = st.sidebar.selectbox('Month', list(range(1, 13)))
    day = st.sidebar.selectbox('Day', list(range(1, 32)))

    holiday = st.sidebar.selectbox('Holiday', ['No Holiday', 'Holiday'])
    functioning_day = st.sidebar.selectbox('Functioning Day', ['Yes', 'No'])
    season = st.sidebar.selectbox('Season', ['Winter', 'Spring', 'Summer', 'Autumn'])

    # Encoding categorical inputs
    holiday_encoded = 1 if holiday == 'Holiday' else 0
    functioning_day_encoded = 1 if functioning_day == 'Yes' else 0
    
    # One-hot encode the season (ensure 2 binary features for seasons)
    season_encoded = {
        'Spring': [1, 0],  # Spring (1, 0)
        'Summer': [0, 1],  # Summer (0, 1)
        'Autumn': [0, 0],  # Autumn (0, 0)
        'Winter': [0, 0]   # Winter (0, 0) - same as Autumn (if needed)
    }[season]

    # Compile all features into a single numpy array (15 features in total)
    features = np.array([hour, temperature, humidity, wind_speed, visibility, solar_radiation, rainfall,
                         snowfall, year, month, day, holiday_encoded, functioning_day_encoded] + season_encoded)

    return features.reshape(1, -1)


# Get user input
input_data = user_input_features()

# Predict button
if st.button('Predict Bike Rentals'):
    try:
        # Feature Scaling
        input_scaled = scaler.transform(input_data)

        # Prediction
        prediction = model.predict(input_scaled)[0]
        st.success(f"✅ Predicted Bike Rentals: {int(prediction)}")

        # Log prediction (convert inputs into list for storing)
        log_entry = {
            'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'Inputs': input_data.flatten().tolist(),
            'Prediction': int(prediction)
        }

        # Read existing log, append new entry, and save
        log_df = pd.read_csv(log_file)
        log_df = pd.concat([log_df, pd.DataFrame([log_entry])], ignore_index=True)
        log_df.to_csv(log_file, index=False)

    except Exception as e:
        st.error(f"Error during prediction or logging: {str(e)}")

# Show prediction logs
if st.checkbox('Show Prediction Logs'):
    try:
        log_df = pd.read_csv(log_file)
        st.dataframe(log_df.tail(10))
    except Exception as e:
        st.error(f"Error loading log file: {str(e)}")


