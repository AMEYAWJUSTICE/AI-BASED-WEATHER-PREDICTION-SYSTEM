
import streamlit as st
import pandas as pd
import joblib
import numpy as np # Ensure numpy is imported for potential internal use by joblib or model

# Load the tuned Random Forest model
# Make sure 'tuned_random_forest_model.pkl' is in the same directory as this app.py file
try:
    model = joblib.load('tuned_random_forest_model.pkl')
    st.success('Tuned Random Forest Model loaded successfully!')
except FileNotFoundError:
    st.error('Error: Model file (tuned_random_forest_model.pkl) not found. Please ensure it is in the same directory.')
    st.stop()

st.title('Weather Temperature Prediction App')
st.write('Predict the temperature based on various weather conditions using a Tuned Random Forest Model.')

# Input fields for features
st.sidebar.header('Input Weather Parameters')

# Station
station_options = ['Station1', 'Station2', 'Station3']
station = st.sidebar.selectbox('Station', station_options)

# Precipitation
precipitation = st.sidebar.slider('Precipitation (mm)', min_value=0.0, max_value=20.0, value=1.0, step=0.1)

# Humidity
humidity = st.sidebar.slider('Humidity (%)', min_value=0.0, max_value=100.0, value=70.0, step=0.1)

# WindSpeed
windspeed = st.sidebar.slider('WindSpeed (m/s)', min_value=0.0, max_value=30.0, value=10.0, step=0.1)

# WeatherCondition
weather_condition_options = ['Cloudy', 'Partly Cloudy', 'Rain', 'Snow', 'Sunny']
weather_condition = st.sidebar.selectbox('Weather Condition', weather_condition_options)

# Month
month = st.sidebar.slider('Month', min_value=1, max_value=12, value=7, step=1)

# Day
day = st.sidebar.slider('Day', min_value=1, max_value=31, value=15, step=1)

# Create a DataFrame from inputs
input_data = pd.DataFrame({
    'Station': [station],
    'Precipitation': [precipitation],
    'Humidity': [humidity],
    'WindSpeed': [windspeed],
    'WeatherCondition': [weather_condition],
    'Month': [month],
    'Day': [day]
})

st.subheader('Input Parameters')
st.write(input_data)

# Make prediction
if st.button('Predict Temperature'):
    try:
        prediction = model.predict(input_data)[0]
        st.success(f'Predicted Temperature: {prediction:.2f} °C')
    except Exception as e:
        st.error(f'An error occurred during prediction: {e}')
        st.warning('Please ensure all input parameters are valid and the model is correctly loaded.')

st.markdown("""
**How to run this app:**
1.  Save the code from this cell to a file named `app.py` in your local environment.
2.  Make sure `tuned_random_forest_model.pkl` is in the same directory as `app.py`.
3.  Open your terminal or command prompt.
4.  Navigate to the directory where you saved `app.py`.
5.  Run the command: `streamlit run app.py`
""")
