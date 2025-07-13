import pandas as pd
import streamlit as st
import joblib
import numpy as np

# App title
st.title("Laptop Price Predictor")

# Load the model and dataframe with error handling
try:
    pipe = joblib.load('pipe.pkl')
except FileNotFoundError:
    st.error("The model file 'pipe.pkl' is missing. Please ensure it is in the project directory.")
    st.stop()
except Exception as e:
    st.error(f"An error occurred while loading the model: {e}")
    st.stop()

try:
    df = joblib.load('df.pkl')
except FileNotFoundError:
    st.error("The data file 'df.pkl' is missing. Please ensure it is in the project directory.")
    st.stop()
except Exception as e:
    st.error(f"An error occurred while loading the data: {e}")
    st.stop()

# Brand selection
company = st.selectbox('Brand', df['Company'].unique())

# Type of Laptop
laptop_type = st.selectbox('Type', df['TypeName'].unique())

# RAM selection
ram = st.selectbox('Ram (in GB)', [2, 4, 6, 8, 12, 16, 24, 32, 64])

# Weight input
weight = st.number_input('Weight of the laptop')

# Touchscreen option
touchscreen = st.selectbox('TouchScreen', ['No', 'Yes'])

# IPS option
ips = st.selectbox('IPS', ['No', 'Yes'])

# Screen size input
screen_size = st.number_input('Screen Size')

# Screen resolution selection
resolution = st.selectbox('Screen Resolution', [
    '1920x1080', '1920x1200', '1366x768', '1600x900', '3840x2160',
    '3200x1800', '2880x1800', '2560x1600', '2560x1440', '2304x1440'
])

# CPU selection
cpu = st.selectbox('CPU', df['Cpu Brand'].unique())

# HDD selection
hdd = st.selectbox('HDD (in GB)', [0, 128, 256, 512, 1024, 2048])

# SSD selection
ssd = st.selectbox('SSD (in GB)', [0, 8, 128, 256, 512, 1024])

# GPU selection
gpu = st.selectbox('GPU', df['Gpu Brand'].unique())

# OS selection
os = st.selectbox('OS', df['os'].unique())

# When the predict button is clicked
if st.button('Predict Price'):

    # Convert TouchScreen and IPS to 1/0
    touchscreen = 1 if touchscreen == 'Yes' else 0
    ips = 1 if ips == 'Yes' else 0

    # Calculate PPI (Pixels Per Inch)
    X_res = int(resolution.split('x')[0])
    Y_res = int(resolution.split('x')[1])
    ppi = ((X_res**2) + (Y_res**2))**0.5 / screen_size

    # Create a DataFrame with appropriate column names
    query_df = pd.DataFrame({
        'Company': [company],
        'TypeName': [laptop_type],
        'Ram': [ram],  # Match column name exactly as 'Ram'
        'Weight': [weight],
        'TouchScreen': [touchscreen],  # Match column name as 'TouchScreen'
        'IPS': [ips],
        'PPI': [ppi],
        'Cpu Brand': [cpu],
        'HDD': [hdd],
        'SSD': [ssd],
        'Gpu Brand': [gpu],
        'os': [os]
    })

    # Predict the price using the pipeline
    try:
        predicted_price = round(int(np.exp(pipe.predict(query_df)[0])))  # assuming log transformation
        st.title(f"Predicted Price : INR {predicted_price:.2f}")
    except Exception as e:
        st.error(f"Error in prediction: {e}")



