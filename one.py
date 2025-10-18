import streamlit as st
import pickle
import pandas as pd
import numpy as np
st.set_page_config(page_title="Laptop Price Converter", layout="wide")


pipe = pickle.load(open('pipe.pkl','rb'))
data = pickle.load(open('data.pkl','rb'))

st.title("💻 Laptop Price Predictor")

#User Inputs 
company = st.selectbox('Brand', data['Company'].unique())
type_name = st.selectbox('Type', data['TypeName'].unique())
ram = st.selectbox('RAM (GB)', [2,4,6,8,12,16,24,32,64])
weight = st.number_input('Weight of the Laptop (kg)', 0.5, 5.0, 1.5, 0.1)
touchscreen = st.selectbox('Touchscreen', ['No', 'Yes'])
ips = st.selectbox('IPS Display', ['No', 'Yes'])
screen_size = st.slider('Screen Size (inches)', 10.0, 18.0, 13.0)
resolution = st.selectbox(
    'Screen Resolution',
    ['1920x1080','1366x768','1600x900','3840x2160','3200x1800','2880x1800','2560x1600','2560x1440','2304x1440']
)
cpu = st.selectbox('CPU', ['Intel Core i5', 'Intel Core i7', 'AMD Processor', 'Intel Core i3', 'Other Intel Processor'])
hdd = st.selectbox('HDD (GB)', [0,128,256,512,1024,2048])
ssd = st.selectbox('SSD (GB)', [0,8,128,256,512,1024])
gpu = st.selectbox('GPU', data['Gpu Brand'].unique())
os = st.selectbox('Operating System', data['os'].unique())

# Prediction 
if st.button('Predict Price'):
    # Convert Yes/No to 1/0
    touchscreen_val = 1 if touchscreen=='Yes' else 0
    ips_val = 1 if ips=='Yes' else 0

    # Calculate PPI
    X_res = int(resolution.split('x')[0])
    Y_res = int(resolution.split('x')[1])
    ppi = ((X_res**2) + (Y_res**2))**0.5 / screen_size

    # Create initial query DataFrame
    query = pd.DataFrame({
        'Company': [company],
        'TypeName': [type_name],
        'Ram': [ram],
        'Weight': [weight],
        'TouchScreen': [touchscreen_val],
        'IPS': [ips_val],
        'ppi': [ppi],
        'Cpu brand': [cpu],  
        'HDD': [hdd],
        'SSD': [ssd],
        'Gpu Brand': [gpu],
        'os': [os]
    })

    expected_cols = pipe.feature_names_in_
    rename_map = {}

    for col in query.columns:
        for exp in expected_cols:
            if col.strip().lower() == exp.strip().lower():
                rename_map[col] = exp

    query.rename(columns=rename_map, inplace=True)

    try:
        pred = pipe.predict(query)[0]
        price = int(np.exp(pred))
        st.success(f"💰 Predicted Laptop Price: ₹{price:,}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
