# import streamlit as st
# import pickle
# import numpy as np
# # import the model
# pipe = pickle.load(open('pipe.pkl','rb'))
# data = pickle.load(open('data.pkl','rb'))

# st.title("Laptop Predictor")

# # brand
# company = st.selectbox('Brand',data['Company'].unique())
# # type of laptop
# type = st.selectbox('Type',data['TypeName'].unique())
# # Ram
# ram = st.selectbox('RAM(in GB)',[2,4,6,8,12,16,24,32,64])
# # weight
# weight = st.number_input('Weight of the Laptop')
# # Touchscreen
# touchscreen = st.selectbox('Touchscreen',['No','Yes'])
# # IPS
# ips = st.selectbox('IPS',['No','Yes'])
# # screen size
# screen_size = st.slider('Scrensize in inches', 10.0, 18.0, 13.0)
# # resolution
# resolution = st.selectbox('Screen Resolution',['1920x1080','1366x768','1600x900','3840x2160','3200x1800','2880x1800','2560x1600','2560x1440','2304x1440'])
# #cpU
# cpu = st.selectbox('CPU',['Intel Core i5', 'Intel Core i7', 'AMD Processor', 'Intel Core i3','Other Intel Processor'])
# hdd = st.selectbox('HDD(in GB)',[0,128,256,512,1024,2048])
# ssd = st.selectbox('SSD(in GB)',[0,8,128,256,512,1024])
# gpu = st.selectbox('GPU',data['Gpu Brand'].unique())
# os = st.selectbox('OS',data['os'].unique())
# if st.button('Predict Price'):
#     #query
#     ppi = None
#     if touchscreen == 'Yes':
#         touchscreen = 1
#     else:
#         touchscreen = 0
#     if ips == 'Yes':
#         ips = 1
#     else :
#         ips = 0

#     X_res = int(resolution.split('x')[0])
#     Y_res = int(resolution.split('x')[1])
#     ppi = ((X_res**2)+(Y_res**2))**0.5/screen_size    


#     query = np.array([company,type,ram,weight,touchscreen,ips,ppi,cpu,hdd,ssd,gpu,os])
#     query = query.reshape(1,12)
#     st.title(int(np.exp(pipe.predict(query)[0])))

# import streamlit as st
# import pickle
# import pandas as pd
# import numpy as np

# # Load model and data
# pipe = pickle.load(open('pipe.pkl','rb'))
# data = pickle.load(open('data.pkl','rb'))

# st.title("Laptop Price Predictor")

# # User Inputs
# company = st.selectbox('Brand', data['Company'].unique())
# type_name = st.selectbox('Type', data['TypeName'].unique())
# ram = st.selectbox('RAM (GB)', [2,4,6,8,12,16,24,32,64])
# weight = st.number_input('Weight of the Laptop (kg)')
# touchscreen = st.selectbox('Touchscreen', ['No', 'Yes'])
# ips = st.selectbox('IPS', ['No', 'Yes'])
# screen_size = st.slider('Screen Size (inches)', 10.0, 18.0, 13.0)
# resolution = st.selectbox('Screen Resolution', ['1920x1080','1366x768','1600x900','3840x2160','3200x1800','2880x1800','2560x1600','2560x1440','2304x1440'])
# cpu = st.selectbox('CPU', ['Intel Core i5', 'Intel Core i7', 'AMD Processor', 'Intel Core i3', 'Other Intel Processor'])
# hdd = st.selectbox('HDD (GB)', [0,128,256,512,1024,2048])
# ssd = st.selectbox('SSD (GB)', [0,8,128,256,512,1024])
# gpu = st.selectbox('GPU', data['Gpu Brand'].unique())
# os = st.selectbox('OS', data['os'].unique())

# if st.button('Predict Price'):
#     # Convert Yes/No to 1/0
#     touchscreen_val = 1 if touchscreen=='Yes' else 0
#     ips_val = 1 if ips=='Yes' else 0

#     # Calculate PPI
#     X_res = int(resolution.split('x')[0])
#     Y_res = int(resolution.split('x')[1])
#     ppi = ((X_res**2)+(Y_res**2))**0.5 / screen_size

#     # Create query DataFrame (important!)
#     query = pd.DataFrame({
#         'Company':[company],
#         'TypeName':[type_name],
#         'Ram':[ram],
#         'Weight':[weight],
#         'Touchscreen':[touchscreen_val],
#         'Ips':[ips_val],
#         'PPI':[ppi],
#         'Cpu brand':[cpu],
#         'HDD':[hdd],
#         'SSD':[ssd],
#         'Gpu Brand':[gpu],
#         'OS':[os]
#     })

#     # Predict
#     pred = pipe.predict(query)[0]
#     st.title(f"Predicted Laptop Price: ₹{int(np.exp(pred))}")
# import streamlit as st
# import pickle
# import pandas as pd
# import numpy as np

# # Load trained pipeline and data
# pipe = pickle.load(open('pipe.pkl','rb'))
# data = pickle.load(open('data.pkl','rb'))

# # Strip any extra spaces in column names just in case
# data.columns = data.columns.str.strip()

# st.title("💻 Laptop Price Predictor")

# # --- User Inputs ---
# company = st.selectbox('Brand', data['Company'].unique())
# type_name = st.selectbox('Type', data['TypeName'].unique())
# ram = st.selectbox('RAM (GB)', [2,4,6,8,12,16,24,32,64])
# weight = st.number_input('Weight of the Laptop (kg)', 0.5, 5.0, 1.5, 0.1)
# touchscreen = st.selectbox('Touchscreen', ['No', 'Yes'])
# ips = st.selectbox('IPS Display', ['No', 'Yes'])
# screen_size = st.slider('Screen Size (inches)', 10.0, 18.0, 13.0)
# resolution = st.selectbox('Screen Resolution', 
#                           ['1920x1080','1366x768','1600x900','3840x2160','3200x1800','2880x1800','2560x1600','2560x1440','2304x1440'])
# cpu = st.selectbox('CPU', ['Intel Core i5', 'Intel Core i7', 'AMD Processor', 'Intel Core i3', 'Other Intel Processor'])
# hdd = st.selectbox('HDD (GB)', [0,128,256,512,1024,2048])
# ssd = st.selectbox('SSD (GB)', [0,8,128,256,512,1024])
# gpu = st.selectbox('GPU', data['Gpu Brand'].unique())
# os = st.selectbox('Operating System', data['os'].unique())

# # --- Prediction ---
# if st.button('Predict Price'):
    
#     # Convert Yes/No to 1/0
#     touchscreen_val = 1 if touchscreen=='Yes' else 0
#     ips_val = 1 if ips=='Yes' else 0

#     # Calculate PPI
#     X_res = int(resolution.split('x')[0])
#     Y_res = int(resolution.split('x')[1])
#     ppi = ((X_res**2) + (Y_res**2))**0.5 / screen_size

#     # Create query DataFrame with exact column names expected by pipeline
#     query = pd.DataFrame({
#         'Company': [company],
#         'TypeName': [type_name],
#         'Ram': [ram],
#         'Weight': [weight],
#         'TouchScreen': [touchscreen_val],
#         'IPS': [ips_val],
#         'ppi': [ppi],
#         'Cpu brand ': [cpu],  # note trailing space
#         'HDD': [hdd],
#         'SSD': [ssd],
#         'Gpu Brand': [gpu],
#         'os': [os]
#     })

#     # Strip spaces in query columns just in case
#     query.columns = query.columns.str.strip()

#     # Predict
#     try:
#         pred = pipe.predict(query)[0]
#         st.success(f"💰 Predicted Laptop Price: ₹{int(np.exp(pred))}")
#     except Exception as e:
#         st.error(f"Prediction failed: {e}")
import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Load trained pipeline and data
pipe = pickle.load(open('pipe.pkl','rb'))
data = pickle.load(open('data.pkl','rb'))

st.title("💻 Laptop Price Predictor")

# --- User Inputs ---
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

# --- Prediction ---
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
        'Cpu brand': [cpu],   # without space first
        'HDD': [hdd],
        'SSD': [ssd],
        'Gpu Brand': [gpu],
        'os': [os]
    })

    # ✅ Ensure column names exactly match what pipeline expects
    expected_cols = pipe.feature_names_in_
    rename_map = {}

    for col in query.columns:
        for exp in expected_cols:
            if col.strip().lower() == exp.strip().lower():
                rename_map[col] = exp

    query.rename(columns=rename_map, inplace=True)

    # Show debug info (optional)
    # st.write("Final columns:", query.columns.tolist())

    # Predict
    try:
        pred = pipe.predict(query)[0]
        price = int(np.exp(pred))
        st.success(f"💰 Predicted Laptop Price: ₹{price:,}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
