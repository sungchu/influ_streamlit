#!/usr/bin/env python
# coding: utf-8

import time
import streamlit as st
import numpy as np
import pandas as pd
import keras
import cv2
import os
from PIL import Image
import pickle
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
st.title('流感分級系統')

col1, col2, col3 = st.columns((1,1,3))

with col1:
    # Respiratory_failure (0 = false/1 = true)
    rf = st.selectbox('Respiratory Failure',options = ['Yes','No'])
    Respiratory_failure = 1 if rf == 'Yes' else 0 
    # Sepsis (0 = false/1 = true)
    se = st.selectbox('Sepsis',options = ['Yes','No'])
    Sepsis = 1 if se == 'Yes' else 0 
    # Septic_shock (0 = false/1 = true)
    ss = st.selectbox('Septic_shock',options = ['Yes','No'])
    Septic_shock = 1 if ss == 'Yes' else 0 
    # Creatinine
    Creatinine = st.number_input(label = 'Creatinine(mg/dL)', min_value = 0.0, step = 0.1)
    # DM_with_complications (0 = false/1 = true)
    dm = st.selectbox('DM with complications',options = ['Yes','No'])
    DM_with_complications = 1 if dm == 'Yes' else 0    
with col2:
    # Congestive_heart_failure (0 = false/1 = true)
    chf = st.selectbox('Congestive heart failure',options = ['Yes','No'])
    Congestive_heart_failure = 1 if chf == 'Yes' else 0
    # Acute_Kidney_Injury (0 = false/1 = true)
    aki = st.selectbox('Acute Kidney Injury',options = ['Yes','No'])
    Acute_Kidney_Injury = 1 if aki == 'Yes' else 0 
    # Hypothermia (0 = false/1 = true)
    hypo = st.selectbox('Hypothermia',options = ['Yes','No'])
    Hypothermia = 1 if hypo == 'Yes' else 0 
    # Hyperkalemia (0 = false/1 = true)
    hype = st.selectbox('Hyperkalemia',options = ['Yes','No'])
    Hyperkalemia = 1 if hype == 'Yes' else 0 
    # Thrombocytopenia (0 = false/1 = true)
    th = st.selectbox('Thrombocytopenia',options = ['Yes','No'])
    Thrombocytopenia = 1 if th == 'Yes' else 0 

with col3:
    # upload X-ray image and return score(1-5)
    uploaded_file = st.file_uploader("請上傳一張X光圖：", type=["jpeg"])

    uploaded_image = []
    if uploaded_file is not None and st.button('Submit'):
        st.write("Loading....")
        image = Image.open(uploaded_file)
        st.image(image, caption='X-ray', width = 320)
        image = cv2.cvtColor(np.array(image), cv2.COLOR_GRAY2BGR)
        image = cv2.resize(image, (299, 299))
        uploaded_image.append(image/255.0)

        model = keras.models.load_model('xception_mse_mse0.9446_val_mse1.1390.h5')
        pred = model.predict(np.array(uploaded_image))
        if pred > 5.0:  pred = 5.0
        if pred < 1.0:  pred = 1.0  
        
                      
        dataset = pd.DataFrame([[float(pred), Respiratory_failure, Sepsis, Septic_shock, Creatinine, DM_with_complications,
        Congestive_heart_failure, Acute_Kidney_Injury, Hypothermia, Hyperkalemia, Thrombocytopenia]], 
        columns = ['pred', 'Respiratory_failure', 'Sepsis', 'Septic_shock', 'Creatinine', 'DM_with_complications',
        'Congestive_heart_failure', 'Acute_Kidney_Injury', 'Hypothermia', 'Hyperkalemia', 'Thrombocytopenia'])
        
        dataset["pred"].astype('float')
        
        with open('LightGBM_clinicaldata_xray_AUC0.8158.pickle', 'rb') as f:
            LightGBM = pickle.load(f)
        result = LightGBM.predict_proba(dataset)

        st.info("CXR嚴重度(1-5)評分為{:.2f}".format(float(pred)))
        #st.write(result)  # alive within 30 days(0 = false/1 = true)
        st.info("三十天內的存活率為{:.2f}%".format(result[0][1]*100))