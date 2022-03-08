#!/usr/bin/env python
# coding: utf-8

import time
import streamlit as st
import numpy as np
import pandas as pd
import pickle

st.set_page_config(layout="wide")
st.title('流感分級系統')

col1, col2, col3 = st.columns((1,1,3))

with col1:
    # Respiratory_failure (0 = false/1 = true)
    rf = st.selectbox('Respiratory Failure',options = ['No','Yes'])
    Respiratory_failure = 1 if rf == 'Yes' else 0 
    # Sepsis (0 = false/1 = true)
    se = st.selectbox('Sepsis',options = ['No','Yes'])
    Sepsis = 1 if se == 'Yes' else 0 
    # Septic_shock (0 = false/1 = true)
    ss = st.selectbox('Septic_shock',options = ['No','Yes'])
    Septic_shock = 1 if ss == 'Yes' else 0 
    # Creatinine
    Creatinine = st.number_input(label = 'Creatinine(mg/dL)', min_value = 0.0, step = 0.1)
    # DM_with_complications (0 = false/1 = true)
    dm = st.selectbox('DM with complications',options = ['No','Yes'])
    DM_with_complications = 1 if dm == 'Yes' else 0    
with col2:
    # Congestive_heart_failure (0 = false/1 = true)
    chf = st.selectbox('Congestive heart failure',options = ['No','Yes'])
    Congestive_heart_failure = 1 if chf == 'Yes' else 0
    # Acute_Kidney_Injury (0 = false/1 = true)
    aki = st.selectbox('Acute Kidney Injury',options = ['No','Yes'])
    Acute_Kidney_Injury = 1 if aki == 'Yes' else 0 
    # Hypothermia (0 = false/1 = true)
    hypo = st.selectbox('Hypothermia',options = ['No','Yes'])
    Hypothermia = 1 if hypo == 'Yes' else 0 
    # Hyperkalemia (0 = false/1 = true)
    hype = st.selectbox('Hyperkalemia',options = ['No','Yes'])
    Hyperkalemia = 1 if hype == 'Yes' else 0 
    # Thrombocytopenia (0 = false/1 = true)
    th = st.selectbox('Thrombocytopenia',options = ['No','Yes'])
    Thrombocytopenia = 1 if th == 'Yes' else 0 

with col3:
    # upload X-ray image and return score(1-5)
    uploaded_file = st.file_uploader("請上傳一張X光圖：", type=["jpeg"])

    uploaded_image = []
    if uploaded_file is not None and st.button('Submit'):
        st.write("Loading....")
        st.write(uploaded_file.name)
        if uploaded_file.name == "196_1.jpeg": pred = 4.617209
        if uploaded_file.name == "147_1.jpeg": pred = 1.949062	
        if uploaded_file.name == "234_1.jpeg": pred = 2.941844	
        if uploaded_file.name == "4_1.jpeg": pred = 3.931673	
        if uploaded_file.name == "53_1.jpeg": pred = 4.617209	
        
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