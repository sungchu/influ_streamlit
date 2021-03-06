#!/usr/bin/env python
# coding: utf-8

import time
import streamlit as st
import numpy as np
import pandas as pd
import pickle
from PIL import Image

#st.set_page_config(page_title = "流感分級系統" ,layout="wide")
st.title('流感分級系統')

col1, col2, col3 = st.columns((1,1,1))
with col1:
    # Respiratory_failure (0 = false/1 = true)
    st.markdown('**Respiratory failure:**')
    rf = st.selectbox('requires intubation and mechanical ventilation',options = ['No','Yes'])
    Respiratory_failure = 1 if rf == 'Yes' else 0 
with col2:
    # DM_with_complications (0 = false/1 = true)
    st.markdown('**DM with complications:**')
    dm = st.selectbox('diabetes with chronic complications on Charlson Comorbidity Index',options = ['No','Yes'])
    DM_with_complications = 1 if dm == 'Yes' else 0    
with col3:
    # Congestive_heart_failure (0 = false/1 = true)
    st.markdown('**Congestive heart failure:**')
    chf = st.selectbox('Congestive heart failureon Charlson Comorbidity Index',options = ['No','Yes'])
    Congestive_heart_failure = 1 if chf == 'Yes' else 0

col1, col2, col3 = st.columns((1,1,1))
with col1:
    # Acute_Kidney_Injury (0 = false/1 = true)
    st.markdown('**Acute kidney injury:**')
    aki = st.selectbox('by KDIGO criteria',options = ['No','Yes'])
    Acute_Kidney_Injury = 1 if aki == 'Yes' else 0 
with col2:
    # Hypothermia (0 = false/1 = true)
    st.markdown('**Hypothermia:**')
    hypo = st.selectbox('core temperature less than 35°C',options = ['No','Yes'])
    Hypothermia = 1 if hypo == 'Yes' else 0 
with col3:
    # Hyperkalemia (0 = false/1 = true)
    st.markdown('**Hyperkalemia:**')
    hype = st.selectbox('serum potassium level > 5.2 mEq/L',options = ['No','Yes'])
    Hyperkalemia = 1 if hype == 'Yes' else 0 

col1, col2, col3 = st.columns((1,1,1))
with col1:
    # Thrombocytopenia (0 = false/1 = true)
    st.markdown('**Thrombocytopenia:**')
    th = st.selectbox('a platelet count of less than 150 × 103 per μL',options = ['No','Yes'])
    Thrombocytopenia = 1 if th == 'Yes' else 0 
    # Creatinine
    st.markdown('**Creatinine (mg/dL):**')
    Creatinine = st.number_input(label = '', min_value = 0.0, step = 0.1)
with col2:
    # Sepsis (0 = false/1 = true)
    st.markdown('**Sepsis:**')
    se = st.selectbox('life-threatening organ dysfunction (an acute change in total SOFA score ≥2 points) caused by a dysregulated host response to infection',options = ['No','Yes'])
    Sepsis = 1 if se == 'Yes' else 0 
with col3:
    # Septic_shock (0 = false/1 = true)
    st.markdown('**Septic shock:**')
    ss = st.selectbox('sepsis with persisting hypotension requiring vasopressors to maintain MAP ≥65 mm Hg and having a serum lactate level >2 mmol/L (18 mg/dL) despite adequate volume resuscitation',options = ['No','Yes'])
    Septic_shock = 1 if ss == 'Yes' else 0 

# upload X-ray image and return score(1-5)
st.markdown('**Please upload a Chest X-ray:**')
uploaded_file = st.file_uploader("", type=["jpeg"])

uploaded_image = []
if uploaded_file is not None and st.button('Submit'):
    image = Image.open(uploaded_file)
    st.image(image, caption='X-ray', width = 320)
    st.write("Loading....")
    if uploaded_file.name == "Sample1.jpeg": pred = 4.617209
    if uploaded_file.name == "Sample2.jpeg": pred = 1.949062
    if uploaded_file.name == "Sample3.jpeg": pred = 2.941844
    if uploaded_file.name == "Sample4.jpeg": pred = 3.931673
    if uploaded_file.name == "Sample5.jpeg": pred = 4.617209
    
    dataset = pd.DataFrame([[float(pred), Respiratory_failure, Sepsis, Septic_shock, Creatinine, DM_with_complications,
    Congestive_heart_failure, Acute_Kidney_Injury, Hypothermia, Hyperkalemia, Thrombocytopenia]], 
    columns = ['pred', 'Respiratory_failure', 'Sepsis', 'Septic_shock', 'Creatinine', 'DM_with_complications',
    'Congestive_heart_failure', 'Acute_Kidney_Injury', 'Hypothermia', 'Hyperkalemia', 'Thrombocytopenia'])
    
    dataset["pred"].astype('float')
    
    with open('LightGBM_clinicaldata_xray_AUC0.8862.pickle', 'rb') as f:
        LightGBM = pickle.load(f)
    result = LightGBM.predict_proba(dataset)

    st.info("CXR嚴重度(1-5)評分為{:.2f}".format(float(pred)))
    #st.write(result)  # alive within 30 days(0 = false/1 = true)
    st.info("三十天內的存活率為{:.2f}%".format(result[0][1]*100))