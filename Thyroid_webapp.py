import pickle
import streamlit as st
import pandas as pd
import numpy as np


#loading the saved models

with open('Thyroid_model.sav', 'rb') as file:
    Thyroid_model = pickle.load(file)

model = Thyroid_model

st.markdown("""
    <h1 style='text-align: center; color: #1E999F; font-size: 95px;'>Thyroid Disease and Condition Type Detection</h1>
    """, unsafe_allow_html=True)

# Page title
st.title('Thyroid Detection')
    
# Input fields
col1, col2, col3 = st.columns(3)
    
with col1:
    age = st.number_input('Age', min_value=0, max_value=120, value=0, help="Enter your age in years.")
with col2:
    sex = st.selectbox('Gender', ['male', 'female'], help="Select your gender.")
with col3:
    on_thyroxine = st.selectbox('Thyroxine', ['yes', 'no'], help="Are you currently on thyroxine medication?")
with col1:
    on_antithyroid_meds = st.selectbox('On antithyroid medicines', ['yes', 'no'], help="Are you currently on antithyroid medications?")
with col2:
    sick = st.selectbox('Sick', ['yes', 'no'], help="Are you currently sick?")
with col3:
    pregnant = st.selectbox('Pregnant', ['yes', 'no'], help="Are you pregnant?")
with col1:
    thyroid_surgery = st.selectbox('Thyroid surgery', ['yes', 'no'], help="Have you had thyroid surgery?")
with col2:
    I131_treatment = st.selectbox('I131 treatment', ['yes', 'no'], help="Have you received I131 treatment?")
with col3:
    query_hypothyroid = st.selectbox('Query hypothyroid', ['yes', 'no'], help="Is there a query about hypothyroidism?")
with col1:
    query_hyperthyroid = st.selectbox('Query hyperthyroid', ['yes', 'no'], help="Is there a query about hyperthyroidism?")
with col2:
    lithium = st.selectbox('Lithium', ['yes', 'no'], help="Are you on lithium medication?")
with col3:
    goitre = st.selectbox('Goitre', ['yes', 'no'], help="Do you have goitre?")
with col1:
    tumor = st.selectbox('Tumor', ['yes', 'no'], help="Do you have a tumor?")
with col2:
    hypopituitary = st.selectbox('Hypopituitary', ['yes', 'no'], help="Do you have hypopituitary condition?")
with col3:
    psych = st.selectbox('Psych', ['yes', 'no'], help="Do you have any psychiatric condition?")
with col1:
    TSH = st.number_input('TSH', value=0.0, help="Enter your TSH (Thyroid-stimulating hormone) level.")
with col2:
    T3 = st.number_input('T3', value=0.0, help="Enter your T3 (Triiodothyronine) level.")
with col3:
    TT4 = st.number_input('TT4', value=0.0, help="Enter your TT4 (Total thyroxine) level.")
with col1:
    T4U = st.number_input('T4U', value=0.0, help="Enter your T4U (Thyroxine uptake) level.")
with col2:
    FTI = st.number_input('FTI', value=0.0, help="Enter your FTI (Free thyroxine index).")
with col3:
    TBG = st.number_input('TBG', value=0.0, help="Enter your TBG (Thyroxine-binding globulin) level.")

# Convert categorical inputs to numerical if necessary
sex = 1 if sex == 'male' else 0
on_thyroxine = 1 if on_thyroxine == 'yes' else 0
on_antithyroid_meds = 1 if on_antithyroid_meds == 'yes' else 0
sick = 1 if sick == 'yes' else 0
pregnant = 1 if pregnant == 'yes' else 0
thyroid_surgery = 1 if thyroid_surgery == 'yes' else 0
I131_treatment = 1 if I131_treatment == 'yes' else 0
query_hypothyroid = 1 if query_hypothyroid == 'yes' else 0
query_hyperthyroid = 1 if query_hyperthyroid == 'yes' else 0
lithium = 1 if lithium == 'yes' else 0
goitre = 1 if goitre == 'yes' else 0
tumor = 1 if tumor == 'yes' else 0
hypopituitary = 1 if hypopituitary == 'yes' else 0
psych = 1 if psych == 'yes' else 0
    
#code for Prediction

#mapping from integer to thyroid condition labels
int_to_label = {
    0: '-', 1: 'A', 2: 'AK', 3: 'B', 4: 'C', 5: 'C|I', 6: 'D', 7: 'D|R', 8: 'E',
    9: 'F', 10: 'FK', 11: 'G', 12: 'GI', 13: 'GK', 14: 'GKJ', 15: 'H|K', 16: 'I',
    17: 'J', 18: 'K', 19: 'KJ', 20: 'L', 21: 'LJ', 22: 'M', 23: 'MI', 24: 'MK',
    25: 'N', 26: 'O', 27: 'OI', 28: 'P', 29: 'Q', 30: 'R', 31: 'S', 32: 'T'
    }
    
    #mapping from labels to descriptive names
label_to_name = {
    '-': 'No thyroid condition',
    'A': 'hyperthyroid',
    'B': 'T3 toxic',
    'C': 'toxic goitre',
    'C|I': 'toxic goitre with increased binding protein',
    'D': 'secondary toxic',
    'D|R': 'secondary toxic with discordant assay results',
    'E': 'hypothyroid',
    'F': 'primary hypothyroid',
    'FK': 'primary hypothyroid with concurrent non-thyroidal illness',
    'G': 'compensated hypothyroid',
    'GI': 'compensated hypothyroid with increased binding protein',
    'GK': 'compensated hypothyroid with concurrent non-thyroidal illness',
    'GKJ': 'compensated hypothyroid with concurrent non-thyroidal illness and decreased binding protein',
    'H|K': 'secondary hypothyroid with concurrent non-thyroidal illness',
    'I': 'increased binding protein',
    'J': 'decreased binding protein',
    'K': 'concurrent non-thyroidal illness',
    'KJ': 'concurrent non-thyroidal illness with decreased binding protein',
    'L': 'consistent with replacement therapy',
    'LJ': 'consistent with replacement therapy with decreased binding protein',
    'M': 'underreplaced',
    'MI': 'underreplaced with increased binding protein',
    'MK': 'underreplaced with concurrent non-thyroidal illness',
    'N': 'overreplaced',
    'O': 'antithyroid drugs',
    'OI': 'antithyroid drugs with increased binding protein',
    'P': 'I131 treatment',
    'Q': 'surgery',
    'R': 'discordant assay results',
    'S': 'elevated TBG',
    'T': 'elevated thyroid hormones',
    'AK': 'hyperthyroid with concurrent non-thyroidal illness',
    'LJ': 'consistent with replacement therapy with decreased binding protein',
    'C|I': 'toxic goitre with increased binding protein',
    'H|K': 'secondary hypothyroid with concurrent non-thyroidal illness',
    'GKJ': 'compensated hypothyroid with concurrent non-thyroidal illness and decreased binding protein'
    }
    
    # Button for prediction
if st.button('Thyroid Test Result'):
    # Create input array
    input_data = (age, sex, on_thyroxine, on_antithyroid_meds, sick, 
                  pregnant, thyroid_surgery, I131_treatment, query_hypothyroid, 
                  query_hyperthyroid, lithium, goitre, tumor, hypopituitary, psych, 
                  TSH, T3, TT4, T4U, FTI, TBG)

    input_data_as_np_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_np_array.reshape(1,-1)
    thyroid_prediction = Thyroid_model.predict(input_data_reshaped)
    label = int_to_label[thyroid_prediction[0]]
    condition_name = label_to_name[label]
    
    # Display result
    st.success(f'Result: {condition_name}')