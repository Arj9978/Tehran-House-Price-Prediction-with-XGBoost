# Import Libraries
import joblib
import sklearn

import numpy as np
import pandas as pd
import streamlit as st

from utils import PrepProcesor, columns 
from sklearn.ensemble import GradientBoostingClassifier

# Set custom CSS styles
page_bg_img = '''
    <style>
        body {
            background-image: url("https://img.staticmb.com/mbcontent/images/uploads/2022/12/Most-Beautiful-House-in-the-World.jpg");
            background-size: cover;
        }
    </style>
'''
st.markdown(page_bg_img, unsafe_allow_html=True)

# Load Model
model = joblib.load('XGBoost.joblib')

# Define the unique addresses used for one-hot encoding during training
addresses = ['Shahran', 'Pardis', 'Shahrake Qods', 'Shahrake Gharb', 'North Program Organization', 'Andisheh', 'West Ferdows Boulevard', 'Narmak', 'Saadat Abad', 'Zafar', 'Islamshahr', 'Pirouzi', 'Shahrake Shahid Bagheri', 'Moniriyeh', 'Velenjak', 'Amirieh', 'Southern Janatabad', 'Salsabil', 'Zargandeh', 'Feiz Garden', 'Water Organization', 'ShahrAra', 'Gisha', 'Ray', 'Abbasabad', 'Ostad Moein', 'Farmanieh', 'Parand', 'Punak', 'Qasr-od-Dasht', 'Aqdasieh', 'Pakdasht', 'Railway', 'Central Janatabad', 'East Ferdows Boulevard', 'Pakdasht KhatunAbad', 'Sattarkhan', 'Baghestan', 'Shahryar', 'Northern Janatabad', 'Daryan No', 'Southern Program Organization', 'Rudhen', 'West Pars', 'Afsarieh', 'Marzdaran', 'Dorous', 'Sadeghieh', 'Chahardangeh', 'Baqershahr', 'Jeyhoon', 'Lavizan', 'Shams Abad', 'Fatemi', 'Keshavarz Boulevard', 'Kahrizak', 'Qarchak', 'Shahr-e-Ziba', 'Pasdaran', 'Northren Jamalzadeh', 'Azarbaijan', 'Bahar', 'Persian Gulf Martyrs Lake', 'Beryanak', 'Heshmatieh', 'Elm-o-Sanat', 'Golestan', 'Chardivari', 'Gheitarieh', 'Kamranieh', 'Gholhak', 'Heravi', 'Hashemi', 'Dehkade Olampic', 'Damavand', 'Republic', 'Zaferanieh', 'Qazvin Imamzadeh Hassan', 'Niavaran', 'Valiasr', 'Qalandari', 'Amir Bahador', 'Ekhtiarieh', 'Ekbatan', 'Absard', 'Haft Tir', 'Mahallati', 'Ozgol', 'Tajrish', 'Abazar', 'Koohsar', 'Hekmat', 'Parastar', 'Lavasan', 'Majidieh', 'Southern Chitgar', 'Karimkhan', 'Si Metri Ji', 'Karoon', 'Northern Chitgar', 'East Pars', 'Kook', 'Air force', 'Sohanak', 'Komeil', 'Azadshahr', 'Zibadasht', 'Amirabad', 'Dezashib', 'Elahieh', 'Mirdamad', 'Razi', 'Jordan', 'Mahmoudieh', 'Shahedshahr', 'Yaftabad', 'Mehran', 'Nasim Shahr', 'Tenant', 'Chardangeh', 'Fallah', 'Eskandari', 'Shahrakeh Naft', 'Ajudaniye', 'Tehransar', 'Nawab', 'Yousef Abad', 'Northern Suhrawardi', 'Villa', 'Hakimiyeh', 'Nezamabad', 'Garden of Saba', 'Tarasht', 'Azari', 'Shahrake Apadana', 'Araj', 'Vahidieh', 'Malard', 'Shahrake Azadi', 'Darband', 'Vanak', 'Tehran Now', 'Darabad', 'Eram', 'Atabak', 'Sabalan', 'SabaShahr', 'Shahrake Madaen', 'Waterfall', 'Ahang', 'Salehabad', 'Pishva', 'Enghelab', 'Islamshahr Elahieh', 'Ray - Montazeri', 'Firoozkooh Kuhsar', 'Ghoba', 'Mehrabad', 'Southern Suhrawardi', 'Abuzar', 'Dolatabad', 'Hor Square', 'Taslihat', 'Kazemabad', 'Robat Karim', 'Ray - Pilgosh', 'Ghiyamdasht', 'Telecommunication', 'Mirza Shirazi', 'Gandhi', 'Argentina', 'Seyed Khandan', 'Shahrake Quds', 'Safadasht', 'Khademabad Garden', 'Hassan Abad', 'Chidz', 'Khavaran', 'Boloorsazi', 'Mehrabad River River', 'Varamin - Beheshti', 'Shoosh', 'Thirteen November', 'Darakeh', 'Aliabad South', 'Alborz Complex', 'Firoozkooh', 'Vahidiyeh', 'Shadabad', 'Naziabad', 'Javadiyeh', 'Yakhchiabad']

st.title('Tehran House Price Prediction')
# Area,	Room, Parking, Warehouse, Elevator, Address
Area = st.text_input("Input Area of House:", '100') 
Room = st.selectbox("Choose Number of Rooms:", [0,1,2,3,4,5])
st.code("0: No, 1: Yes")
Parking  = st.selectbox("The House Has Parking:", [0,1])
Warehouse  = st.selectbox("The House Has Warehouse:", [0,1])
Elevator  = st.selectbox("The House Has Elevator:", [0,1])
Address  = st.selectbox("Where is the House:", addresses)

def predict(): 
    row = np.array([Area, Room, Parking, Warehouse, Elevator])
    st.write(row)
    # Create a DataFrame with the row data and columns matching the training data
    X = pd.DataFrame([row], columns=['Area', 'Room', 'Parking', 'Warehouse', 'Elevator'])
    st.write(X)
    # Perform one-hot encoding for the address input
    address_dummy = pd.get_dummies([Address], columns=['Address'], prefix='', prefix_sep='')

    # Align the address columns with the training data columns
    address_dummy = address_dummy.reindex(columns=addresses, fill_value=0)

    # Concatenate the address columns with the input data
    X = pd.concat([X, address_dummy], axis=1)
    st.write(X)

    X = np.array(X)
    st.write(X)
    prediction = model.predict(X)
    st.write(prediction)

trigger = st.button('Predict', on_click=predict)
