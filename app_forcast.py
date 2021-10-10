# -*- coding: utf-8 -*-
"""
Created on Tue Sep  7 22:45:59 2021

@author: HP
"""

import streamlit as st
from PIL import Image
from sklearn.neural_network import MLPRegressor
import numpy as np
import pandas as pd
import pickle
import joblib
import matplotlib.pyplot as plt
import datetime

df = pd.read_excel(r'C:\Users\HP\Desktop\AI_Data_science\Data_sci\ExcelR_project2\Dataset.xlsx',header=10, names=['Date','Price'])
df = df.set_index(['Date'])
#@st.cache

st.title('Streamlit Gold Price Prediction ML App')
image=Image.open(r"C:\Users\HP\Desktop\AI_Data_science\Data_sci\ExcelR_project2\gd.jpg")
st.image(image, use_column_width='auto')
#st.write(df)
base="dark"
primaryColor="purple"
model= joblib.load(r'C:\Users\HP\Desktop\AI_Data_science\Data_sci\ExcelR_project2\forcast_model.pkl')

#pickle_in = open("classifier.pkl","rb")
#classifier=pickle.load(pickle_in)



def create_lag_feature(df, no_of_days):
    
    for day in range(1, no_of_days+1):
        df[f"lag_{day}"] = df["Price"].shift(day)
        
    return df

df_features = create_lag_feature(df, 14)
df_features =df_features.dropna()
df_forcast =df_features.iloc[-50:]
#@app.route('/')
def welcome():
    return "Welcome All"


#@app.route('/predict',methods=["Get"])
def pred(Days):
    x=df_forcast.index[-1]
    df_1=df_forcast.copy()
    for i in range(1,Days+1):
    #days=[]
        excluded=(6, 7)
         #df.index[-1]+datetime.timedelta(days=i)
        a=x+datetime.timedelta(days=i)
        if a.isoweekday() not in excluded:
            #days.append(a)
            dx=pd.DataFrame({'Date':a},index=[0])
            dx.set_index('Date',inplace=True)
            df_1=df_1.append(dx)
            for day in range(1,15):
                df_1[f"lag_{day}"] = df_1["Price"].shift(day)
            #if df.Price.isnull().any():
                #place=df[df.Price.isnull()].index
                xi=df_1.iloc[-1:,]
                xi=xi.drop("Price",axis=1)
        predict=model.predict(xi)
        df_1.iloc[-1:,]['Price']=predict[0]
        
    return df_1


#<div style="background-color:tomato;padding:10px">
#    <h2 style="color:white;text-align:center;">Streamlit Gold Price Prediction ML App </h2>
#    </div> st.line_chart(plot(df_result)

def plot(df):
    plt.figure(figsize=(12,5), dpi=100)
    plt.plot(df['Price'], label='actual')
    plt.plot(df.loc['20210721':]['Price'], label='forecast')
    plt.title('Forecast')
    plt.legend(loc='upper left', fontsize=8)


def main():
    #st.title("Bank Authenticator")
    html_temp = """
    
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    st.subheader('History Of Gold')
    st.line_chart(df.Price)
    st.subheader('Select days here to predict Gold Price')
#    Days_slider =st.slider('Days',1,30)
#    Days_etry = st.text_input("Days","Type Here")
    Days = st.slider('Days',1,30)
    result=""
    Date_forcast=""
    df_result=st.write()
    if st.button("Predict"):
        df_result=pred(int(Days))
        result= round(df_result.iloc[-1]['Price'])
        Date_forcast=df_result.index[-1]  
#        st.line_chart(df_result['Price'])
        
    st.success('The Gold Price is {} on {}'.format(result,Date_forcast))
    
if __name__=='__main__':
    main()
    
    