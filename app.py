import streamlit as st
import pandas as pd
import os
import pandas_profiling
from streamlit_pandas_profiling import st_profile_report
from pycaret.regression import setup,compare_models,pull,save_model



with st.sidebar:
    st.title('Autobot')
    st.image('https://en.pimg.jp/069/791/012/1/69791012.jpg')
    choice = st.radio('Navigation',['Upload','Profiling','ML','Download'])
    st.info('This application allows you to build an automated ML pipeline using Streamlit ')

if os.path.exists('sourcedata.csv'):
    df =pd.read_csv('sourcedata.csv',index_col=None)


if choice == 'Upload':
    st.title('Upload your data for modeling !!')
    file = st.file_uploader('Upload Your Dataset Here')
    if file:
        df =pd.read_csv(file,index_col=None)
        df.to_csv('sourcedata.csv',index=None)
        st.dataframe(df)
        

if choice=='Profiling':
    st.title(' Automated Exploratory Data Analysis')
    profile_report=df.profile_report()
    st_profile_report(profile_report)

if choice =='ML':
    st.title(' I Choice You !! Machine Learning Goo!')
    target=st.selectbox('Select Your Target',df.columns)
    if st.button('Train Model'):
        setup(df,target=target)
        setup_df=pull()
        st.info('This The ML Experiment Settings')
        best_model =compare_models()
        compare_df=pull()
        st.info('THis is the ML Model')
        st.dataframe(compare_df)
        best_model
        save_model(best_model,'best_model')
    
if choice=='Download':
    with open('best_model.pkl','rb') as f:
        st.download_button('Download the Model',f,'trained_model.pkl')