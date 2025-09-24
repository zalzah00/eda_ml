# app.py

import streamlit as st
import pandas as pd

st.set_page_config(layout="wide", page_title="Multi-Page ML App")
st.title("Interactive Data Analysis and Prediction App 📈")

st.header("1. Upload Your Data")

uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=['csv', 'xlsx'])

if uploaded_file:
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
    
    # Store the DataFrame in session state
    st.session_state['df'] = df

    st.success("File uploaded successfully! You can now navigate to other pages.")
    st.subheader("Data Preview")
    st.dataframe(df.head())
    
    st.markdown("---")
    st.markdown("### Navigate to the pages in the sidebar to continue.")