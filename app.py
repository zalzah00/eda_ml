import streamlit as st
import pandas as pd

st.set_page_config(page_title="Data Analysis App", layout="wide")

# Initialize session state
if 'df' not in st.session_state:
    st.session_state.df = None
if 'file_name' not in st.session_state:
    st.session_state.file_name = None
if 'target_col' not in st.session_state:
    st.session_state.target_col = None

st.title("Data Analysis App")
st.write("Upload your dataset to get started with analysis and machine learning.")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.session_state.df = df
        st.session_state.file_name = uploaded_file.name  # Store the actual filename
        
        st.success(f"File '{uploaded_file.name}' uploaded successfully!")
        
        # Display basic info
        st.subheader("Data Preview")
        st.write(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")
        st.dataframe(df.head())
        
        st.subheader("Data Types")
        st.write(df.dtypes)
        
    except Exception as e:
        st.error(f"Error reading file: {e}")
else:
    st.info("Please upload a CSV file to proceed.")
