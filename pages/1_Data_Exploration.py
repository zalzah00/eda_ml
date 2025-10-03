# pages/1_Data_Exploration.py

import streamlit as st
import pandas as pd

if 'df' not in st.session_state or st.session_state['df'] is None:
    st.warning("Please upload a data file on the Home page first.")
else:
    st.header("2. Data Exploration and Selection")
    df = st.session_state['df']

    # Function to describe the dataframe
    @st.cache_data
    def describe_dataframe(data):
        descriptions = data.describe(include='all').T
        descriptions['Missing Values'] = data.isnull().sum()
        descriptions['Unique Values'] = data.nunique()
        return descriptions[['Missing Values', 'Unique Values', 'count', 'mean', 'std']]

    st.dataframe(describe_dataframe(df), use_container_width=True)

    target_col = st.selectbox("Select the target column for prediction", 
                              options=df.columns, 
                              index=0) # Set a default value

    # Store the selected target column in session state
    st.session_state['target_col'] = target_col
    
    st.info("The selected target column has been saved. Proceed to the next page for visualizations.")
