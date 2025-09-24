# pages/2_Visualizations.py

import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

if 'df' not in st.session_state or 'target_col' not in st.session_state:
    st.warning("Please upload data and select a target column on previous pages.")
else:
    st.header("3. Visualizations")
    df = st.session_state['df']
    target_col = st.session_state['target_col']

    feature_cols = [col for col in df.columns if col != target_col]
    selected_features = st.multiselect("Select features to visualize", 
                                       options=feature_cols, 
                                       default=feature_cols[:5])

    if st.button("Generate Plots"):
        if not selected_features:
            st.error("Please select at least one feature.")
        else:
            # Generate and display plots using selected features and target
            numerical_features = df[selected_features].select_dtypes(include=np.number).columns
            
            if len(numerical_features) > 1:
                st.subheader("Pair Plot of Selected Features")
                fig = sns.pairplot(df, vars=numerical_features)
                st.pyplot(fig)
            else:
                st.info("Select at least two numerical features for a pair plot.")
            
            # Additional plot logic (e.g., histograms, heatmaps) would go here