# pages/4_Stats_Analysis.py

import streamlit as st
import pandas as pd
import statsmodels.api as sm
import numpy as np 

st.set_page_config(page_title="Statistical Analysis", layout="wide")

if 'df' not in st.session_state or 'target_col' not in st.session_state:
    st.warning("Please upload a data file and select a target column on the previous pages.")
else:
    st.header("5. Statistical Analysis with Statsmodels")
    st.info("This section uses a statistical model to analyze the significance and impact of each predictor.")

    df = st.session_state['df'].copy() 
    target_col = st.session_state['target_col']

    # --- Model Type Selection based on Target Column ---
    target_is_numerical = pd.api.types.is_numeric_dtype(df[target_col])
    
    # Check if target is binary (suitable for Logit)
    # We dropna() just for the unique count check, as the main data is cleaned later.
    if not target_is_numerical and len(df[target_col].dropna().unique()) <= 2:
        model_name = "Logistic Regression (Logit)"
        sm_model = sm.Logit
    elif target_is_numerical:
        model_name = "Ordinary Least Squares (OLS) - Regression"
        sm_model = sm.OLS
    else:
        st.error(f"Target column '{target_col}' is categorical with more than 2 unique values. Only OLS (numerical) or Logit (binary) models are supported here.")
        sm_model = None 
        
    if sm_model:
        st.subheader(f"Model Selected: {model_name}")

        # --- Predictor Selection ---
        feature_cols = [col for col in df.columns if col != target_col]
        selected_features = st.multiselect(
            "Select predictor variables for the statistical model:",
            options=feature_cols,
            default=st.session_state.get('selected_features_model', feature_cols)
        )

        if st.button("Run Statistical Analysis"):
            if not selected_features:
                st.error("Please select at least one predictor variable.")
            else:
                try:
                    # --- Prepare data for Statsmodels ---
                    
                    # 1. Drop rows with any missing values in the selected columns
                    data_for_stats = df[[target_col] + selected_features].dropna()
                    
                    if data_for_stats.empty:
                        st.error("After dropping rows with missing data, no records remain. Cannot run analysis.")
                        return

                    # Separate features (X) and target (y)
                    X_stats = data_for_stats[selected_features]
                    y_stats = data_for_stats[target_col]
                    
                    # 2. Identify categorical features
                    categorical_features = X_stats.select_dtypes(include=['object', 'category']).columns.tolist()

                    # 3. One-hot encode categorical variables (FIXES PREVIOUS TYPE ERROR)
                    if categorical_features:
                        X_stats = pd.get_dummies(X_stats, columns=categorical_features, drop_first=True)
                        st.success(f"One-Hot Encoded categorical features: {', '.join(categorical_features)}")

                    # 4. Add a constant (intercept)
                    X_stats = sm.add_constant(X_stats, prepend=False)

                    # --- Run the Statistical Model ---
                    with st.spinner('Running statistical analysis...'):
                        model = sm_model(y_stats, X_stats).fit()

                    st.success("Analysis complete! 🎉")
                    
                    # Display the results
                    st.subheader("Model Summary")
                    # Use st.code() to display the summary string neatly
                    st.code(model.summary().as_text(), language='text')
                    
                    st.info("The summary table shows key metrics like R-squared / Pseudo-R-squared and the p-value for each predictor. A low **p-value (typically < 0.05)** indicates the predictor is statistically significant.")

                except Exception as e:
                    st.error(f"A runtime error occurred during model fitting: {e}. Please check for high correlation (multicollinearity) or issues with the target class balance.")
