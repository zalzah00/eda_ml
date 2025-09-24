import streamlit as st
import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import OneHotEncoder

if 'df' not in st.session_state or 'target_col' not in st.session_state:
    st.warning("Please upload a data file and select a target column on the previous pages.")
else:
    st.header("5. Statistical Analysis with Statsmodels")
    st.info("This section uses a statistical model to analyze the significance and impact of each predictor.")

    df = st.session_state['df']
    target_col = st.session_state['target_col']

    feature_cols = [col for col in df.columns if col != target_col]
    selected_features = st.multiselect(
        "Select predictor variables for the statistical model:",
        options=feature_cols,
        default=feature_cols
    )

    if st.button("Run Statistical Analysis"):
        if not selected_features:
            st.error("Please select at least one predictor variable.")
        else:
            try:
                # --- Prepare data for Statsmodels ---
                # Drop rows with any missing values in the selected columns
                data_for_stats = df[[target_col] + selected_features].dropna()
                
                # Separate features (X) and target (y)
                X_stats = data_for_stats[selected_features]
                y_stats = data_for_stats[target_col]
                
                # Identify numerical and categorical features
                numerical_features = X_stats.select_dtypes(include=['int64', 'float64']).columns.tolist()
                categorical_features = X_stats.select_dtypes(include=['object', 'category']).columns.tolist()

                # One-hot encode categorical variables
                if categorical_features:
                    X_stats = pd.get_dummies(X_stats, columns=categorical_features, drop_first=True)

                # Add a constant (intercept) for the OLS model
                X_stats = sm.add_constant(X_stats)

                # --- Run the OLS (Ordinary Least Squares) Model ---
                with st.spinner('Running statistical analysis...'):
                    model = sm.OLS(y_stats, X_stats).fit()

                st.success("Analysis complete!")
                
                # Display the results
                st.subheader("Statistical Summary")
                st.write(model.summary())
                
                st.info("The summary table shows key metrics like R-squared and the p-value for each predictor. A low p-value (typically < 0.05) indicates that the predictor is statistically significant.")

            except Exception as e:
                st.error(f"An error occurred: {e}. Please check your data and selections.")