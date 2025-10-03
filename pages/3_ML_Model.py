# pages/3_ML_Model.py

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
import matplotlib.pyplot as plt

st.set_page_config(page_title="ML Model", layout="wide")

if 'df' not in st.session_state or 'target_col' not in st.session_state:
    st.warning("Please upload data and select a target column on previous pages.")
else:
    st.header("4. Machine Learning Model")
    df = st.session_state['df'].copy()
    target_col = st.session_state['target_col']

    # Determine problem type
    target_is_numerical = pd.api.types.is_numeric_dtype(df[target_col])
    problem_type = "Regression" if target_is_numerical else "Classification"
    st.info(f"Detected Problem Type: **{problem_type}**")

    # --- Feature Selection ---
    feature_cols = [col for col in df.columns if col != target_col]
    default_selections = st.session_state.get('selected_features_viz', feature_cols)
    safe_default_selections = [col for col in default_selections if col in feature_cols]
    
    st.subheader("Feature Selection for Model")
    selected_features = st.multiselect(
        "Select features for model training",
        options=feature_cols,
        default=safe_default_selections
    )

    # **CRITICAL FIX 1: Save features using the key expected by 5_Summary.py**
    st.session_state['selected_features_model'] = selected_features
    st.session_state['selected_features'] = selected_features # For 5_Summary.py consistency
    
    st.markdown("---")

    if not selected_features:
        st.error("Please select at least one feature to train the model.")
    else:
        # --- Model Selection & Parameters ---
        st.subheader("Model Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if problem_type == "Classification":
                model_options = {
                    "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
                    "Random Forest Classifier": RandomForestClassifier(random_state=42)
                }
                model_name = st.selectbox("Select Model", list(model_options.keys()))
                model = model_options[model_name]
            else: # Regression
                model_options = {
                    "Linear Regression": LinearRegression(),
                    "Random Forest Regressor": RandomForestRegressor(random_state=42)
                }
                model_name = st.selectbox("Select Model", list(model_options.keys()))
                model = model_options[model_name]

        with col2:
            test_size = st.slider("Test Set Size", min_value=0.1, max_value=0.5, value=0.2, step=0.05)
            
        # --- Preprocessing Steps ---
        numerical_features = df[selected_features].select_dtypes(include=np.number).columns.tolist()
        categorical_features = df[selected_features].select_dtypes(include=['object', 'category']).columns.tolist()
        
        df_model = df[selected_features + [target_col]].dropna() # Simple dropna
        
        if len(df_model) < len(df):
             st.warning(f"Note: Dropping {len(df) - len(df_model)} rows with missing values for model training.")

        X = df_model[selected_features]
        y = df_model[target_col]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        numerical_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_features),
                ('cat', categorical_transformer, categorical_features)
            ],
            remainder='passthrough'
        )

        # **CRITICAL FIX 2: Set default preprocessing strategies for 5_Summary.py**
        st.session_state['imputer_strategy'] = 'dropna_pre_split' # Indicate rows were dropped
        st.session_state['scaler_method'] = 'StandardScaler'
        st.session_state['handle_unknown'] = True
        st.session_state['k_neighbors'] = 'N/A' # Default for non-KNN models
        
        # **CRITICAL FIX 3: Store core model configuration**
        st.session_state['selected_model_name'] = model_name
        st.session_state['is_classification'] = (problem_type == "Classification")
        st.session_state['test_size'] = test_size

        # --- Training and Evaluation ---
        if st.button("Train and Evaluate Model"):
            with st.spinner(f"Training {model_name}..."):
                full_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                                ('model', model)])
                
                full_pipeline.fit(X_train, y_train)
                y_pred = full_pipeline.predict(X_test)

                # --- Results ---
                st.subheader("Model Evaluation Results")

                if problem_type == "Classification":
                    # For Classification
                    accuracy = accuracy_score(y_test, y_pred)
                    st.success(f"**Accuracy:** {accuracy:.4f}")
                    st.metric("Test Set Accuracy", f"{accuracy:.2%}")
                    
                    # **CRITICAL FIX 4: Save Classification Results**
                    st.session_state['accuracy'] = accuracy
                    # Set other regression metrics to N/A
                    st.session_state['rmse'] = 'N/A'
                    st.session_state['r2'] = 'N/A'
                    
                else:
                    # For Regression
                    mse = mean_squared_error(y_test, y_pred)
                    rmse = np.sqrt(mse) 
                    r2 = r2_score(y_test, y_pred)
                    
                    st.success(f"**$R^2$ Score:** {r2:.4f}")
                    st.metric("Root Mean Squared Error (RMSE)", f"{rmse:.2f}")
                    st.metric("Coefficient of Determination ($R^2$)", f"{r2:.2f}")
                    
                    # **CRITICAL FIX 5: Save Regression Results**
                    st.session_state['rmse'] = rmse
                    st.session_state['r2'] = r2
                    # Set other classification metrics to N/A
                    st.session_state['accuracy'] = 'N/A'

                    # Plotting predicted vs actual
                    fig, ax = plt.subplots(figsize=(8, 6))
                    ax.scatter(y_test, y_pred, alpha=0.6)
                    min_val = min(y_test.min(), y_pred.min())
                    max_val = max(y_test.max(), y_pred.max())
                    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
                    ax.set_xlabel('Actual Values')
                    ax.set_ylabel('Predicted Values')
                    ax.set_title(f'{model_name}: Actual vs Predicted')
                    st.pyplot(fig)

                st.balloons()
