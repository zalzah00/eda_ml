# pages/3_ML_Model.py

import streamlit as st
import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, mean_absolute_error
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

    # Save features to session state
    st.session_state['selected_features_model'] = selected_features
    st.session_state['selected_features'] = selected_features 
    
    st.markdown("---")

    if not selected_features:
        st.error("Please select at least one feature to train the model.")
    else:
        # --- Model Selection & Parameters ---
        st.subheader("Model Configuration")
        
        col1, col2 = st.columns(2)
        
        # --- Define available models ---
        if problem_type == "Classification":
            model_options = {
                "Logistic Regression": LogisticRegression,
                "Decision Tree Classifier": DecisionTreeClassifier,
                "Random Forest Classifier": RandomForestClassifier
            }
        else: # Regression
            model_options = {
                "Linear Regression": LinearRegression,
                "Ridge Regression": Ridge,
                "Lasso Regression": Lasso,
                "Decision Tree Regressor": DecisionTreeRegressor,
                "Random Forest Regressor": RandomForestRegressor
            }

        alpha = None
        with col1:
            model_name = st.selectbox("Select Model", list(model_options.keys()))
            model_class = model_options[model_name]
            
        with col2:
            test_size = st.slider("Test Set Size", min_value=0.1, max_value=0.5, value=0.2, step=0.05)
            
            # --- CONDITIONAL ALPHA SLIDER for Regularized Regression Models ---
            if problem_type == "Regression" and model_name in ["Ridge Regression", "Lasso Regression"]:
                st.markdown("---")
                alpha = st.slider(
                    "Regularization Strength (Alpha)", 
                    min_value=0.01, max_value=10.0, value=1.0, step=0.1, 
                    help="Higher Alpha means stronger regularization (coefficient shrinking/zeroing)."
                )
                st.info(f"Using Alpha = **{alpha}**")
        
        # --- Training and Evaluation ---
        if st.button("Train and Evaluate Model"):
            
            # 1. Instantiate the selected model with parameters
            model_params = {'random_state': 42} 
            
            if model_name == "Logistic Regression":
                model_params.update({'max_iter': 1000})
            elif model_name in ["Ridge Regression", "Lasso Regression"]:
                model_params = {'alpha': alpha, 'random_state': 42} 
            elif model_name == "Linear Regression":
                model_params = {}
            
            # Instantiate the model class with the determined parameters
            model = model_class(**model_params)
            
            # 2. Prepare data for Pipeline
            df_model = df[selected_features + [target_col]].dropna() 

            if len(df_model) < len(df):
                st.warning(f"Note: Dropping {len(df) - len(df_model)} rows with missing values for model training.")
                
            if df_model.empty:
                st.error("After dropping rows with missing data, no records remain. Cannot run analysis.")
                st.stop()

            X = df_model[selected_features]
            y = df_model[target_col]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

            numerical_features = X.select_dtypes(include=np.number).columns.tolist()
            categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

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

            # 4. Train
            with st.spinner(f"Training {model_name}..."):
                try:
                    full_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                                    ('model', model)])
                    
                    full_pipeline.fit(X_train, y_train)
                    
                    # Make predictions for both training and test sets
                    y_train_pred = full_pipeline.predict(X_train)
                    y_test_pred = full_pipeline.predict(X_test)

                    # --- Store COMPLETE Pipeline Configuration for Summary Page ---
                    st.session_state['pipeline_config'] = {
                        'problem_type': problem_type,
                        'model_name': model_name,
                        'model_class': model_class.__name__,
                        'model_params': model_params,
                        'preprocessor_type': 'ColumnTransformer',
                        'numerical_features': numerical_features,
                        'categorical_features': categorical_features,
                        'numerical_transformer': {
                            'steps': ['scaler'],
                            'scaler_type': 'StandardScaler'
                        },
                        'categorical_transformer': {
                            'steps': ['onehot'],
                            'encoder_type': 'OneHotEncoder',
                            'handle_unknown': 'ignore'
                        },
                        'imputation_strategy': 'dropna_pre_split',
                        'test_size': test_size,
                        'selected_features': selected_features,
                        'target_column': target_col
                    }
                    
                    # --- Store Metrics ---
                    if problem_type == "Classification":
                        train_accuracy = accuracy_score(y_train, y_train_pred)
                        test_accuracy = accuracy_score(y_test, y_test_pred)
                        
                        st.session_state['metrics'] = {
                            'train_accuracy': train_accuracy,
                            'test_accuracy': test_accuracy,
                            'train_mae': 'N/A',
                            'test_mae': 'N/A', 
                            'test_rmse': 'N/A',
                            'test_r2': 'N/A'
                        }
                        
                    else:
                        train_mae = mean_absolute_error(y_train, y_train_pred)
                        test_mae = mean_absolute_error(y_test, y_test_pred)
                        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
                        test_r2 = r2_score(y_test, y_test_pred)
                        
                        st.session_state['metrics'] = {
                            'train_mae': train_mae,
                            'test_mae': test_mae,
                            'test_rmse': test_rmse,
                            'test_r2': test_r2,
                            'train_accuracy': 'N/A',
                            'test_accuracy': 'N/A'
                        }

                    # --- Results Display ---
                    st.subheader("Model Evaluation Results")
                    
                    if problem_type == "Classification":
                        st.success(f"**Training Accuracy:** {train_accuracy:.4f}")
                        st.success(f"**Test Accuracy:** {test_accuracy:.4f}")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Training Accuracy", f"{train_accuracy:.2%}")
                        with col2:
                            st.metric("Test Accuracy", f"{test_accuracy:.2%}")
                        
                    else:
                        st.success(f"**{model_name} Performance:**")
                        st.write(f"**Training MAE:** {train_mae:.4f}")
                        st.write(f"**Test MAE:** {test_mae:.4f}")
                        st.write(f"**Test RMSE:** {test_rmse:.4f}")
                        st.write(f"**Test R²:** {test_r2:.4f}")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Training MAE", f"{train_mae:.4f}")
                        with col2:
                            st.metric("Test MAE", f"{test_mae:.4f}")
                        with col3:
                            st.metric("Test RMSE", f"{test_rmse:.4f}")
                        with col4:
                            st.metric("Test R²", f"{test_r2:.4f}")

                        # Plotting predicted vs actual
                        fig, ax = plt.subplots(figsize=(8, 6))
                        ax.scatter(y_test, y_test_pred, alpha=0.6)
                        min_val = min(y_test.min(), y_test_pred.min())
                        max_val = max(y_test.max(), y_test_pred.max())
                        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
                        ax.set_xlabel('Actual Values')
                        ax.set_ylabel('Predicted Values')
                        ax.set_title(f'{model_name}: Actual vs Predicted')
                        st.pyplot(fig)

                    st.balloons()
                    
                    # Save the trained pipeline to session state
                    st.session_state['trained_pipeline'] = full_pipeline
                    
                except Exception as e:
                    st.error(f"An error occurred during model training: {str(e)}")
                    st.stop()
