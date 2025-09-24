import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier

from sklearn.metrics import (
    mean_squared_error, r2_score, accuracy_score, precision_score, 
    recall_score, f1_score, roc_curve, auc
)

if 'df' not in st.session_state or 'target_col' not in st.session_state:
    st.warning("Please upload data and select a target column first.")
else:
    st.header("4. Train Your Machine Learning Model")
    df = st.session_state['df']
    target_col = st.session_state['target_col']
    
    feature_cols = [col for col in df.columns if col != target_col]
    selected_features = st.multiselect("Select features for model training", 
                                       options=feature_cols, 
                                       default=st.session_state.get('selected_features', feature_cols))

    # --- Determine Problem Type ---
    is_classification = df[target_col].nunique() <= 10 and df[target_col].dtype in ['object', 'category', 'int64']

    # --- Preprocessing Options UI ---
    st.subheader("Model Preprocessing Options")
    col1, col2 = st.columns(2)
    with col1:
        imputer_strategy = st.selectbox(
            "Select Imputation Strategy for Missing Values:",
            ('median', 'mean', 'most_frequent'),
            index=('median', 'mean', 'most_frequent').index(st.session_state.get('imputer_strategy', 'median'))
        )
    with col2:
        scaler_method = st.selectbox(
            "Select Scaling Method for Numerical Features:",
            ('StandardScaler', 'MinMaxScaler', 'None'),
            index=('StandardScaler', 'MinMaxScaler', 'None').index(st.session_state.get('scaler_method', 'StandardScaler'))
        )
    handle_unknown = st.checkbox(
        "Ignore unknown categorical values?", 
        value=st.session_state.get('handle_unknown', True)
    )
    
    # --- Model Selection and Hyperparameters UI ---
    st.subheader("Model Selection and Hyperparameters")
    
    col3, col4 = st.columns(2)
    with col3:
        if is_classification:
            model_options = {
                "Logistic Regression": LogisticRegression,
                "Random Forest Classifier": RandomForestClassifier,
                "K-Nearest Neighbors Classifier": KNeighborsClassifier,
                "Decision Tree Classifier": DecisionTreeClassifier
            }
        else:
            model_options = {
                "Linear Regression": LinearRegression,
                "Ridge Regression": Ridge,
                "Lasso Regression": Lasso,
                "Random Forest Regressor": RandomForestRegressor,
                "K-Nearest Neighbors Regressor": KNeighborsRegressor,
                "Decision Tree Regressor": DecisionTreeRegressor
            }
        
        # Determine the default selected model
        default_model_index = 0
        if 'selected_model_name' in st.session_state and st.session_state['selected_model_name'] in model_options:
            default_model_index = list(model_options.keys()).index(st.session_state['selected_model_name'])
        
        selected_model_name = st.selectbox(
            "Choose an ML Model", 
            list(model_options.keys()),
            index=default_model_index
        )

    with col4:
        test_size = st.slider(
            "Select Test Set Size (%):", 
            min_value=10, 
            max_value=50, 
            value=int(st.session_state.get('test_size', 0.2) * 100), 
            step=5
        ) / 100
    
    # --- KNN specific hyperparameter (k) ---
    k_neighbors = None
    if 'K-Nearest Neighbors' in selected_model_name:
        k_neighbors = st.slider(
            "Number of Neighbors (k):",
            min_value=1,
            max_value=20,
            value=st.session_state.get('k_neighbors', 5),
            step=1
        )

    if st.button("Train Model"):
        # Store user choices in session state
        st.session_state['selected_model_name'] = selected_model_name
        st.session_state['imputer_strategy'] = imputer_strategy
        st.session_state['scaler_method'] = scaler_method
        st.session_state['handle_unknown'] = handle_unknown
        st.session_state['test_size'] = test_size
        st.session_state['selected_features'] = selected_features
        st.session_state['is_classification'] = is_classification
        if k_neighbors is not None:
            st.session_state['k_neighbors'] = k_neighbors

        X = df[selected_features]
        y = df[target_col]
        
        numerical_features = X.select_dtypes(include=np.number).columns.tolist()
        categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

        # Build numerical transformer based on user choices
        steps_numerical = [('imputer', SimpleImputer(strategy=imputer_strategy))]
        if scaler_method == 'StandardScaler':
            steps_numerical.append(('scaler', StandardScaler()))
        elif scaler_method == 'MinMaxScaler':
            steps_numerical.append(('scaler', MinMaxScaler()))
        numerical_transformer = Pipeline(steps=steps_numerical)
        
        # Build categorical transformer based on user choices
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')), 
            ('onehot', OneHotEncoder(handle_unknown='ignore' if handle_unknown else 'error'))
        ])
        
        # Bundle preprocessing
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_features),
                ('cat', categorical_transformer, categorical_features)
            ])

        # Create the full pipeline with the chosen model and hyperparameters
        model_instance = model_options[selected_model_name](n_neighbors=k_neighbors) if k_neighbors is not None else model_options[selected_model_name]()
            
        full_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                        ('model', model_instance)])
        
        # Split data and train
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        with st.spinner('Training model...'):
            full_pipeline.fit(X_train, y_train)
            y_pred = full_pipeline.predict(X_test)
        
        st.success("Model training complete! 🎉")

        # --- Displaying Metrics Based on Problem Type ---
        st.header("5. Model Performance")
        
        if is_classification:
            y_pred_proba = full_pipeline.predict_proba(X_test)[:, 1] if hasattr(full_pipeline, 'predict_proba') else None
            
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            
            # Save metrics to session state
            st.session_state['accuracy'] = accuracy
            st.session_state['precision'] = precision
            st.session_state['recall'] = recall
            st.session_state['f1'] = f1
            
            col1, col2, col3, col4 = st.columns(4)
            with col1: st.metric("Accuracy", f"{accuracy:.2f}")
            with col2: st.metric("Precision", f"{precision:.2f}")
            with col3: st.metric("Recall", f"{recall:.2f}")
            with col4: st.metric("F1 Score", f"{f1:.2f}")

            if y_pred_proba is not None:
                fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
                roc_auc = auc(fpr, tpr)
                st.metric("AUC", f"{roc_auc:.2f}")
                st.session_state['roc_auc'] = roc_auc
        else:
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            
            # Save metrics to session state
            st.session_state['rmse'] = rmse
            st.session_state['r2'] = r2
            
            col1, col2 = st.columns(2)
            with col1: st.metric("Root Mean Squared Error (RMSE)", f"{rmse:.2f}")
            with col2: st.metric("R-squared (R²)", f"{r2:.2f}")

        st.subheader("Predicted vs. Actual Values")
        results_df = pd.DataFrame({
            "Actual": y_test.values,
            "Predicted": y_pred
        })
        st.dataframe(results_df.head(), use_container_width=True)