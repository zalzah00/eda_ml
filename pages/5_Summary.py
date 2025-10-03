# pages/5_Summary.py

import streamlit as st
import pandas as pd
import numpy as np
import json

def generate_notebook_content(session_state):
    """Generates the content for a Jupyter Notebook based on ACTUAL pipeline configuration."""
    
    # Get the actual pipeline configuration from session state
    pipeline_config = session_state.get('pipeline_config', {})
    metrics = session_state.get('metrics', {})
    file_name = session_state.get('file_name', 'your_data.csv')
    
    if not pipeline_config:
        return json.dumps({"cells": [{"cell_type": "markdown", "source": "# No pipeline configuration found"}]})

    # Extract configuration
    problem_type = pipeline_config.get('problem_type', 'Unknown')
    model_name = pipeline_config.get('model_name', 'Unknown Model')
    model_class = pipeline_config.get('model_class', 'Unknown')
    model_params = pipeline_config.get('model_params', {})
    numerical_features = pipeline_config.get('numerical_features', [])
    categorical_features = pipeline_config.get('categorical_features', [])
    test_size = pipeline_config.get('test_size', 0.2)
    selected_features = pipeline_config.get('selected_features', [])
    target_column = pipeline_config.get('target_column', 'target')
    random_state = pipeline_config.get('random_state', 42)
    
    # Convert features to string representations
    features_str = ', '.join(f"'{f}'" for f in selected_features)
    numerical_features_str = ', '.join(f"'{f}'" for f in numerical_features)
    categorical_features_str = ', '.join(f"'{f}'" for f in categorical_features)
    
    # Model parameters string (filter out empty params)
    model_params_str = ', '.join([f"{key}={repr(value)}" for key, value in model_params.items() if value is not None])
    
    # --- Construct the notebook cells based on ACTUAL configuration ---
    notebook_cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": "# Analysis Summary\n\nThis notebook reproduces the analysis performed in the Streamlit app. It contains the code for data loading, preprocessing, model training, and evaluation."
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": "## 1. Data Loading"
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": f"""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, mean_absolute_error

# Configuration from Streamlit app
file_name = '{file_name}'
target_column = '{target_column}'
selected_features = [{features_str}]
test_size = {test_size}
random_state = {random_state}

# Load data
try:
    df = pd.read_csv(file_name)
    print(f"Data loaded successfully from {{file_name}}")
    print(f"Data shape: {{df.shape}}")
except FileNotFoundError:
    print(f"Error: The file '{{file_name}}' was not found. Please update the file_name variable.")
    df = None
"""
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": "## 2. Data Exploration"
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": """if df is not None:
    print("Data Head:")
    print(df.head())
    print("\\nData Info:")
    df.info()
    print("\\nDescriptive Statistics:")
    print(df.describe(include='all'))
"""
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": f"## 3. Machine Learning Pipeline\n\n**Problem Type:** {problem_type}  \n**Model:** {model_name}  \n**Target:** {target_column}"
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": f"""if df is not None:
    # Prepare data (drop rows with missing values as done in the app)
    data_for_model = df[[target_column] + selected_features].dropna()
    X = data_for_model[selected_features]
    y = data_for_model[target_column]
    
    print(f"Data after dropping missing values: {{X.shape[0]}} rows")
    
    # Feature types
    numerical_features = [{numerical_features_str}]
    categorical_features = [{categorical_features_str}]
    
    print(f"Numerical features: {{numerical_features}}")
    print(f"Categorical features: {{categorical_features}}")
    
    # Preprocessing pipelines
    numerical_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # Bundle preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    # Model selection based on actual model class
    model_class = '{model_class}'
    {model_params_str}
    
    if model_class == 'LogisticRegression':
        model = LogisticRegression({model_params_str})
    elif model_class == 'LinearRegression':
        model = LinearRegression({model_params_str})
    elif model_class == 'Ridge':
        model = Ridge({model_params_str})
    elif model_class == 'Lasso':
        model = Lasso({model_params_str})
    elif model_class == 'DecisionTreeClassifier':
        model = DecisionTreeClassifier({model_params_str})
    elif model_class == 'DecisionTreeRegressor':
        model = DecisionTreeRegressor({model_params_str})
    elif model_class == 'RandomForestClassifier':
        model = RandomForestClassifier({model_params_str})
    elif model_class == 'RandomForestRegressor':
        model = RandomForestRegressor({model_params_str})
    else:
        raise ValueError(f"Unknown model class: {{model_class}}")
    
    # Create the full pipeline
    full_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])
    
    print("Pipeline built successfully!")
    print(f"Problem Type: {problem_type}")
    print(f"Model: {model_name}")
    print(f"Model Class: {{model_class}}")
"""
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": "## 4. Model Training and Evaluation"
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": f"""if df is not None:
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    print(f"Training set: {{X_train.shape[0]}} rows")
    print(f"Test set: {{X_test.shape[0]}} rows")
    
    # Train model
    full_pipeline.fit(X_train, y_train)
    
    # Make predictions
    y_train_pred = full_pipeline.predict(X_train)
    y_test_pred = full_pipeline.predict(X_test)
    
    print("Model Training Complete!")

    # Evaluation based on problem type
    if '{problem_type.lower()}' == 'classification':
        train_accuracy = accuracy_score(y_train, y_train_pred)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        
        print("\\n--- Classification Results ---")
        print(f"Training Accuracy: {{train_accuracy:.4f}}")
        print(f"Test Accuracy: {{test_accuracy:.4f}}")
        
    else:
        train_mae = mean_absolute_error(y_train, y_train_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        test_r2 = r2_score(y_test, y_test_pred)
        
        print("\\n--- Regression Results ---")
        print(f"Training MAE: {{train_mae:.4f}}")
        print(f"Test MAE: {{test_mae:.4f}}")
        print(f"Test RMSE: {{test_rmse:.4f}}")
        print(f"Test R²: {{test_r2:.4f}}")
        
        # Plot actual vs predicted
        plt.figure(figsize=(8, 6))
        plt.scatter(y_test, y_test_pred, alpha=0.6)
        min_val = min(y_test.min(), y_test_pred.min())
        max_val = max(y_test.max(), y_test_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title('Actual vs Predicted Values')
        plt.show()
"""
        },
        {
            "cell_type": "markdown", 
            "metadata": {},
            "source": "## 5. Expected Results\n\nBased on the Streamlit app analysis, you should see results similar to:"
        },
        {
            "cell_type": "code",
            "execution_count": None, 
            "metadata": {},
            "outputs": [],
            "source": f"""print("Expected Results from Streamlit App:")
print("Problem Type: {problem_type}")
print("Model: {model_name}")

if '{problem_type.lower()}' == 'classification':
    print(f"Training Accuracy: {metrics.get('train_accuracy', 'N/A')}")
    print(f"Test Accuracy: {metrics.get('test_accuracy', 'N/A')}")
else:
    print(f"Training MAE: {metrics.get('train_mae', 'N/A')}")
    print(f"Test MAE: {metrics.get('test_mae', 'N/A')}") 
    print(f"Test RMSE: {metrics.get('test_rmse', 'N/A')}")
    print(f"Test R²: {metrics.get('test_r2', 'N/A')}")
"""
        }
    ]

    # Combine all parts into a valid Jupyter Notebook JSON structure
    notebook = {
        "cells": notebook_cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {
                    "name": "ipython",
                    "version": 3
                },
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.8.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    return json.dumps(notebook)

# --- Start of the Streamlit page content ---
st.header("6. Analysis Summary")
st.info("A review of all the options and results from your analysis.")

if 'df' not in st.session_state:
    st.warning("Please upload a data file on the Home page to see the summary.")
else:
    # --- Data Overview ---
    st.subheader("Data & Analysis Overview")
    df_desc = pd.DataFrame({
        "Column Name": st.session_state['df'].columns,
        "Data Type": [str(dtype) for dtype in st.session_state['df'].dtypes]
    }).set_index('Column Name')
    
    file_name = st.session_state.get('file_name', 'N/A')
    st.markdown(f"**Loaded File:** `{file_name}`")
    st.markdown(f"**Number of Rows:** `{st.session_state['df'].shape[0]}`")
    st.markdown(f"**Number of Columns:** `{st.session_state['df'].shape[1]}`")
    st.dataframe(df_desc.T)
    
    target_col = st.session_state.get('target_col', 'N/A')
    if target_col != 'N/A':
        st.markdown(f"**Selected Target Variable:** `{target_col}`")
        st.markdown(f"**Target Type:** `{st.session_state['df'][target_col].dtype}`")
    
    # --- Pipeline Configuration ---
    st.markdown("---")
    st.subheader("Machine Learning Pipeline")
    
    pipeline_config = st.session_state.get('pipeline_config', {})
    if pipeline_config:
        st.markdown(f"**Problem Type:** `{pipeline_config.get('problem_type', 'N/A')}`")
        st.markdown(f"**Selected Model:** `{pipeline_config.get('model_name', 'N/A')}`")
        st.markdown("**Preprocessing Configuration:**")
        st.write(f"- Numerical Features: `{pipeline_config.get('numerical_features', [])}`")
        st.write(f"- Categorical Features: `{pipeline_config.get('categorical_features', [])}`")
        st.write(f"- Scaler: `{pipeline_config.get('numerical_transformer', {}).get('scaler_type', 'StandardScaler')}`")
        st.write(f"- Encoder: `{pipeline_config.get('categorical_transformer', {}).get('encoder_type', 'OneHotEncoder')}`")
        st.write(f"- Handle Unknown: `{pipeline_config.get('categorical_transformer', {}).get('handle_unknown', 'ignore')}`")
        st.write(f"- Imputation: `{pipeline_config.get('imputation_strategy', 'dropna_pre_split')}`")
        st.markdown(f"**Features Used:** `{pipeline_config.get('selected_features', [])}`")
        st.markdown(f"**Train/Test Split:** `{100 - pipeline_config.get('test_size', 0.2) * 100:.0f}% train / {pipeline_config.get('test_size', 0.2) * 100:.0f}% test`")
        st.markdown(f"**Random State:** `{pipeline_config.get('random_state', 42)}`")
        
        # --- Results Summary ---
        st.markdown("---")
        st.subheader("Results Summary")
        
        metrics = st.session_state.get('metrics', {})
        if pipeline_config.get('problem_type') == 'Classification':
            st.markdown(f"**Training Accuracy:** `{metrics.get('train_accuracy', 'N/A')}`")
            st.markdown(f"**Test Accuracy:** `{metrics.get('test_accuracy', 'N/A')}`")
        else:
            st.markdown(f"**Training MAE:** `{metrics.get('train_mae', 'N/A')}`")
            st.markdown(f"**Test MAE:** `{metrics.get('test_mae', 'N/A')}`")
            st.markdown(f"**Test RMSE:** `{metrics.get('test_rmse', 'N/A')}`")
            st.markdown(f"**Test R²:** `{metrics.get('test_r2', 'N/A')}`")
    else:
        st.info("No machine learning model has been trained yet. Please navigate to Page 4 to run an analysis.")

    # --- Export Button ---
    st.markdown("---")
    st.subheader("Export Analysis")
    st.download_button(
        label="Export to Jupyter Notebook",
        data=generate_notebook_content(st.session_state),
        file_name='analysis_summary.ipynb',
        mime="application/x-ipynb+json"
    )
