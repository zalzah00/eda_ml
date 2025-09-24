import streamlit as st
import pandas as pd
import numpy as np
import json

def generate_notebook_content(session_state):
    """Generates the content for a Jupyter Notebook based on session state."""

    # Get data from session state
    target_col = session_state.get('target_col', 'N/A')
    selected_features = session_state.get('selected_features', [])
    model_name = session_state.get('selected_model_name', 'N/A')
    is_classification = session_state.get('is_classification', False)
    imputer_strategy = session_state.get('imputer_strategy', 'median')
    scaler_method = session_state.get('scaler_method', 'StandardScaler')
    handle_unknown_str = "True" if session_state.get('handle_unknown', True) else "False"
    test_size = session_state.get('test_size', 0.2)
    k_neighbors = session_state.get('k_neighbors', 5)

    # Convert features to a string representation that can be used in the notebook
    features_str = ', '.join(f"'{f}'" for f in selected_features)

    # --- Construct the notebook cells ---
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
            "source": """import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, f1_score, precision_score, recall_score, roc_curve, auc

# --- IMPORTANT: Change this path to your file location ---
file_path = 'your_data.csv' 
try:
    df = pd.read_csv(file_path)
except FileNotFoundError:
    print(f\"Error: The file at {file_path} was not found. Please update the path.\")
    df = None # Set df to None to prevent errors
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
    print(\"Data Head:\")
    print(df.head())
    print(\"\\nData Info:\")
    df.info()
    print(\"\\nDescriptive Statistics:\")
    print(df.describe(include='all'))
"""
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": "## 3. Machine Learning Pipeline"
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": f"""if df is not None:
    target_col = '{target_col}'
    selected_features = [{features_str}]

    X = df[selected_features]
    y = df[target_col]
    
    numerical_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Preprocessing pipelines based on app choices
    numerical_transformer_steps = [('imputer', SimpleImputer(strategy='{imputer_strategy}'))]
    if '{scaler_method}' == 'StandardScaler':
        numerical_transformer_steps.append(('scaler', StandardScaler()))
    elif '{scaler_method}' == 'MinMaxScaler':
        numerical_transformer_steps.append(('scaler', MinMaxScaler()))
    
    numerical_transformer = Pipeline(steps=numerical_transformer_steps)
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')), 
        ('onehot', OneHotEncoder(handle_unknown='ignore' if {handle_unknown_str} else 'error'))
    ])
    
    # Bundle preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    # Model selection based on app choice
    if '{model_name}' == 'Logistic Regression':
        model = LogisticRegression()
    elif '{model_name}' == 'K-Nearest Neighbors Classifier':
        model = KNeighborsClassifier(n_neighbors={k_neighbors})
    # Add other models here
    
    # Create the full pipeline
    full_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])
    
    print(\"Pipeline built successfully!\")
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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size={test_size}, random_state=42)
    
    full_pipeline.fit(X_train, y_train)
    y_pred = full_pipeline.predict(X_test)
    
    print(\"Model Training Complete!\")

    # Evaluation based on problem type
    if {str(is_classification).lower()}:
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        print(\"\\n--- Classification Results ---\")
        print(f\"Accuracy: {{accuracy:.2f}}\")
        print(f\"Precision: {{precision:.2f}}\")
        print(f\"Recall: {{recall:.2f}}\")
        print(f\"F1 Score: {{f1:.2f}}\")
    else:
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        print(\"\\n--- Regression Results ---\")
        print(f\"RMSE: {{rmse:.2f}}\")
        print(f\"R^2 Score: {{r2:.2f}}\")
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
    # --- Existing summary code ... ---
    st.subheader("Data & Analysis Overview")
    df_desc = pd.DataFrame({
        "Column Name": st.session_state['df'].columns,
        "Data Type": [str(dtype) for dtype in st.session_state['df'].dtypes]
    }).set_index('Column Name')
    
    st.markdown(f"**Loaded File:** `{st.session_state.get('file_name', 'N/A')}`")
    st.markdown(f"**Number of Rows:** `{st.session_state['df'].shape[0]}`")
    st.markdown(f"**Number of Columns:** `{st.session_state['df'].shape[1]}`")
    st.dataframe(df_desc.T)
    target_col = st.session_state.get('target_col', 'N/A')
    if target_col != 'N/A':
        st.markdown(f"**Selected Target Variable:** `{target_col}`")
        st.markdown(f"**Target Type:** `{st.session_state['df'][target_col].dtype}`")
    st.markdown("---")
    st.subheader("Machine Learning Pipeline")
    if 'selected_model_name' in st.session_state:
        st.markdown(f"**Selected Model:** `{st.session_state['selected_model_name']}`")
        st.markdown("**Preprocessing Options:**")
        st.write(f"- Imputation Strategy: `{st.session_state.get('imputer_strategy', 'N/A')}`")
        st.write(f"- Scaling Method: `{st.session_state.get('scaler_method', 'N/A')}`")
        st.write(f"- Ignore Unknown Categories: `{st.session_state.get('handle_unknown', 'N/A')}`")
        st.markdown(f"**Features Used:** `{st.session_state.get('selected_features', 'N/A')}`")
        st.markdown(f"**Train/Test Split:** `{100 - st.session_state.get('test_size', 0.2) * 100:.0f}% train / {st.session_state.get('test_size', 0.2) * 100:.0f}% test`")
        if 'k_neighbors' in st.session_state:
            st.markdown(f"**`k` Neighbors:** `{st.session_state['k_neighbors']}`")
        st.markdown("---")
        st.subheader("Results Summary")
        if 'is_classification' in st.session_state:
            if st.session_state['is_classification']:
                st.markdown(f"**Accuracy:** `{st.session_state.get('accuracy', 'N/A'):.2f}`")
                st.markdown(f"**Precision:** `{st.session_state.get('precision', 'N/A'):.2f}`")
                st.markdown(f"**Recall:** `{st.session_state.get('recall', 'N/A'):.2f}`")
                st.markdown(f"**F1 Score:** `{st.session_state.get('f1', 'N/A'):.2f}`")
                if 'roc_auc' in st.session_state:
                    st.markdown(f"**AUC:** `{st.session_state['roc_auc']:.2f}`")
            else:
                st.markdown(f"**RMSE:** `{st.session_state.get('rmse', 'N/A'):.2f}`")
                st.markdown(f"**R² Score:** `{st.session_state.get('r2', 'N/A'):.2f}`")
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