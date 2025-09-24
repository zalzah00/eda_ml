import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, auc

# Set a wide page layout
st.set_page_config(layout="wide")

# --- Helper Functions ---

def describe_dataframe(df):
    """
    Analyzes a DataFrame and returns a descriptive DataFrame for the app.
    """
    desc_df = []
    for col in df.columns:
        dtype = df[col].dtype
        missing_count = df[col].isnull().sum()
        unique_count = df[col].nunique()
        
        # Suggest type based on unique values and dtype
        if dtype in ['int64', 'float64']:
            if unique_count <= 20: # Threshold for categorical vs numerical
                suggested_type = 'Categorical (Nominal)'
            else:
                suggested_type = 'Numerical'
        else:
            suggested_type = 'Categorical (Nominal)'
            
        desc_df.append({
            'Feature': col,
            'Data Type': str(dtype),
            'Missing Values': missing_count,
            'Unique Values': unique_count,
            'Suggested Type': suggested_type
        })
    return pd.DataFrame(desc_df)

def get_plot_style():
    """Returns a dictionary for plot styling."""
    return {
        'font.family': 'sans-serif',
        'axes.facecolor': '#f0f2f6',
        'figure.facecolor': '#f0f2f6',
        'axes.edgecolor': '#9fa3a9',
        'grid.color': '#cccccc',
        'grid.linestyle': '--',
        'axes.spines.top': False,
        'axes.spines.right': False
    }

def generate_eda_plots(df, selected_features):
    """
    Generates and displays EDA plots based on selected features.
    """
    with st.spinner("Generating plots..."):
        # Create a copy to avoid modifying the original DataFrame
        plot_df = df[selected_features].copy()
        
        numerical_features = plot_df.select_dtypes(include=np.number).columns.tolist()
        categorical_features = plot_df.select_dtypes(include=['object', 'category']).columns.tolist()

        # Set plot style
        plt.style.use('fivethirtyeight')
        
        st.subheader("Distribution Plots (Numerical Features)")
        if numerical_features:
            for feature in numerical_features:
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.histplot(plot_df[feature].dropna(), kde=True, ax=ax, color='skyblue')
                ax.set_title(f'Distribution of {feature}', fontsize=16, fontweight='bold')
                st.pyplot(fig)
                plt.close(fig)
        else:
            st.info("No numerical features selected for distribution plots.")

        st.subheader("Correlation Heatmap")
        if len(numerical_features) > 1:
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(plot_df[numerical_features].corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
            ax.set_title('Correlation Heatmap', fontsize=16, fontweight='bold')
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info("Select at least two numerical features for a correlation heatmap.")
            
        st.subheader("Pairplot")
        if len(selected_features) > 1 and len(selected_features) <= 5:
            with st.spinner("Generating pairplot... This may take a moment."):
                fig = sns.pairplot(plot_df)
                st.pyplot(fig)
                plt.close(fig)
        else:
            st.info("Select between 2 and 5 features for a pairplot.")


def train_and_evaluate_model(df, target_column, selected_features, ml_algorithm):
    """
    Implements the full ML workflow with a pipeline.
    """
    with st.spinner(f"Training {ml_algorithm} model..."):
        # Drop rows with any missing data for simplicity
        df = df.dropna(subset=[target_column] + selected_features)
        
        X = df[selected_features]
        y = df[target_column]
        
        # Check if the problem is regression or classification
        # Simple heuristic: If the target has fewer than 10 unique values, treat as classification
        is_classification = y.nunique() <= 10
        
        # Define the numerical and categorical features
        numerical_features = X.select_dtypes(include=np.number).columns.tolist()
        categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Create a preprocessor pipeline for numerical features
        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])
        
        # Create a preprocessor pipeline for categorical features
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        # Create a ColumnTransformer to apply different transformations
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_features),
                ('cat', categorical_transformer, categorical_features)
            ])
            
        # Select the model based on the user's choice
        if ml_algorithm == 'Linear Regression':
            model = LinearRegression()
        elif ml_algorithm == 'Ridge Regression':
            model = Ridge()
        elif ml_algorithm == 'Lasso Regression':
            model = Lasso()
        elif ml_algorithm == 'Random Forest Regressor':
            model = RandomForestRegressor(n_estimators=100, random_state=42)
        elif ml_algorithm == 'KNN Regressor':
            model = KNeighborsRegressor()
        else:
            st.error("Selected algorithm is not supported.")
            return None

        # Create the final pipeline that includes preprocessing and the model
        full_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', model)
        ])

        # Split data and train the model
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        full_pipeline.fit(X_train, y_train)
        y_pred = full_pipeline.predict(X_test)
        
        # Evaluation based on problem type
        st.subheader("Model Evaluation")
        if is_classification:
            st.write("This is a classification problem.")
            y_pred_class = (y_pred > 0.5).astype(int) # simple threshold for binary
            
            # Display metrics
            st.metric("Accuracy", f"{accuracy_score(y_test, y_pred_class):.2f}")
            st.metric("Precision", f"{precision_score(y_test, y_pred_class, average='binary'):.2f}")
            st.metric("Recall", f"{recall_score(y_test, y_pred_class, average='binary'):.2f}")
            st.metric("F1 Score", f"{f1_score(y_test, y_pred_class, average='binary'):.2f}")
            
        else:
            st.write("This is a regression problem.")
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            
            # Display metrics
            st.metric("R² Score", f"{r2:.2f}")
            st.metric("RMSE", f"{rmse:.2f}")
            
            # Display a plot of actual vs predicted values
            results_df = pd.DataFrame({
                'Actual': y_test.values,
                'Predicted': y_pred
            }).reset_index(drop=True)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.scatterplot(x='Actual', y='Predicted', data=results_df, ax=ax, alpha=0.6)
            ax.plot([results_df['Actual'].min(), results_df['Actual'].max()],
                    [results_df['Actual'].min(), results_df['Actual'].max()],
                    'r--', lw=2, label='Perfect Prediction Line')
            ax.set_title('Actual vs Predicted Values', fontsize=16, fontweight='bold')
            ax.set_xlabel('Actual Values')
            ax.set_ylabel('Predicted Values')
            ax.legend()
            st.pyplot(fig)
            plt.close(fig)

    st.success("Model training and evaluation complete!")


# --- Main Application Logic ---

def main():
    st.title("Data Analysis & ML App")
    st.markdown("### A simple app to upload, explore, and predict on your data.")

    # Step 1: Data Upload
    st.header("1. Upload Your Data")
    uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=['csv', 'xlsx'])

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            st.session_state.df = df
            st.success("File uploaded successfully!")
            st.write("First 5 rows of your data:")
            st.dataframe(df.head())

        except Exception as e:
            st.error(f"Error reading file: {e}")
            st.session_state.df = None

    if 'df' in st.session_state and st.session_state.df is not None:
        df = st.session_state.df
        
        # Step 2: Data Exploration
        st.header("2. Data Exploration")
        st.subheader("Data Description")
        st.dataframe(describe_dataframe(df))

        # Step 3: Feature Selection and Visualization
        st.header("3. Feature Selection & Visualization")
        
        # User selects the target and features
        all_cols = df.columns.tolist()
        target_column = st.selectbox("Select the Target Column (y)", all_cols)
        feature_cols = [col for col in all_cols if col != target_column]
        selected_features = st.multiselect(
            "Select Features to include in Analysis (X)",
            options=feature_cols,
            default=feature_cols
        )
        
        if st.button("Generate Plots"):
            if not selected_features:
                st.warning("Please select at least one feature to generate plots.")
            else:
                generate_eda_plots(df, selected_features)
        
        # Step 4: Model Training
        st.header("4. Model Training")
        
        ml_algorithms = [
            "Linear Regression",
            "Ridge Regression",
            "Lasso Regression",
            "Random Forest Regressor",
            "KNN Regressor"
        ]
        
        model_choice = st.selectbox("Choose a Machine Learning Algorithm", ml_algorithms)
        
        if st.button(f"Train {model_choice} Model"):
            if not selected_features:
                st.error("Please select features to train the model.")
            else:
                train_and_evaluate_model(df, target_column, selected_features, model_choice)


if __name__ == '__main__':
    main()
