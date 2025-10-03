# pages/2_Visualizations.py

import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

if 'df' not in st.session_state or 'target_col' not in st.session_state or st.session_state['df'] is None:
    st.warning("Please upload data and select a target column on previous pages.")
else:
    st.header("3. Visualizations")
    df = st.session_state['df']
    target_col = st.session_state['target_col']

    feature_cols = [col for col in df.columns if col != target_col]
    
    # Check if a selection was previously made on this page or the next
    # Fallback to the first 5 features if no previous selection exists
    default_features = st.session_state.get('selected_features_viz', feature_cols[:5])
    
    # Filter the default list to ensure all columns exist in the current DataFrame
    # This prevents the StreamlitAPIException if the DataFrame changed.
    safe_default_features = [col for col in default_features if col in feature_cols]

    selected_features = st.multiselect("Select features to visualize", 
                                       options=feature_cols, 
                                       default=safe_default_features)
    
    # ⭐ ADD THIS LINE: Save the current selection to session state
    # Use a new key to avoid conflicts: 'selected_features_viz' for this page,
    # and 'selected_features_model' (or just 'selected_features') for the next page
    st.session_state['selected_features_viz'] = selected_features
    
    # Check if target is numerical for correlation analysis
    target_is_numerical = pd.api.types.is_numeric_dtype(df[target_col])

    if st.button("Generate Plots"):
        if not selected_features:
            st.error("Please select at least one feature.")
        else:
            # Create tabs for different types of visualizations
            tab1, tab2, tab3 = st.tabs(["Pair Plot", "Correlation Heatmap", "Distribution Plots"])
            
            with tab1:
                st.subheader("Pair Plot of Selected Features")
                
                # Include target if it's numerical for enhanced analysis
                if target_is_numerical:
                    plot_data = df[selected_features + [target_col]]
                    st.info("Target variable included in pair plot (numerical target detected)")
                else:
                    plot_data = df[selected_features]
                    st.info("Categorical target variable - using hue for coloring")
                
                numerical_features = plot_data.select_dtypes(include=np.number).columns
                
                if len(numerical_features) > 1:
                    # For categorical targets, use hue coloring
                    if not target_is_numerical and len(numerical_features) >= 2:
                        fig = sns.pairplot(plot_data, 
                                           vars=numerical_features, 
                                           hue=target_col,
                                           diag_kind='hist',
                                           palette='viridis')
                        plt.suptitle(f"Pair Plot with Target '{target_col}' as Hue", y=1.02)
                    else:
                        fig = sns.pairplot(plot_data, 
                                           vars=numerical_features,
                                           diag_kind='hist')
                        plt.suptitle("Pair Plot of Numerical Features", y=1.02)
                    
                    st.pyplot(fig)
                else:
                    st.info("Select at least two numerical features for a pair plot.")
            
            with tab2:
                st.subheader("Correlation Heatmap")
                
                # Select only numerical features for correlation
                numerical_data = df[selected_features].select_dtypes(include=np.number)
                
                # Include target if it's numerical
                if target_is_numerical:
                    numerical_data[target_col] = df[target_col]
                    st.info("Target variable included in correlation analysis")
                
                if len(numerical_data.columns) > 1:
                    # Calculate correlation matrix
                    corr_matrix = numerical_data.corr()
                    
                    # Create heatmap
                    fig, ax = plt.subplots(figsize=(10, 8))
                    
                    # Create mask for upper triangle (optional)
                    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
                    
                    # Plot heatmap
                    sns.heatmap(corr_matrix, 
                                mask=mask,
                                annot=True, 
                                cmap='coolwarm', 
                                center=0,
                                square=True, 
                                fmt='.2f',
                                cbar_kws={"shrink": .8},
                                ax=ax)
                    
                    plt.title('Correlation Heatmap')
                    plt.xticks(rotation=45, ha='right')
                    plt.yticks(rotation=0)
                    plt.tight_layout()
                    
                    st.pyplot(fig)
                    
                    # Interpretation guidance
                    st.markdown("""
                    **Interpretation Guide:**
                    - Values close to **+1**: Strong positive correlation
                    - Values close to **-1**: Strong negative correlation  
                    - Values close to **0**: Weak or no linear correlation
                    """)
                    
                    # Highlight strong correlations with target if included
                    if target_is_numerical and target_col in corr_matrix.columns:
                        st.subheader("Feature Correlations with Target")
                        target_correlations = corr_matrix[target_col].drop(target_col).sort_values(key=abs, ascending=False)
                        
                        corr_df = pd.DataFrame({
                            'Feature': target_correlations.index,
                            'Correlation with Target': target_correlations.values
                        })
                        
                        st.dataframe(corr_df, use_container_width=True)
                else:
                    st.info("Need at least two numerical features for correlation heatmap.")
            
            with tab3:
                st.subheader("Distribution Plots")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Target distribution
                    st.write(f"**Distribution of Target: {target_col}**")
                    fig, ax = plt.subplots(figsize=(8, 4))
                    
                    if target_is_numerical:
                        df[target_col].hist(bins=30, ax=ax, alpha=0.7)
                        ax.set_xlabel(target_col)
                        ax.set_ylabel('Frequency')
                        ax.set_title(f'Distribution of {target_col}')
                    else:
                        df[target_col].value_counts().plot(kind='bar', ax=ax, alpha=0.7)
                        ax.set_xlabel(target_col)
                        ax.set_ylabel('Count')
                        ax.set_title(f'Distribution of {target_col}')
                        plt.xticks(rotation=45)
                    
                    st.pyplot(fig)
                
                with col2:
                    # Feature distributions
                    if selected_features:
                        feature_to_plot = st.selectbox("Select feature to view distribution:", 
                                                       options=selected_features)
                        
                        fig, ax = plt.subplots(figsize=(8, 4))
                        
                        if pd.api.types.is_numeric_dtype(df[feature_to_plot]):
                            df[feature_to_plot].hist(bins=30, ax=ax, alpha=0.7)
                            ax.set_xlabel(feature_to_plot)
                            ax.set_ylabel('Frequency')
                            ax.set_title(f'Distribution of {feature_to_plot}')
                        else:
                            # Use .head(10) to prevent bar charts with too many categories
                            df[feature_to_plot].value_counts().head(10).plot(kind='bar', ax=ax, alpha=0.7)
                            ax.set_xlabel(feature_to_plot)
                            ax.set_ylabel('Count')
                            ax.set_title(f'Distribution of {feature_to_plot} (Top 10)')
                            plt.xticks(rotation=45)
                        
                        st.pyplot(fig)
            
            # Additional visualization: Feature vs Target scatter plots for numerical targets
            if target_is_numerical:
                st.subheader("Feature vs Target Relationships (Numerical Only)")
                
                # Select numerical features only for scatter plots
                numerical_features_for_scatter = [f for f in selected_features 
                                                  if pd.api.types.is_numeric_dtype(df[f])]
                
                if numerical_features_for_scatter:
                    # Create a grid of scatter plots
                    n_plots = len(numerical_features_for_scatter)
                    n_cols = 3
                    n_rows = (n_plots + n_cols - 1) // n_cols
                    
                    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
                    axes = axes.flatten() if n_plots > 1 else [axes]
                    
                    for idx, feature in enumerate(numerical_features_for_scatter):
                        if idx < len(axes):
                            axes[idx].scatter(df[feature], df[target_col], alpha=0.5)
                            axes[idx].set_xlabel(feature)
                            axes[idx].set_ylabel(target_col)
                            axes[idx].set_title(f'{feature} vs {target_col}')
                            
                            # Add correlation coefficient
                            corr = df[feature].corr(df[target_col])
                            axes[idx].annotate(f'r = {corr:.2f}', 
                                               xy=(0.05, 0.95), 
                                               xycoords='axes fraction',
                                               bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
                    
                    # Hide empty subplots
                    for idx in range(len(numerical_features_for_scatter), len(axes)):
                        axes[idx].set_visible(False)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                else:
                    st.info("No numerical features selected for scatter plots with target.")
