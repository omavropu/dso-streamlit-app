import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import statsmodels.api as sm
import matplotlib.pyplot as plt

# --- Page Configuration ---
st.set_page_config(
    page_title="DSO Prediction and Simulation",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Helper Functions ---
def get_performance_metrics(y_true, y_pred):
    """Calculates and returns a dictionary of performance metrics."""
    metrics = {
        "R-squared": r2_score(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAE": mean_absolute_error(y_true, y_pred)
    }
    return metrics

def train_xgb_model(X_train, y_train, params):
    """Trains an XGBoost model and returns it."""
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=params['n_estimators'],
        max_depth=params['max_depth'],
        learning_rate=params['learning_rate'],
        random_state=42
    )
    model.fit(X_train, y_train)
    return model
    
def get_clustered_ses(df_train, features, target, cluster_col):
    """Fits an OLS model and returns results with clustered standard errors."""
    try:
        # Add a constant for the intercept
        X = sm.add_constant(df_train[features])
        y = df_train[target]
        
        # Fit the model
        model = sm.OLS(y, X).fit(cov_type='cluster', cov_kwds={'groups': df_train[cluster_col]})
        return model.summary()
    except Exception as e:
        return f"Could not compute clustered standard errors: {e}"


# --- App State Initialization ---
if 'data' not in st.session_state:
    st.session_state.data = None
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'xgb_model' not in st.session_state:
    st.session_state.xgb_model = None
if 'lin_model' not in st.session_state:
    st.session_state.lin_model = None
if 'X_test' not in st.session_state:
    st.session_state.X_test = None
if 'y_test' not in st.session_state:
    st.session_state.y_test = None
if 'features' not in st.session_state:
    st.session_state.features = []


# --- Sidebar ---
with st.sidebar:
    st.title("⚙️ Configuration")
    st.markdown("Upload your data and configure the model parameters.")

    uploaded_file = st.file_uploader("Upload your CSV data", type=["csv"])

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            # Clean column names
            df.columns = [col.replace('.', '').replace('_', ' ').strip() for col in df.columns]
            st.session_state.data = df
            st.success("Data loaded successfully!")
        except Exception as e:
            st.error(f"Error loading file: {e}")
            st.session_state.data = None
    else:
        st.info("Using default simulated data. Upload your own CSV to analyze.")
        if st.session_state.data is None:
            # Generate sample data if none is loaded
            np.random.seed(42)
            n_samples = 200
            simulated_data = {
                'Customer ID': [f'CUST_{i}' for i in range(1, n_samples + 1)],
                'Payment Terms Days': np.random.choice([30, 60, 90, 120], size=n_samples, p=[0.4, 0.3, 0.2, 0.1]),
                'Invoice Error Rate': np.random.uniform(0.01, 0.15, size=n_samples),
                'Forecast Accuracy': np.random.uniform(0.75, 0.99, size=n_samples),
                'Contract Extension Days': np.random.choice([0, 15, 30], size=n_samples, p=[0.8, 0.15, 0.05]),
                'Avg Days Late Last3 Days': np.random.poisson(lam=5, size=n_samples)
            }
            sim_df = pd.DataFrame(simulated_data)
            sim_df['DSO actual Days'] = (
                sim_df['Payment Terms Days'] + sim_df['Avg Days Late Last3 Days'] +
                (sim_df['Invoice Error Rate'] * 100) + ((1 - sim_df['Forecast Accuracy']) * 50) +
                (sim_df['Contract Extension Days'] * 0.5) + np.random.normal(0, 5, size=n_samples)
            )
            sim_df['DSO actual Days'] = sim_df['DSO actual Days'].apply(lambda x: max(0, x)).round(1)
            st.session_state.data = sim_df

    if st.session_state.data is not None:
        df = st.session_state.data
        all_cols = df.columns.tolist()

        st.subheader("Variable Selection")
        target_variable = st.selectbox("Select Target Variable (DSO)", all_cols, index=len(all_cols)-1 if all_cols else 0)
        
        default_features = [
            'Payment Terms Days', 'Invoice Error Rate', 'Forecast Accuracy',
            'Contract Extension Days', 'Avg Days Late Last3 Days'
        ]
        
        available_features = [col for col in all_cols if col != target_variable and col != 'Customer ID']
        
        # Filter default_features to only those present in the uploaded data
        valid_default_features = [f for f in default_features if f in available_features]

        features = st.multiselect("Select Feature Variables", available_features, default=valid_default_features)
        st.session_state.features = features


        st.subheader("Model Parameters")
        test_size = st.slider("Test Set Size", 0.1, 0.5, 0.2, 0.05)
        
        with st.expander("XGBoost Hyperparameters"):
            n_estimators = st.slider("Number of Estimators (n_estimators)", 50, 500, 100, 10)
            max_depth = st.slider("Max Depth", 3, 15, 5, 1)
            learning_rate = st.slider("Learning Rate", 0.01, 0.3, 0.1, 0.01)
        
        xgb_params = {'n_estimators': n_estimators, 'max_depth': max_depth, 'learning_rate': learning_rate}

        if st.button("🚀 Train Models", use_container_width=True):
            with st.spinner("Training models... This may take a moment."):
                if len(features) > 0:
                    # Prepare data
                    df_clean = df.dropna(subset=features + [target_variable])
                    X = df_clean[features]
                    y = df_clean[target_variable]
                    
                    # Store data for OLS with clustered SEs
                    df_train_ols = df_clean.loc[X.index]


                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
                    st.session_state.X_train_df = X_train
                    st.session_state.df_train_ols = df_train_ols.loc[X_train.index]


                    # Train XGBoost
                    st.session_state.xgb_model = train_xgb_model(X_train, y_train, xgb_params)

                    # Train Linear Regression
                    lin_model = LinearRegression()
                    lin_model.fit(X_train, y_train)
                    st.session_state.lin_model = lin_model

                    # Store for later use
                    st.session_state.X_test = X_test
                    st.session_state.y_test = y_test
                    st.session_state.model_trained = True
                else:
                    st.warning("Please select at least one feature variable.")
            st.success("Models trained successfully!")

# --- Main Page ---
st.title("💼 Days Sales Outstanding (DSO) Analysis")
st.markdown("An interactive tool to predict DSO, understand its key drivers, and simulate the impact of business decisions.")

if st.session_state.data is None:
    st.warning("Please upload a CSV file or use the default data via the sidebar to begin.")
else:
    df = st.session_state.data
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Exploratory Data Analysis", "🤖 Model Performance", "🧠 Prediction Explanations", "🔮 DSO Simulation"])

    with tab1:
        st.header("Exploratory Data Analysis")
        st.markdown("A first look at your data.")
        
        st.subheader("Data Preview")
        st.dataframe(df.head())

        st.subheader("Summary Statistics")
        st.dataframe(df.describe())

        if len(st.session_state.features) > 0:
            st.subheader("Visualizations")
            
            # Create two columns for charts
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### Feature Distributions")
                feature_to_plot = st.selectbox("Select a feature to see its distribution", st.session_state.features)
                fig_hist = px.histogram(df, x=feature_to_plot, marginal="box", title=f"Distribution of {feature_to_plot}", color_discrete_sequence=px.colors.qualitative.Pastel)
                st.plotly_chart(fig_hist, use_container_width=True)
            
            with col2:
                st.markdown(f"#### Relationship with DSO")
                fig_scatter = px.scatter(df, x=feature_to_plot, y=target_variable, trendline="ols", title=f"{feature_to_plot} vs. {target_variable}", color_discrete_sequence=px.colors.qualitative.Pastel1)
                st.plotly_chart(fig_scatter, use_container_width=True)

            st.markdown("#### Correlation Heatmap")
            corr_df = df[st.session_state.features + [target_variable]]
            corr_matrix = corr_df.corr()
            fig_heatmap = go.Figure(data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                colorscale='Viridis',
                colorbar=dict(title='Correlation')
            ))
            fig_heatmap.update_layout(title="Feature Correlation Matrix")
            st.plotly_chart(fig_heatmap, use_container_width=True)


    with tab2:
        st.header("Model Performance")
        if not st.session_state.model_trained:
            st.info("Train the models in the sidebar to see performance metrics.")
        else:
            y_pred_xgb = st.session_state.xgb_model.predict(st.session_state.X_test)
            y_pred_lin = st.session_state.lin_model.predict(st.session_state.X_test)

            metrics_xgb = get_performance_metrics(st.session_state.y_test, y_pred_xgb)
            metrics_lin = get_performance_metrics(st.session_state.y_test, y_pred_lin)
            
            st.subheader("Performance on Test Set")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### 🌳 XGBoost Model")
                for name, val in metrics_xgb.items():
                    st.metric(label=name, value=f"{val:.3f}")

            with col2:
                st.markdown("#### 📈 Linear Regression")
                for name, val in metrics_lin.items():
                    st.metric(label=name, value=f"{val:.3f}")
            
            st.subheader("Feature Importance (XGBoost)")
            importance = pd.DataFrame({
                'feature': st.session_state.features,
                'importance': st.session_state.xgb_model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            fig_importance = px.bar(importance, x='importance', y='feature', orientation='h', title="Feature Importance")
            fig_importance.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig_importance, use_container_width=True)
            
            st.subheader("Linear Regression Coefficients")
            st.markdown("Coefficients from a standard OLS model. For robust inference, especially with grouped data (like customers), using clustered standard errors is recommended.")
            
            cluster_col = st.selectbox("Select a column for clustering standard errors (e.g., Customer ID)", df.columns)
            if cluster_col:
                with st.spinner("Calculating Clustered Standard Errors..."):
                    summary = get_clustered_ses(st.session_state.df_train_ols, st.session_state.features, target_variable, cluster_col)
                    st.text(summary)


    with tab3:
        st.header("Prediction Explanations (SHAP)")
        if not st.session_state.model_trained:
            st.info("Train the models in the sidebar to generate explanations.")
        else:
            explainer = shap.Explainer(st.session_state.xgb_model)
            shap_values = explainer(st.session_state.X_test)
            
            st.subheader("Global Feature Impact")
            st.markdown("The SHAP summary plot shows the impact of each feature on the model's output. Each point is a single observation. Red means a high feature value, blue means low.")
            st.pyplot(shap.summary_plot(shap_values, st.session_state.X_test, show=False))
            plt.gcf().set_size_inches(10, 5) # Adjust figure size
            plt.tight_layout() # Adjust layout
            
            st.subheader("Individual Prediction Breakdown")
            st.markdown("Select a single observation from the test set to see how the model arrived at its prediction.")
            
            observation_index = st.slider("Select an observation index", 0, len(st.session_state.X_test)-1, 0, 1)
            
            st.markdown(f"**Explaining Observation {observation_index}**")
            
            # Force plot
            fig, ax = plt.subplots(nrows=1, ncols=1)
            shap.force_plot(explainer.expected_value, shap_values.values[observation_index,:], st.session_state.X_test.iloc[observation_index,:], show=False, matplotlib=True)
            st.pyplot(fig, bbox_inches='tight')
            plt.close(fig)


            # Display actual vs predicted
            actual_val = st.session_state.y_test.iloc[observation_index]
            predicted_val = st.session_state.xgb_model.predict(st.session_state.X_test.iloc[[observation_index]])[0]
            
            col1, col2 = st.columns(2)
            col1.metric("Actual DSO", f"{actual_val:.2f} days")
            col2.metric("Predicted DSO", f"{predicted_val:.2f} days")

            # Show feature values for the selected observation
            with st.expander("View feature values for this observation"):
                st.dataframe(st.session_state.X_test.iloc[[observation_index]])


    with tab4:
        st.header("DSO Simulation Tool")
        if not st.session_state.model_trained:
            st.info("Train the models in the sidebar to run simulations.")
        else:
            st.markdown("Use the sliders to simulate changes to business drivers and see the potential impact on the average predicted DSO for the test set.")
            
            # Make a copy of the test data for modification
            X_test_modified = st.session_state.X_test.copy()
            
            st.subheader("Simulation Controls")
            
            simulation_cols = st.columns(3)
            col_idx = 0

            # Create sliders for each feature
            for feature in st.session_state.features:
                with simulation_cols[col_idx % 3]:
                    min_val = X_test_modified[feature].min()
                    max_val = X_test_modified[feature].max()
                    mean_val = X_test_modified[feature].mean()
                    
                    if min_val < 1 and max_val <= 1: # Likely a rate or percentage
                        change = st.slider(f"Change {feature} (pp)", -0.25, 0.25, 0.0, 0.01, key=f"sim_{feature}")
                        X_test_modified[feature] += change
                        X_test_modified[feature] = np.clip(X_test_modified[feature], 0, 1)
                    else: # Likely a count or day value
                        change = st.slider(f"Change {feature} (days)", -int(mean_val), int(mean_val), 0, 1, key=f"sim_{feature}")
                        X_test_modified[feature] += change
                        X_test_modified[feature] = np.clip(X_test_modified[feature], 0, None)
                col_idx += 1
            
            st.subheader("Simulation Results")

            # Predict on original and modified data
            pred_before = st.session_state.xgb_model.predict(st.session_state.X_test)
            pred_after = st.session_state.xgb_model.predict(X_test_modified)

            avg_before = np.mean(pred_before)
            avg_after = np.mean(pred_after)
            avg_impact = avg_after - avg_before
            
            res_col1, res_col2, res_col3 = st.columns(3)
            res_col1.metric("Original Avg. Predicted DSO", f"{avg_before:.2f} days")
            res_col2.metric("Simulated Avg. Predicted DSO", f"{avg_after:.2f} days", delta=f"{avg_impact:.2f} days")
            
            if abs(avg_impact) > 0.01:
                if avg_impact < 0:
                    st.success(f"**This combination of changes could reduce the average DSO by {abs(avg_impact):.2f} days.**")
                else:
                    st.warning(f"**This combination of changes could increase the average DSO by {avg_impact:.2f} days.**")
            else:
                st.info("No significant change in average DSO with current simulation settings.")