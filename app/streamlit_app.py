import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shap
import seaborn as sns

# Fix imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

st.set_page_config(page_title='X-Maintain', page_icon='🔧', layout='wide')

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

@st.cache_resource
def load_models():
    try:
        model_path = os.path.join(BASE_DIR, 'models', 'best_model.joblib')
        if not os.path.exists(model_path):
            model_path = os.path.join(BASE_DIR, 'models', 'model.pkl')
        model = joblib.load(model_path)
        
        preprocessor = joblib.load(os.path.join(BASE_DIR, 'models', 'preprocessor.joblib'))
        
        with open(os.path.join(BASE_DIR, 'models', 'feature_names.json'), 'r') as f:
            feature_names = json.load(f)
            
        return model, preprocessor, feature_names
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None

@st.cache_data
def load_data():
    try:
        X_train = pd.read_csv(os.path.join(BASE_DIR, 'data', 'processed', 'X_train.csv'))
        return X_train
    except Exception as e:
        return None

@st.cache_resource
def get_explainer(_model, _X_train):
    try:
        explainer = shap.TreeExplainer(_model)
        return explainer
    except Exception as e:
        return None

model, preprocessor, feature_names = load_models()
X_train = load_data()
if model is not None and X_train is not None:
    explainer = get_explainer(model, X_train)
else:
    explainer = None

def preprocess_input(input_dict, preprocessor):
    df = pd.DataFrame([input_dict])
    
    # Preprocessor expects brackets
    expected_cols = [
        'Air temperature [K]', 'Process temperature [K]', 
        'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]', 
        'Type_H', 'Type_L', 'Type_M'
    ]
    
    for col in expected_cols:
        if col not in df.columns:
            df[col] = False if col.startswith('Type_') else 0.0
            
    df = df[expected_cols]
    
    transformed = preprocessor.transform(df)
    
    # Rename to parentheses for model
    model_cols = [
        'Air temperature (K)', 'Process temperature (K)', 
        'Rotational speed (rpm)', 'Torque (Nm)', 'Tool wear (min)', 
        'Type_H', 'Type_L', 'Type_M'
    ]
    
    transformed_df = pd.DataFrame(transformed, columns=model_cols)
    return transformed_df

def plot_waterfall(shap_values, max_display=10):
    fig = plt.figure(figsize=(10, 6))
    shap.plots.waterfall(shap_values, max_display=max_display, show=False)
    plt.tight_layout()
    return fig

# --- SIDEBAR NAVIGATION ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", [
    "Home", 
    "Prediction", 
    "Explainability", 
    "What-If Analysis", 
    "Model Performance", 
    "Dataset Insights"
])

if page == "Home":
    st.title("X-Maintain")
    st.subheader("Explainable AI for Predictive Maintenance")
    st.markdown("""
    Welcome to the X-Maintain dashboard. This system utilizes advanced Machine Learning models to predict equipment failures before they happen, allowing for timely maintenance and reducing downtime.
    
    **Key Highlights of the System:**
    - **Accurate Predictions**: Employs robust ML algorithms like XGBoost.
    - **Explainability**: Understand *why* the model makes a prediction using SHAP (SHapley Additive exPlanations).
    - **What-If Analysis**: Explore how changes in operating conditions affect failure probability.
    - **Comprehensive Insights**: Explore dataset statistics and model performance metrics.
    """)

elif page == "Prediction":
    st.title("Predictive Maintenance")
    
    st.write("Enter the operating conditions to predict the failure risk.")
    
    col1, col2 = st.columns(2)
    with col1:
        product_type = st.selectbox("Product Type", ["L", "M", "H"])
        air_temp = st.slider("Air Temperature (K)", 295.0, 305.0, 300.0, step=0.1)
        process_temp = st.slider("Process Temperature (K)", 305.0, 315.0, 310.0, step=0.1)
    with col2:
        rot_speed = st.slider("Rotational Speed (rpm)", 1100, 2900, 1500, step=10)
        torque = st.slider("Torque (Nm)", 3.0, 80.0, 40.0, step=0.5)
        tool_wear = st.slider("Tool Wear (min)", 0, 260, 100, step=1)
        
    if st.button("PREDICT FAILURE RISK", type="primary", use_container_width=True):
        if model is None or preprocessor is None:
            st.error("Model or preprocessor not loaded.")
        else:
            input_data = {
                'Air temperature [K]': air_temp,
                'Process temperature [K]': process_temp,
                'Rotational speed [rpm]': rot_speed,
                'Torque [Nm]': torque,
                'Tool wear [min]': tool_wear,
                'Type_H': product_type == 'H',
                'Type_L': product_type == 'L',
                'Type_M': product_type == 'M'
            }
            
            try:
                processed_df = preprocess_input(input_data, preprocessor)
                prob = model.predict_proba(processed_df)[0][1]
                
                st.markdown("---")
                
                res_col1, res_col2 = st.columns([1, 2])
                
                with res_col1:
                    if prob < 0.3:
                        risk_level = "Low"
                        color = "green"
                    elif prob < 0.7:
                        risk_level = "Medium"
                        color = "orange"
                    else:
                        risk_level = "High"
                        color = "red"
                        
                    st.markdown(f"### Risk Level: <span style='color:{color}'>{risk_level}</span>", unsafe_allow_html=True)
                    st.metric("Failure Probability", f"{prob:.2%}")
                    
                    st.markdown(f"**Risk Category:** {risk_level}")
                    
                with res_col2:
                    if explainer is not None:
                        st.subheader("Why did the model predict this?")
                        shap_vals = explainer(processed_df)
                        # Handle 3D output for some classifiers
                        if len(shap_vals.shape) == 3:
                            sv = shap_vals[0, :, 1]
                        else:
                            sv = shap_vals[0]
                            
                        fig = plot_waterfall(sv)
                        st.pyplot(fig)
                        
                        st.markdown("#### Feature Contributions")
                        contributions = pd.DataFrame({
                            'Feature': processed_df.columns,
                            'Value': processed_df.iloc[0].values,
                            'SHAP Value': sv.values
                        }).sort_values('SHAP Value', ascending=False)
                        
                        st.dataframe(contributions.style.background_gradient(cmap='RdYlGn_r', subset=['SHAP Value']))
                        
                        # Natural language explanation
                        top_positive = contributions[contributions['SHAP Value'] > 0].head(2)
                        top_negative = contributions[contributions['SHAP Value'] < 0].head(2)
                        
                        explanation = "The main factors increasing the risk of failure are "
                        if not top_positive.empty:
                            explanation += ", ".join([f"**{row['Feature']}**" for _, row in top_positive.iterrows()]) + ". "
                        else:
                            explanation += "none. "
                            
                        explanation += "The factors reducing the risk are "
                        if not top_negative.empty:
                            explanation += ", ".join([f"**{row['Feature']}**" for _, row in top_negative.iterrows()]) + "."
                        else:
                            explanation += "none."
                            
                        st.info(explanation)
                    else:
                        st.warning("Explainer could not be loaded for explanations.")
            except Exception as e:
                st.error(f"Error during prediction: {e}")

elif page == "Explainability":
    st.title("Model Explainability")
    st.write("Understand the global behavior of the model.")
    
    artifacts_dir = os.path.join(BASE_DIR, 'artifacts', 'figures')
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Global Feature Importance")
        global_imp_path = os.path.join(artifacts_dir, 'shap_global_importance.png')
        if os.path.exists(global_imp_path):
            st.image(global_imp_path, use_column_width=True)
        else:
            st.warning("Global importance plot not found.")
            
    with col2:
        st.subheader("Beeswarm Plot")
        beeswarm_path = os.path.join(artifacts_dir, 'shap_beeswarm.png')
        if os.path.exists(beeswarm_path):
            st.image(beeswarm_path, use_column_width=True)
        else:
            st.warning("Beeswarm plot not found.")
            
    st.markdown("---")
    st.subheader("Live Sample Failure Case Explanation")
    if st.button("Generate Explanation for a Failure Case"):
        if model is not None and preprocessor is not None and X_train is not None and explainer is not None:
            try:
                # Find a predicted failure in the training set
                preds = model.predict(X_train)
                failure_idx = np.where(preds == 1)[0]
                
                if len(failure_idx) > 0:
                    idx = failure_idx[0]
                    sample = X_train.iloc[[idx]]
                    st.write(f"Showing explanation for instance index {idx}")
                    
                    shap_vals = explainer(sample)
                    if len(shap_vals.shape) == 3:
                        sv = shap_vals[0, :, 1]
                    else:
                        sv = shap_vals[0]
                        
                    fig = plot_waterfall(sv)
                    st.pyplot(fig)
                    
                    st.write("**Feature Importance Ranking for this Sample:**")
                    contributions = pd.DataFrame({
                        'Feature': sample.columns,
                        'Value': sample.iloc[0].values,
                        'SHAP Value': sv.values
                    }).sort_values('SHAP Value', key=abs, ascending=False)
                    st.dataframe(contributions)
                else:
                    st.warning("No predicted failures found in the training sample.")
            except Exception as e:
                st.error(f"Error generating sample explanation: {e}")
        else:
            st.error("Components missing to generate live explanation.")

elif page == "What-If Analysis":
    st.title("What-If Analysis")
    st.info("This is model-based what-if analysis. It shows how the model prediction changes, not a causal guarantee.")
    
    st.subheader("Base Scenario")
    col1, col2 = st.columns(2)
    with col1:
        base_product_type = st.selectbox("Base Product Type", ["L", "M", "H"], key="b_pt")
        base_air_temp = st.slider("Base Air Temperature (K)", 295.0, 305.0, 300.0, step=0.1, key="b_at")
        base_process_temp = st.slider("Base Process Temperature (K)", 305.0, 315.0, 310.0, step=0.1, key="b_ptemp")
    with col2:
        base_rot_speed = st.slider("Base Rotational Speed (rpm)", 1100, 2900, 1500, step=10, key="b_rs")
        base_torque = st.slider("Base Torque (Nm)", 3.0, 80.0, 40.0, step=0.5, key="b_t")
        base_tool_wear = st.slider("Base Tool Wear (min)", 0, 260, 100, step=1, key="b_tw")
        
    st.subheader("Modified Scenario")
    col3, col4 = st.columns(2)
    with col3:
        mod_product_type = st.selectbox("Modified Product Type", ["L", "M", "H"], index=["L", "M", "H"].index(base_product_type), key="m_pt")
        mod_air_temp = st.slider("Modified Air Temperature (K)", 295.0, 305.0, base_air_temp, step=0.1, key="m_at")
        mod_process_temp = st.slider("Modified Process Temperature (K)", 305.0, 315.0, base_process_temp, step=0.1, key="m_ptemp")
    with col4:
        mod_rot_speed = st.slider("Modified Rotational Speed (rpm)", 1100, 2900, base_rot_speed, step=10, key="m_rs")
        mod_torque = st.slider("Modified Torque (Nm)", 3.0, 80.0, base_torque, step=0.5, key="m_t")
        mod_tool_wear = st.slider("Modified Tool Wear (min)", 0, 260, base_tool_wear, step=1, key="m_tw")
        
    if st.button("COMPARE SCENARIOS", type="primary"):
        if model is not None and preprocessor is not None:
            base_input = {
                'Air temperature [K]': base_air_temp,
                'Process temperature [K]': base_process_temp,
                'Rotational speed [rpm]': base_rot_speed,
                'Torque [Nm]': base_torque,
                'Tool wear [min]': base_tool_wear,
                'Type_H': base_product_type == 'H',
                'Type_L': base_product_type == 'L',
                'Type_M': base_product_type == 'M'
            }
            
            mod_input = {
                'Air temperature [K]': mod_air_temp,
                'Process temperature [K]': mod_process_temp,
                'Rotational speed [rpm]': mod_rot_speed,
                'Torque [Nm]': mod_torque,
                'Tool wear [min]': mod_tool_wear,
                'Type_H': mod_product_type == 'H',
                'Type_L': mod_product_type == 'L',
                'Type_M': mod_product_type == 'M'
            }
            
            try:
                base_df = preprocess_input(base_input, preprocessor)
                mod_df = preprocess_input(mod_input, preprocessor)
                
                base_prob = model.predict_proba(base_df)[0][1]
                mod_prob = model.predict_proba(mod_df)[0][1]
                
                diff = mod_prob - base_prob
                
                st.markdown("---")
                res1, res2, res3 = st.columns(3)
                
                res1.metric("Original Prediction", f"{base_prob:.2%}")
                res2.metric("Modified Prediction", f"{mod_prob:.2%}")
                res3.metric("Probability Difference", f"{diff:+.2%}", delta=f"{diff:+.2%}", delta_color="inverse")
                
            except Exception as e:
                st.error(f"Error during comparison: {e}")

elif page == "Model Performance":
    st.title("Model Performance Metrics")
    
    metrics_dir = os.path.join(BASE_DIR, 'artifacts', 'metrics')
    figures_dir = os.path.join(BASE_DIR, 'artifacts', 'figures')
    
    best_info_file = os.path.join(metrics_dir, 'best_model_info.json')
    if os.path.exists(best_info_file):
        try:
            with open(best_info_file, 'r') as f:
                best_info = json.load(f)
            st.success(f"**Selected Production Model:** {best_info.get('model_name', 'XGBoost')}")
            st.markdown(f"**Selection Rationale:** {best_info.get('reasoning', '')}")
        except Exception:
            pass

    comp_file = os.path.join(metrics_dir, 'comparison_table.csv')
    if os.path.exists(comp_file):
        st.subheader("Model Comparison Benchmark")
        comp_df = pd.read_csv(comp_file)
        st.dataframe(comp_df, use_container_width=True)

    research_file = os.path.join(metrics_dir, 'research_experiments_summary.csv')
    if os.path.exists(research_file):
        st.subheader("Research Experiments: Model & Imbalance Handling Strategies")
        res_df = pd.read_csv(research_file)
        st.dataframe(res_df, use_container_width=True)
        
        imb_chart = os.path.join(figures_dir, 'imbalance_experiments_comparison.png')
        if os.path.exists(imb_chart):
            st.image(imb_chart, caption="Comparison across Class Imbalance Strategies", use_column_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Confusion Matrix")
        cm_path = os.path.join(figures_dir, 'cm_xgboost.png')
        if os.path.exists(cm_path):
            st.image(cm_path, use_column_width=True)
        else:
            st.warning("Confusion matrix not found.")
            
        st.subheader("Precision-Recall Curve")
        pr_path = os.path.join(figures_dir, 'pr_curves.png')
        if os.path.exists(pr_path):
            st.image(pr_path, use_column_width=True)
            
    with col2:
        st.subheader("ROC Curve")
        roc_path = os.path.join(figures_dir, 'roc_curves.png')
        if os.path.exists(roc_path):
            st.image(roc_path, use_column_width=True)
        else:
            st.warning("ROC curve not found.")

elif page == "Dataset Insights":
    st.title("Dataset Insights")
    
    raw_data_dir = os.path.join(BASE_DIR, 'data', 'raw')
    raw_files = [f for f in os.listdir(raw_data_dir) if f.endswith('.csv')] if os.path.exists(raw_data_dir) else []
    
    if raw_files:
        try:
            raw_data = pd.read_csv(os.path.join(raw_data_dir, raw_files[0]))
            st.subheader("Raw Data Statistics")
            st.dataframe(raw_data.describe())
            
            col1, col2 = st.columns(2)
            with col1:
                if 'Target' in raw_data.columns or 'Machine failure' in raw_data.columns:
                    target_col = 'Machine failure' if 'Machine failure' in raw_data.columns else 'Target'
                    st.subheader("Target Distribution")
                    fig, ax = plt.subplots(figsize=(6, 4))
                    sns.countplot(x=target_col, data=raw_data, ax=ax)
                    st.pyplot(fig)
            
            with col2:
                st.subheader("Correlation Heatmap")
                fig, ax = plt.subplots(figsize=(8, 6))
                numeric_df = raw_data.select_dtypes(include=[np.number])
                sns.heatmap(numeric_df.corr(), annot=False, cmap='coolwarm', ax=ax)
                st.pyplot(fig)
                
            st.subheader("Feature Distributions")
            num_cols = numeric_df.columns[:6] # Limit to a few features
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            for i, col in enumerate(num_cols):
                r, c = i // 3, i % 3
                sns.histplot(raw_data[col], ax=axes[r, c], kde=True)
                axes[r, c].set_title(col)
            plt.tight_layout()
            st.pyplot(fig)
            
        except Exception as e:
            st.error(f"Error loading dataset insights: {e}")
    else:
        st.warning("No raw data file found for insights.")
