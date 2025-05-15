import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Set page config
st.set_page_config(
    page_title="Credit Risk Predictor",
    page_icon="💰",
    layout="wide"
)

# Load all trained models
@st.cache_resource
def load_models():
    models = {
        'Logistic Regression': joblib.load('LogisticRegression_model.pkl'),
        'Decision Tree': joblib.load('DecisionTree_model.pkl'),
        'Random Forest': joblib.load('RandomForest_model.pkl'),
        'Gradient Boosting': joblib.load('GradientBoosting_model.pkl')
    }
    return models

models = load_models()

# Main app
def main():
    st.title("💰 Credit Risk Prediction Dashboard")
    st.markdown("Predict the likelihood of loan default based on customer characteristics")
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.radio("Choose a page", ["Prediction", "Model Comparison"])
    
    if app_mode == "Prediction":
        show_prediction_page()
    else:
        show_comparison_page()

# Prediction page
def show_prediction_page():
    st.header("Make a Prediction")
    
    # Create input form
    with st.form("input_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            new_repeat = st.selectbox("Customer Type", ["New", "Repeat"])
            loan_duration = st.number_input("Loan Duration (months)", min_value=1, max_value=120, value=12)
        
        with col2:
            loan_amount = st.number_input("Loan Amount ($)", min_value=100, max_value=100000, value=5000)
            interest_amount = st.number_input("Interest Amount ($)", min_value=0, max_value=10000, value=500)
        
        model_choice = st.selectbox("Select Model", list(models.keys()))
        
        submitted = st.form_submit_button("Predict")
    
    if submitted:
        # Prepare input data
        input_data = pd.DataFrame({
            'new_repeat': [new_repeat],
            'loan_duration': [loan_duration],
            'loan_amount': [loan_amount],
            'interest_amount': [interest_amount]
        })
        
        # Make prediction
        model = models[model_choice]
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]
        
        # Display results
        st.subheader("Prediction Results")
        
        if prediction == 1:
            st.error(f"🚨 High Risk of Default (Probability: {probability:.2%})")
        else:
            st.success(f"✅ Low Risk of Default (Probability: {probability:.2%})")
        
        # Show probability gauge
        st.progress(probability)
        st.caption(f"Default probability: {probability:.2%}")

# Model comparison page
def show_comparison_page():
    st.header("Model Performance Comparison")
    
    # Sample performance data - replace with your actual metrics
    performance_data = {
        'Model': ['Logistic Regression', 'Decision Tree', 'Random Forest', 'Gradient Boosting'],
        'AUC-ROC': [0.85, 0.82, 0.89, 0.88],
        'Accuracy': [0.78, 0.76, 0.82, 0.81],
        'Precision': [0.75, 0.72, 0.80, 0.79],
        'Recall': [0.70, 0.75, 0.82, 0.81]
    }
    
    df = pd.DataFrame(performance_data)
    
    # Display metrics table
    st.dataframe(df.style.highlight_max(axis=0), use_container_width=True)
    
    # Visualizations
    st.subheader("Performance Metrics")
    metric_choice = st.selectbox("Select metric to visualize", ['AUC-ROC', 'Accuracy', 'Precision', 'Recall'])
    
    st.bar_chart(df.set_index('Model')[metric_choice])

if __name__ == "__main__":
    main()