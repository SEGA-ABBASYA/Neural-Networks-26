import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from preprocessing import preprocess_data, preprocess_sample
from perceptron import PerceptronModel
from adaline import AdalineModel
from visualization import visualize_data

# Page configuration
st.set_page_config(
    page_title="Neural Networks - Perceptron & Adaline",
    page_icon="🧠",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        color: #2c3e50;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #3498db;
        padding-bottom: 0.5rem;
    }
    .metric-container {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #3498db;
    }
</style>
""", unsafe_allow_html=True)

def main():
    # Main header
    st.markdown('<h1 class="main-header">🧠 Neural Networks - Perceptron & Adaline</h1>', unsafe_allow_html=True)

    # Sidebar for user input
    st.sidebar.markdown("## ⚙️ Configuration")

    # Feature selection
    st.sidebar.markdown("### 📊 Feature Selection")
    numeric_features = ['CulmenLength', 'CulmenDepth', 'FlipperLength', 'BodyMass', 'OriginLocation']

    feature1 = st.sidebar.selectbox(
        "Select First Feature:",
        numeric_features,
        index=0
    )

    feature2 = st.sidebar.selectbox(
        "Select Second Feature:",
        [f for f in numeric_features if f != feature1],
        index=0
    )

    # Class selection
    st.sidebar.markdown("### 🎯 Class Selection")
    unique_classes =  ['Gentoo', 'Chinstrap', 'Adelie']
    class_options = {
        f"{unique_classes[0]} & {unique_classes[1]}": (unique_classes[0], unique_classes[1]),
        f"{unique_classes[0]} & {unique_classes[2]}": (unique_classes[0], unique_classes[2]),
        f"{unique_classes[1]} & {unique_classes[2]}": (unique_classes[1], unique_classes[2])
    }

    selected_classes = st.sidebar.selectbox(
        "Select Two Classes:",
        list(class_options.keys())
    )

    class1, class2 = class_options[selected_classes]

    # Model parameters
    st.sidebar.markdown("### 🔧 Model Parameters")

    learning_rate = st.sidebar.number_input(
        "Learning Rate (η):",
        min_value=0.001,
        max_value=1.0,
        value=0.01,
        step=0.001,
        format="%.3f"
    )

    epochs = st.sidebar.number_input(
        "Number of Epochs (m):",
        min_value=1,
        max_value=1000,
        value=100,
        step=1
    )

    mse_threshold = st.sidebar.number_input(
        "MSE Threshold:",
        min_value=0.001,
        max_value=1.0,
        value=0.01,
        step=0.001,
        format="%.3f"
    )

    use_bias = st.sidebar.checkbox("Add Bias", value=True)

    # Algorithm selection
    st.sidebar.markdown("### 🤖 Algorithm Selection")
    algorithm = st.sidebar.radio(
        "Choose Algorithm:",
        ["Perceptron", "Adaline"]
    )

    # Preprocess data
    X_train, y_train, X_test, y_test, min_val, max_val = preprocess_data(feature1, feature2, class1, class2)

    # Main content area
    st.markdown('<h2 class="section-header">🎯 Model Training</h2>', unsafe_allow_html=True)

    # Train button
    if st.button("🚀 Train Model", type="primary"):
        with st.spinner("Training model..."):
            if algorithm == "Perceptron":
                model = PerceptronModel(
                    learning_rate=learning_rate,
                    n_epochs=epochs,
                    add_bias=use_bias,
                    min=min_val,
                    max=max_val
                )
            else:
                model = AdalineModel(
                    learning_rate=learning_rate,
                    n_epochs=epochs,
                    mse_threshold=mse_threshold,
                    add_bias=use_bias,
                )
            model.train(X_train, y_train)
            weights, bias = model.get_weights()
            y_pred = model.test(X_test)
            st.session_state['model'] = model
            st.success("✅ Model trained successfully!")
            # Data visualization
            visualize_data(feature1, feature2, class1, class2, learning_rate, epochs, mse_threshold, use_bias, algorithm, X_train, y_train, X_test, y_test, y_pred, weights, bias)

    # Classification section
    st.markdown('<h2 class="section-header">🔍 Classification</h2>', unsafe_allow_html=True)
    # Input fields for sample classification
    sample_feature1 = st.number_input(
        f"{feature1}:"
    )
    sample_feature2 = st.number_input(
        f"{feature2}:"
    )
    if st.button("🔮 Classify Sample"):
        if 'model' in st.session_state:
            # Prepare sample for prediction
            sample = preprocess_sample(sample_feature1, sample_feature2)
            prediction_value = st.session_state['model'].predict(sample)

            # Map prediction value (-1 or 1) to class names
            if prediction_value == 1:
                predicted_class = class1
            else:  # prediction_value == -1
                predicted_class = class2

            st.success(f"🎯 Predicted Class: **{predicted_class}**")
        else:
            st.error("❌ Please train a model first!")

if __name__ == "__main__":
    main()
