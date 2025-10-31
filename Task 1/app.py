import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go
from preprocessing import preprocess_data, preprocess_sample
from perceptron import PerceptronModel
from adaline import AdalineModel
from visualization import visualize_data

st.set_page_config(
    page_title="Penguin Classification",
    page_icon="🐧",
    layout="wide"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #0EA5E9;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        color: #075985;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 1px solid #0EA5E9;
        padding-bottom: 0.5rem;
    }
    /* Style the primary button */
    button[kind="primary"] {
        background-color: #0284C7 !important;
        border-color: #0284C7 !important;
        height: 38px !important;
        width: 150px !important;
    }
    button[kind="primary"]:hover {
        background-color: #0369A1 !important;
        border-color: #0369A1 !important;
    }
</style>
""", unsafe_allow_html=True)

def main():
    st.markdown('<h1 class="main-header">🐧 Penguin Classification 🐧</h1>', unsafe_allow_html=True)
    st.sidebar.markdown("## ❄️ Configuration")

    # Features
    st.sidebar.markdown("### ⛄ Feature Selection")
    numeric_features = ['CulmenLength', 'CulmenDepth', 'FlipperLength', 'BodyMass', 'OriginLocation']

    feature1 = st.sidebar.selectbox(
        "Select First Feature:",
        features,
        index=0
    )

    feature2 = st.sidebar.selectbox(
        "Select Second Feature:",
        [f for f in features if f != feature1],
        index=0
    )

    # Classes
    st.sidebar.markdown("### ⛄ Class Selection")
    classes = ['Gentoo', 'Chinstrap', 'Adelie']
    class_options = {
        f"{c1} & {c2}": (c1, c2)
        for c1, c2 in combinations(classes, 2)
    }
    selected_classes = st.sidebar.selectbox(
        "Select Two Classes:",
        list(class_options.keys())
    )

    class1, class2 = class_options[selected_classes]

    # Hyperparameters
    st.sidebar.markdown("### ⛄ Model Hyperparameters")

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

    # Algorithms
    st.sidebar.markdown("### ⛄ Algorithm Selection")
    algorithm = st.sidebar.radio(
        "Choose Algorithm:",
        ["Perceptron", "Adaline"]
    )

    # Preprocess data
    X_train, y_train, X_test, y_test, min_val, max_val = preprocess_data(feature1, feature2, class1, class2)

    st.markdown('<h2 class="section-header">🌨️ Model Training</h2>', unsafe_allow_html=True)
    # Model Training Section
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("☃️ Train Model", type="primary"):
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
            st.success("✔️ Model trained successfully!")
            # Data visualization
            visualize_data(feature1, feature2, class1, class2, learning_rate, epochs, mse_threshold, use_bias, algorithm, X_train, y_train, X_test, y_test, y_pred, weights, bias)

    # Classification section
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<h2 class="section-header">🌨️ Classification</h2>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    # Input fields for sample classification
    sample_feature1 = st.number_input(
        f"{feature1}:"
    )
    sample_feature2 = st.number_input(
        f"{feature2}:"
    )
    if st.button("❄️ Classify Sample"):
        if 'model' in st.session_state:
            # Sample preprocessing
            sample = preprocess_sample(sample_feature1, sample_feature2)
            prediction_value = st.session_state['model'].predict(sample)

            # Map prediction value (-1 or 1) to class names
            if prediction_value == 1:
                predicted_class = class1
            else:  # prediction_value == -1
                predicted_class = class2
            st.success(f"🐧 Predicted Class: **{predicted_class}** ")
        else:
            st.error("❌ Please train a model first!")

if __name__ == "__main__":
    main()
