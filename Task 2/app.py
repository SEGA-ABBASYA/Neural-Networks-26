import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from preprocessing import preprocess_data, preprocess_sample_backprop
from backpropagation import BackpropagationModel

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

    num_hidden_layers = st.sidebar.number_input(
        "Number of hidden layers (H):",
        min_value=1,
        max_value=10,
        value=1,
        step=1
    )
    st.sidebar.markdown("**Number of neurons in each hidden layer:**")
    neurons_per_layer = []
    for i in range(num_hidden_layers):
        neurons = st.sidebar.number_input(
            f"Hidden Layer {i+1} neurons (N{i+1}):",
            min_value=1,
            max_value=100,
            value=5,
            step=1,
            key=f"neurons_{i}"
        )
        neurons_per_layer.append(int(neurons))

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
        max_value=10000,
        value=100,
        step=1
    )

    use_bias = st.sidebar.checkbox("Add Bias", value=True)
    activation_function = st.sidebar.radio(
        "Activation Function:",
        ["Sigmoid", "Hyperbolic Tangent"]
    )

    X_train, y_train, X_test, y_test = preprocess_data(activation_function)


    st.markdown('<h2 class="section-header">🌨️ Model Training</h2>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("☃️ Train Model", type="primary"):
        with st.spinner("Training model..."):
            model = BackpropagationModel(
                num_features=5,
                num_classes=3,
                num_hidden_layers=num_hidden_layers,
                neurons_per_layer=neurons_per_layer,
                learning_rate=learning_rate,
                n_epochs=epochs,
                use_bias=use_bias,
                activation_function=activation_function.lower()
            )
            model.train(X_train, y_train)
            st.session_state['model'] = model
            st.success("✔️ Model trained successfully!")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<h2 class="section-header">🌨️ Classification</h2>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)


    col1, col2 = st.columns(2)

    with col1:
        culmen_length = st.number_input(
            "CulmenLength:",
            min_value=0.0,
            value=0.0,
            step=0.1
        )
        culmen_depth = st.number_input(
            "CulmenDepth:",
            min_value=0.0,
            value=0.0,
            step=0.1
        )
        flipper_length = st.number_input(
            "FlipperLength:",
            min_value=0.0,
            value=0.0,
            step=0.1
        )

    with col2:
        body_mass = st.number_input(
            "BodyMass:",
            min_value=0.0,
            value=0.0,
            step=0.1
        )
        origin_location_options = ['Torgersen', 'Dream', 'Biscoe']
        origin_location = st.selectbox(
            "OriginLocation:",
            options=origin_location_options,
            index=0
        )

    if st.button("❄️ Classify Sample", type="primary"):
        if 'model' in st.session_state:
            sample = preprocess_sample(
                culmen_length,
                culmen_depth,
                flipper_length,
                body_mass,
                origin_location
            )
            # Predict class
            prediction = st.session_state['model'].predict(sample)
            # Map prediction to class name
            class_names = ['Adelie', 'Chinstrap', 'Gentoo']
            predicted_class = class_names[prediction]
            st.success(f"🐧 Predicted Class: **{predicted_class}**")
        else:
            st.error("❌ Please train a model first!")

if __name__ == "__main__":
    main()
