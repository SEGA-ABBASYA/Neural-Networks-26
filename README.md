# Neural Networks 26 🧠

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Framework-TensorFlow%20%7C%20PyTorch-orange.svg)](https://pytorch.org/)


## 📌 Project Overview
This repository contains the implementations, experiments, and architectural designs for **Neural Networks 26**. It serves as a comprehensive exploration of deep learning concepts, encompassing data preprocessing, model building, training loops, and performance evaluation. 

The project focuses on applying neural network architectures to solve complex classification and regression tasks, emphasizing optimized training methodologies and accurate predictive modeling.

## ✨ Key Features
* **Custom Model Architectures:** Implementation of Multi-Layer Perceptrons (MLPs), CNNs,  RNNs tailored for specific dataset characteristics.
* **Robust Data Pipeline:** Automated scripts for data cleaning, normalization, feature scaling, and train/test splitting.
* **Hyperparameter Tuning:** Systematic optimization of learning rates, batch sizes, activation functions, and dropout layers to prevent overfitting.
* **Performance Metrics:** Comprehensive evaluation visualizing loss curves, accuracy metrics, and confusion matrices using `Matplotlib` and `Seaborn`.

## 🛠️ Tech Stack & Tools
* **Language:** Python
* **Data Manipulation:** NumPy, Pandas
* **Visualization:** Matplotlib, Seaborn
* **Environment:** Jupyter Notebook / Google Colab

## 📂 Project Structure
```text
📦 Neural-Networks-26
 ┣ 📂 data               # Raw and processed datasets
 ┣ 📂 models             # Saved model weights
 ┣ 📂 notebooks          # Jupyter notebooks for EDA and model experimentation
 ┣ 📂 src                # Source code for data pipelines and network architectures
 ┃ ┣ 📜 data_loader.py   # Scripts for data ingestion and preprocessing
 ┃ ┣ 📜 network.py       # Neural network class definitions
 ┃ ┗ 📜 train.py         # Main training loop and evaluation scripts
 ┣ 📜 requirements.txt   # Project dependencies
 ┗ 📜 README.md          # Project documentation