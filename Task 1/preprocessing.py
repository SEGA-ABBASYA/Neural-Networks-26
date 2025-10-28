import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("penguins.csv")


def label_encode_species(series):
    le = LabelEncoder()
    return le.fit_transform(series)

def one_hot_encode_location(df):
    if( 'OriginLocation' in df.columns):
        df = pd.get_dummies(df, columns=['OriginLocation'], drop_first=True)
    return df

def preprocess_data(feature1, feature2, class1, class2):
    
    new_df = df[df['Species'].isin([class1, class2])].copy()
    new_df = new_df[[feature1, feature2, 'Species']]
    X = new_df[[feature1, feature2]]
    y = new_df['Species']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=42, stratify=y
    )

    X_train = X_train.fillna(X_train.mean(numeric_only=True))
    X_test = X_test.fillna(X_train.mean(numeric_only=True))
    
    y_train = label_encode_species(y_train)
    y_test = label_encode_species(y_test)
    
    X_train = one_hot_encode_location(X_train)
    X_test = one_hot_encode_location(X_test)

    scaler = StandardScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

    st.session_state['feature_columns'] = X_train.columns.tolist()
    st.session_state['scaler'] = scaler
    st.session_state['mean_values'] = X_train.mean(numeric_only=True)

    return X_train, y_train, X_test, y_test

def preprocess_sample(feature1, feature2):
    sample_df = pd.DataFrame([[feature1, feature2]], columns=st.session_state['feature_columns'][:2])   

    sample_df =sample_df.fillna(st.session_state['mean_values'])
    sample_df = one_hot_encode_location(sample_df)

    sample_df = sample_df.reindex(columns=st.session_state['feature_columns'], fill_value=0)
    
    scaler = st.session_state['scaler']
    sample_df = pd.DataFrame(scaler.transform(sample_df), columns=sample_df.columns)

    return sample_df