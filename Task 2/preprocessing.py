import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder

df = pd.read_csv("penguins.csv")

def label_encode_species(series):
    le = LabelEncoder()
    return le.fit_transform(series)

def one_hot_encode_location(df):
    if( 'OriginLocation' in df.columns):
        df = pd.get_dummies(df, columns=['OriginLocation'], drop_first=True)
    return df


def tanh_one_hot(y, num_classes=3):
    y_tanh = -np.ones((len(y), num_classes))
    for i, label in enumerate(y):
        y_tanh[i, label] = 1
    return y_tanh

def class_encode_species(y_train, y_test, activationFN):
    if (activationFN == "sigmoid"): 
        encoder = OneHotEncoder(sparse=False)
        y_train_encoded = encoder.fit_transform(y_train.reshape(-1, 1))
        y_test_encoded = encoder.transform(y_test.reshape(-1, 1))
    else:
        y_train_encoded = tanh_one_hot(y_train)
        y_test_encoded = tanh_one_hot(y_test)

    return y_train_encoded, y_test_encoded


def splitting_data(df):
    X_train_list = []
    X_test_list = []
    y_train_list = []
    y_test_list = []

    for cls in sorted(df['Species'].unique()):
        class_df = df[df['Species'] == cls]

        class_df = class_df.head(50)     
        train_df = class_df.iloc[:30]     
        test_df = class_df.iloc[30:50]

        X_train_list.append(train_df.drop(columns=['Species']))
        y_train_list.append(train_df['Species'])

        X_test_list.append(test_df.drop(columns=['Species']))
        y_test_list.append(test_df['Species'])

    X_train = pd.concat(X_train_list).reset_index(drop=True)
    y_train = pd.concat(y_train_list).reset_index(drop=True)

    X_test = pd.concat(X_test_list).reset_index(drop=True)
    y_test = pd.concat(y_test_list).reset_index(drop=True)
    return X_train, X_test, y_train, y_test

def preprocess_data(activationFN):

    df_clean = df.copy()

    X = df_clean.drop(columns=['Species'])
    y = df_clean['Species']
   
    X_train, X_test, y_train, y_test = splitting_data(df_clean)

    X_train = X_train.fillna(X_train.mean(numeric_only=True))
    X_test = X_test.fillna(X_train.mean(numeric_only=True))
   
    y_train = label_encode_species(y_train)
    y_test = label_encode_species(y_test)

    activationFN = str(activationFN).lower()
    y_train, y_test = class_encode_species(y_train, y_test, activationFN)

    X_train = one_hot_encode_location(X_train)
    X_test = one_hot_encode_location(X_test)

    scaler = MinMaxScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

    min_value = scaler.data_min_
    max_value = scaler.data_max_
    st.session_state['feature_columns'] = X_train.columns.tolist()
    st.session_state['scaler'] = scaler
    st.session_state['mean_values'] = X_train.mean(numeric_only=True)
    st.session_state['min_values'] = min_value
    st.session_state['max_values'] = max_value
    return (
        X_train.to_numpy(),
        y_train,
        X_test.to_numpy(),
        y_test,
        min_value,
        max_value,
    )


def preprocess_sample(culmen_length, culmen_depth, flipper_length, body_mass, origin_location):
    sample_dict = {
        'culmen_length': culmen_length,
        'culmen_depth': culmen_depth,
        'flipper_length': flipper_length,
        'body_mass': body_mass,
        'origin_location': origin_location
    }

    sample_df = pd.DataFrame([sample_dict])
    sample_df = sample_df.fillna(st.session_state['mean_values'])
    sample_df = one_hot_encode_location(sample_df)
    sample_df = sample_df.reindex(columns=st.session_state['feature_columns'], fill_value=0)
    scaler = st.session_state['scaler']
    sample_df = pd.DataFrame(scaler.transform(sample_df), columns=sample_df.columns)

    return sample_df
