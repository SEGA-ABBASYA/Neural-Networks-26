import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
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

def preprocess_data(activation_function):
    pass
def preprocess_sample(culmen_length, culmen_depth, flipper_length, body_mass, origin_location):
    pass