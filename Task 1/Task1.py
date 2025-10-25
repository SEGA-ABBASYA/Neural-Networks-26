"""# 1: Import Libraries"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt

"""# 2: Uploading Dataset"""

df = pd.read_csv("penguins.csv")

"""# 3: Exploratory Data Analysis (EDA)"""

df.head()

df.info()
# Found only two rows with missing values

df.describe()

df.hist()

"""# 4: Splitting Data"""

X = df.drop('Species', axis=1)
y = df['Species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
train_df = pd.concat([X_train, y_train], axis=1)
test_df = pd.concat([X_test, y_test], axis=1)

"""# 5: Log BodyMass (optional)"""

def log_bodymass(df):
    df['BodyMass'] = df['BodyMass'].apply(lambda x: np.log(x))
    return df
train_df = log_bodymass(train_df)

"""# 6: Handle Missing Values"""

def handle_missing_values(df):
    # CHANGE BODYMASS TO MEAN IF YOU USE THE LOG
    df = df.fillna({
        'BodyMass': df['BodyMass'].median(),
        'FlipperLength': df['FlipperLength'].mean(),
        'CulmenLength': df['CulmenLength'].mean(),
        'CulmenDepth': df['CulmenDepth'].mean()
    })
    return df
train_df = handle_missing_values(train_df)

"""# 7: Label Encode Target Column"""

def label_encode_species(df):
    le = LabelEncoder()
    df['Species'] = le.fit_transform(df['Species'])
    return df
train_df = label_encode_species(train_df)

"""# 8: One Hot Encode Origin Location"""

def one_hot_encode_location(df):
    df = pd.get_dummies(df, columns=['OriginLocation'], drop_first=True)
    return df
train_df = one_hot_encode_location(train_df)

"""# 9: Feature Engineering (optional)"""

def feature_engineering(df):
    #df["BeakArea"] = df["CulmenLength"] * df["CulmenDepth"]
    return df
train_df = feature_engineering(train_df)

"""# 10: MinMax Scale"""

def min_max_scale(df):
    scaler = StandardScaler()
    df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    return df
train_df = min_max_scale(train_df)

"""# 11: Calculate Correlation"""

def corr_target(df):
    corr_matrix = df.corr(method='pearson', numeric_only= True)

    # Plot heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", square=True)
    plt.title("Pearson Correlation Heatmap")
    plt.show()
corr_target(train_df)

# train_df.to_csv("processed_penguins_train.csv", index=False)