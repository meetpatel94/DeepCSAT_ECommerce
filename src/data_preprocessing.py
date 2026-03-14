import pandas as pd
from sklearn.preprocessing import LabelEncoder


def load_data():

    df = pd.read_csv("data/eCommerce_Customer_support_data.csv")

    return df


def preprocess_data(df):

    df = df.dropna()

    cat_cols = df.select_dtypes(include="object").columns

    le = LabelEncoder()

    for col in cat_cols:

        df[col] = le.fit_transform(df[col])

    return df