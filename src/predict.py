import pandas as pd
import joblib

model = joblib.load("models/csat_model.pkl")
encoders = joblib.load("models/encoders.pkl")

def predict_csat(data):

    df = pd.DataFrame([data])

    for col in encoders:
        if col in df:
            df[col] = encoders[col].transform(df[col])

    prediction = model.predict(df)

    return prediction[0]