from flask import Flask, render_template, request, jsonify
import pandas as pd
import pickle
# from tensorflow.keras.models import load_model
import numpy as np

app = Flask(__name__)

# Dataset
df = pd.read_csv("data/eCommerce_Customer_support_data.csv")
df = df.fillna("")

# Load ANN model
model = load_model("models/ann_model.h5", compile=False)

# Load scaler + encoders
scaler = pickle.load(open("models/scaler.pkl","rb"))
encoders = pickle.load(open("models/encoders.pkl","rb"))

@app.route("/")
def home():

    return render_template(
        "index.html",
        channels=df["channel_name"].unique(),
        categories=df["category"].unique(),
        products=df["Product_category"].unique(),
        shifts=df["Agent Shift"].unique(),
        cities=df["Customer_City"].unique()
    )


@app.route("/ann")
def ann_page():

    return render_template(
        "ann.html",
        channels=df["channel_name"].unique(),
        categories=df["category"].unique(),
        products=df["Product_category"].unique(),
        shifts=df["Agent Shift"].unique(),
        cities=df["Customer_City"].unique(),
        prediction=None
    )


@app.route("/predict", methods=["POST"])
def predict():

    channel = request.form["channel"]
    category = request.form["category"]
    city = request.form["city"]
    product = request.form["product"]
    shift = request.form["shift"]
    price = float(request.form["price"])

    input_data = pd.DataFrame([{
        "channel_name": encoders["channel_name"].transform([channel])[0],
        "category": encoders["category"].transform([category])[0],
        "Customer_City": encoders["Customer_City"].transform([city])[0],
        "Product_category": encoders["Product_category"].transform([product])[0],
        "Agent Shift": encoders["Agent Shift"].transform([shift])[0],
        "Item_price": price
    }])

    scaled = scaler.transform(input_data)

    prediction = model.predict(scaled)

    csat = round(float(prediction[0][0]),2)

    csat = max(1, min(5, csat))

    return render_template(
        "ann.html",
        channels=df["channel_name"].unique(),
        categories=df["category"].unique(),
        products=df["Product_category"].unique(),
        shifts=df["Agent Shift"].unique(),
        cities=df["Customer_City"].unique(),
        prediction=csat
    )

@app.route("/analytics")
def analytics():

    df = pd.read_csv("data/eCommerce_Customer_support_data.csv")

    csat = df["CSAT Score"].value_counts().sort_index()

    city = df.groupby("Customer_City")["CSAT Score"].mean().sort_values(ascending=False).head(10)

    product = df.groupby("Product_category")["CSAT Score"].mean()

    return render_template(
        "analytics.html",
        csat=csat,
        city=city,
        product=product
    )

@app.route("/about")
def about():
    return render_template("about.html")
@app.route("/filter_data")
def filter_data():

    channel = request.args.get("channel")
    category = request.args.get("category")
    city = request.args.get("city")
    product = request.args.get("product")
    shift = request.args.get("shift")

    min_price = request.args.get("min_price")
    max_price = request.args.get("max_price")

    page = int(request.args.get("page", 1))
    per_page = 10

    filtered = df.copy()

    if channel:
        filtered = filtered[filtered["channel_name"] == channel]

    if category:
        filtered = filtered[filtered["category"] == category]

    if city:
        filtered = filtered[filtered["Customer_City"].str.contains(city, case=False)]

    if product:
        filtered = filtered[filtered["Product_category"] == product]

    if shift:
        filtered = filtered[filtered["Agent Shift"] == shift]

    if min_price:
        filtered = filtered[filtered["Item_price"] >= float(min_price)]

    if max_price:
        filtered = filtered[filtered["Item_price"] <= float(max_price)]

    start = (page - 1) * per_page
    end = start + per_page

    page_data = filtered.iloc[start:end]

    return jsonify({
        "data": page_data.to_dict(orient="records")
    })


if __name__ == "__main__":
    app.run(debug=True)