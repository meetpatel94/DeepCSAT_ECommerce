import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Load dataset
df = pd.read_csv("data/eCommerce_Customer_support_data.csv")

# Required columns
df = df[[
"channel_name",
"category",
"Customer_City",
"Product_category",
"Agent Shift",
"Item_price",
"CSAT Score"
]]

# Encoding
encoders = {}

for col in ["channel_name","category","Customer_City","Product_category","Agent Shift"]:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# Features
X = df.drop("CSAT Score", axis=1)
y = df["CSAT Score"]

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(
X_scaled,
y,
test_size=0.2,
random_state=42
)

# ANN Model
model = Sequential()

model.add(Dense(64, activation="relu", input_shape=(6,)))
model.add(Dense(32, activation="relu"))
model.add(Dense(1))

model.compile(
optimizer="adam",
loss="mse",
metrics=["mae"]
)

# Train
model.fit(
X_train,
y_train,
epochs=30,
batch_size=32,
validation_data=(X_test,y_test)
)

# Save model
model.save("models/ann_model.h5")

# Save scaler + encoders
pickle.dump(scaler, open("models/scaler.pkl","wb"))
pickle.dump(encoders, open("models/encoders.pkl","wb"))

print("ANN Model Training Completed")