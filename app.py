import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Load dataset
df = pd.read_csv("weather_data.csv")

# Features and target
X = df[["Temperature", "Humidity", "Pressure", "WindSpeed", "Rainfall"]]
y = df["Weather"]

# Train model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
model.fit(X, y)

# App title
st.title("🌦️ Weather Prediction System")
st.subheader("📊 Weather Data Overview")

st.write("Temperature Trend")
st.line_chart(df.set_index("Date")["Temperature"])

st.write("Humidity Trend")
st.line_chart(df.set_index("Date")["Humidity"])

st.write("Rainfall Trend")
st.line_chart(df.set_index("Date")["Rainfall"])
st.write("Enter weather conditions to predict the weather.")

# User inputs
temperature = st.number_input(
    "Temperature (°C)", value=25.0
)

humidity = st.number_input(
    "Humidity (%)", value=60.0
)

pressure = st.number_input(
    "Pressure (hPa)", value=1013.0
)

wind_speed = st.number_input(
    "Wind Speed", value=10.0
)

rainfall = st.number_input(
    "Rainfall", value=2.0
)

# Prediction button
if st.button("Predict Weather"):
    new_data = pd.DataFrame({
        "Temperature": [temperature],
        "Humidity": [humidity],
        "Pressure": [pressure],
        "WindSpeed": [wind_speed],
        "Rainfall": [rainfall]
    })

    prediction = model.predict(new_data)

    st.success(f"🌤️ Predicted Weather: {prediction[0]}")