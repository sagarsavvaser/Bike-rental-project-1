import streamlit as st
import pickle
import numpy as np
import pandas as pd

# -----------------------------
# Load trained model
# -----------------------------
model = pickle.load(open("model.pkl", "rb"))

st.title("🚲 Bike Rental Demand Prediction")

st.write("Enter input values")

# -----------------------------
# USER INPUTS
# -----------------------------
hr = st.slider("Hour", 0, 23, 8)
weekday = st.slider("Weekday (0=Sun)", 0, 6, 2)

temp = st.slider("Temperature", 0.0, 1.0, 0.5)
atemp = st.slider("Feels Like Temperature", 0.0, 1.0, 0.5)
hum = st.slider("Humidity", 0.0, 1.0, 0.5)
windspeed = st.slider("Wind Speed", 0.0, 1.0, 0.2)

year = st.selectbox("Year", [2011, 2012])
month = st.slider("Month", 1, 12, 7)
day = st.slider("Day", 1, 31, 15)
dayofweek = st.slider("Day of Week", 0, 6, 2)

season = st.selectbox("Season", ["spring", "summer", "fall", "winter"])
holiday = st.selectbox("Holiday", ["No", "Yes"])
workingday = st.selectbox("Working Day", ["Working Day", "Holiday"])
weather = st.selectbox("Weather", ["Clear", "Mist", "Light Snow", "Heavy Rain"])

# -----------------------------
# DERIVED FEATURES
# -----------------------------
is_peak_hour = 1 if (7 <= hr <= 9 or 17 <= hr <= 19) else 0

season_springer = 1 if season == "spring" else 0
season_summer = 1 if season == "summer" else 0
season_winter = 1 if season == "winter" else 0

yr_2012 = 1 if year == 2012 else 0

mnth_2 = 1 if month == 2 else 0
mnth_3 = 1 if month == 3 else 0
mnth_4 = 1 if month == 4 else 0
mnth_5 = 1 if month == 5 else 0
mnth_6 = 1 if month == 6 else 0
mnth_7 = 1 if month == 7 else 0
mnth_8 = 1 if month == 8 else 0
mnth_9 = 1 if month == 9 else 0
mnth_10 = 1 if month == 10 else 0
mnth_11 = 1 if month == 11 else 0
mnth_12 = 1 if month == 12 else 0

holiday_Yes = 1 if holiday == "Yes" else 0
workingday_Working_Day = 1 if workingday == "Working Day" else 0

weathersit_Mist = 1 if weather == "Mist" else 0
weathersit_Light_Snow = 1 if weather == "Light Snow" else 0
weathersit_Heavy_Rain = 1 if weather == "Heavy Rain" else 0

# -----------------------------
# CREATE DATAFRAME IN EXACT ORDER
# -----------------------------
input_df = pd.DataFrame([[
    hr, weekday, temp, atemp, hum, windspeed,
    year, month, day, dayofweek,
    season_springer, season_summer, season_winter,
    yr_2012,
    mnth_10, mnth_11, mnth_12, mnth_2, mnth_3, mnth_4,
    mnth_5, mnth_6, mnth_7, mnth_8, mnth_9,
    holiday_Yes, workingday_Working_Day,
    weathersit_Heavy_Rain, weathersit_Light_Snow, weathersit_Mist,
    is_peak_hour
]])

input_df.columns = [
    'hr','weekday','temp','atemp','hum','windspeed','year','month','day','dayofweek',
    'season_springer','season_summer','season_winter','yr_2012',
    'mnth_10','mnth_11','mnth_12','mnth_2','mnth_3','mnth_4','mnth_5','mnth_6','mnth_7','mnth_8','mnth_9',
    'holiday_Yes','workingday_Working Day',
    'weathersit_Heavy Rain','weathersit_Light Snow','weathersit_Mist',
    'is_peak_hour'
]

# -----------------------------
# PREDICTION
# -----------------------------
if st.button("Predict Bike Rentals"):
    prediction = model.predict(input_df)
    st.success(f"Predicted Bike Rentals: {int(prediction[0])}")												
												
												
												
												
												
												
												
												
												
												
												
												
												