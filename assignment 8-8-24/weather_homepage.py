# 1. Make a web app for Weather prediction using Naive Bayes algorithm,
#    do eda in one page and prediction in another page.

import streamlit as st

pg=st.navigation([
st.Page("weather_eda_nb.py",title="Weather EDA (Exploratory Data Analysis)"),
st.Page("weather_prediction_nb.py",title="Weather Prediction"),
])

pg.run()