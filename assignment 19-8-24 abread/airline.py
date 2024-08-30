


import streamlit as st
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose

df = pd.read_csv("airline_passengers.csv", index_col = "Month", parse_dates = True)

st.header("Airline Dataset")
st.dataframe(df.head())

st.line_chart(data = df, y = "Passengers")


components = seasonal_decompose(df['Passengers'], model = "multiplicative")
fig1 = components.plot()
st.pyplot(fig1)

