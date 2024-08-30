# 4. Make a web app for Salary prediction using polynomial regression, do eda
# in one page and prediction in another page.
# EDA page


import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


st.set_page_config(page_title="Salary",page_icon=" :moneybag: ",layout="wide")
st.title(" :money_with_wings: Salary Data Analysis :moneybag: ")

sdf = pd.read_csv("Position_Salaries.csv")
st.subheader("Salary Dataset")
st.dataframe(sdf.head())

lin_model = LinearRegression()
poly = PolynomialFeatures(degree=4)

#=== variables
x = sdf[['Level']]
y = sdf[['Salary']]

#== Linear Model training
xpoly = poly.fit_transform(x)
lin_model.fit(xpoly, y)

#== plotting the graph
fig, ax = plt.subplots(figsize=(2,2))
ax.scatter(x,y,c="blue")
ax.plot(x, lin_model.predict(poly.fit_transform(x)), c='g')
st.pyplot(fig)

#== saving the model to a file for prediction
saved_model = pickle.dump(lin_model, open('salary.pkl', 'wb'))





