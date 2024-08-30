# 2. Make a web app for spam detection using BernoulliNB and MultinomialNB algorithm, 
#    do eda in one page and prediction in another page.
import streamlit as st

pg=st.navigation([
st.Page("spam_eda_models.py",title="EDA Exploratory Data Analysis"),
st.Page("spam_prediction.py",title="Prediction"),
])

pg.run()