import seaborn as sns
import streamlit as st
import pandas as pd
from class sklearn.preprocessing.LabelEncoder

from sklearn.tree.import DecisionTreeClassifier as dtc
from sklearn.model_selection.train_test_split

st.set_page_config(page_title="WEATHER DATA PREDICTION",page_icone=":moon:",layout="wide")

st.tile(":moon:" weather data analysis")
wdf=pd.read_csv("weather.csv")
st.header("weather dataset")
st.dataframe(wdf.head())

