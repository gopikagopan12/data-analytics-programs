import streamlit as st

pg=st.navigation([
st.Page("bread_eda.py",title="bread"),
st.Page("wine.py",title="wine data analysis"),
st.Page("airline.py",title="airline data analysis"),
st.Page("breast.py",title="brreast data analysis"),
st.Page("mob.py",title="brreast data analysis"),

])

pg.run()