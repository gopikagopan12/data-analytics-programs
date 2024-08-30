import pandas as pd
import streamlit as st
import numpy as np
import plotly.express as px
st.set_page_config(page_title="myfirst.py",page_icon="None",layout="centered",initial_sidebar_state="auto",menu_items=None)
st.write("hello, *world!* :smile:")
df1=pd.DataFrame({'column1':[1,2,3,4,5],'column2':[11,12,13,14,15]})
st.write("this is a DataFrame",df1)

st.title("this is a title")
st.title("_python_is :blue[cool]:smile")
st.header("one",divider=True)
st.markdown("#Header1")
st.caption(":copyright: cdac tvm")
st.divider()
code='''def hello():
      print("helloworld")'''
st.code(code,language="python")
with st.echo():
     st.write("this code willl be printed")
     st.write(code)
st.latex(r'''\frac{3}{4}''')

## data element



st.table(df1)

st.metric(label="temperature",value="35 *degree celsius",delta="1.2 degree celsius")

st.json({"1":"one","numbers":[1,2,3,4,5]})

## chart elements

chart_data=pd.DataFrame(np.random.randn(20,3),columns=["a","b","c"])
st.area_chart(chart_data)
st.bar_chart(chart_data)
st.line_chart(chart_data)
st.scatter_chart(chart_data)
df2=pd.DataFrame({"Latitude":np.random.randn(1000)/50+37.76,"Longitude":np.random.randn(1000)/50-122.4,"sizes":np.random.randn(1000)*100,"colors":np.random.randn(1000,4).tolist()})
st.map(df2,latitude="Latitude",longitude="Longitude",size="sizes",color='colors')

