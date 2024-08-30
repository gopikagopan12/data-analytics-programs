import streamlit as st
import time
import pandas as pd
import numpy as np
import warnings
import plotly.express as px
from datetime import *
warnings.filterwarnings('ignore')
st.set_page_config(page_title="superstore analystics dashboard!!!",page_icon=":smile:",layout="wide")
st.title(":smile: superstore analystics dashboard !!!")
df=pd.read_excel('Superstore.xlsx',sheet_name='Superstore')
#st.subheader("First 5 rows of HR data")
#st.dataframe(df.head())
col1,col2=st.columns(2)


st.sidebar.header("Choose your filter: ")
Region=st.sidebar.multiselect("Pick your region",df["Region"].unique())
State=st.sidebar.multiselect("Pick your state",df["State"].unique())
City=st.sidebar.multiselect("Pick your city",df["City"].unique())
col1.subheader("select start date")

d1=col1.date_input("when do you start",datetime(2000,7,6))
#st.write("your start is",d1)

col2.subheader("select end date")
d2=col2.date_input("when do u finish",datetime(2000,7,6))

#st.write("your end date is",d2)






col1.subheader("category wise sales") 
    
table1=pd.pivot_table(df,values='Sales',index=['Category'],aggfunc=np.sum).reset_index()

fig1=px.bar(table1,x="Sales",y="Category")
col1.plotly_chart(fig1,use_container_width=True)
    
col2.subheader("Region wise sales")
if not Region:
    st.warning("select region")
else:   
    table2=pd.pivot_table(df,values='Sales',index=['Region'],aggfunc=np.sum).reset_index()
    fig2=px.pie(table2,values="Sales",names="Region",hole=0.5)
    col2.plotly_chart(fig2,use_container_width=True)

col1.write(table1)
col2.write(table2)

col1.subheader("Time seires Analysis")
table3=pd.pivot_table(df,values="Sales",index=['Order Date'],aggfunc=np.sum).reset_index()
fig3=px.line(table3,x="Order Date",y="Sales")
col1.plotly_chart(fig3,use_container_width=True)
st.divider()



col1.subheader("Segment wise sales")
table5=pd.pivot_table(df,values="Sales",index=['Segment'],aggfunc=np.sum).reset_index()
fig5=px.pie(table5,values="Sales",names="Segment",hole=0)
col2.plotly_chart(fig5,use_container_width=True)
st.divider()

col2.subheader("hierarchial view of sales using treemap")
df.fillna("None")
fig=px.treemap(df,path=["Region","Category","Sub-Category"],values='Sales')
col2.plotly_chart(fig)
st.divider()







col2.subheader("category wise SALES")
table6=pd.pivot_table(df,values='Sales',index=['Category'],aggfunc=np.sum).reset_index()
fig6=px.pie(table6,values="Sales",names="Category",hole=0)
col2.plotly_chart(fig6,use_container_width=True)






