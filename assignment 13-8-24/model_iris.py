import streamlit as st
import pickle

st.set_page_config(page_title="Iris Species prediction",page_icon=" :tulip: ",layout="wide")
st.title(" :tulip: iris prediction for species :tulip: ")


model1=pickle.load(open('svcm1.pkl','rb'))

c1,c2=st.columns(2)
n1=c1.number_input("Sepal length")
n2=c2.number_input("Sepal width")
n3=c1.number_input("petal length")
n4=c2.number_input("petal width")

new_features=[[n1,n2,n3,n4]]

if st.button("Predict the species"):
    t1=model1.predict(new_features)
    c1.subheader("predicted species is ")
    if t1==0:
        st.write("Iris setosa")
    elif t1==1:
        st.write("Iris versicolor")
    elif t1==2:
        st.write("Iris virginica")
    else:
        st.write("flower not listed")