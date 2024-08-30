import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split as tts
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
