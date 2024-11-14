import streamlit as st
import pandas as pd

st.title('🐧 Penguin Predictor App')

st.info('This app builds machine learning model to predict penguin species!')

df = pd.read_csv('https://raw.githubusercontent.com/dataprofessor/data/master/penguins_cleaned.csv')
df
