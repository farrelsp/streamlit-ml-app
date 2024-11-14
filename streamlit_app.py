import streamlit as st
import pandas as pd

st.title('🐧 Penguin Predictor App')

st.info('This app builds machine learning model to predict penguin species!')

with st.expander('Data'):
  st.write('**Raw Data**')
  df = pd.read_csv('https://raw.githubusercontent.com/dataprofessor/data/master/penguins_cleaned.csv')
  df

  st.write('**X**')
  X = df.drop('species', axis=1)
  X

  st.write('**y**')
  y = df.species
  y

with st.expander('Data Visualization'):
  st.scatter_chart(data=df, 
                   x='bill_length_mm', 
                   y='body_mass_g', 
                   x_label='Bill length (mm)',
                   y_label='Body mass (g)',
                   color='species')
with st.sidebar:
  st.header('Input features')
  island = st.selectbox('Island', ('Torgersen', 'Biscoe', 'Dream'))
  gender = st.selectbox('Gender', ('male', 'female'))
  bill_length_mm = st.slider('Bill length (mm)', 
                             df['df.bill_length_mm'].min(), 
                             df['bill_length_mm'].max(), 
                             df['bill_length_mm'].mean())
