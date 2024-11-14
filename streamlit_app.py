import streamlit as st
import pandas as pd

st.title('🐧 Penguin Predictor App')

st.info('This app builds machine learning model to predict penguin species!')

with st.expander('Data'):
  st.write('**Raw Data**')
  df = pd.read_csv('https://raw.githubusercontent.com/dataprofessor/data/master/penguins_cleaned.csv')
  df

  st.write('**X**')
  X_raw = df.drop('species', axis=1)
  X_raw

  st.write('**y**')
  y_raw = df.species
  y_raw

with st.expander('Data Visualization'):
  st.scatter_chart(data=df, 
                   x='bill_length_mm', 
                   y='body_mass_g', 
                   x_label='Bill length (mm)',
                   y_label='Body mass (g)',
                   color='species')

# Input feature
with st.sidebar:
  st.header('Input features')
  island = st.selectbox('Island', ('Torgersen', 'Biscoe', 'Dream'))
  gender = st.selectbox('Gender', ('male', 'female'))
  bill_length_mm = st.slider('Bill length (mm)', 32.1, 59.6, 43.9)
  bill_depth_mm = st.slider('Bill depth (mm)', 13.1, 21.5, 17.2)
  flipper_length_mm = st.slider('Flipper length (mm)', 172.0, 231.0, 201.0)
  body_mass_g = st.slider('Body mass (g)', 2700.0, 6300.0, 4207.0)

input_data = {
  "island": island,
  "bill_length_mm": bill_length_mm,
  "bill_depth_mm": bill_depth_mm,
  "flipper_length_mm": flipper_length_mm,
  "body_mass_g": body_mass_g,
  "sex": gender
}

input_df = pd.DataFrame(input_data, index=[0])
combined_df = pd.concat([input_df, X_raw], axis=0)

with st.expander('Input Features'):
  st.write("**Input penguin**")
  input_df
  st.write("**Combined penguins data**")
  combined_df
  
# Data preparation
# Encode X
encode = ['island', 'sex']
df_penguins = pd.get_dummies(combined_df, prefix=encode)
df_penguins
