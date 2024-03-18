# Importing necessary libraries
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import streamlit as st
import joblib

# Load data
@st.cache
def load_data():
    return pd.read_csv('vehicles2_rec.csv')

df = load_data()

# Load price prediction model
price_model = joblib.load('rec_price.pkl')

# Function for recommending cars based on car manufacturer country. 
# It takes car manufacturer country, color group, type group, and price range as input.
def recommend(made, color_group, type_group, price_range):
    # Your existing code for recommendation here...

# Streamlit UI
st.title('Car Recommendation System')

# Sidebar for user inputs
st.sidebar.title('Filters')
made = st.sidebar.selectbox('Car Manufacturer Country', df['Made'].unique())
color_group = st.sidebar.selectbox('Color Group', df['color_group'].unique())
type_group = st.sidebar.selectbox('Type Group', df['type_group'].unique())
price_range = st.sidebar.slider('Price Range', min_value=0, max_value=50000, value=(5000,10000), step=1000)

# Display recommendations
if st.sidebar.button('Recommend'):
    recommendations = recommend(made, color_group, type_group, price_range)
    st.subheader('Top Car Recommendations')
    st.dataframe(recommendations)

# Price prediction section
st.title('Price Prediction')

# Input fields for price prediction
car_engine = st.number_input('Car Engine')
car_accident = st.number_input('Car Accident')
car_year = st.number_input('Car Year')
car_owner = st.number_input('Car Owner')

# Predict price
if st.button('Predict Price'):
    features = [[car_engine, car_accident, car_year, car_owner]]
    price_prediction = price_model.predict(features)
    st.success(f'Predicted Price: ${price_prediction[0]:,.2f}')
