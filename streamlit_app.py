# Importing necessary libraries
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import streamlit as st

# Load data
@st.cache
def load_data():
    return pd.read_csv('vehicles2_rec.csv')

df = load_data()

# Function for recommending cars based on car manufacturer country. 
# It takes car manufacturer country, color group, type group  and price range as input.

def recommend(made,color_group,type_group,price_range):
    
    # Matching the type with the dataset and reset the index
    data = df.loc[(df['color_group']==color_group) 
                  & (df['type_group']==type_group) & ((df['price']>=price_range[0]) & (df['price']<=price_range[1]))]  
    data.reset_index(level = 0, inplace = True) 
  
    # Convert the index into series
    indices = pd.Series(data.index, index = data['Made'])
    
    #Converting the car manufacturer country into vectors and used unigram
    tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 1), min_df = 1, stop_words='english')
    tfidf_matrix = tf.fit_transform(data['Made'])
    
    # Calculating the similarity measures based on Cosine Similarity
    sg = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    # Get the index corresponding to original_manufacturer
    idx = indices[made]
    # Get the pairwise similarity scores 
    sig = list(enumerate(sg[idx]))
    # Sort the cars
    sig = sorted(sig, reverse=True)
    # Scores of the 6 most similar cars 
    sig = sig[0:6]
    # car indices
    movie_indices = [i[0] for i in sig]
   
    # Top 6 car recommendations
    rec = data[['price','Made','manufacturer', 'model','type','year','Age','condition','fuel','title_status'
                ,'transmission','paint_color','mil_rating','state']].iloc[movie_indices]
    return rec

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
