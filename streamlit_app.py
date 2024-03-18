import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load data for car recommendation app
@st.cache
def load_data():
    return pd.read_csv('vehicles2_rec.csv')

df_recommendation = load_data()

# Load model for car price prediction
filename = 'Auto_Price_Pred_Model'
model = pickle.load(open(filename, 'rb'))

html_temp = """
<div style="background-color:yellow;padding:1.5px">
<h1 style="color:black;text-align:center;">Car Recommendation and Price Prediction</h1>
</div><br>"""
st.markdown(html_temp, unsafe_allow_html=True)

# Function for recommending cars based on car manufacturer country.
def recommend(made,color_group,type_group,price_range):
    # Your existing recommendation function code here
    data = df_recommendation.loc[(df_recommendation['color_group']==color_group) 
                  & (df_recommendation['type_group']==type_group) & ((df_recommendation['price']>=price_range[0]) & (df_recommendation['price']<=price_range[1]))]  
    data.reset_index(level = 0, inplace = True) 
    indices = pd.Series(data.index, index = data['Made'])
    #Converting the car manufacturer country into vectors and used unigram
    tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 1), min_df = 1, stop_words='english')
    tfidf_matrix = tf.fit_transform(data['Made'])
    sg = cosine_similarity(tfidf_matrix, tfidf_matrix)
    idx = indices[made]
    sig = list(enumerate(sg[idx]))
    sig = sorted(sig, reverse=True)
    sig = sig[0:6]
    movie_indices = [i[0] for i in sig]
    rec = data[['price','Made','manufacturer', 'model','type','year','Age','condition','fuel','title_status'
                ,'transmission','paint_color','mil_rating','state']].iloc[movie_indices]
    return rec

# Function to predict car price
def predict_price(make_model, hp_kW, age, km, Gears, Gearing_Type):
    # Your existing prediction function code here
    my_dict = {"make_model":make_model, "hp_kW":hp_kW, "age":age, "km":km, "Gears":Gears, "Gearing_Type":Gearing_Type}
    df = pd.DataFrame.from_dict([my_dict])
    pred = model.predict(df)
    return pred[0].astype(int)

# Streamlit UI
st.title('Car Recommendation and Price Prediction')

# Sidebar for navigation
page = st.sidebar.selectbox("Select Page", ["Car Recommendation", "Car Price Prediction"])

if page == "Car Recommendation":
    st.subheader('Car Recommendation')
    
    # Streamlit UI for car recommendation
    with st.sidebar:
        st.subheader('Car Specs to Recommend')
        made = st.selectbox("Car Manufacturer Country", df_recommendation['Made'].unique())
        color_group = st.selectbox("Color Group", df_recommendation['color_group'].unique())
        type_group = st.selectbox("Type Group", df_recommendation['type_group'].unique())
        price_range = st.slider("Price Range", min_value=0, max_value=50000, value=(5000,10000), step=1000)
    
    # Display recommendations
    if st.button("Recommend"):
        recommendations = recommend(made, color_group, type_group, price_range)
        st.subheader('Top Car Recommendations')
        st.table(recommendations)

elif page == "Car Price Prediction":
    st.subheader('Car Price Prediction')
    
    # Streamlit UI for car price prediction
    with st.sidebar:
        st.subheader('Car Specs to Predict Price')
        make_model = st.selectbox("Model Selection", ("Audi A3", "Audi A1", "Opel Insignia", "Opel Astra", "Opel Corsa", "Renault Clio", "Renault Espace", "Renault Duster"))
        hp_kW = st.number_input("Horse Power:", min_value=40, max_value=294, value=120, step=5)
        age = st.number_input("Age:", min_value=0, max_value=3, value=0, step=1)
        km = st.number_input("km:", min_value=0, max_value=317000, value=10000, step=5000)
        Gears = st.number_input("Gears:", min_value=5, max_value=8, value=5, step=1)
        Gearing_Type = st.radio("Gearing Type", ("Manual", "Automatic", "Semi-automatic"))
    
    # Display selected car specs
    st.write("Selected Specs:")
    st.write(f"- Car Model: {make_model}")
    st.write(f"- Horse Power: {hp_kW}")
    st.write(f"- Age: {age}")
    st.write(f"- km: {km}")
    st.write(f"- Gears: {Gears}")
    st.write(f"- Gearing Type: {Gearing_Type}")

    if st.button("Predict"):
        pred_price = predict_price(make_model, hp_kW, age, km, Gears, Gearing_Type)
        st.write(f"The estimated value of car price is € {pred_price}")

