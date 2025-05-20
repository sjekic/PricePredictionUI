import streamlit as st

st.set_page_config(
    page_title="Computer Price Predictor",
    page_icon="💻",
)


st.title("💻 Computer Price Predictor")

st.image("https://images.pexels.com/photos/115655/pexels-photo-115655.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2")

st.markdown(
    """
    ### Welcome to our Final Project!
    
    **Navigate using the sidebar** to explore different sections:
    
    - **KNN**: See the 5 most similar laptops to the one whose specifications you choose using a KNN model
    - **PricePredictor**: Predict computer prices based on specifications using an XGBoost model
    - **Data Explorer**: Explore the data we used to train the model
    
    ### Project Overview
    This project was developed for our Machine Learning Foundations course. It was done entirely in Python, with Streamlit for the UI. Done by Group 1. 
    """
)
