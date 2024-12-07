import pandas as pd
import numpy as np
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
import streamlit as st

# Load the imdb dataset
word_index = imdb.get_word_index()
reversed_word_index = {value: key for key,value in word_index.items()}

#Load the pre-trained model
model = load_model('Simple_rnn_imdb.h5')

## Helper function to decode review
def decode_review(encoded_review):
    return ' '.join([reversed_word_index.get(i - 3,'?') for i in encoded_review])

## Function to preprocess user input
def preprocess_text(text):
    words = text.lower().split()
    encoded_review =[word_index.get(word,2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review

## Prediction function
def predict_sentiment(review):
    review_array = preprocess_text(review)
    prediction = model.predict(review_array)
    sentiment = 'positive' if prediction[0][0] > 0.5 else 'negative'
    return sentiment,prediction[0][0]

# App title
st.title('Sentiment Analysis')
st.write('Enter sentiment or review to classify it as positive or negative.')

# User input
user_input = st.text_area('Movie Review')

# Prediction

if st.button('Classify'):
    preprocess_input = preprocess_text(user_input)

    ## Make Prediction
    prediction = model.predict(preprocess_input)
    sentiment = 'positive' if prediction[0][0] > 0.5 else 'negative'

    st.write(f'Predicted Sentiment: {sentiment}')
    st.write(f'Prediction Score: {prediction[0][0]*100:.2f}%')
else:
    st.write('Please enter  movie review.')    