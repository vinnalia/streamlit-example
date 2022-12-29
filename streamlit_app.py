import nltk
import re
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from collections import namedtuple
import altair as alt
import math
import pandas as pd
import streamlit as st
import pickle
nltk.download('stopwords')
from nltk.corpus import stopwords

"""
## Kelompok 2
"""

st.write("halo ini vinna")

with open("model_pkl", 'rb') as file:
    model = pickle.load(file)

cv = CountVectorizer(max_features=1500)
corpus = []
ps = PorterStemmer()

df = pd.read_csv('Restaurant_Reviews.tsv', delimiter='\t', quoting=3)
for i in range(0,df.shape[0]):
    message = re.sub(pattern='[^a-zA-Z]', repl=' ', string=df.Review[i]) #Cleaning special character from the message
    message = message.lower() #Converting the entire message into lower case
    words = message.split() # Tokenizing the review by words
    words = [word for word in words if word not in set(stopwords.words('english'))] #Removing the stop words
    words = [ps.stem(word) for word in words] #Stemming the words
    message = ' '.join(words) #Joining the stemmed words
    corpus.append(message) #Building a corpus of messages
# corpus[0:10]
X = cv.fit_transform(corpus).toarray()

def predict_review(sample_message):
    sample_message = re.sub(
        pattern='[^a-zA-Z]', repl=' ', string=sample_message)
    sample_message = sample_message.lower()
    sample_message_words = sample_message.split()
    sample_message_words = [word for word in sample_message_words if not word in set(
        stopwords.words('english'))]
    ps = PorterStemmer()
    final_message = [ps.stem(word) for word in sample_message_words]
    final_message = ' '.join(final_message)
    temp = cv.transform([final_message]).toarray()
    return model.predict(temp)


input_review = st.text_area(label="Masukkan review (dalam bahasa Inggris):",
                            placeholder="Contoh: I like this course...")
analisis_button = st.button(label="Analisis")

result = ['OH NO! Negative Review','OMG! Positive Review']

if analisis_button:
    hasil_analisis = predict_review(input_review)
    if predict_review(input_review):
        st.write("Hasil Analisis " + result[1])
    else:
        st.write("Hasil Analisis " + result[0])
