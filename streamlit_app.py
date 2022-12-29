from collections import namedtuple
import altair as alt
import math
import pandas as pd
import streamlit as st
import pickle
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.porter import PorterStemmer
import re

"""
## Kelompok 2
"""

st.write("halo ini vinna")

model = pickle.load(open("model_pkl", 'rb'))

cv = CountVectorizer(max_features=1500)


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

# if analisis_button:
#     hasil_analisis = predict_review(input_review)
#     st.write("Hasil Analisis : " + hasil_analisis)
