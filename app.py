import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

import streamlit as st

tfidf = pickle.load(open('vectorizer.pkl', 'rb'))

model = pickle.load(open('model.pkl', 'rb'))


st.title("Email Classifier")
input_m = st.text_area("Enter the email")

ps = PorterStemmer()
def transform_(text):
    text = text.lower().strip("\'")
    text = nltk.word_tokenize(text)

    list = []
    for i in text:
        if i.isalnum():
            list.append(i)

    list2 = list[:]
    list.clear()

    for i in list2:
        if i not in stopwords.words('english') and i not in string.punctuation:
            list.append(i)

    list2 = list[:]
    list.clear()

    for i in list2:
        list.append(ps.stem(i))


    return " ".join(list)


if st.button("Predict"):
    transformed = transform_(input_m)
    vector = tfidf.transform([transformed])
    result = model.predict(vector)[0]


    if result == 1:
       st.header("Spam")
    else:
      st.header("Not Spam")    