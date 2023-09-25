import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import sklearn

import streamlit as st
var =  st.sidebar.radio("Navigation", ["Home", "About"])

if var == "Home":
    tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
    model = pickle.load(open('model.pkl', 'rb'))


    st.title("Email Classifier")
    input_m = st.text_area(label="",placeholder="Enter the email to check", height=200)

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
          st.header("Your input is a SPAM email or message.")
        else:
          st.header("Your input is NOT A SPAM email or message.")    

elif var == "About":
    st.subheader("Hi there, this is my first project of NLP(machine learning). As by name of the project, this is a spam classifier. The accuracy of this model is of about 96%. I have used Multonomial Naive Bias Classification algorithm to train this model along with some important  NLP concepts like stammering, tokenization, stopwords removal etc. ")      