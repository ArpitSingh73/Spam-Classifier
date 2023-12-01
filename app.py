# import pickle
# import string
# import nltk
# from nltk.corpus import stopwords
# from nltk.stem.porter import PorterStemmer
# import sklearn
# nltk.download('punkt')
# nltk.download('stopwords')
# import streamlit as st
# var =  st.sidebar.radio("Navigation", ["Home", "About"])

# if var == "Home":
#     tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
#     model = pickle.load(open('model.pkl', 'rb'))


#     st.title("Email Classifier")
#     input_m = st.text_area(label="",placeholder="Enter the email to check", height=200)

#     ps = PorterStemmer()
#     def transform_(text):
#         text = text.lower().strip("\'")
#         text = nltk.word_tokenize(text)

#         list = []
#         for i in text:
#             if i.isalnum():
#                 list.append(i)

#         list2 = list[:]
#         list.clear()

#         for i in list2:
#             if i not in stopwords.words('english') and i not in string.punctuation:
#                 list.append(i)

#         list2 = list[:]
#         list.clear()

#         for i in list2:
#             list.append(ps.stem(i))


#         return " ".join(list)


#     if st.button("Predict"):
#         transformed = transform_(input_m)
#         vector = tfidf.transform([transformed])
#         result = model.predict(vector)[0]


#         if result == 1:
#           st.header("Your input is a SPAM email or message.")
#         else:
#           st.header("Your input is NOT A SPAM email or message.")    

# elif var == "About":
#     st.subheader("Hi there, this is my first project of NLP(machine learning). As by name of the project, this is a spam classifier. The accuracy of this model is of about 96%. I have used Multonomial Naive Bias Classification algorithm to train this model along with some important  NLP concepts like stammering, tokenization, stopwords removal etc. ")      




import streamlit as st 
import pickle
import sklearn
from nltk.corpus import stopwords
import nltk 
import string 
from nltk.stem.porter import PorterStemmer





var =  st.sidebar.radio("Navigation", ["Home", "About"])
if var == "Home":

    tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
    model = pickle.load(open('model.pkl', 'rb'))
    st.title('Spam Classifier')

    input_sms = st.text_area(label="",height=200)
    ps = PorterStemmer()
    nltk.download('punkt')
    nltk.download('stopwords')

    def transform_text(text):
        text = text.lower()
        
        text= nltk.word_tokenize(text)
        y = []
        for i in text:
            if i.isalnum():
                y.append(i)
                
        
        text = y[:]
        y.clear()
        
        for i in text:
            if i not in stopwords.words('english') and  i not in string.punctuation:
                y.append(i)
                
        text = y[:]       
        y.clear()
        
        for i in text:
            y.append(ps.stem(i))
            
        
        return " ".join(y)  


    if st.button('Click to Predict'):
        transform_sms = transform_text(input_sms)
        vector_input = tfidf.transform([transform_sms])
        result = model.predict(vector_input)[0]


        if result == 1:
            st.error("Your input is a SPAM email or message.")
        else:
            st.success('Your input is NOT A SPAM email or message.')



elif var == "About":
    st.subheader("Hi there, this is my first project of NLP(machine learning). As by name of the project, this is a spam classifier. The accuracy of this model is of about 96%. I have used Multonomial Naive Bias Classification algorithm to train this model along with some important  NLP concepts like stemming, tokenization, stopwords removal etc. The data I used, was downloaded from Kaggle.")      
