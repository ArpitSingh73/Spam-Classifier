# from collections import Counter
# from sklearn.preprocessing import LabelEncoder
# import chardet
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import nltk
# import seaborn as sns
# import string
# import sklearn
# nltk.download('stopwords')
# nltk.download('punkt')
# from nltk.corpus import stopwords
# from nltk.stem.porter import PorterStemmer
# import pickle

# data = 'spam.csv'
# # because of encoding error, I used this method to read the file-->
# with open(data, 'rb') as rawdata:
#     result = chardet.detect(rawdata.read(5572))
# result

# data = pd.read_csv(data, encoding='ISO-8859-1')


# # print(data.sample(5))
# # print(data.info)

# # ##### -------------------Steps to be performed----------------------------
# # Data info
# # Data cleaning
# # EDA
# # Text Preprocessing
# # Model building
# # Evaluation
# # Required changes
# # Deploy

# data = data[['v1', 'v2']]
# # print(data.sample(10))
# # renaming the columns name-->
# data.rename(columns={'v1': 'type', 'v2': 'text'}, inplace=True)


# # performing encoding on target column-->
# encoder = LabelEncoder()
# data['type'] = encoder.fit_transform(data['type'])

# # print(data.sample(10))
# # print(data.isnull().sum())
# # print(data.duplicated().sum())

# # dropping duplicats values-->
# data = data.drop_duplicates(keep='first')
# # print(data.duplicated().sum())

# # pie chart for visualization of ham vs spam values-->
# plt.pie(data['type'].value_counts(), labels=['ham', 'spam'], autopct="0.2f")
# # plt.show()

# # adding 3 more columns which contains number of characters, words and sentences respectively-->
# data['char_count'] = data['text'].apply(lambda m: len(m))
# data['word_count'] = data['text'].apply(lambda m: nltk.word_tokenize(m))
# data['sent_count'] = data['text'].apply(lambda m: nltk.sent_tokenize(m))


# x = data[data['type'] == 1][['char_count', 'word_count', 'sent_count']].describe()
# y = data[data['type'] == 0][['char_count', 'word_count', 'sent_count']].describe()


# # print(x)
# # print()
# # print(y)


# # plotting the histplot chart of ham vs spam character count-->
# sns.histplot(data[data['type'] == 0]['char_count'])
# sns.histplot(data[data['type'] == 1]['char_count'], color='green')
# sns.pairplot(data, hue='type')
# # from resulting graph we can see that number of words per spam messages are grrater than ham message-->>


# # creating object of PortStammer-->
# ps = PorterStemmer()



# # Creating a function for further data preprocesiing like-->

# # converting to lowecase
# # removal of characters other than alphanumeric values
# # removal of stopwords 
# # punctuations
# # stemming(reducing words to its nearest possible root words)

# def transform_(text):
#     text = text.lower().strip("\'")
#     text = nltk.word_tokenize(text)

#     list = []
#     for i in text:
#         if i.isalnum():
#             list.append(i)

#     list2 = list[:]
#     list.clear()

#     for i in list2:
#         if i not in stopwords.words('english') and i not in string.punctuation:
#             list.append(i)

#     list2 = list[:]
#     list.clear()

#     for i in list2:
#         list.append(ps.stem(i))


#     return " ".join(list)


# # creting a new column in existing dataframe to store the result of above function-->
# data['tranf_text'] = data['text'].apply(transform_)
# # print(data.head())
# print("line 123")


# # visualizing the top 30 words-->
# spam_corpus = []
# for msg in data[data['type'] == 1]['tranf_text'].tolist():
#     for word in msg.split():
#         spam_corpus.append(word)

# print(len(spam_corpus))

# sns.barplot(pd.DataFrame(Counter(spam_corpus).most_common(30))[
#             0], pd.DataFrame(Counter(spam_corpus).most_common(30))[1])
# plt.xticks(rotation='vertical')
# # plt.show()


# ham_corpus = []
# for msg in data[data['type'] == 0]['tranf_text'].tolist():
#     for word in msg.split():
#         ham_corpus.append(word)
# print(len(ham_corpus)
#       )
# sns.barplot(pd.DataFrame(Counter(ham_corpus).most_common(30))[
#             0], pd.DataFrame(Counter(ham_corpus).most_common(30))[1])
# plt.xticks(rotation='vertical')
# # plt.show()



# # 


# from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# cv = CountVectorizer()
# tfidf = TfidfVectorizer(max_features=3000)

# print("161")
# x = tfidf.fit_transform(data['tranf_text']).toarray()
# y = data['type'].values

# from sklearn.model_selection import train_test_split

# x_train, x_test , y_train , y_test = train_test_split(x, y, test_size=0.2, random_state=2)


# from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
# from sklearn.metrics import accuracy_score, confusion_matrix, precision_score

# gb = GaussianNB()
# mnb = MultinomialNB()
# br = BernoulliNB()

# mnb.fit(x_train, y_train)
# y_pred = mnb.predict(x_test)
# print(accuracy_score(y_test, y_pred))
# print(confusion_matrix(y_test, y_pred))
# print(precision_score(y_test, y_pred))
# pickle.dump(tfidf, open('vectorizer.pkl', 'wb'))

# pickle.dump(mnb, open('model.pkl', 'wb'))













import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
df = pd.read_csv('spam.csv' , encoding = 'latin-1')

# 1. Data cleaning 
df.drop( columns = ['Unnamed: 2','Unnamed: 3','Unnamed: 4'], inplace = True)


#rename 
df.rename(columns = {'v1' :'target','v2':'text'}, inplace = True)



from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
df['target']=encoder.fit_transform(df['target'])

# check missing value 

df.isna().sum()


df.duplicated().sum()



df.drop_duplicates(keep = 'first' , inplace = True)





df.duplicated().sum()

# print(df.head())


import matplotlib.pyplot as plt 
plt.pie(df['target'].value_counts() , labels = ['ham', 'spam'] ,autopct = '%0.2f')
plt.show()

import nltk
nltk.download('punkt')


df['num_character']=df['text'].apply(len)


df['num_word']=df['text'].apply( lambda x: nltk.word_tokenize(x)).apply(len)
df['num_sentence']=df['text'].apply( lambda x: nltk.sent_tokenize(x)).apply(len)


# df[['num_character', 'num_word', 'num_sentence']].describe()


# import seaborn as sns
# plt.figure(figsize=(10,5))
# sns.histplot(df[df['target']==0]['num_character'], color = 'yellow')
# sns.histplot(df[df['target']==1]['num_character'] , color = 'red')
# plt.show()


# plt.figure(figsize=(10,5))
# sns.histplot(df[df['target']==0]['num_word'], color = 'yellow')
# sns.histplot(df[df['target']==1]['num_word'] , color = 'red')
# plt.show()


# sns.pairplot(df,hue= ('target'))
# plt.show()




import string 
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

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


from nltk.corpus import stopwords 
nltk.download('stopwords')

df['transform_text']=df['text'].apply(transform_text)



#word cloud

# from wordcloud import WordCloud
# wc= WordCloud(width = 700 , height = 700 , min_font_size = 10 ,background_color = 'yellow')
# spam_wc = wc.generate(df[df['target'] == 1]['transform_text'].str.cat(sep = ' '))
# plt.imshow(spam_wc)


# ham_wc = wc.generate(df[df['target'] == 0]['transform_text'].str.cat(sep = ' '))
# plt.imshow(ham_wc)



#top 50 words 

spam_corpus = []
for msg in df[df['target'] == 1]['transform_text'].tolist():
    for word in msg.split():
        spam_corpus.append(word)
    


ham_corpus = []
for msg in df[df['target'] == 0]['transform_text'].tolist():
    for word in msg.split():
        ham_corpus.append(word)    

from collections import Counter
pd.DataFrame(Counter(spam_corpus).most_common(50))

pd.DataFrame(Counter(ham_corpus).most_common(50))        



from sklearn.feature_extraction.text import CountVectorizer , TfidfVectorizer
cv = CountVectorizer()
tfidf = TfidfVectorizer()
X = tfidf.fit_transform(df['transform_text']).toarray()


y = df['target'].values


from sklearn.model_selection import train_test_split
X_train, X_test , y_train , y_test = train_test_split(X,y , test_size = 0.2 , random_state = 2 )
from sklearn.naive_bayes import GaussianNB, MultinomialNB ,BernoulliNB
from sklearn.metrics import accuracy_score ,confusion_matrix, precision_score 
gnb = GaussianNB()
mnb = MultinomialNB()
bnb = BernoulliNB()
gnb.fit(X_train, y_train)
y_pred1 = gnb.predict(X_test)
print(accuracy_score(y_test, y_pred1))
print(confusion_matrix(y_test, y_pred1))
print(precision_score(y_test, y_pred1))



mnb.fit(X_train, y_train)
y_pred2 = mnb.predict(X_test)
print(accuracy_score(y_test, y_pred2))
print(confusion_matrix(y_test, y_pred2))
print(precision_score(y_test, y_pred2))


bnb.fit(X_train, y_train)
y_pred3 = bnb.predict(X_test)
print(accuracy_score(y_test, y_pred3))
print(confusion_matrix(y_test, y_pred3))
print(precision_score(y_test, y_pred3))


# from sklearn.linear_model import LogisticRegression
# from sklearn.svm import SVC
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier , ExtraTreesClassifier, GradientBoostingClassifier
# from xgboost import XGBClassifier
import pickle
pickle.dump(tfidf,open('vectorizer.pkl','wb'))
pickle.dump(mnb,open('model.pkl','wb'))
df['transform_text'][4532]