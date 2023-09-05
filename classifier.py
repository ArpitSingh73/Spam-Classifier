from collections import Counter
from sklearn.preprocessing import LabelEncoder
import chardet
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
import seaborn as sns
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


data = 'spam.csv'
# because of encoding error, I used this method to read the file-->
with open(data, 'rb') as rawdata:
    result = chardet.detect(rawdata.read(5572))
result

data = pd.read_csv(data, encoding='ISO-8859-1')


print(data.sample(5))
print(data.info)

# ##### -------------------Steps to be performed----------------------------
# Data info
# Data cleaning
# EDA
# Text Preprocessing
# Model building
# Evaluation
# Required changes
# Deploy

data = data[['v1', 'v2']]
print(data.sample(10))
# renaming the columns name-->
data.rename(columns={'v1': 'type', 'v2': 'text'}, inplace=True)


# performing encoding on target column-->
encoder = LabelEncoder()
data['type'] = encoder.fit_transform(data['type'])

print(data.sample(10))
print(data.isnull().sum())
print(data.duplicated().sum())

# dropping duplicats values-->
data = data.drop_duplicates(keep='first')
print(data.duplicated().sum())


# pie chart for visualization of ham vs spam values-->
plt.pie(data['type'].value_counts(), labels=['ham', 'spam'], autopct="0.2f")
plt.show()

# adding 3 more columns which contains number of characters, words and sentences respectively-->
data['char_count'] = data['text'].apply(lambda m: len(m))
data['word_count'] = data['text'].apply(lambda m: nltk.word_tokenize(m))
data['sent_count'] = data['text'].apply(lambda m: nltk.sent_tokenize(m))


x = data[data['type'] == 1][['char_count', 'word_count', 'sent_count']].describe()
y = data[data['type'] == 0][['char_count', 'word_count', 'sent_count']].describe()


print(x)
print()
print(y)


# plotting the histplot chart of ham vs spam character count-->
sns.histplot(data[data['type'] == 0]['char_count'])
sns.histplot(data[data['type'] == 1]['char_count'], color='green')
sns.pairplot(data, hue='type')
# from resulting graph we can see that number of words per spam messages are grrater than ham message-->>


# creating object of PortStammer-->
ps = PorterStemmer()



# Creating a function for further data preprocesiing like-->

# converting to lowecase
# removal of characters other than alphanumeric values
# removal of stopwords 
# punctuations
# stemming(reducing words to its nearest possible root words)

def transform_(text):
    text = text.lower()
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


# creting a new column in existing dataframe to store the result of above function-->
data['tranf_text'] = data['text'].apply(transform_)
print(data.head())


# visualizing the top 30 words-->
spam_corpus = []
for msg in data[data['type'] == 1]['tranf_text'].tolist():
    for word in msg.split():
        spam_corpus.append(word)

print(len(spam_corpus))

sns.barplot(pd.DataFrame(Counter(spam_corpus).most_common(30))[
            0], pd.DataFrame(Counter(spam_corpus).most_common(30))[1])
plt.xticks(rotation='vertical')
plt.show()


ham_corpus = []
for msg in data[data['type'] == 0]['tranf_text'].tolist():
    for word in msg.split():
        ham_corpus.append(word)
print(len(ham_corpus)
      )
sns.barplot(pd.DataFrame(Counter(ham_corpus).most_common(30))[
            0], pd.DataFrame(Counter(ham_corpus).most_common(30))[1])
plt.xticks(rotation='vertical')
plt.show()



# 