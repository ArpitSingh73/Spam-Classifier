import pandas as pd
import numpy as np


df = 'spam.csv'



import chardet
with open(df, 'rb') as rawdata:
    result = chardet.detect(rawdata.read(5572))
result


data=pd.read_csv(df , encoding='ISO-8859-1')


# print(data.sample(5))
# print(data.info)




# Data info
# Data cleaning
# EDA
# Text Preprocessing
# Model building
# Evaluation
# Required changes
# Deploy

data = data[['v1', 'v2']]
# print(data.sample(10))
data.rename(columns = {'v1':'type', 'v2':'text' }, inplace=True)



from sklearn.preprocessing import LabelEncoder


encoder = LabelEncoder()


data['type'] = encoder.fit_transform(data['type'])

# print(data.sample(10))


print(data.isnull().sum())


print(data.duplicated().sum())

data = data.drop_duplicates(keep='first') 

print(data.duplicated().sum())


