#encoding=utf-8
import pandas as pd
import mysql.connector
import numpy as np
import re
from pandas import DataFrame
from snowballstemmer import TurkishStemmer
turkStem=TurkishStemmer()
from sklearn import metrics
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfTransformer
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import pickle


mydb = mysql.connector.connect(
  host="localhost",
  user="root",
  passwd="123hal123",
  database="comments",
)

mycursor = mydb.cursor()

mycursor.execute("SELECT comment, opinion FROM commenttableN11other ")

df = DataFrame(mycursor.fetchall())

df.columns=mycursor.column_names

words = stopwords.words("turkish")

df['cleaned'] = df['comment'].apply(lambda x: " ".join([i for i in x.split() if i not in words]).lower())

X_train ,X_test, y_train ,y_test = train_test_split(df['cleaned'],df.opinion,test_size=0.2)

print(X_train)
print(y_train)
print(X_test)
print(y_test)


count = CountVectorizer(min_df=10,ngram_range=(1,2),max_df=0.9)

temp=count.fit_transform(X_train)

tdif = TfidfTransformer()

temp2 = tdif.fit_transform(temp)

text_regression=LogisticRegression()

model=text_regression.fit(temp2,y_train)

prediction_data = tdif.transform((count.transform(X_test)))

predicted=model.predict(prediction_data)



print(model.get_params())
print(np.mean(predicted == y_test))
print(model.predict(tdif.transform(count.transform(["laptopu beğenmedim"]))))

print("Accuracy:",metrics.accuracy_score(y_test, predicted))
# Get precision and recall

#print("Precision:",metrics.precision_score(y_test, predicted))
#cv = CountVectorizer(ngram_range=(3,10),min_df=10, analyzer = 'char')

#print(cv.fit_transform(df).toarray())

# now you can save it to a file
#with open('filename.pkl', 'wb') as f:
  #  pickle.dump(clf, f)
