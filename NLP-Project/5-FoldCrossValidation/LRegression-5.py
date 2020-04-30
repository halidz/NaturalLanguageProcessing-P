import pandas as pd

#encoding=utf-8
import mysql.connector
import numpy as np
import re
from pandas import DataFrame
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from snowballstemmer import TurkishStemmer
turkStem=TurkishStemmer()
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2
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

mycursor.execute("SELECT comment, opinion FROM generaltable ")

df=DataFrame(mycursor.fetchall())
df.columns=mycursor.column_names

print(len(df))



print(df.head())

print(df)
print(df.columns)
print(len(df))

print(len(df[df.opinion=="positive"]))


#words = stopwords.words("english")
words = stopwords.words("turkish")



#df['cleaned'] = df['comment'].apply(lambda x: " ".join([turkStem.stemWord(i) for i in re.sub("[^a-zA-Z]", " ", x).split() if i not in words]).lower())
df['cleaned'] = df['comment'].apply(lambda x: " ".join([i for i in x.split() if i not in words]).lower())

xtrain=df.cleaned[800:]

ytrain=df.opinion[800:]

xtest=df.cleaned[:800]
ytest=df.opinion[:800]


X_train, X_test, y_train, y_test = train_test_split(df['cleaned'], df.opinion, test_size=0.2)

count = CountVectorizer(ngram_range=(1,2),analyzer="word",min_df=5,max_df=0.9)
#count = CountVectorizer(ngram_range=(1,2),analyzer="word")
#count = CountVectorizer(lowercase=False)

#count=CountVectorizer()
temp=count.fit_transform(xtrain)
print(count.vocabulary_.__len__())
#print(count.get_feature_names())
print(count.get_params())    #her satırda bir cümle nin vectorizeri

tdif = TfidfTransformer()

temp2 = tdif.fit_transform(temp)
print(temp2)
text_regression=LogisticRegression()

model=text_regression.fit(temp2,ytrain)

prediction_data = tdif.transform((count.transform(xtest)))
#prediction_data = count.transform(xtest)

predicted=model.predict(prediction_data)



print(model.get_params())

print(np.mean(predicted == ytest))

print(model.predict(tdif.transform(count.transform(["laptopu beğenmedim"]))))




# Get precision and recall

#print("Precision:",metrics.precision_score(y_test, predicted))
#cv = CountVectorizer(ngram_range=(3,10),min_df=10, analyzer = 'char')

#print(cv.fit_transform(df).toarray())

# now you can save it to a file
#with open('filename.pkl', 'wb') as f:
  #  pickle.dump(clf, f)
"""""
stemmer = SnowballStemmer('english')
words = stopwords.words("turkish")


#df['cleaned'] = df['comment'].apply(lambda x: " ".join([turkStem.stemWord(i) for i in re.sub("[^a-zA-Z]", " ", x).split() if i not in words]).lower())
df['cleaned'] = df['comment'].apply(lambda x: " ".join([turkStem.stemWord(i) for i in x.split() if i not in words]).lower())

xtrain=df.cleaned[2000:]

ytrain=df.opinion[2000:]

xtest=df.cleaned[:2000]
ytest=df.opinion[:2000]


X_train, X_test, y_train, y_test = train_test_split(df['cleaned'], df.opinion, test_size=0.2)

count = CountVectorizer(min_df=2,ngram_range=(1,2))

#count=CountVectorizer()
temp=count.fit_transform(xtrain)
print(count.vocabulary_)
print(count.get_feature_names())
print(count.get_params())    #her satırda bir cümle nin vectorizeri
print(temp[1])
tdif = TfidfTransformer()

temp2 = tdif.fit_transform(temp)
print(temp2)
text_regression=LogisticRegression()

model=text_regression.fit(temp2,ytrain)
print(model)
prediction_data = tdif.transform((count.transform(xtest)))
print(prediction_data)
predicted=model.predict(prediction_data)

print(model.get_params())

print(np.mean(predicted == ytest))

print(model.predict(tdif.transform(count.transform(["laptopu beğendim"]))))


print("Accuracy:",metrics.accuracy_score(ytest, predicted))
"""""