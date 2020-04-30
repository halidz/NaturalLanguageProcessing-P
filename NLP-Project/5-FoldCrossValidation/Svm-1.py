import time

import pandas as pd

#encoding=utf-8
import mysql.connector
import numpy as np
import re
from pandas import DataFrame
from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from snowballstemmer import TurkishStemmer
turkStem=TurkishStemmer()
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2
from sklearn import metrics, svm
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

df = DataFrame(mycursor.fetchall())
df.columns=mycursor.column_names

print(len(df))

print(df.head())

print(df)
print(df.columns)
print(len(df))

print(len(df[df.opinion=="positive"]))


words = stopwords.words("english")
#words = stopwords.words("turkish")



df['cleaned'] = df['comment'].apply(lambda x: " ".join([i for i in x.split() if i not in words]))

msk=8000
xtrain=df.cleaned[:8000]
ytrain=df.opinion[:8000]
xtest=df.cleaned[8000:]
ytest=df.opinion[8000:]
print(len(xtrain))
print(len(ytrain))
print(len(xtest))
X_train, X_test, y_train, y_test = train_test_split(df['cleaned'], df.opinion, test_size=0.2)
print(type(X_train))

pipeline = Pipeline([('vect', TfidfVectorizer( ngram_range=(1,2), sublinear_tf=True)),
                     ('clf', LinearSVC(C=1.0, penalty='l1', max_iter=3000, dual=False))])


model = pipeline.fit(xtrain, ytrain)

print("accuracy score: " + str(model.score(xtest, ytest)))


print(model.predict(['Kötü bir laptop ve begenmedim açıkcası']))
vectorizer = TfidfVectorizer(min_df = 5,
                             max_df = 0.8,
                             sublinear_tf = True,
                             use_idf = True)

train_vectors = vectorizer.fit_transform(xtrain)
test_vectors = vectorizer.transform(xtest)
classifier_linear = svm.SVC(kernel='linear')

t0 = time.time()
classifier_linear.fit(train_vectors, ytrain)
t1 = time.time()
prediction_linear = classifier_linear.predict(test_vectors)
t2 = time.time()
time_linear_train = t1-t0
time_linear_predict = t2-t1
# results
print("Training time: %fs; Prediction time: %fs" % (time_linear_train, time_linear_predict))
report = classification_report(ytest, prediction_linear, output_dict=True)
print('positive: ', report['positive'])
print('negative: ', report['negative'])

