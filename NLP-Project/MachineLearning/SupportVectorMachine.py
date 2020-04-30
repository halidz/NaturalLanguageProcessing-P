#encoding=utf-8
import json as j
import pandas as pd
import re
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import mysql.connector
import numpy as np
import re
from pandas import DataFrame
from snowballstemmer import TurkishStemmer
from TurkishStemmer import TurkishStemmer
import pickle

turkStem=TurkishStemmer()

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfTransformer
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

mydb = mysql.connector.connect(
  host="localhost",
  user="root",
  passwd="123hal123",
  database="comments",
)

mycursor = mydb.cursor()

mycursor.execute("SELECT comment, opinion FROM commenttablehepsinew where opinion='negative'")

df = DataFrame(mycursor.fetchall())
df.columns=mycursor.column_names

mycursor.execute("SELECT comment, opinion FROM commenttablen11dx where opinion='negative'")

dfx = DataFrame(mycursor.fetchall())

dfx.columns=mycursor.column_names

df=pd.concat([df,dfx],ignore_index=True)

mycursor.execute("SELECT comment, opinion FROM commenttablen11dx where opinion='positive' LIMIT 1300")


dfxx = DataFrame(mycursor.fetchall())
dfxx.columns=mycursor.column_names
print(len(dfxx))
df=pd.concat([df,dfxx],ignore_index=True)

#mycursor.execute("SELECT comment, opinion FROM commenttablehepsinew")

#df = DataFrame(mycursor.fetchall())
#df.columns=mycursor.column_names

#mycursor.execute("SELECT comment, opinion FROM commenttablen11dx")


#dfx = DataFrame(mycursor.fetchall())
#dfx.columns=mycursor.column_names

#df=pd.concat([df,dfx])


stemmer = SnowballStemmer('english')
words = stopwords.words("turkish")


df['cleaned'] = df['comment'].apply(lambda x: " ".join([i for i in x.split() if i not in words]).lower())
X_train, X_test, y_train, y_test = train_test_split(df['cleaned'], df.opinion, test_size=0.2)


pipeline = Pipeline([('vect', TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)),
                     ('chi',  SelectKBest(chi2, k=10000)),
                     ('clf', LinearSVC(C=1.0, penalty='l1', max_iter=3000, dual=False))])


model = pipeline.fit(X_train, y_train)

vectorizer = model.named_steps['vect']
chi = model.named_steps['chi']
clf = model.named_steps['clf']


print("accuracy score: " + str(model.score(X_test, y_test)))


print(model.predict(['iyi bir laptop ve begendim açıkcası']))

feature_names = vectorizer.get_feature_names()
feature_names = [feature_names[i] for i in chi.get_support(indices=True)]
feature_names = np.asarray(feature_names)

# now you can save it to a file
#with open('filename.pkl', 'wb') as f:
 #   pickle.dump(clf, f)
