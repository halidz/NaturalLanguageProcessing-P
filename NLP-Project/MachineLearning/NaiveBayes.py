import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import mysql.connector
import numpy as np
import re
from pandas import DataFrame
from snowballstemmer import TurkishStemmer
turkStem=TurkishStemmer()
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfTransformer
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

import pickle

# and later you can load it
#with open('filename.pkl', 'rb') as f:
   # clf = pickle.load(f)


mydb = mysql.connector.connect(
  host="localhost",
  user="root",
  passwd="123hal123",
  database="comments",
)

mycursor = mydb.cursor()

mycursor.execute("SELECT comment, opinion FROM generaltable")

df = DataFrame(mycursor.fetchall())
df.columns=mycursor.column_names

print(len(df))


#mycursor.execute("SELECT comment, opinion FROM commenttablehepsinew")

#df = DataFrame(mycursor.fetchall())
#df.columns=mycursor.column_names

#mycursor.execute("SELECT comment, opinion FROM commenttablen11dx")


#dfx = DataFrame(mycursor.fetchall())
#dfx.columns=mycursor.column_names

#df=pd.concat([df,dfx],ignore_index=True)



print(df.head())
print(df)
print(len(df))
print(len(df[df.opinion=="positive"]))

stemmer = SnowballStemmer('english')
words = stopwords.words("turkish")
words.append("bir")

df['cleaned'] = df['comment'].apply(lambda x: " ".join([i for i in x.split() if i not in words]).lower())

X_train ,X_test, y_train ,y_test = train_test_split(df['cleaned'],df.opinion,test_size=0.2)

count = CountVectorizer(ngram_range=(1,2),min_df=10, analyzer = 'word')

temp=count.fit_transform(X_train)          #extracting feature from data


wordlist=[]
text = " ".join(review for review in count.vocabulary_)

print(text)
print(count.vocabulary_.__len__())
print(count.get_feature_names())
tdif = TfidfTransformer()                 #gives low weights to common words by dividinf word freq to # of words in document

temp2 = tdif.fit_transform(temp)

gnb = GaussianNB()

mnb=MultinomialNB()

model=mnb.fit(temp2,y_train)

prediction_data = tdif.transform((count.transform(X_test)))

predicted=model.predict(prediction_data)


print(np.mean(predicted == y_test))

print(model.predict(tdif.transform(count.transform(["Alırken acaba ısınır mı diye düşünmüştüm ancak kullandığım program ve yoğun oyun temposuna rağmen sıcaklık adına hiç bir şey olmadı"]))))


from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt




# Create and generate a word cloud image:
wordcloud = WordCloud(background_color="white").generate(text)

# Display the generated image:
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")

plt.savefig("wordcloud3.png")

plt.show()


# now you can save it to a file
# open('naivebayes.pkl', 'wb') as f:
 #   pickle.dump(clf, f)
