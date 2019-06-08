import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn_evaluation.plot import confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem import PorterStemmer
import nltk
from sklearn.svm import SVC
import random

news = pd.read_csv('News_Category_Dataset_v3.csv')

print(news.head())
print(news['CATEGORY'].unique())

news.CATEGORY.value_counts().plot(kind='pie',
                                  figsize=(8,6),
                                  fontsize=13,
                                  autopct='%1.1f%%',
                                  wedgeprops={'linewidth': 5}
                                  )
plt.axis('off')
plt.axis('equal')
plt.show()

real_news = news['short_description'].str.replace('[^\w\s]','').str.lower()
news['TITLE'] = news['TITLE'].str.replace('[^\w\s]','').str.lower() # unpunctuate and lower case
ps = PorterStemmer()
"""
for new in news['TITLE']:
    words = nltk.word_tokenize(str(new))
    sent = ''
    for word in words:
        word = ps.stem(word)
        sent = sent + word + ' '
    real_news.append(sent)
"""
print(real_news[0:100])

# convert data to vectors
vectorizer = CountVectorizer(stop_words='english')
X = vectorizer.fit_transform(news['TITLE'].values.astype('U'))

y = news['CATEGORY']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1) # 30% split

# fit and score the bayesian classifier
mnb = MultinomialNB(alpha=0.1)
mnb.fit(X_train, y_train)
print(mnb.score(X_test, y_test))

test = random.sample(news, 1000)

results = mnb.predict(test['TITLE'].values.astype('U'))

for result in results:
    print(test['CATEGORY'] + ' and ' + result)
