import numpy as np
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

def getModel (news):
    news['TITLE'] = news['TITLE'].str.replace('[^\w\s]', '').str.lower()
    vectorizer = CountVectorizer(stop_words='english')
    x_train, x_test, y_train, y_test = train_test_split(news['TITLE'], news['CATEGORY'], test_size=0.1)  # 10% split
    x_train = vectorizer.fit_transform(x_train.values.astype('U'))
    model = MultinomialNB(alpha=0.1)
    model.fit(x_train, y_train)
    return model, vectorizer


def getInputClasses(input, model, vectorizer):
    print('\n' + 'Your input is: ' + input + '\n')
    real_input = vectorizer.transform(pd.Series(input).str.replace('[^\w\s]', '').str.lower().values.astype('U'))
    print('We predict it as: ' + str(model.predict(real_input)) + '\n')
    print("It's probability vector is:" + '\n')
    result = model.predict_proba(real_input)
    probabilities = list(result[0])
    for i in range(len(probabilities)):
        probabilities[i] = '{:.6f}'.format(probabilities[i])
    for i in range(len(model.classes_)):
        print(str(model.classes_[i]) + ": " + str(probabilities[i]))
    index = result.ravel().argsort()[-1:-3 - 1:-1]
    print()
    print("It's highest 3 probability classes are: " + str(model.classes_[index]))
    print("Their probabilities are: " + str(np.array(probabilities)[index]))


news = pd.read_csv('News_Category_Dataset_v41.csv')
model, vectorizer = getModel(news)

input = input('Input the sentence you want to predict: ')
getInputClasses(input, model, vectorizer)