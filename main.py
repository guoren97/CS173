import pandas as pd
import matplotlib.pyplot as plt
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer


def getModel (news):
    news['TITLE'] = news['TITLE'].str.replace('[^\w\s]', '').str.lower()  # Remove punctuation and replace upper letter
    vectorizer = CountVectorizer(stop_words='english')  # Initial the vectorizer
    x_train, x_test, y_train, y_test = train_test_split(news['TITLE'], news['CATEGORY'], test_size=0.1)  # 10% split
    x_train = vectorizer.fit_transform(x_train.values.astype('U'))  # transform the format of training data
    model = MultinomialNB(alpha=0.1)  # Initial the model
    model.fit(x_train, y_train)  # Training data
    return model, vectorizer, x_test, y_test


news4 = pd.read_csv('News_Category_Dataset_v4.csv')
news8 = pd.read_csv('News_Category_Dataset_v8.csv')
news20 = pd.read_csv('News_Category_Dataset_v20.csv')
news41 = pd.read_csv('News_Category_Dataset_v41.csv')

model4, vectorizer4, x_test4, y_test4 = getModel(news4)
model8, vectorizer8, x_test8, y_test8 = getModel(news8)
model20, vectorizer20, x_test20, y_test20 = getModel(news20)
model41, vectorizer41, x_test41, y_test41 = getModel(news41)

score4 = model4.score(vectorizer4.transform(x_test4.values.astype('U')), y_test4)
score8 = model8.score(vectorizer8.transform(x_test8.values.astype('U')), y_test8)
score20 = model20.score(vectorizer20.transform(x_test20.values.astype('U')), y_test20)
score41 = model41.score(vectorizer41.transform(x_test41.values.astype('U')), y_test41)
score = [score4, score8, score20, score41]

print('4 classes are:   ' + str(news4['CATEGORY'].unique()))
print('accuracy of choosing 1 class from 4 classes:     ' + str(score4) + '\n')
print('8 classes are:   ' + str(news8['CATEGORY'].unique()))
print('accuracy of choosing 1 class from 8 classes:     ' + str(score8) + '\n')
print('20 classes are:  ' + str(news20['CATEGORY'].unique()))
print('accuracy of choosing 1 class from 20 classes:    ' + str(score20) + '\n')
print('41 classes are:  ' + str(news41['CATEGORY'].unique()))
print('accuracy of choosing 1 class from 41 classes:    ' + str(score41) + '\n')

classes = ['4', '8', '20', '41']
plt.bar(classes, score)
plt.xlabel('classes')
plt.ylabel('accuracy')
plt.title('Accuracy of choosing 1 classes from n classes')
for a, b in zip(classes, score):
    plt.text(a, b, '%.3f' % b, ha='center', va='bottom', fontsize=7)
plt.show()


def getMost3ClassAccuracy(model, vectorizer, x_test, y_test):
    count = 0
    x_test = vectorizer.transform(x_test.values.astype('U'))
    results = model.predict_proba(x_test)  # Get the probability vector
    for i in range(len(results)):
        index = results[i].ravel().argsort()[-1:-3 - 1:-1]
        if str(y_test.tolist()[i]) in list(model.classes_[index]):
            count = count + 1
    return count / len(results)


most3score4 = getMost3ClassAccuracy(model4, vectorizer4, x_test4, y_test4)
most3score8 = getMost3ClassAccuracy(model8, vectorizer8, x_test8, y_test8)
most3score20 = getMost3ClassAccuracy(model20, vectorizer20, x_test20, y_test20)
most3score41 = getMost3ClassAccuracy(model41, vectorizer41, x_test41, y_test41)
most3score = [most3score4, most3score8, most3score20, most3score41]

print('accuracy of choosing 3 class from 4 classes:     ' + str(most3score4))
print('accuracy of choosing 3 class from 8 classes:     ' + str(most3score8))
print('accuracy of choosing 3 class from 20 classes:    ' + str(most3score20))
print('accuracy of choosing 3 class from 41 classes:    ' + str(most3score41))

plt.bar(classes, most3score)
plt.xlabel('classes')
plt.ylabel('accuracy')
plt.title('Accuracy of choosing 3 classes from n classes')
for a, b in zip(classes, most3score):
    plt.text(a, b, '%.3f' % b, ha='center', va='bottom', fontsize=7)
plt.show()


def getInputClasses(input, model, vectorizer):
    print('\n' + 'Sample input is: ' + input + '\n')
    real_input = vectorizer.transform(pd.Series(input).str.replace('[^\w\s]', '').str.lower().values.astype('U'))
    print('We predict it as: ' + str(model.predict(real_input)) + '\n')
    print("It's probability vector is:" + '\n')
    result = model.predict_proba(real_input)
    for i in range(len(model.classes_)):
        print(str(model.classes_[i]) + ": " + str(result[0][i]))
    index = result.ravel().argsort()[-1:-3 - 1:-1]
    print()
    print("It's highest 3 probability classes are: " + str(model.classes_[index]))


input = 'A Dizzyingly High Rooftop Infinity Pool Is Coming to London, and It Will Have 360-degree Skyline Views'
getInputClasses(input, model41, vectorizer41)
