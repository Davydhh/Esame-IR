import pickle
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

df = pd.read_csv("NationalNames.csv").drop(["Id", "Year", "Count"], axis=1)

x = df["Name"]
y = df["Gender"]

cv = CountVectorizer()
X = cv.fit_transform(x)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

models = {
    'Logistic Regression': LogisticRegression(),
    'Support Vector Machine': SVC(),
    'Multinomial Naive Bayes': MultinomialNB(),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'K-Nearest Neighbors': KNeighborsClassifier()
}

models_acc = {}

for model_name, model in models.items():
    print("Training: {}".format(model_name))
    model.fit(X_train, y_train)
    word = ["Francis"]
    prediction = model.predict(cv.transform(word).toarray())
    print("Predictions --> {}".format(list(zip(word, prediction))))
    accuracy = model.score(X_test, y_test)*100
    print("The accuracy is {}%".format(accuracy), '\n')
    models_acc[model_name] = accuracy
    filename = '{}.sav'.format(model_name)
    pickle.dump(model, open(filename, 'wb'))

print("The best classifier is {}".format(max(accuracy, key=accuracy.get)))