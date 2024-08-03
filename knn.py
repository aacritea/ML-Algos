#Iris Flower Classification using K-Nearest Neighbors Algorithm
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

import seaborn as sns

df = pd.read_csv("iris.csv")
df.head()

df.shape

X = df.drop("species", axis=1)
X.head()
y = df["species"]
y.head()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

classifier = KNeighborsClassifier(n_neighbors=50)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
y_pred

acc = accuracy_score(y_test, y_pred)
print("Accuracy:", acc)

classifier2 = KNeighborsClassifier(n_neighbors=51)
classifier2.fit(X_train, y_train)

X_test.head()
y_test.head()

y_pred2 = classifier2.predict(X_test)
y_pred2

acc2 = accuracy_score(y_test, y_pred2)
print("Accuracy:", acc2)

# Confusion matrix and heatmap of y_pred (for y_pred accuracy score is 1.0)
cm = confusion_matrix(y_test, y_pred)
print(cm)
sns.heatmap(cm, annot=True)

# Confusion matrix and heatmap of y_pred2 (in this we don't have 1.0 accuracy so we got some false positives/false negatives in the heatmap)
cm2 = confusion_matrix(y_test, y_pred2)
print(cm2)
sns.heatmap(cm2, annot=True)

df['species'].value_counts()

list_k=[5,7,9,11,13,15]
from sklearn.model_selection import cross_val_score

for k in list_k:
    clf_knn= KNeighborsClassifier(n_neighbors=k)
    score_knn=cross_val_score(clf_knn,X,y,cv=5)
    print("K:", k)
    print("Cross validation score:" ,score_knn)
    print("Cross validation Mean Score:", score_knn.mean())
    print("")

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)


