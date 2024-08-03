import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("iris.csv")
df.head()

df['species'].value_counts()
df.isnull().any()

sns.pairplot(data=df, hue = 'species')

# correlation matrix
sns.heatmap(df.corr())

y = df['species']
df1 = df.copy()
df1 = df1.drop('species', axis =1)

X = df1
y

#label encoding
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)
y

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X , y, test_size = 0.2, random_state = 42)
print("Training split input- ", X_train.shape)
print("Testing split input- ", X_test.shape)
print("Training split output- ", y_train.shape)
print("Training split output- ", y_test.shape)

from sklearn.tree import DecisionTreeClassifier
DT=DecisionTreeClassifier()
DT.fit(X_train,y_train)
print('Decision Tree Classifier Created')

y_pred = DT.predict(X_test)
y_pred

from sklearn.metrics import classification_report, confusion_matrix
print("Classification report - \n", classification_report(y_test,y_pred))

cm = confusion_matrix(y_test, y_pred)
cm

from sklearn.metrics import accuracy_score
acc = accuracy_score(y_test, y_pred)
acc

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5,5))
sns.heatmap(data=cm,linewidths=.5, annot=True,square = True,  cmap = 'Blues')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
all_sample_title = 'Accuracy Score: {0}'.format(DT.score(X_test, y_test))
plt.title(all_sample_title)

from sklearn import tree
import graphviz
plt.figure(figsize=(10,6))
tree.plot_tree(DT, filled=True, feature_names=list(X_train))
plt.show()

from sklearn.model_selection import GridSearchCV
from sklearn import metrics

import warnings
warnings.filterwarnings('ignore')

dt_params = [{'max_depth': list(range(3, 10)), 'max_features': list(range(0,4))}]

clf = GridSearchCV(DT, dt_params, cv = 5, scoring='accuracy')

clf.fit(X_train, y_train)

print(clf.best_params_)

print(clf.best_score_)

my_tree_one=tree.DecisionTreeClassifier(criterion='entropy',max_depth=3, random_state=101)
my_tree_one=my_tree_one.fit(X_train,y_train)

plt.figure(figsize=(10,6))
tree.plot_tree(my_tree_one, filled=True, feature_names=list(X_train))
plt.show()

cm = confusion_matrix(y_test, y_pred)
cm
