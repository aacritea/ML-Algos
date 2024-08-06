import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# import data from csv file
df=pd.read_csv('attachment_titanic.csv')

# Checking the number of rows(observations) and colmumns(variables)
df.shape

df.info()

# Column names
df.columns

# Number of passengers in each class
df.groupby('Pclass')['Pclass'].count()

df.groupby('Sex')['Sex'].count()

# Number of men and women in each of the passenger class
df.groupby(['Sex', 'Pclass'])['Sex'].count()

# Number of passengers who survived in each class grouped by sex. Also total was found for each class grouped by sex.
df.pivot_table('Survived', 'Sex', 'Pclass', aggfunc=np.sum, margins=True)

df.describe()

df[(df['Age']>=0) & (df['Age']<=1)]

df.isnull().sum()
df.corr()

sns.countplot(x='Survived', data=df)
sns.catplot(x='Survived',col='Sex',kind='count',data=df)
sns.countplot(x='Pclass', data=df, hue='Sex')
sns.catplot(x='Survived',col='Pclass', kind='count', data=df)
sns.catplot(x='Survived',col='Embarked', kind='count', data=df)
sns.swarmplot(x='Survived',y='Fare', data=df)

#Feature Engineering
#Family members or alone
df['Family']=df['SibSp']+df['Parch']+1
df['IsAlone']=0
df.loc[df['Family']==1,'IsAlone']=1

df.head()
sns.catplot(x='Survived',kind='count',hue='Family',data=df)

# Male, female or child
def male_female_child(passenger):
    age,sex=passenger
    if age<10:
        return 'child'
    else:
        return sex

df['person']=df[['Age','Sex']].apply(male_female_child,axis=1)
df['person'].unique()

sns.countplot(x='Survived',hue='person',  data=df)

df['person']=df['person'].apply(lambda x:0 if x=='male' else (1 if x=='female' else 2))
df['Survived'].groupby(df['person']).mean()
df['Name'].str.split(', |\\.',expand=True)[1].value_counts()
df['title']=df['Name'].str.split(', |\\.',expand=True)[1]

a={'Capt':'Military',
            'Col':'Military',
            'Don':'Noble',
            'Dona':'Noble',
            'Dr':'Dr',
            'Jonkheer':'Noble',
            'Lady':'Noble',
            'Major':'Military',
            'Master':'Common',
            'Miss':'Common',
            'Mlle':'Common',
            'Mme':'Common',
            'Mr':'Common',
            'Mrs':'Common',
            'Ms':'Common',
            'Rev':'Clergy',
            'Sir':'Noble',
            'the Countess':'Noble',
            }

df['social status'] = df['title'].map(a)
df.head()

df['Survived'].groupby(df['social status']).mean().plot(kind='bar')

columns=['Pclass','Age','Fare','Family','Embarked','person','social status','Survived']
feature=df[list(columns)].values

feature
df['Embarked'].unique(), df['social status'].unique()

from sklearn.preprocessing import LabelEncoder
lc=LabelEncoder()
df['Embarked']=lc.fit_transform(df['Embarked'])
df['social status']=lc.fit_transform(df['social status'])
df['Embarked'].unique(), df['social status'].unique()

columns=['Pclass','Age','Fare','Family','Embarked','person','social status','Survived']
feature=df[list(columns)].values

from sklearn.impute import SimpleImputer
imp=SimpleImputer(missing_values=np.nan,strategy='mean')
X=pd.DataFrame(imp.fit_transform(feature))
X.columns=columns
X.index=df.index
X.head()

df2=X.filter(['Pclass','Age','Fare','Family','Embarked','person','social status','Survived'])
df2.head()

from sklearn.model_selection import train_test_split
X=df2.drop(['Survived'],axis=1)
y=df2['Survived']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=101)
X_train.shape, X_test.shape, y_train.shape, y_test.shape

from sklearn import tree
my_tree_one=tree.DecisionTreeClassifier(criterion='entropy',max_depth=3, random_state=101)
my_tree_one=my_tree_one.fit(X_train,y_train)

import graphviz
plt.figure(figsize=(10,6))
tree.plot_tree(my_tree_one, filled=True, feature_names=list(feature))
plt.show()

list(zip(columns,my_tree_one.feature_importances_))

from sklearn.preprocessing import StandardScaler
SSC=StandardScaler()
X_train_Sc=SSC.fit_transform(X_train)
X_test_Sc=SSC.transform(X_test)

from sklearn.model_selection import GridSearchCV

param_grid = { "max_depth": [3,4,5], "min_samples_split" : [10, 12,15],"min_samples_leaf" : [2,5,10],"criterion": ["gini", "entropy"]}
model=GridSearchCV(estimator=my_tree_one,
                       param_grid=param_grid,
                       scoring='accuracy',
                       verbose=1)
model.fit(X_train_Sc,y_train)

from sklearn.model_selection import RandomizedSearchCV
param_grid = { "max_depth": [3,4], "min_samples_split" : [5,10, 12],"min_samples_leaf" :[5,10,20], "criterion": ["gini", "entropy"]}
model=RandomizedSearchCV(my_tree_one, param_distributions=param_grid, n_iter=10, cv=5)
model.fit(X_train_Sc, y_train)

model.best_params_, model.best_score_

#GridSearchCV evaluated model
DT=tree.DecisionTreeClassifier(criterion='entropy',max_depth=3, min_samples_split=15,min_samples_leaf=5,random_state=101)
DT=DT.fit(X_train_Sc,y_train)
plt.figure(figsize=(10,6))
tree.plot_tree(DT, filled=True, feature_names=list(feature))
plt.show()

#RandomizedSearchCV evaluated model
DT=tree.DecisionTreeClassifier(criterion='gini',max_depth=3, min_samples_split=5,min_samples_leaf=10,random_state=101)
DT.fit(X_train_Sc,y_train)

y_pred=DT.predict(X_test_Sc)
DT.score(X_train_Sc,y_train)

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
accuracy_score(y_test,y_pred)
confusion_matrix(y_test, y_pred)

TN,FP,FN,TP = confusion_matrix(y_test, y_pred).ravel()
TN,FP,FN,TP

precision_score(y_test, y_pred)
recall_score(y_test, y_pred)
f1_score(y_test, y_pred)

#Random Forest
from sklearn.ensemble import RandomForestClassifier
forest=RandomForestClassifier()

from sklearn.model_selection import GridSearchCV
param_grid = { "criterion" : ["gini", "entropy"], "min_samples_leaf" : [2, 5, 10], "min_samples_split" : [4, 10, 12], "n_estimators": [10,50,100]}
model=GridSearchCV(forest, param_grid=param_grid, cv=5)
model.fit(X_train_Sc, y_train)

model.best_params_, model.best_score_

my_forest=RandomForestClassifier(criterion='gini',n_estimators=10, min_samples_split=4,min_samples_leaf=5, random_state=1)
my_forest.fit(X_train_Sc,y_train)

y_pred=my_forest.predict(X_test)

my_forest.score(X_train_Sc,y_train)

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
accuracy_score(y_test,y_pred)

confusion_matrix(y_test, y_pred)
precision_score(y_test, y_pred)
recall_score(y_test, y_pred)
f1_score(y_test, y_pred)

from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
param_grid = { "max_depth": randint(2,5), "n_estimators": [10,20,40],  "criterion": ["gini", "entropy"]}
grid_model=RandomizedSearchCV(forest, param_distributions=param_grid, n_iter=10, cv=5)
grid_model.fit(X_train_Sc, y_train)

grid_model.best_params_, grid_model.best_score_
my_forest=RandomForestClassifier(n_estimators=20, min_samples_split=5, max_depth=4, random_state=1)
my_forest.fit(X_train_Sc,y_train)
