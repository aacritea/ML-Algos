import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df=pd.read_csv("hearing_test.csv")

df.head()
df.info()
df.describe()

df['test_result'].value_counts()
sns.countplot(data=df, x='test_result')

# EDA
sns.boxplot(x='test_result', y='age', data=df)
sns.boxplot(x='test_result', y='physical_score', data=df)
sns.scatterplot(x='age', y='physical_score', data=df, hue='test_result')
sns.pairplot(df,hue='test_result')

from mpl_toolkits.mplot3d import Axes3D
fig=plt.figure(figsize=(12,8))
ax=fig.add_subplot(111, projection='3d')
ax.scatter(df['age'], df['physical_score'], df['test_result'],c=df['test_result'])

X=df.drop('test_result', axis=1)
y=df['test_result']

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# train-test split
X_train,X_test,y_train,y_test=train_test_split(X,y, test_size=0.2, random_state=1)
scaler=StandardScaler()
scaled_X_train=scaler.fit_transform(X_train)
scaled_X_test=scaler.transform(X_test)

# performs classification
from sklearn.linear_model import LogisticRegression
log_model=LogisticRegression()
log_model.fit(scaled_X_train, y_train)

from sklearn.metrics import accuracy_score, confusion_matrix
y_pred=log_model.predict(scaled_X_test)
y_pred

y_pred_prob=log_model.predict_proba(scaled_X_test)
y_pred_prob

# checks accuracy
accuracy_score(y_test,y_pred)
confusion_matrix(y_test,y_pred)
