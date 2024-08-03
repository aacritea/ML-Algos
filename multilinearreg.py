import numpy as np
from sklearn.datasets import load_diabetes

X, y = load_diabetes(return_X_y=True)
X
X.shape
y
y.shape

# Using Sklearn's Linear Regression
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=2)

print(X_train.shape)
print(X_test.shape)

from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(X_train,y_train)

y_pred = reg.predict(X_test)
from sklearn.metrics import r2_score
r2_score(y_test,y_pred)

reg.coef_
reg.intercept_
