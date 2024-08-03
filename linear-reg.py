# imports modules
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# initialises dataframe
df = pd.read_csv('placement_SLR.csv')

df.head()

# makes scatter plot
plt.scatter(df['cgpa'],df['package'])
plt.xlabel('CGPA')
plt.ylabel('Package(in lpa)')

# configues index locations
X = df.iloc[:,0:1]
y = df.iloc[:,-1]

print("Gunnu is a smart girl.")

# divides dataset into training and testing data
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=2)

# performs linear regression
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train,y_train)

X_test
y_test

# prediction modelling
lr.predict(X_test.iloc[0].values.reshape(1,1))
lr.predict(X_test.iloc[1].values.reshape(1,1))
lr.predict(X_test.iloc[2].values.reshape(1,1))

plt.scatter(df['cgpa'],df['package'])
plt.plot(X_train,lr.predict(X_train),color='red')
plt.xlabel('CGPA')
plt.ylabel('Package(in lpa)')

y_pred = lr.predict(X_test)

m = lr.coef_
m

b = lr.intercept_
b

# y = mx + b
m * 8.58 + b
m * 9.5 + b
m * 100 + b

from sklearn.metrics import mean_squared_error
mean_squared_error(y_test, y_pred)

test_predictions = lr.predict(X_test).flatten()
print(test_predictions)

# shows the true value and predicted value in dataframe
true_predicted = pd.DataFrame(list(zip(y_test, test_predictions)), 
                    columns=['True Value','Predicted Value'])
true_predicted.head(10)

# calculates accuracy
from sklearn.metrics import r2_score
r2_score(y_test,y_pred)
