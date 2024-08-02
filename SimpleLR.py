import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df = pd.read_csv('placement_SLR.csv')

df.head()

plt.scatter(df['cgpa'],df['package'])
plt.xlabel('CGPA')
plt.ylabel('Package(in lpa)')

X = df.iloc[:,0:1]
y = df.iloc[:,-1]

y

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=2)
