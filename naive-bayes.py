# A cancer dataset considered for naive bayes implementation.It contains total 569 rows and 32 column. We shall examine malignant and benign tumors

# Naive Bayes Classifiers are classified into three categories —
# i) Gaussian Naive Bayes
# This classifier is employed when the predictor values are continuous and are expected to follow a Gaussian distribution.
# ii) Bernoulli Naive Bayes
# When the predictors are boolean in nature and are supposed to follow the Bernoulli distribution, this classifier is utilized.
# iii) Multinomial Naive Bayes
# This classifier makes use of a multinomial distribution and is often used to solve issues involving document or text classification.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv("cancer.csv")
dataset.sample(10)

dataset.info()

dataset = dataset.drop(["id"], axis = 1)
dataset = dataset.drop(["Unnamed: 32"], axis = 1)

M = dataset[dataset.diagnosis == "M"]
B = dataset[dataset.diagnosis == "B"]

plt.title("Malignant vs Benign Tumor")
plt.xlabel("Radius Mean")
plt.ylabel("Texture Mean")
plt.scatter(M.radius_mean, M.texture_mean, color = "red", label = "Malignant", alpha = 0.3)
plt.scatter(B.radius_mean, B.texture_mean, color = "lime", label = "Benign", alpha = 0.3)
plt.legend()
plt.show()

dataset.diagnosis = [1 if i=='M' else 0 for i in dataset.diagnosis]
dataset.sample(10)

x = dataset.drop(["diagnosis"], axis = 1)
y = dataset.diagnosis.values

# Normalization:
x= (x-np.min(x))/(np.max(x)-np.min(x))

print(x)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 42)

from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(x_train, y_train)

print("Naive Bayes score: ",nb.score(x_test, y_test))

from sklearn.naive_bayes import BernoulliNB
classifier = BernoulliNB()
classifier.fit(x_train, y_train)

print(" BernoulliNB Bayes score: ",classifier.score(x_test, y_test))
