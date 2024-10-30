# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle

# loading the dataset into a pandas dataframe
sonar_data = pd.read_csv('sonar data.csv', header=None)

# Separating Data and labels
X = sonar_data.drop(columns=60, axis=1)  # storing all the columns except the 60th (label)
Y = sonar_data[60]  # storing the 60th column (label)

# Splitting into Training and Test Data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, stratify=Y, random_state=1)

# Training the Logistic Regression model
model = LogisticRegression()
model.fit(X_train, Y_train)

# Evaluating the model
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print("Accuracy on training data:", training_data_accuracy)

X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print("Accuracy on test data:", test_data_accuracy)

# Saving the trained model
with open('rock_vs_mine_model.pkl', 'wb') as f:
    pickle.dump(model, f)