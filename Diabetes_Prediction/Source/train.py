import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler # used to standarize the data to a common range
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
import joblib

# loading dataset
diabetes_dataset = pd.read_csv('diabetes.csv')
diabetes_dataset= diabetes_dataset.fillna('')
# separating Data and Labels
X=diabetes_dataset.drop(columns='Outcome',axis=1)
Y=diabetes_dataset['Outcome']
# Data standarization
scaler=StandardScaler()
scaler.fit(X)
standarized_data= scaler.transform(X)
X= standarized_data
Y= diabetes_dataset['Outcome']

# train- test-split
X_train ,X_test,Y_train,Y_test= train_test_split(X,Y,test_size=0.2,stratify=Y,random_state=2)
# train the model
classifier=svm.SVC(kernel='linear')
classifier.fit(X_train,Y_train)
# Evaluating the model 
train_accuracy = accuracy_score(Y_train, classifier.predict(X_train))
test_accuracy = accuracy_score(Y_test, classifier.predict(X_test))
print(f"Training Accuracy: {train_accuracy}")
print(f"Testing Accuracy: {test_accuracy}")
# Save the model and vectorizer
joblib.dump(classifier, 'diabetes_dataset_classifier.joblib')
joblib.dump(standarized_data, 'standarized_data.joblib')

print("Model and standarized data saved successfully.")