# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.datasets
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn import metrics
import joblib
# Loading and analyzing data
from sklearn.datasets import fetch_california_housing
housing = fetch_california_housing()
# Loading the dataset into a pandas dataframe
house_price_dataframe = pd.DataFrame(housing.data,columns=housing.feature_names)
house_price_dataframe['price']=housing.target
# Splitting the Data
X= house_price_dataframe.drop(['price'],axis=1) 
Y= house_price_dataframe['price']
#train-test-split
X_train,X_test,Y_train,Y_test= train_test_split(X,Y,test_size=0.2,random_state=2)
#Moedl training
model = XGBRegressor()
model.fit(X_train,Y_train)
#Evaluating the model
training_data_prediction = model.predict(X_train)
test_data_prediction= model.predict(X_test)
print("Training Data - R-squared Error:", metrics.r2_score(Y_train, training_data_prediction))
print("Training Data - Mean Absolute Error:", metrics.mean_absolute_error(Y_train, training_data_prediction))
print("Test Data - R-squared Error:", metrics.r2_score(Y_test, test_data_prediction))
print("Test Data - Mean Absolute Error:", metrics.mean_absolute_error(Y_test, test_data_prediction))
# Save the model
joblib.dump(model, 'house_price_model.joblib')
print("Model saved as 'house_price_model.joblib'")