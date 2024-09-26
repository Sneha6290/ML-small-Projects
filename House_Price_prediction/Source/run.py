import joblib
import numpy as np
from sklearn.datasets import fetch_california_housing

# Loading the saved model
model = joblib.load('house_price_model.joblib')

# Function to get user input for features
def get_user_input():
    housing = fetch_california_housing()
    feature_names = housing.feature_names
    user_input = []
    for feature in feature_names:
        value = float(input(f"Enter value for {feature}: "))
        user_input.append(value)
    return np.array([user_input])

# Main inference loop
while True:
    print("\nEnter house features for price prediction (or 'quit' to exit):")
    user_input = input("Ready to enter values? (yes/quit): ")
    
    if user_input.lower() == 'quit':
        break
    
    features = get_user_input()
    prediction = model.predict(features)
    
    print(f"\nPredicted house price: ${prediction[0]:.2f}")

print("Thank you for using the House Price Prediction model!")