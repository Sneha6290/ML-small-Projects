import numpy as np
import joblib

# Load the saved model and scaler
classifier = joblib.load('diabetes_dataset_classifier.joblib')
scaler = joblib.load('scaler.joblib')  # Ensure this loads the StandardScaler object

# Define a function to take user input for prediction
def get_user_input():
    print("Please enter the following details for diabetes prediction:")
    
    pregnancies = float(input("Number of Pregnancies: "))
    glucose = float(input("Glucose Level: "))
    blood_pressure = float(input("Blood Pressure: "))
    skin_thickness = float(input("Skin Thickness: "))
    insulin = float(input("Insulin Level: "))
    bmi = float(input("BMI (Body Mass Index): "))
    dpf = float(input("Diabetes Pedigree Function: "))
    age = float(input("Age: "))
    
    # Combine all inputs into a single array
    user_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]])
    
    # Scale the user data using the loaded scaler
    user_data_scaled = scaler.transform(user_data)  # Use the correct StandardScaler

    return user_data_scaled

# Make predictions
def predict():
    user_data = get_user_input()
    
    # Predict using the loaded model
    prediction = classifier.predict(user_data)
    
    # Output the result
    if prediction[0] == 1:
        print("\nPrediction: The person is likely to have diabetes.")
    else:
        print("\nPrediction: The person is unlikely to have diabetes.")

if __name__ == "__main__":
    print("Welcome to the Diabetes Prediction System!")
    predict()
