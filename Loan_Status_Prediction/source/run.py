import numpy as np
import joblib
import os
import logging
from typing import List, Union, Tuple

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class LoanPredictor:
    def __init__(self, model_dir: str = 'models'):
        """Initialize the loan predictor."""
        self.model_dir = model_dir
        self.model = None
        self.feature_names = None
        self.load_model()
    
    def load_model(self) -> None:
        """Load the trained model and feature names."""
        try:
            model_path = os.path.join(self.model_dir, 'loan_prediction_model.joblib')
            features_path = os.path.join(self.model_dir, 'feature_names.npy')
            
            self.model = joblib.load(model_path)
            self.feature_names = np.load(features_path)
            
            logger.info("Model and features loaded successfully")
        except FileNotFoundError:
            logger.error("Model files not found. Please run train.py first")
            raise
    
    def validate_input(self, input_data: List[float]) -> bool:
        """Validate the input data."""
        if len(input_data) != len(self.feature_names):
            logger.error(f"Expected {len(self.feature_names)} features, got {len(input_data)}")
            return False
        
        # Validate ranges for categorical variables
        validations = {
            'Gender': (0, 1),
            'Married': (0, 1),
            'Dependents': (0, 4),
            'Education': (0, 1),
            'Self_Employed': (0, 1),
            'Credit_History': (0, 1),
            'Property_Area': (0, 2)
        }
        
        for feature, value, (min_val, max_val) in zip(
            self.feature_names, 
            input_data, 
            [validations.get(feat, (float('-inf'), float('inf'))) for feat in self.feature_names]
        ):
            if not min_val <= value <= max_val:
                logger.error(f"Invalid value {value} for {feature}. Expected range: [{min_val}, {max_val}]")
                return False
        
        return True
    
    def predict(self, input_data: List[float]) -> Tuple[int, float]:
        """Make a prediction for the input data."""
        if not self.validate_input(input_data):
            raise ValueError("Invalid input data")
        
        input_array = np.array(input_data).reshape(1, -1)
        prediction = self.model.predict(input_array)[0]
        
        # Get decision function value for confidence
        confidence = abs(self.model.decision_function(input_array)[0])
        
        return int(prediction), confidence
    
    @staticmethod
    def print_input_format() -> None:
        """Print the expected input format."""
        print("\nInput Format Required:")
        print("Please provide 11 values in the following order:")
        print("1. Gender (1 for Male, 0 for Female)")
        print("2. Married (1 for Yes, 0 for No)")
        print("3. Dependents (0-4)")
        print("4. Education (1 for Graduate, 0 for Not Graduate)")
        print("5. Self_Employed (1 for Yes, 0 for No)")
        print("6. ApplicantIncome (numeric)")
        print("7. CoapplicantIncome (numeric)")
        print("8. LoanAmount (numeric)")
        print("9. Loan_Amount_Term (numeric)")
        print("10. Credit_History (0 or 1)")
        print("11. Property_Area (0 for Rural, 1 for Semiurban, 2 for Urban)")
        print("\nExample: 1 1 4 1 0 3036 2504 158 360 0 1")

def get_user_input() -> List[float]:
    """Get and validate user input."""
    while True:
        try:
            input_string = input("\nEnter the values separated by spaces: ")
            return [float(x) for x in input_string.split()]
        except ValueError:
            print("Error: All inputs must be numeric. Please try again.")

def main():
    """Main function to run predictions."""
    try:
        predictor = LoanPredictor()
        
        # Print input format
        predictor.print_input_format()
        
        # Get user input
        input_data = get_user_input()
        
        # Make prediction
        prediction, confidence = predictor.predict(input_data)
        
        # Print result
        result = "Approved" if prediction == 1 else "Not Approved"
        print(f"\nLoan Status Prediction: {result}")
        print(f"Confidence Score: {confidence:.2f}")
        
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())