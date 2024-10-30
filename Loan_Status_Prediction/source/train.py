import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
import joblib
import os
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class LoanModelTrainer:
    def __init__(self, data_path='loan_status_prediction.csv'):
        """Initialize the model trainer with data path."""
        self.data_path = data_path
        self.model = None
        self.feature_names = None
    
    def load_data(self):
        """Load and preprocess the dataset."""
        logger.info("Loading dataset from %s", self.data_path)
        try:
            df = pd.read_csv(self.data_path)
            return df
        except FileNotFoundError:
            logger.error("Dataset file not found at %s", self.data_path)
            raise
    
    def preprocess_data(self, df):
        """Preprocess the input dataframe."""
        logger.info("Preprocessing data...")
        
        # Drop missing values
        df = df.dropna()
        
        # Label encoding for Loan_Status
        df.replace({"Loan_Status":{'N':0, 'Y':1}}, inplace=True)
        
        # Replace '3+' with 4 in Dependents
        df = df.replace(to_replace='3+', value=4)
        
        # Convert categorical columns to numerical
        categorical_mapping = {
            'Married': {'No': 0, 'Yes': 1},
            'Gender': {'Male': 1, 'Female': 0},
            'Self_Employed': {'No': 0, 'Yes': 1},
            'Property_Area': {'Rural': 0, 'Semiurban': 1, 'Urban': 2},
            'Education': {'Graduate': 1, 'Not Graduate': 0}
        }
        
        df.replace(categorical_mapping, inplace=True)
        return df
    
    def prepare_features(self, df):
        """Prepare features and target variables."""
        logger.info("Preparing features and target variables...")
        
        # Separate features and target
        X = df.drop(columns=['Loan_ID', 'Loan_Status'], axis=1)
        y = df['Loan_Status']
        
        self.feature_names = X.columns.tolist()
        return X, y
    
    def train_model(self, X_train, y_train):
        """Train the SVM model."""
        logger.info("Training SVM model...")
        
        self.model = svm.SVC(kernel='linear')
        self.model.fit(X_train, y_train)
        return self.model
    
    def evaluate_model(self, X_train, X_test, y_train, y_test):
        """Evaluate model performance."""
        logger.info("Evaluating model performance...")
        
        train_predictions = self.model.predict(X_train)
        test_predictions = self.model.predict(X_test)
        
        train_accuracy = accuracy_score(train_predictions, y_train)
        test_accuracy = accuracy_score(test_predictions, y_test)
        
        logger.info(f"Training Accuracy: {train_accuracy:.4f}")
        logger.info(f"Test Accuracy: {test_accuracy:.4f}")
        
        return train_accuracy, test_accuracy
    
    def save_model(self, model_dir='models'):
        """Save the trained model and feature names."""
        logger.info("Saving model and features...")
        
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        
        model_path = os.path.join(model_dir, 'loan_prediction_model.joblib')
        features_path = os.path.join(model_dir, 'feature_names.npy')
        
        joblib.dump(self.model, model_path)
        np.save(features_path, self.feature_names)
        
        logger.info("Model saved to %s", model_path)
        logger.info("Features saved to %s", features_path)
    
    def train(self):
        """Execute the full training pipeline."""
        try:
            # Load and preprocess data
            df = self.load_data()
            df = self.preprocess_data(df)
            
            # Prepare features
            X, y = self.prepare_features(df)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.1, stratify=y, random_state=2
            )
            
            # Train model
            self.train_model(X_train, y_train)
            
            # Evaluate model
            self.evaluate_model(X_train, X_test, y_train, y_test)
            
            # Save model
            self.save_model()
            
            logger.info("Training completed successfully!")
            return True
            
        except Exception as e:
            logger.error("Error during training: %s", str(e))
            return False

def main():
    """Main function to run the training pipeline."""
    trainer = LoanModelTrainer()
    success = trainer.train()
    
    if not success:
        logger.error("Training failed!")
        exit(1)

if __name__ == "__main__":
    main()