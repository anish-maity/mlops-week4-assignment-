# test.py 
#check pull
import unittest
import os
import pandas as pd
import joblib
from sklearn import metrics

class TestIrisModelPipeline(unittest.TestCase):

    # --- 1. DEFINE PARAMETERS AND PATHS ---
    # These paths must match the output paths from train.py
    SPLITS_DIR = 'data/splits'
    MODEL_DIR = 'model'
    TEST_DATA_PATH = os.path.join(SPLITS_DIR, 'test.csv')
    # CORRECTED TYPO: 'iris_classifier_model.joblib'
    MODEL_FILE_PATH = os.path.join(MODEL_DIR, 'iris_classifier_model.joblib')

    # --- 2. SETUP METHOD ---
    # This method runs once before all tests in the class
    @classmethod
    def setUpClass(cls):
        """
        Set up for the tests. This method loads the model and test data
        that were created by the train.py script.
        It assumes 'dvc pull' has already been run to fetch these files in the CI environment.
        """
        print("\n--- Setting up for tests ---")
        try:
            # Load the trained model
            cls.model = joblib.load(cls.MODEL_FILE_PATH)
            print(f"Model loaded from {cls.MODEL_FILE_PATH}")
            
            # Load the test data
            cls.test_data = pd.read_csv(cls.TEST_DATA_PATH)
            print(f"Test data loaded from {cls.TEST_DATA_PATH}")
        except FileNotFoundError as e:
            cls.fail(f"Setup failed: Required file not found. Have you run 'dvc pull'? Error: {e}")
        print("--- Setup complete. ---\n")

    # --- 3. TEST CASES ---

    def test_data_validation(self):
        """
        (Test 1: Data Validation)
        Tests if the test dataset has the expected columns, ensuring data integrity.
        """
        print("Running data validation test...")
        expected_cols = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
        actual_cols = list(self.test_data.columns)
        
        self.assertListEqual(expected_cols, actual_cols, "Data columns do not match expected schema.")
        print("Data validation test PASSED.")

    def test_model_evaluation(self):
        """
        (Test 2: Model Evaluation)
        Tests if the model's accuracy on the test set is above a reasonable threshold.
        """
        print("Running model evaluation test...")
        features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
        target = 'species'

        X_test = self.test_data[features]
        y_test = self.test_data[target]
        
        prediction = self.model.predict(X_test)
        
        accuracy = metrics.accuracy_score(y_test, prediction)
        print(f"Model accuracy on test set: {accuracy:.3f}")
        
        self.assertGreater(accuracy, 0.9, f"Model accuracy {accuracy:.3f} is below the 0.9 threshold.")
        print("Model evaluation test PASSED.")

# --- 4. SCRIPT EXECUTION ---
if __name__ == '__main__':
    # This allows running the tests directly from the command line
    unittest.main(verbosity=2)
