# train.py
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
import joblib

# --- 1. DEFINE PARAMETERS AND PATHS ---

# Input Data Path (assumes data is in this folder)
RAW_DATA_PATH = 'samples/data.csv'

# Output Paths for data splits, model, and metrics
# These directories will be created if they don't exist.
SPLITS_DIR = 'data/splits'
MODEL_DIR = 'model'
METRICS_FILE = 'metrics.csv'
MODEL_FILE = os.path.join(MODEL_DIR, 'iris_classifier_model.joblib')

# Model Parameters
TEST_SPLIT_RATIO = 0.4
RANDOM_STATE = 42
MODEL_MAX_DEPTH = 3

# --- 2. DATA PREPARATION FUNCTION ---
def prepare_and_split_data():
    """
    Loads the raw Iris dataset, splits it into training and testing sets,
    and saves them to the splits directory.
    """
    print("--- Starting data preparation and splitting ---")
    # Create output directories if they don't exist
    os.makedirs(SPLITS_DIR, exist_ok=True)
    
    print(f"Loading data from {RAW_DATA_PATH}")
    data = pd.read_csv(RAW_DATA_PATH)
    
    print(f"Splitting data with test ratio: {TEST_SPLIT_RATIO}")
    train, test = train_test_split(
        data, 
        test_size=TEST_SPLIT_RATIO, 
        stratify=data['species'], 
        random_state=RANDOM_STATE
    )
    
    # Define paths for saving the splits
    train_path = os.path.join(SPLITS_DIR, 'train.csv')
    test_path = os.path.join(SPLITS_DIR, 'test.csv')
    
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)
    
    print(f"Train data saved to {train_path}")
    print(f"Test data saved to {test_path}")
    print("--- Data preparation complete. ---\n")
    

# --- 3. MODEL TRAINING AND EVALUATION FUNCTION ---
def train_and_evaluate_model():
    """
    Loads the split data, trains a Decision Tree model, evaluates it,
    and saves the model and its performance metrics.
    """
    print("--- Starting model training and evaluation ---")
    # Create model directory if it doesn't exist
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # Load the split data
    train_path = os.path.join(SPLITS_DIR, 'train.csv')
    test_path = os.path.join(SPLITS_DIR, 'test.csv')
    
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    
    # Prepare data for training
    features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    target = 'species'
    
    X_train = train_data[features]
    y_train = train_data[target]
    X_test = test_data[features]
    y_test = test_data[target]
    
    print(f"Training Decision Tree with max_depth={MODEL_MAX_DEPTH}...")
    mod_dt = DecisionTreeClassifier(max_depth=MODEL_MAX_DEPTH, random_state=RANDOM_STATE)
    mod_dt.fit(X_train, y_train)
    
    # Evaluate the model
    prediction = mod_dt.predict(X_test)
    accuracy = metrics.accuracy_score(y_test, prediction)
    print(f'The accuracy of the Decision Tree is: {accuracy:.3f}')
    
    # Save the trained model
    joblib.dump(mod_dt, MODEL_FILE)
    print(f"Model saved to {MODEL_FILE}")
    
    # Save metrics to a file
    with open(METRICS_FILE, 'w') as f:
        f.write(f'accuracy,{accuracy:.3f}\n')
    print(f"Metrics saved to {METRICS_FILE}")
    print("--- Model training and evaluation complete. ---")


# --- 4. SCRIPT EXECUTION ---
if __name__ == '__main__':
    prepare_and_split_data()
    train_and_evaluate_model()
