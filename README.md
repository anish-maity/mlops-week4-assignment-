train.py

Purpose: This script is the heart of the machine learning pipeline. It handles the entire training process from start to finish.

Functionality:

Loads the raw Iris dataset from samples/data.csv.

Splits the data into training and testing sets and saves them to the data/splits/ directory.

Initializes and trains a DecisionTreeClassifier model using the training data.

Evaluates the trained model on the test data to calculate its accuracy.

Saves the final trained model object to model/iris_classifier_model.joblib.

Saves the accuracy score to metrics.csv.

How to Run: python train.py

test.py

Purpose: This script contains automated tests to ensure the quality and reliability of the data and the trained model. It acts as a quality gate in our CI pipeline.

Functionality:

Data Validation Test: Checks if the test dataset (data/splits/test.csv) contains the correct columns. This prevents errors from changes in the data schema.

Model Evaluation Test: Loads the trained model from model/iris_classifier_model.joblib and verifies that its accuracy on the test data is above a predefined threshold (e.g., 90%).

How to Run: pytest or python -m unittest test.py

CI/CD and Automation

.github/workflows/runtest.yml

Purpose: This is the configuration file for our Continuous Integration (CI) pipeline, powered by GitHub Actions. It defines the automated workflow that runs whenever code is pushed or a pull request is created.

Functionality:

Triggers: The workflow runs automatically on a push to the dev branch and on a pull_request to the main branch.

Environment Setup: Sets up a clean Ubuntu runner, installs Python, and all dependencies from requirements.txt.

Authentication: Securely installs an SSH key (from repository secrets) that grants the runner access to the DVC remote storage.

Data Fetching: Runs dvc pull to download the version-controlled data and model from the remote storage.

Testing: Executes the tests in test.py using pytest.

Reporting: Uses cml to generate a report containing test results and model metrics, and posts it as a comment on the relevant commit or pull request.

Dependencies and Configuration

requirements.txt

Purpose: A standard Python file that lists all the external libraries required to run this project.

Utility: It allows anyone (or any machine, like the GitHub Actions runner) to create an identical environment by running a single command (pip install -r requirements.txt). This ensures reproducibility.

Key Libraries:

pandas: For data manipulation.

scikit-learn: For model training and evaluation.

dvc[ssh]: For data version control, with SSH support for remote storage.

pytest: For running the automated tests.

cml: For generating reports in the CI pipeline.
