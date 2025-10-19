MLOps Pipeline for Iris Classification with DVC and GCP

This repository contains a complete, automated MLOps pipeline for training, versioning, and testing an Iris flower classification model. It demonstrates best practices by integrating Git for code versioning, DVC for data and model versioning, Google Cloud Storage (GCS) for remote storage, and GitHub Actions for Continuous Integration (CI).

MLOps Pipeline Overview

This project follows a modern MLOps workflow. When a developer pushes code to the dev branch, a GitHub Actions workflow is automatically triggered. This workflow authenticates with Google Cloud, pulls the version-controlled data from GCS using DVC, runs a suite of automated tests, and posts a detailed report back to the commit.

Key Features

Reproducibility: DVC ensures that every version of the data and model is tracked, making experiments fully reproducible.

Automation: GitHub Actions automates the entire testing process, providing rapid feedback on code changes.

Scalable Storage: Google Cloud Storage provides a robust and scalable backend for storing large data and model files.

Continuous Integration: New code is automatically tested for data schema validity and model performance before it can be merged into the main branch.

Automated Reporting: CML (Continuous Machine Learning) is used to post clear, concise reports directly on commits and pull requests.

Detailed File Descriptions

Core Machine Learning Scripts

📄 train.py

Purpose: This is the primary script that orchestrates the entire machine learning training process. It acts as the "engine" of the pipeline, taking raw data as input and producing a trained model and other artifacts as output.

Functionality:

Loads Data: Reads the raw Iris dataset from samples/data.csv.

Splits Data: Divides the data into training and testing sets to ensure the model is evaluated on unseen data. These splits are saved to data/splits/.

Trains Model: Uses a Decision Tree Classifier from scikit-learn to train a model on the training data.

Evaluates & Saves: Calculates the model's accuracy on the test set and saves three key artifacts:

The final trained model (model/iris_classifier_model.joblib).

The exact data splits used (data/splits/).

The performance metrics (metrics.csv).

📄 test.py

Purpose: This script serves as the project's "quality assurance gate." It contains a suite of automated tests that validate the outputs of train.py. The CI pipeline runs these tests to prevent bugs and performance regressions.

Functionality (using unittest):

Data Schema Validation: A test that checks if the columns in the test data are exactly as expected. This crucial test catches any unexpected changes in the data format.

Model Performance Validation: A test that loads the trained model and asserts that its accuracy is above a predefined threshold (e.g., 90%). This ensures that new code changes do not accidentally degrade the model's performance.

Automation and Configuration

📄 .github/workflows/runtest.yml

Purpose: This is the brain of the entire automation process. It is a configuration file that tells GitHub Actions exactly what steps to execute when triggered.

Functionality:

Triggers: The workflow is configured to run automatically on a push to the dev branch and on any pull_request targeting the main branch.

Environment Setup: It prepares a clean virtual machine, installs Python, and all the project dependencies listed in requirements.txt.

Secure Authentication: It securely authenticates with Google Cloud Platform using a Service Account key stored in GitHub's encrypted secrets.

Data Synchronization: It runs dvc pull to download the specific versions of the model and data from the GCS bucket that correspond to the current code commit.

Automated Testing: It executes the test suite using pytest.

Report Generation: It uses CML to build a Markdown report that includes the model's metrics and the test results, then posts this report as a comment on the relevant commit or pull request.

📄 requirements.txt

Purpose: This file ensures a reproducible Python environment. It lists all the external libraries the project depends on.

Key Dependencies:

pandas & scikit-learn: For data handling and machine learning.

dvc: The core library for data version control.

dvc-gs: The specific plugin that enables DVC to communicate with Google Cloud Storage.

pytest: The framework used for running the automated tests in test.py.

cml: The tool used for generating reports in the CI workflow.
