import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.datasets import load_iris

import mlflow
import mlflow.sklearn

# --- MLflow Setup (Optional: Set tracking URI if not using default local) ---
# If you want to log to a remote MLflow Tracking Server, uncomment and set the URI:
# mlflow.set_tracking_uri("http://localhost:5000") # Replace with your tracking server URI
# mlflow.set_experiment("Iris_RandomForest_Experiment") # Set a name for your experiment

# --- start with MLflow run ---

with mlflow.start_run():
    # Load the Iris dataset
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = pd.Series(iris.target, name='species')


    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize the Random Forest Classifier
    model = RandomForestClassifier(n_estimators=20, random_state=42)

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions
    predictions = model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, average='weighted')
    recall = recall_score(y_test, predictions, average='weighted')
    f1 = f1_score(y_test, predictions, average='weighted')

    # Log metrics to MLflow
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)

    # Log the model
    mlflow.sklearn.log_model(model, "model")
    print("MLflow logging complete. View your runs by running 'mlflow ui' in your terminal.")
    print("Open your web browser and navigate to the address shown in the terminal (usually http://localhost:5000).")
