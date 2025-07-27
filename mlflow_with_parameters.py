# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.datasets import load_iris

# Import MLflow
import mlflow
import mlflow.sklearn

# --- MLflow Setup (Optional: Set tracking URI if not using default local) ---
# If you want to log to a remote MLflow Tracking Server, uncomment and set the URI:
# mlflow.set_tracking_uri("http://localhost:5000") # Replace with your tracking server URI
# mlflow.set_experiment("Iris_RandomForest_Hyperparameter_Tuning") # Set a name for your experiment

# --- Data Collection and Preparation (outside the tuning loop as it's static) ---
# Load the Iris dataset
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target, name='species')

print("Dataset loaded successfully. Shape:", X.shape)

# Split the data into training and testing sets
test_size = 0.2
random_state_split = 42
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=random_state_split, stratify=y
)

print(f"\nTraining data shape: {X_train.shape}")
print(f"Testing data shape: {X_test.shape}")

# --- Hyperparameter Tuning with MLflow ---
# Define the hyperparameters to tune
# We'll try different combinations of n_estimators and max_depth
param_combinations = [
    {"n_estimators": 20, "max_depth": 5},
    {"n_estimators": 100, "max_depth": 8},
    {"n_estimators": 150, "max_depth": 10},
    {"n_estimators": 200, "max_depth": 15}, # None means nodes are expanded until all leaves are pure
]

print("\nStarting hyperparameter tuning with MLflow...")

for i, params in enumerate(param_combinations):
    # Start a new MLflow run for each combination of hyperparameters
    # Each run will have its own set of logged parameters, metrics, and model
    with mlflow.start_run(run_name=f"Run_{i+1}_n{params['n_estimators']}_d{params['max_depth']}"):
        print(f"\n--- Running experiment with n_estimators={params['n_estimators']}, max_depth={params['max_depth']} ---")

        # --- Model Training ---
        # Initialize and train a RandomForestClassifier with current parameters
        model = RandomForestClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            random_state=random_state_split # Use the same random state for consistency
        )
        model.fit(X_train, y_train)

        print("Model training complete for current run.")

        # --- Model Evaluation ---
        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')

        print(f"Model Evaluation Results for this run:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")

        # --- MLflow Logging ---
        print("Logging parameters, metrics, and model to MLflow for this run...")

        # Log parameters for the current run
        mlflow.log_param("n_estimators", params["n_estimators"])
        mlflow.log_param("max_depth", params["max_depth"])
        mlflow.log_param("test_size", test_size) # Log static parameters as well for completeness
        mlflow.log_param("random_state_split", random_state_split)

        # Log metrics for the current run
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        # Log the model for the current run
        mlflow.sklearn.log_model(model, "random_forest_model")

        print(f"MLflow logging complete for run {i+1}.")

print("\nHyperparameter tuning complete. View all runs by running 'mlflow ui' in your terminal.")
