# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.datasets import load_iris

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

# Import MLflow
import mlflow
import mlflow.tensorflow

# --- MLflow Setup ---
# Set the experiment name for better organization in the MLflow UI
mlflow.set_experiment("Iris_DeepLearning_Experiment")

# Enable automatic logging for TensorFlow/Keras.
# This will automatically log parameters, metrics, and the model itself.
mlflow.tensorflow.autolog()

# --- Data Collection and Preparation ---
# Load the Iris dataset
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target, name='species')

print("Dataset loaded successfully. Shape:", X.shape)
print("First 5 rows of features:\n", X.head())
print("First 5 rows of target:\n", y.head())

# Split the data into training and testing sets
test_size = 0.2
random_state_split = 42
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=random_state_split, stratify=y
)

print(f"\nTraining data shape: {X_train.shape}")
print(f"Testing data shape: {X_test.shape}")

# Preprocessing for Deep Learning
# Scale numerical features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# One-hot encode the target variable
# Keras expects target labels to be one-hot encoded for categorical cross-entropy
num_classes = len(iris.target_names)
y_train_encoded = to_categorical(y_train, num_classes=num_classes)
y_test_encoded = to_categorical(y_test, num_classes=num_classes)

print(f"Target variable one-hot encoded. Shape: {y_train_encoded.shape}")

# --- Deep Learning Model Definition ---
# Define a simple Sequential Keras model
model = Sequential([
    # Input layer with 4 features (sepal length, sepal width, petal length, petal width)
    Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    # Hidden layer
    Dense(32, activation='relu'),
    # Output layer with 3 units (for 3 Iris species), using softmax for multi-class classification
    Dense(num_classes, activation='softmax')
])

# Compile the model
# Using Adam optimizer and categorical_crossentropy for multi-class classification
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# --- Model Training ---
epochs = 50
batch_size = 8

print(f"\nStarting deep learning model training for {epochs} epochs with batch size {batch_size}...")
history = model.fit(
    X_train_scaled, y_train_encoded,
    epochs=epochs,
    batch_size=batch_size,
    validation_split=0.1, # Use a small portion of training data for validation during training
    verbose=1
)
print("Deep learning model training complete.")

# --- Model Evaluation ---
# Evaluate the model on the test set
loss, accuracy = model.evaluate(X_test_scaled, y_test_encoded, verbose=0)
print(f"\nModel Evaluation Results on Test Set:")
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")

# For precision, recall, f1-score, we need to convert predictions back to labels
y_pred_probs = model.predict(X_test_scaled)
y_pred_labels = tf.argmax(y_pred_probs, axis=1).numpy() # Convert probabilities to class labels

precision = precision_score(y_test, y_pred_labels, average='weighted', zero_division=0)
recall = recall_score(y_test, y_pred_labels, average='weighted', zero_division=0)
f1 = f1_score(y_test, y_pred_labels, average='weighted', zero_division=0)

print(f"Precision (weighted): {precision:.4f}")
print(f"Recall (weighted): {recall:.4f}")
print(f"F1 Score (weighted): {f1:.4f}")

# --- MLflow Logging (autolog takes care of most, but you can add more) ---
# Since mlflow.tensorflow.autolog() is enabled, most parameters and metrics
# from model.fit() and model.evaluate() are automatically logged.
# You can manually log additional parameters if they are not automatically captured.
with mlflow.start_run(nested=True): # Use nested run to group under the autologged run
    mlflow.log_param("test_size", test_size)
    mlflow.log_param("random_state_split", random_state_split)
    mlflow.log_param("num_classes", num_classes)
    mlflow.log_param("epochs", epochs)
    mlflow.log_param("batch_size", batch_size)
    mlflow.log_param("optimizer", "Adam")
    mlflow.log_param("loss_function", "categorical_crossentropy")

    # Log evaluation metrics explicitly if not fully covered by autolog
    mlflow.log_metric("test_accuracy", accuracy)
    mlflow.log_metric("test_loss", loss)
    mlflow.log_metric("test_precision_weighted", precision)
    mlflow.log_metric("test_recall_weighted", recall)
    mlflow.log_metric("test_f1_score_weighted", f1)

    # The model is also automatically logged by mlflow.tensorflow.autolog()
    # You can explicitly log it if you need a custom artifact path or signature:
    # mlflow.tensorflow.log_model(model, "iris_dl_model")

print("\nDeep learning experiment complete. View your runs by running 'mlflow ui' in your terminal.")
