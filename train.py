import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.datasets import load_iris


### data collection and loading
iris = load_iris()

### Data preprocessing
# Convert to DataFrame for easier manipulation
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target, name='target')


print("Data loaded and preprocessed successfully.")
print("first 5 rows of the dataset: \n", X.head())
print("first 5 target values:", y.head().values)

### Data splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)   

print("Data split into training and testing sets successfully.")    
print("Training set size:", X_train.shape)
print("Testing set size:", X_test.shape)    

### Model training
model = RandomForestClassifier(n_estimators=20, random_state=42)

model.fit(X_train, y_train)

print("Model trained successfully.")

### Model evaluation
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')  
f1 = f1_score(y_test, y_pred, average='weighted')

print("Model evaluation metrics:")
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")

print("this model is baseline machine learning model for iris dataset.")

