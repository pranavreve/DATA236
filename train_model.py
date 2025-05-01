import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
accuracy = model.score(X_test, y_test)
print(f"Model accuracy: {accuracy:.4f}")

# Save the model to a .pkl file
joblib.dump(model, 'model.pkl')
print("Model saved as model.pkl")

# Save feature names for later use
feature_names = iris.feature_names
joblib.dump(feature_names, 'feature_names.pkl')
print("Feature names saved as feature_names.pkl") 