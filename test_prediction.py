import joblib
import numpy as np

# Load the model
model = joblib.load('model.pkl')
feature_names = joblib.load('feature_names.pkl')
class_names = ['setosa', 'versicolor', 'virginica']

# Create sample data (example of Iris setosa)
sample_data = np.array([[5.1, 3.5, 1.4, 0.2]])  # sepal length, sepal width, petal length, petal width

# Make a prediction
prediction = int(model.predict(sample_data)[0])
probabilities = model.predict_proba(sample_data)[0]

# Format the result like our API would
result = {
    "predicted_class": class_names[prediction],
    "predicted_class_id": prediction,
    "probabilities": {class_names[i]: float(probabilities[i]) for i in range(len(class_names))}
}

print("\n=== PREDICTION TEST ===")
print(f"Input features: {dict(zip(feature_names, sample_data[0]))}")
print(f"Predicted class: {result['predicted_class']}")
print(f"Prediction probabilities: {result['probabilities']}")
print("=======================\n")

print("This demonstrates the model working correctly and how the API would process predictions.") 