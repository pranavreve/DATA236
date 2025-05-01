from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import os

app = FastAPI(title="Iris Classification API", 
              description="API for predicting Iris flower species",
              version="1.0.0")

# Load the model and feature names
try:
    model = joblib.load('model.pkl')
    feature_names = joblib.load('feature_names.pkl')
    class_names = ['setosa', 'versicolor', 'virginica']
except Exception as e:
    print(f"Error loading model: {e}")
    raise

# Define the input data model
class IrisFeatures(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

class PredictionResponse(BaseModel):
    predicted_class: str
    predicted_class_id: int
    probabilities: dict

@app.get("/")
def read_root():
    return {"message": "Welcome to the Iris Classification API"}

@app.post("/predict", response_model=PredictionResponse)
def predict(features: IrisFeatures):
    try:
        # Convert input features to numpy array
        feature_array = np.array([
            [
                features.sepal_length,
                features.sepal_width,
                features.petal_length,
                features.petal_width
            ]
        ])
        
        # Make prediction
        prediction = int(model.predict(feature_array)[0])
        
        # Get prediction probabilities
        probabilities = model.predict_proba(feature_array)[0]
        prob_dict = {class_names[i]: float(probabilities[i]) for i in range(len(class_names))}
        
        return {
            "predicted_class": class_names[prediction],
            "predicted_class_id": prediction,
            "probabilities": prob_dict
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# For direct execution with uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 