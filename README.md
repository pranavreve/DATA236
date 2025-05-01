# ML-OPS Project

A mini MLOps pipeline with an Iris classification model and FastAPI endpoint.

## Project Structure

- `train_model.py`: Trains a Random Forest model on the Iris dataset and saves it as a .pkl file
- `app.py`: FastAPI application that loads the model and provides a /predict endpoint
- `test_model.py`: Tests that the model can be loaded correctly
- `.github/workflows/ci.yml`: GitHub Actions CI workflow

## Setup

1. Install dependencies:
```
pip install -r requirements.txt
```

2. Train the model:
```
python train_model.py
```

3. Run the API:
```
uvicorn app:app --reload
```

## API Usage

Send a POST request to `/predict` with the following JSON structure:

```json
{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}
```

Response:
```json
{
  "predicted_class": "setosa",
  "predicted_class_id": 0,
  "probabilities": {
    "setosa": 1.0,
    "versicolor": 0.0,
    "virginica": 0.0
  }
}
```

## CI/CD Pipeline

The GitHub Actions workflow in `.github/workflows/ci.yml` performs the following steps:

1. Sets up Python environment
2. Installs dependencies
3. Trains the model
4. Tests that the model can be loaded correctly
5. Verifies that the FastAPI application can start

This ensures that the model training and API functionalities work correctly with each code change. 