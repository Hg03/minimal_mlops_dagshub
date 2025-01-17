from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import mlflow
from typing import List, Dict, Union
import json
import os

# Initialize FastAPI app
app = FastAPI(
    title="MLflow Model Prediction API",
    description="API for making predictions using an MLflow model",
    version="1.0.0"
)

# Define input data model
class PredictionInput(BaseModel):
    data: List[Dict[str, Union[float, int, str]]]

# Load the MLflow model at startup
try:
    import pickle
    # logged_model = 'runs:/8ca2ec3d6fc0413a9a3a193813a1e6ae/model'
    # loaded_model = mlflow.pyfunc.load_model(logged_model)
    model_file = open("minimal_mlops/src/models/random_forest_model.pkl", "rb")
    loaded_model = pickle.load(model_file)
except Exception as e:
    raise RuntimeError("Failed to initialize model")

@app.post("/predict")
async def predict(input_data: PredictionInput):
    try:
        # Convert input data to pandas DataFrame
        df = pd.DataFrame(input_data.data)
        
        # Make predictions
        predictions = loaded_model.predict(df)
        
        # Convert numpy array to list for JSON serialization
        predictions_list = predictions.tolist() if hasattr(predictions, 'tolist') else predictions
        
        return {
            "status": "success",
            "predictions": predictions_list
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": loaded_model is not None}

# Example usage in documentation
@app.get("/")
async def root():
    return {
        "message": "Welcome to the MLflow Model Prediction API",
        "usage": {
            "endpoint": "/predict",
            "method": "POST",
            "example_input": {
                "data": [
                    {'Trip_Distance_km': 19.35, 'Time_of_Day': 'Morning', 'Day_of_Week': 'Weekday', 'Passenger_Count': 3.0, 'Traffic_Conditions': 'Low', 'Weather': 'Clear', 'Base_Fare': 3.56, 'Per_Km_Rate': 0.8, 'Per_Minute_Rate': 0.32, 'Trip_Duration_Minutes': 53.82},
                ]
            }
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)