from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import mlflow
from typing import List, Dict, Union
import json

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
    logged_model = 'runs:/cc6d0fd01d1b4711895746df4fb16a7e/model'
    loaded_model = mlflow.pyfunc.load_model(logged_model)
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
                    {"feature1": 1.0, "feature2": 2.0},
                    {"feature1": 3.0, "feature2": 4.0}
                ]
            }
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)