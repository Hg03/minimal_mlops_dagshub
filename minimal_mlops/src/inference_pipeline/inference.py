from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
import pickle
import uvicorn
from typing import List, Dict
from minimal_mlops.src.confs.config import load_config


class InferenceAPI:
    def __init__(self, config: Dict):
        # Initialize FastAPI app
        self.app = FastAPI(
            title="MLflow Model Prediction API",
            description="API for making predictions using an MLflow model",
            version="1.0.0",
        )

        # Add routes
        self.add_routes()

        # Load model
        self.loaded_model = self.load_model()

    def load_model(self):
        try:
            model_path = "minimal_mlops/src/models/random_forest_model.pkl"
            with open(model_path, "rb") as model_file:
                return pickle.load(model_file)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize model: {e}")

    def add_routes(self):
        @self.app.post("/predict")
        async def predict(input_data: List[PredictionInput]):
            try:
                # Convert input data to pandas DataFrame
                df = pd.DataFrame([item.dict() for item in input_data])

                # Make predictions
                predictions = self.loaded_model.predict(df)

                # Convert numpy array to list for JSON serialization
                predictions_list = predictions.tolist() if hasattr(predictions, "tolist") else predictions

                return {
                    "status": "success",
                    "predictions": predictions_list,
                }
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/health")
        async def health_check():
            return {"status": "healthy", "model_loaded": self.loaded_model is not None}

        @self.app.get("/")
        async def root():
            return {
                "message": "Welcome to the MLflow Model Prediction API",
                "usage": {
                    "endpoint": "/predict",
                    "method": "POST",
                    "example_input": [
                        {
                            "Trip_Distance_km": 19.35,
                            "Time_of_Day": "Morning",
                            "Day_of_Week": "Weekday",
                            "Passenger_Count": 3.0,
                            "Traffic_Conditions": "Low",
                            "Weather": "Clear",
                            "Base_Fare": 3.56,
                            "Per_Km_Rate": 0.8,
                            "Per_Minute_Rate": 0.32,
                            "Trip_Duration_Minutes": 53.82,
                        },
                    ],
                },
            }

    def run(self, host="0.0.0.0", port=8000):
        uvicorn.run(self.app, host=host, port=port)


# Define input data model
class PredictionInput(BaseModel):
    Trip_Distance_km: float = Field(..., description="Distance of the trip in kilometers")
    Time_of_Day: str = Field(..., description="Time of the day (e.g., Morning)")
    Day_of_Week: str = Field(..., description="Day of the week (e.g., Weekday)")
    Passenger_Count: float = Field(..., description="Number of passengers")
    Traffic_Conditions: str = Field(..., description="Traffic conditions (e.g., Low)")
    Weather: str = Field(..., description="Weather conditions (e.g., Clear)")
    Base_Fare: float = Field(..., description="Base fare for the trip")
    Per_Km_Rate: float = Field(..., description="Rate per kilometer")
    Per_Minute_Rate: float = Field(..., description="Rate per minute")
    Trip_Duration_Minutes: float = Field(..., description="Duration of the trip in minutes")


# Example instantiation and usage
if __name__ == "__main__":
    api = InferenceAPI(config=load_config())
    api.run()
