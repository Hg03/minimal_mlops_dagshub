from fastapi import FastAPI, HTTPException, Form
from pydantic import BaseModel, Field
import pandas as pd
import pickle
import uvicorn
from typing import Dict, Annotated
from minimal_mlops.src.utils.common import Time_of_Day_Enum, Traffic_Conditions_Enum, Weather_Enum, Day_of_Week_Enum
from minimal_mlops.src.confs.config import load_config
from pathlib import Path
import os


class InferenceAPI:
    def __init__(self, config: Dict):
        # Initialize FastAPI app
        self.config = config
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
            model_path = Path(os.path.join(self.config["path"]["root"], self.config["path"]["models"]["random-forest-regressor"]["name"]))
            with open(model_path, "rb") as model_file:
                return pickle.load(model_file)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize model: {e}")

    def add_routes(self):
        @self.app.post("/predict")
        async def predict(
            Trip_Distance_km: float = Annotated[float, Form(...)],
            Time_of_Day: Time_of_Day_Enum = Annotated[Time_of_Day_Enum, Form(...)],
            Day_of_Week: Day_of_Week_Enum = Annotated[Day_of_Week_Enum, Form(...)],
            Passenger_Count: float = Annotated[float, Form(...)],
            Traffic_Conditions: Traffic_Conditions_Enum = Annotated[Traffic_Conditions_Enum, Form(...)],
            Weather: Weather_Enum = Annotated[Weather_Enum, Form(...)],
            Base_Fare: float = Annotated[float, Form(...)],
            Per_Km_Rate: float = Annotated[float, Form(...)],
            Per_Minute_Rate: float = Annotated[float, Form(...)],
            Trip_Duration_Minutes: float = Annotated[float, Form(...)],
        ):
            try:

                # Construct DataFrame from form fields
                input_data = {
                    "Trip_Distance_km": [Trip_Distance_km],
                    "Time_of_Day": [Time_of_Day.value],
                    "Day_of_Week": [Day_of_Week.value],
                    "Passenger_Count": [Passenger_Count],
                    "Traffic_Conditions": [Traffic_Conditions.value],
                    "Weather": [Weather.value],
                    "Base_Fare": [Base_Fare],
                    "Per_Km_Rate": [Per_Km_Rate],
                    "Per_Minute_Rate": [Per_Minute_Rate],
                    "Trip_Duration_Minutes": [Trip_Duration_Minutes],
                }
                
                # Convert input data to pandas DataFrame
                df = pd.DataFrame(input_data)

                # Make predictions
                predictions = self.loaded_model.predict(df)

                # Convert numpy array to list for JSON serialization
                predictions_list = predictions.tolist() if hasattr(predictions, "tolist") else predictions

                return {
                    "status": "success",
                    # "input_data": input_data,
                    # "pred": 100
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


# Example instantiation and usage
if __name__ == "__main__":
    api = InferenceAPI(config=load_config())
    api.run()
