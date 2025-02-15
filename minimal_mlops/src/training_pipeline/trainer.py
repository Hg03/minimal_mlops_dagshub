from typing import Dict
import polars as pl
from minimal_mlops.src.feature_pipeline import feature_engineer
from minimal_mlops.src.training_pipeline.trainer_utils import get_model_with_hyperparams, tune_and_predict

class training_engine:
    def __init__(self, config: Dict):
        self.config = config
        self.model_name = "random-forest-regressor"
        self.preprocessed_data = feature_engineer.feature_engine(config=config).preprocessed_data
        self.training_data = self.preprocessed_data["train"]
        self.testing_data = self.preprocessed_data["test"]
        self.preprocessor = self.preprocessed_data["preprocessor"]
        self.train()
        
    def train(self):
        model, hyperparams = get_model_with_hyperparams(config=self.config, model_name=self.model_name)
        tune_and_predict(config=self.config, preprocessor=self.preprocessor, model_name=self.model_name, model=model, hyperparams=hyperparams)
        
if __name__ == "__main__":
    from minimal_mlops.src.confs.config import load_config
    obj = training_engine(config=load_config())