import polars as pl
from typing import Dict, Union
import os
from pathlib import Path
from sklearn.pipeline import Pipeline
from minimal_mlops.src.feature_pipeline.feature_engineer_utils import load_data_from_supabase, split_data, preprocess


class feature_engine:
    def __init__(self, config: Dict):
        self.config: Dict = config
        self.raw_data_path: Path = Path(os.path.join(config["path"]["root"], config["path"]["data"], config["file"]["raw_data"]))
        self.preprocessed_data_path: Path = Path(os.path.join(config["path"]["root"], config["path"]["data"], config["file"]["raw_data"]))
        self.preprocessed_data = self.preprocess_raw()
        
    def get_raw(self) -> pl.DataFrame:
        return load_data_from_supabase(config=self.config, raw_data_path=self.raw_data_path)
    
    def preprocess_raw(self) -> Dict[str, Union[pl.DataFrame, Pipeline]]:
        split_data(config=self.config, raw_data=self.get_raw())
        return preprocess(config=self.config)
    
if __name__ == "__main__":
    from minimal_mlops.src.confs.config import load_config
    engine = feature_engine(config=load_config())