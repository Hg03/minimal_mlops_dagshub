from dotenv import load_dotenv
from supabase import create_client
import os
import polars as pl
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, FunctionTransformer
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from pathlib import Path
from typing import Dict
import numpy as np
from tqdm import tqdm

def load_data_from_supabase(config: Dict, raw_data_path: Path) -> pl.DataFrame:
    load_dotenv()
    conn = create_client(supabase_url=os.getenv("supabase_url"), supabase_key=os.getenv("supabase_key"))
    json_data = []
    batch_size, offset = config["load_params"]["batch_size"], config["load_params"]["offset"]
    total_rows = conn.table(config["load_params"]["table"]).select("count", count="exact").execute().count
    # Create progress bar
    progress_bar = tqdm(total=total_rows,desc="Loading data from Supabase",unit=" rows")
    while True:
        response = conn.table(config["load_params"]["table"]).select("*").limit(batch_size).offset(offset).execute()
        batch = response.data
        if not batch:
            break
        json_data.extend(batch)
        offset+=batch_size
        progress_bar.update(len(batch))
    progress_bar.close()
    raw_data = pl.DataFrame(json_data)
    raw_data.write_parquet(raw_data_path)
    return raw_data


def split_data(config: Dict, raw_data: pl.DataFrame) -> None:
    train, test = train_test_split(raw_data, test_size=config["split"]["test_size"], shuffle=config["split"]["shuffle"], random_state=config["split"]["random_state"])
    train.write_parquet(Path(os.path.join(config["path"]["root"], config["path"]["data"], config["file"]["raw_training_data"])))
    test.write_parquet(Path(os.path.join(config["path"]["root"], config["path"]["data"], config["file"]["raw_testing_data"])))
    

def load_raw_from_local(config: Dict) -> Dict[str, pl.DataFrame]:
    return {
        "train": pl.read_parquet(Path(os.path.join(config["path"]["root"], config["path"]["data"], config["file"]["raw_training_data"]))),
        "test": pl.read_parquet(Path(os.path.join(config["path"]["root"], config["path"]["data"], config["file"]["raw_testing_data"]))),
    }


def column_preserver(data: pl.DataFrame):
    original_columns = [col[col.rfind('__') + 2:] for col in data.columns]
    data = data.rename(dict(zip(data.columns, original_columns)))
    return data

def transformation_pipeline(config: Dict):  
    numerical_imputer = SimpleImputer(strategy=config["preprocessing_strategies"]["numerical_impute"])
    categorical_imputer = SimpleImputer(strategy=config["preprocessing_strategies"]["categorical_impute"])
    keep_the_name_same = FunctionTransformer(column_preserver)
    imputation_transformer = make_column_transformer(
        (numerical_imputer, config["columns"]["numeric"]),
        (categorical_imputer, config["columns"]["categoric"]),
    remainder='passthrough').set_output(transform='polars')
    pipe_I =   make_pipeline(imputation_transformer, keep_the_name_same)
    encoding_transformer = make_column_transformer(
        (OneHotEncoder(sparse_output=False, handle_unknown='ignore'), config["columns"]["nominal"]),
        (OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), config["columns"]["ordinal"]),
        remainder='passthrough'
    ).set_output(transform='polars')
    pipe_II = make_pipeline(encoding_transformer, keep_the_name_same)
    
    final_pipe = make_pipeline(pipe_I, pipe_II)
    return final_pipe
    

def preprocess(config: Dict) -> Dict[str, pl.DataFrame]:
    raw = load_raw_from_local(config=config)
    raw_train, raw_test = raw["train"], raw["test"]
    preprocessor = transformation_pipeline(config=config)
    preprocessed_train = preprocessor.fit_transform(raw_train)
    preprocessed_test = preprocessor.transform(raw_test)
    preprocessed_train.write_parquet(Path(os.path.join(config["path"]["root"], config["path"]["data"], config["file"]["processed_training_data"])))
    preprocessed_test.write_parquet(Path(os.path.join(config["path"]["root"], config["path"]["data"], config["file"]["processed_testing_data"])))
    return {
        "train": preprocessed_train.drop_nulls(),
        "test": preprocessed_test.drop_nulls()
    }