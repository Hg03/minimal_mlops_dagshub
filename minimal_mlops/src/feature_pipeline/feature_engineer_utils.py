from dotenv import load_dotenv
from supabase import create_client
import os
import polars as pl
from sklearn.model_selection import train_test_split
from pathlib import Path
from typing import Dict
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
    

def preprocess(config: Dict) -> Dict[str, pl.DataFrame]:
    pass