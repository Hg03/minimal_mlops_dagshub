from minimal_mlops.src.confs.config import load_config
from minimal_mlops.src.training_pipeline.trainer import training_engine
from minimal_mlops.src.inference_pipeline.inference import InferenceAPI
from typing import Dict
from pathlib import Path
import os
import argparse

def main(mode: str, config: Dict):
    config = load_config()
    if mode == "train":
        inst = training_engine(config=config)
    elif mode == "infer":
        if len(os.listdir(Path(os.path.join(config["path"]["root"], config["path"]["models"])))) == 1:
            print("Train First, then initiate inference")
        else:
            inst = InferenceAPI(config=config)
            inst.run()
    elif mode == "train_with_infer":
        inst = training_engine(config=config)
        inst1 = InferenceAPI(config=config)
        inst1.run()
    
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["train", "infer", "train_with_infer"])
    args = parser.parse_args()
    main(mode=args.mode, config=load_config())
