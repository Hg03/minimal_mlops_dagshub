from minimal_mlops.src.confs.config import load_config
from minimal_mlops.src.training_pipeline.trainer import training_engine
# from minimal_mlops.src.inference_pipeline import inference
import argparse

def main(mode: str):
    config = load_config()
    if mode == "train":
        inst = training_engine(config=config)
    elif mode == "infer":
        # inst = inference
        pass
    elif mode == "train_with_infer":
        inst = training_engine(config=config)
        # inst1 = inference
    
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["train", "infer", "train_with_infer"])
    args = parser.parse_args()
    main(mode=args.mode)
