from minimal_mlops.src.confs.config import load_config
from minimal_mlops.src.training_pipeline.trainer import training_engine
from minimal_mlops.src.inference_pipeline.inference import InferenceAPI
import argparse

def main(mode: str):
    config = load_config()
    if mode == "train":
        inst = training_engine(config=config)
    elif mode == "infer":
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
    main(mode=args.mode)
