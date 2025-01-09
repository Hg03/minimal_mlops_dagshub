import tomli

def load_config():
    with open("minimal_mlops/src/confs/config.toml", "rb") as f:
        config = tomli.load(f)
        return config