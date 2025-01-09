from minimal_mlops.src.confs.config import load_config

def main():
    config = load_config()
    print(config["path"])


if __name__ == "__main__":
    main()
