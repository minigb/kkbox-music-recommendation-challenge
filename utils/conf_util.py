import yaml
from omegaconf import OmegaConf

def load_conf(path):
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    return OmegaConf.create(config)

def save_conf(config, path):
    with open(path, "w") as f:
        yaml.dump(config, f)