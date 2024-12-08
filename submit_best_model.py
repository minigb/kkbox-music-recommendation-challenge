import os
import hydra
import wandb
from omegaconf import OmegaConf
from pathlib import Path

from modules.predict import run_inference
from utils import load_json

def fetch_config_from_wandb(wandb_config):
    # Initialize the wandb API
    api = wandb.Api()

    # Fetch all runs in the project
    runs = api.runs(path=f"{wandb_config.entity}/{wandb_config.project}")

    # Find the specific run by name
    for run in runs:
        if run.name == wandb_config.name:
            print(f"Found run: {run.name}")
            return run.config  # The config is a dictionary
    
    raise ValueError(f"Run with name '{wandb_config.name}' not found in project '{wandb_config.project}'.")



@hydra.main(config_path=".", config_name="config", version_base=None)
def main(config):
    best_model_json = load_json(config.best_model.json_path)
    config.wandb.name = best_model_json['output_dir'].split('/')[-1]
    config = OmegaConf.create(fetch_config_from_wandb(config.wandb))

    if not Path(config.output.submission_path).exists():
        run_inference(config)

    os.system(f"kaggle competitions submit -c kkbox-music-recommendation-challenge -f {config.output.submission_path} -m 'Best model: {config.wandb.name}'")

if __name__ == "__main__":
    main()