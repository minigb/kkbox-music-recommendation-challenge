import os
import sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

import hydra
from pathlib import Path

from modules.predict import run_inference
from utils import load_json, fetch_config_from_wandb


@hydra.main(config_path="../", config_name="config", version_base=None)
def main(config):
    best_model_json = load_json(config.best_model.json_path)
    config.wandb.name = best_model_json['output_dir'].split('/')[-1]
    config = fetch_config_from_wandb(config.wandb)

    if not Path(config.output.submission_path).exists():
        run_inference(config)

    os.system(f"kaggle competitions submit -c kkbox-music-recommendation-challenge -f {config.output.submission_path} -m 'Best model: {config.wandb.name}'")

if __name__ == "__main__":
    main()