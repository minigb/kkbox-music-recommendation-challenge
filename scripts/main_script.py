import os
import sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

import itertools
from copy import deepcopy
from pathlib import Path
from omegaconf import OmegaConf
import hydra
import wandb
from tqdm import tqdm
from argparse import ArgumentParser

from modules.preprocessing import encode_categorical_features
from modules.train_model import train_model
from utils import *

def save_best_model(auroc, config):
    best_auroc = 0
    if Path(config.best_model.json_path).exists():
        best_model = load_json(config.best_model.json_path)
        best_auroc = best_model['val_auroc']

    if auroc['val_auroc'] > best_auroc:
        best_model = {
            'output_dir': config.output.dir,
        }
        best_model.update(auroc)
        Path(config.best_model.json_path).parent.mkdir(parents=True, exist_ok=True)
        save_json(best_model, config.best_model.json_path)

def save_config(config, config_path):
    Path(config_path).parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config, config_path)

def train(config):
    print('Training ...')
    # Step 1: Preprocessing
    processed_data, encoder, categorical_features = encode_categorical_features(config)
    # save_processed_data(processed_data, config.output.processed_data_path)
    save_pkl(encoder, config.output.encoder_path)
    save_pkl(categorical_features, config.output.cat_features_path)

    # Step 2: Model Training
    data = processed_data
    model, aurocs = train_model(data, 'target', categorical_features, config.model_train)
    model.save_model(config.output.model_path)

    # Step 3: Save the validation AUROC score
    save_json(aurocs, config.output.auroc_path)
    save_best_model(aurocs, config)

def wandb_log_data(config, aliases=['latest']):
    # model
    model_artifact = wandb.Artifact('trained_model', type='model')
    model_artifact.add_file(config.output.model_path)
    wandb.log_artifact(model_artifact, aliases=aliases)

    # config details
    wandb.config.update(OmegaConf.to_container(config, resolve=True))

def run(config):
    # TODO(minigb): Remove duplicated code here
    PREFIX = 'run_'
    feature_keys = [key for key in config.feature_engineering if key.startswith(PREFIX)]
    enabled_features = [key[len(PREFIX):] for key, value in config.feature_engineering.items() if key in feature_keys and value]
    features_string = ','.join(enabled_features) if len(enabled_features) > 0 else 'baseline'
    config.wandb.name = f"{features_string}_{config.wandb.name}"

    # Initialize wandb and run the pipeline
    wandb.init(project=config.wandb.project, entity=config.wandb.entity, name=config.wandb.name)
    
    # Call your pipeline functions
    train(config)
    wandb_log_data(config)
    save_config(config, Path(config.output.dir) / 'config.yaml')

    # Finish wandb run
    wandb.finish()

def run_all(config):
    PREFIX = 'run_'
    feature_keys = [key for key in config.feature_engineering if key.startswith(PREFIX)]
    feature_combinations = list(itertools.product([True, False], repeat=len(feature_keys)))

    for combination in tqdm(feature_combinations):
        # Create a deep copy of the config to modify for each combination
        current_config = deepcopy(config)

        # Update the feature_engineering section with the current combination
        for key, value in zip(feature_keys, combination):
            current_config.feature_engineering[key] = value

        # Update the wandb run name to include the enabled features for this combination
        enabled_features = [key[len(PREFIX):] for key, value in current_config.feature_engineering.items() if key in feature_keys and value]
        features_string = ','.join(enabled_features) if len(enabled_features) > 0 else 'baseline'

        current_config.wandb.name = f"{features_string}_{current_config.wandb.name}"

        # Run the pipeline
        run(current_config)

@hydra.main(config_path="../", config_name="config", version_base=None)
def main(config):
    argparser = ArgumentParser()
    argparser.add_argument('--run_all', action='store_true')
    args = argparser.parse_args()

    if args.run_all:
        run_all(config)
    else:
        run(config)

if __name__ == "__main__":
    main()