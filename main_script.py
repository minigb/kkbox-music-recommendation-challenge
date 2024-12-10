import itertools
from copy import deepcopy
from pathlib import Path
from omegaconf import OmegaConf
import hydra
import wandb
from tqdm import tqdm

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

@hydra.main(config_path=".", config_name="config", version_base=None)
def main(config):
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
        # if any(features_string in existing_out_dirs.name for existing_out_dirs in Path(current_config.output.main_dir).iterdir()):
        #     continue

        current_config.wandb.name = f"{features_string}_{current_config.wandb.name}"

        # # skip if it is already ran
        # config_dict = OmegaConf.to_container(current_config, resolve=True)
        # keys = ['feature_engineering', 'model_train']
        # for existing_out_dirs in Path(current_config.output.main_dir).iterdir():
        #     if not existing_out_dirs.is_dir():
        #         continue
        #     config_path = existing_out_dirs / 'config.yaml'
        #     saved_config = load_conf(config_path)
        #     saved_config_dict = OmegaConf.to_container(saved_config, resolve=True)
        #     is_same = True
        #     for key in keys:
        #         if saved_config_dict.get(key) is None:
        #             is_same = False
        #             break
        #         if config_dict[key] != saved_config_dict[key]:
        #             is_same = False
        #             break
        #     if is_same:
        #         continue
            
        # Initialize wandb and run the pipeline
        wandb.init(project=current_config.wandb.project, entity=current_config.wandb.entity, name=current_config.wandb.name)
        
        # Call your pipeline functions
        train(current_config)
        wandb_log_data(current_config)
        save_config(current_config, Path(current_config.output.main_dir) / 'config.yaml')

        # Finish wandb run
        wandb.finish()


if __name__ == "__main__":
    main()