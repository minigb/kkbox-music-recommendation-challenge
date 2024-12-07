import itertools
from copy import deepcopy
from pathlib import Path
from omegaconf import OmegaConf
import hydra
import wandb
from tqdm import tqdm

from modules.preprocessing import preprocess_data, encode_categorical_features, save_processed_data
from modules.train_model import train_model, save_model
from modules.predict import load_model, predict as predict_fn, save_predictions
from utils import *

def save_best_model(auroc, config):
    best_auroc = 0
    if Path(config.best_model.json_path).exists():
        best_model = load_json(config.best_model.json_path)
        best_auroc = best_model['val_auroc']

    if auroc > best_auroc:
        best_model = {
            'output_dir': config.output.dir,
            'val_auroc': auroc
        }
        Path(config.best_model.json_path).parent.mkdir(parents=True, exist_ok=True)
        save_json(best_model, config.best_model.json_path)

def save_auroc(auroc, auroc_path):
    save_json({'val_auroc': auroc}, auroc_path)

def train(config):
    # Step 1: Preprocessing
    print('Step 1: Preprocessing')
    processed_data, encoder, categorical_features = encode_categorical_features(config)
    # save_processed_data(processed_data, config.output.processed_data_path)

    # Save the encoder and categorical_features locally
    save_pkl(encoder, config.output.encoder_path)
    save_pkl(categorical_features, config.output.cat_features_path)

    # Step 2: Model Training
    print('Step 2: Model Training')
    data = processed_data
    model, val_auroc = train_model(data, target_column='target', categorical_features=categorical_features)
    save_model(model, config.output.model_path)
    save_auroc(val_auroc, config.output.auroc_path)
    save_best_model(val_auroc, config)

def run_inference(config):
    # Load data and model
    encoder = load_pkl(config.output.encoder_path)
    categorical_features = load_pkl(config.output.cat_features_path)
    model = load_model(config.output.model_path)

    # Encode categorical features using the saved OrdinalEncoder
    test_data = preprocess_data(config, is_train=False)
    test_data[categorical_features] = encoder.transform(test_data[categorical_features].astype(str))

    # Make predictions
    # Ensure that 'id' column exists in your test_df
    predictions_df = predict_fn(model, test_data, id_column_name='id')

    # Save predictions locally
    save_predictions(predictions_df, config.output.submission_path)

def wandb_log_data(config, aliases=['latest']):
    # model
    model_artifact = wandb.Artifact('trained_model', type='model')
    model_artifact.add_file(config.output.model_path)
    wandb.log_artifact(model_artifact, aliases=aliases)

    # # predictions
    # inference_artifact = wandb.Artifact('inference_results', type='inference')
    # inference_artifact.add_file(config.output.submission_path)
    # wandb.log_artifact(inference_artifact, aliases=['latest'])

    # config details
    wandb.config.update(OmegaConf.to_container(config, resolve=True))   
    
@hydra.main(config_path=".", config_name="config", version_base=None)
def main(config):
    feature_keys = [key for key in config.feature_engineering if not key.startswith('const')]
    feature_combinations = list(itertools.product([True, False], repeat=len(feature_keys)))

    for combination in tqdm(feature_combinations):
        # Create a deep copy of the config to modify for each combination
        current_config = deepcopy(config)

        # Update the feature_engineering section with the current combination
        for key, value in zip(feature_keys, combination):
            current_config.feature_engineering[key] = value

        # Update the wandb run name to include the enabled features for this combination
        enabled_features = [key for key, value in current_config.feature_engineering.items() if key in feature_keys and value]
        features_string = ','.join(enabled_features) if len(enabled_features) > 0 else 'baseline'
        for existing_out_files in Path(current_config.output.dir).glob('*'):
            if features_string in existing_out_files.name:
                continue

        current_config.wandb.name = f"{features_string}_{current_config.wandb.name}"

        # Initialize wandb and run the pipeline
        wandb.init(project=current_config.wandb.project, entity=current_config.wandb.entity, name=current_config.wandb.name)
        
        # Call your pipeline functions
        train(current_config)
        run_inference(current_config)
        wandb_log_data(current_config)

        # Finish wandb run
        wandb.finish()


if __name__ == "__main__":
    main()