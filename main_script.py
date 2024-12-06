import pickle
from omegaconf import OmegaConf
import pandas as pd
import wandb
import hydra
from pathlib import Path

from feature_engineering import load_data, preprocess_data, save_processed_data
from train_model import train_model, save_model
from predict import load_model, preprocess_test_data, predict as predict_fn, save_predictions

def train(config):
    # Step 1: Feature Engineering
    print('Step 1: Feature Engineering')
    user_df, item_df, interaction_df = load_data(
        config.dataset.members_path,
        config.dataset.songs_path,
        config.dataset.train_path
    )
    processed_data, encoder, categorical_features = preprocess_data(user_df, item_df, interaction_df)
    save_processed_data(processed_data, config.output.processed_data_path)

    # Save the encoder and categorical_features locally
    with open(config.output.encoder_path, 'wb') as f:
        pickle.dump(encoder, f)
    with open(config.output.cat_features_path, 'wb') as f:
        pickle.dump(categorical_features, f)

    # Step 2: Model Training
    print('Step 2: Model Training')
    data = processed_data
    model = train_model(data, target_column='target', categorical_features=categorical_features)
    save_model(model, config.output.model_path)

def run_inference(config):
    # Load data and model
    with open(config.output.encoder_path, 'rb') as f:
        encoder = pickle.load(f)
    with open(config.output.cat_features_path, 'rb') as f:
        categorical_features = pickle.load(f)
    user_df = pd.read_csv(config.dataset.members_path)
    item_df = pd.read_csv(config.dataset.songs_path)
    model = load_model(config.output.model_path)

    # Preprocess the test data using the retrieved encoder and categorical features
    test_df = pd.read_csv(config.dataset.test_path)
    test_data = preprocess_test_data(user_df, item_df, test_df, encoder, categorical_features)

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

    # predictions
    inference_artifact = wandb.Artifact('inference_results', type='inference')
    inference_artifact.add_file(config.output.submission_path)
    wandb.log_artifact(inference_artifact, aliases=['latest'])

    # config details
    wandb.config.update(OmegaConf.to_container(config, resolve=True))    

@hydra.main(config_path=".", config_name="config")
def main(config):
    wandb.init(project=config.wandb.project, entity=config.wandb.entity, name=config.wandb.name)

    # Train and log artifacts
    train(config)

    # Predict using artifacts from wandb
    run_inference(config)

    # Log the data
    wandb_log_data(config)

    # Finish the wandb run
    wandb.finish()

if __name__ == "__main__":
    main()