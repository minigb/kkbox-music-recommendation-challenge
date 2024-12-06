import pickle
from omegaconf import OmegaConf
import pandas as pd
import wandb
import hydra
from pathlib import Path

from feature_engineering import load_data, preprocess_data, save_processed_data
from train_model import train_model, save_model
from predict import load_model, preprocess_test_data, predict as predict_fn, save_predictions

def wandb_log_preprocessed_artifact(config, aliases=['latest']):
    # Log preprocessing files as an artifact
    preprocessing_artifact = wandb.Artifact('preprocessing_artifacts', type='dataset')
    preprocessing_artifact.add_file(config.output.processed_data_path)
    preprocessing_artifact.add_file(config.output.encoder_path)
    preprocessing_artifact.add_file(config.output.cat_features_path)
    wandb.log_artifact(preprocessing_artifact, aliases=aliases)

    # Log the model as an artifact
    model_artifact = wandb.Artifact('trained_model', type='model')
    model_artifact.add_file(config.output.model_path)
    wandb.log_artifact(model_artifact, aliases=aliases)

    # Optionally, store config details in wandb
    wandb.config.update(OmegaConf.to_container(config, resolve=True))

def wandb_load_artifact(config):
    # Use the preprocessing artifact from wandb
    preprocessing_art = wandb.use_artifact(f'{wandb.run.entity}/{wandb.run.project}/preprocessing_artifacts:latest', type='dataset')
    preprocessing_dir = Path(preprocessing_art.download())
    encoder_path = preprocessing_dir / config.output.encoder_path
    cat_features_path = preprocessing_dir / config.output.cat_features_path

    with open(encoder_path, 'rb') as f:
        encoder = pickle.load(f)
    with open(cat_features_path, 'rb') as f:
        categorical_features = pickle.load(f)

    # Use the model artifact from wandb
    model_art = wandb.use_artifact(f'{wandb.run.entity}/{wandb.run.project}/trained_model:latest', type='model')
    model_dir = Path(model_art.download())
    model_path = model_dir / config.output.model_path

    # Load the test data
    test_df = pd.read_csv(config.dataset.test_path)

    # Also need user_df and item_df for preprocessing test data
    user_df = pd.read_csv(config.dataset.members_path)
    item_df = pd.read_csv(config.dataset.songs_path)

    # Load the model from the downloaded artifact directory
    model = load_model(model_path)

    return user_df, item_df, test_df, encoder, categorical_features, model
    
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

    # Log artifacts to wandb
    wandb_log_preprocessed_artifact(config)

def run_inference(config):
    # Load artifacts from wandb
    user_df, item_df, test_df, encoder, categorical_features, model = wandb_load_artifact(config)

    # Preprocess the test data using the retrieved encoder and categorical features
    test_data = preprocess_test_data(user_df, item_df, test_df, encoder, categorical_features)

    # Make predictions
    # Ensure that 'id' column exists in your test_df
    predictions_df = predict_fn(model, test_data, id_column_name='id')

    # Save predictions locally
    save_predictions(predictions_df, config.output.submission_path)

    # Log predictions as an artifact
    inference_artifact = wandb.Artifact('inference_results', type='inference')
    inference_artifact.add_file(config.output.submission_path)
    wandb.log_artifact(inference_artifact, aliases=['latest'])

@hydra.main(config_path=".", config_name="config")
def main(config):
    wandb.init(project=config.wandb.project, entity=config.wandb.entity, name=config.wandb.name)

    # Train and log artifacts
    train(config)

    # Predict using artifacts from wandb
    run_inference(config)

    # Finish the wandb run
    wandb.finish()

if __name__ == "__main__":
    main()