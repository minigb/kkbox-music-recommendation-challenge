import os
import pickle
from omegaconf import OmegaConf
import pandas as pd
import wandb

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
    save_processed_data(processed_data, config.processed_data_path)

    # Save the encoder and categorical_features locally
    encoder_path = 'ordinal_encoder.pkl'
    cat_features_path = 'categorical_features.pkl'
    with open(encoder_path, 'wb') as f:
        pickle.dump(encoder, f)
    with open(cat_features_path, 'wb') as f:
        pickle.dump(categorical_features, f)

    # Step 2: Model Training
    print('Step 2: Model Training')
    data = processed_data
    model = train_model(data, target_column='target', categorical_features=categorical_features)
    save_model(model, config.model_path)

    # Log preprocessing files as an artifact
    preprocessing_artifact = wandb.Artifact('preprocessing_artifacts', type='dataset')
    preprocessing_artifact.add_file(config.processed_data_path)
    preprocessing_artifact.add_file(encoder_path)
    preprocessing_artifact.add_file(cat_features_path)
    wandb.log_artifact(preprocessing_artifact, aliases=['latest'])

    # Log the model as an artifact
    model_artifact = wandb.Artifact('trained_model', type='model')
    model_artifact.add_file(config.model_path)
    wandb.log_artifact(model_artifact, aliases=['latest'])

    # Optionally, store config details in wandb
    wandb.config.update(OmegaConf.to_container(config, resolve=True))

def run_inference(config):
    # Use the preprocessing artifact from wandb
    preprocessing_art = wandb.use_artifact(f'{wandb.run.entity}/{wandb.run.project}/preprocessing_artifacts:latest', type='dataset')
    preprocessing_dir = preprocessing_art.download()
    encoder_path = os.path.join(preprocessing_dir, 'ordinal_encoder.pkl')
    cat_features_path = os.path.join(preprocessing_dir, 'categorical_features.pkl')

    with open(encoder_path, 'rb') as f:
        encoder = pickle.load(f)
    with open(cat_features_path, 'rb') as f:
        categorical_features = pickle.load(f)

    # Use the model artifact from wandb
    model_art = wandb.use_artifact(f'{wandb.run.entity}/{wandb.run.project}/trained_model:latest', type='model')
    model_dir = model_art.download()
    model_path = os.path.join(model_dir, 'lightgbm_model.txt')

    # Load the test data
    test_df = pd.read_csv(config.dataset.test_path)

    # Also need user_df and item_df for preprocessing test data
    user_df = pd.read_csv(config.dataset.members_path)
    item_df = pd.read_csv(config.dataset.songs_path)

    # Preprocess the test data using the retrieved encoder and categorical features
    test_data = preprocess_test_data(user_df, item_df, test_df, encoder, categorical_features)

    # Load the model from the downloaded artifact directory
    model = load_model(model_path)

    # Make predictions
    # Ensure that 'id' column exists in your test_df
    predictions_df = predict_fn(model, test_data, id_column_name='id')

    # Save predictions locally
    save_predictions(predictions_df, config.submission_path)

    # Log predictions as an artifact
    inference_artifact = wandb.Artifact('inference_results', type='inference')
    inference_artifact.add_file(config.submission_path)
    wandb.log_artifact(inference_artifact, aliases=['latest'])

if __name__ == "__main__":
    config = OmegaConf.load('config.yaml')
    wandb.init(project=config.wandb.project, entity=config.wandb.entity, name=config.wandb.name)

    # Train and log artifacts
    train(config)

    # Predict using artifacts from wandb
    run_inference(config)

    # Finish the wandb run
    wandb.finish()
