import pickle
from omegaconf import OmegaConf
import pandas as pd
import wandb

from feature_engineering import load_data, preprocess_data, save_processed_data
from train_model import train_model, save_model
from predict import load_model, preprocess_test_data, predict, save_predictions

def train(config):
    # Step 1: Feature Engineering
    print('Step 1: Feature Engineering')
    user_df, item_df, interaction_df = load_data(config.dataset.members_path, config.dataset.songs_path, config.dataset.train_path)
    processed_data, encoder, categorical_features = preprocess_data(user_df, item_df, interaction_df)
    save_processed_data(processed_data, config.processed_data_path)

    # Save the encoder
    with open('ordinal_encoder.pkl', 'wb') as f:
        pickle.dump(encoder, f)

    # Save the list of categorical features
    with open('categorical_features.pkl', 'wb') as f:
        pickle.dump(categorical_features, f)

    # Step 2: Model Training
    print('Step 2: Model Training')
    data = processed_data  # Data is already loaded
    model = train_model(data, target_column='target', categorical_features=categorical_features)
    save_model(model, config.model_path)


def predict(config):
    # Load the model
    model = load_model(config.model_path)

    # Load the test data
    test_df = pd.read_csv(config.dataset.test_path)  # Replace with your actual test data path

    # Load the saved OrdinalEncoder and categorical features
    with open('ordinal_encoder.pkl', 'rb') as f:
        encoder = pickle.load(f)

    with open('categorical_features.pkl', 'rb') as f:
        categorical_features = pickle.load(f)

    user_df = pd.read_csv(config.dataset.members_path)
    item_df = pd.read_csv(config.dataset.songs_path)
    # Preprocess the test data
    test_data = preprocess_test_data(user_df, item_df, test_df, encoder, categorical_features)

    # Make predictions
    predictions_df = predict(model, test_data, id_column_name='id')  # Replace 'id' with your actual ID column name

    # Save predictions
    save_predictions(predictions_df, config.submission_path)


if __name__ == "__main__":
    config = OmegaConf.load('config.yaml')
    wandb.init(project=config.wandb.project, entity=config.wandb.entity, name=config.wandb.name)
    
    train(config)
    predict(config)