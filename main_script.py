from pathlib import Path
import pandas as pd
import hydra

from feature_engineering import load_data, preprocess_data, save_processed_data
from train_model import train_model, save_model
from predict import load_model, preprocess_test_data, predict as predict_fn, save_predictions
from utils import *

def save_best_model(model, auroc, config):
    if not Path(config.best_model.json_path).exists():
        Path(config.best_model.json_path).parent.mkdir(parents=True, exist_ok=True)
        best_model = {
            'output_dir': config.output.dir,
            'val_auroc': auroc
        }
    else:
        best_model = load_json(config.best_model.json_path)
        if auroc > best_model['val_auroc']:
            best_model = {
                'output_dir': config.output.dir,
                'val_auroc': auroc
            }
    save_json(best_model, config.best_model.json_path)

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
    save_pkl(encoder, config.output.encoder_path)
    save_pkl(categorical_features, config.output.cat_features_path)

    # Step 2: Model Training
    print('Step 2: Model Training')
    data = processed_data
    model, val_auroc = train_model(data, target_column='target', categorical_features=categorical_features)
    save_model(model, config.output.model_path)
    save_best_model(model, val_auroc, config)

def run_inference(config):
    # Load data and model
    encoder = load_pkl(config.output.encoder_path)
    categorical_features = load_pkl(config.output.cat_features_path)
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
    
@hydra.main(config_path=".", config_name="config", version_base=None)
def main(config):
    train(config)
    run_inference(config)

if __name__ == "__main__":
    main()