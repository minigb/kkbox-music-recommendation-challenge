# main.py

from omegaconf import OmegaConf
import pandas as pd
from pathlib import Path

from feature_engineering import load_data, preprocess_data, save_processed_data
from train_model import load_data as load_processed_data, train_model, save_model
from predict import load_model, predict, save_predictions

def main(config):
    # Step 1: Feature Engineering
    if config.do_preprocess or not Path(config.processed_data_path).exists():
        user_df, item_df, interaction_df = load_data(config.dataset.members_path, config.dataset.songs_path, config.dataset.train_path)
        processed_data, label_encoders, categorical_features = preprocess_data(user_df, item_df, interaction_df)
        save_processed_data(processed_data, 'processed_data.csv')

    # Step 2: Model Training
    data = load_processed_data(config.processed_data_path)
    model = train_model(data, target_column='target', categorical_features=categorical_features)
    save_model(model, 'lightgbm_model.txt')

    # Step 3: Prediction
    model = load_model('lightgbm_model.txt')
    test_data = data.drop(columns=['target'])  # Replace with your actual test data
    predictions = predict(model, test_data)
    save_predictions(predictions, 'predictions.csv')

if __name__ == '__main__':
    config = OmegaConf.load('config.yaml')
    main(config)