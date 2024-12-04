# main.py

from omegaconf import OmegaConf
import pandas as pd

from feature_engineering import load_data, preprocess_data, save_processed_data
from train_model import load_data as load_processed_data, train_model, save_model
from predict import load_model, preprocess_test_data, predict, save_predictions

def main(config):
    # Step 1: Feature Engineering
    print('Step 1: Feature Engineering')
    user_df, item_df, interaction_df = load_data(config.dataset.members_path, config.dataset.songs_path, config.dataset.train_path)
    processed_data, label_encoders, categorical_features = preprocess_data(user_df, item_df, interaction_df)
    save_processed_data(processed_data, config.processed_data_path)

    # Step 2: Model Training
    print('Step 2: Model Training')
    data = load_processed_data(config.processed_data_path)
    model = train_model(data, target_column='target', categorical_features=categorical_features)
    save_model(model, config.model_path)

    # Step 3: Prediction
    print('Step 3: Prediction')
    model = load_model(config.model_path)
    test_data = pd.read_csv(config.dataset.test_path)
    test_data = preprocess_test_data(test_data, label_encoders)
    predictions = predict(model, test_data, id_column_name='id')
    save_predictions(predictions, config.submission_path)

if __name__ == '__main__':
    config = OmegaConf.load('config.yaml')
    main(config)