# main_predict.py

from omegaconf import OmegaConf
import pickle
import pandas as pd
from predict import load_model, preprocess_test_data, predict, save_predictions

def main(config):
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
    main(config)
