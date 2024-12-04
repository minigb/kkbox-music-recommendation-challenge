# main.py

import pickle
from omegaconf import OmegaConf
from feature_engineering import load_data, preprocess_data, save_processed_data
from train_model import train_model, save_model

def main(config):
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

if __name__ == "__main__":
    config = OmegaConf.load('config.yaml')
    main(config)