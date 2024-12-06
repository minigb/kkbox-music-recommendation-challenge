# feature_engineering.py

import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from pathlib import Path

from modules.feature_engineering import FeatureEngineering

def merge_song_data(config):
    """
    Merge song information and song extra information datasets.
    """
    if Path(config.dataset.songs_path).exists():
        return
    songs = pd.read_csv(config.dataset.songs_original_path)
    songs_extra = pd.read_csv(config.dataset.songs_extra_original_path)
    data = songs.merge(songs_extra, on='song_id', how='left')
    data.to_csv(config.dataset.songs_path, index=False)

def clean_column_names(df):
    """
    Clean column names by replacing special characters with underscores.
    """
    df.columns = df.columns.str.replace(r'[^\w]', '_', regex=True)
    return df

def merge_user_item_to_interaction_data(user_df, item_df, interaction_df):
    """
    Merge datasets on 'msno' and 'song_id'.
    """
    data = interaction_df.merge(user_df, on='msno', how='left')
    data = data.merge(item_df, on='song_id', how='left')
    data = clean_column_names(data)

    return data

def preprocess_data(config, is_train=True):
    """
    Merge datasets, clean column names, and perform feature engineering.
    """
    merge_song_data(config)
    user_df = pd.read_csv(config.dataset.members_path)
    item_df = pd.read_csv(config.dataset.songs_path)
    interaction_df = pd.read_csv(config.dataset.train_path) if is_train else pd.read_csv(config.dataset.test_path)
    data = merge_user_item_to_interaction_data(user_df, item_df, interaction_df)

    # Perform feature engineering
    feature_engineering = FeatureEngineering(data, config)
    data = feature_engineering.run()

    return data

def encode_categorical_features(config):
    data = preprocess_data(config)
    
    # Identify categorical features before encoding
    categorical_features = data.select_dtypes(include=['object']).columns.tolist()
    
    # Save original categorical feature names before encoding
    original_categorical_features = categorical_features.copy()

    # Encode categorical features using OrdinalEncoder
    encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    data[categorical_features] = encoder.fit_transform(data[categorical_features].astype(str))

    # Return the encoder so it can be saved and reused during prediction
    return data, encoder, original_categorical_features

def save_processed_data(data, output_path):
    """
    Save the processed data to a CSV file.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output_path, index=False)
