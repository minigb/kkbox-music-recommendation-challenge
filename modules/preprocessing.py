# feature_engineering.py

import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from pathlib import Path

def load_data(user_data_path, item_data_path, interaction_data_path):
    """
    Load user data, item data, and interaction data from CSV files.
    """
    user_df = pd.read_csv(user_data_path)
    item_df = pd.read_csv(item_data_path)
    interaction_df = pd.read_csv(interaction_data_path)
    return user_df, item_df, interaction_df

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

def preprocess_data(user_df, item_df, interaction_df):
    """
    Merge datasets, clean column names, and perform feature engineering.
    """
    data = merge_user_item_to_interaction_data(user_df, item_df, interaction_df)

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
