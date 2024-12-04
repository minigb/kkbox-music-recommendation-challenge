# feature_engineering.py

import pandas as pd
from sklearn.preprocessing import LabelEncoder

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

def preprocess_data(user_df, item_df, interaction_df):
    """
    Merge datasets, clean column names, and perform feature engineering.
    """
    # Merge user and interaction data on 'msno' (user ID)
    data = interaction_df.merge(user_df, on='msno', how='left')
    # Merge item data on 'song_id' (item ID)
    data = data.merge(item_df, on='song_id', how='left')

    # Handle missing values if any
    data.fillna(-1, inplace=True)

    # Clean column names to remove special characters
    data = clean_column_names(data)

    # Identify categorical features before encoding
    categorical_features = data.select_dtypes(include=['object']).columns.tolist()
    
    # Save original categorical feature names before encoding
    original_categorical_features = categorical_features.copy()

    # Encode categorical features
    label_encoders = {}
    for col in categorical_features:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col].astype(str))
        label_encoders[col] = le

    # After encoding, update the list of categorical features if needed
    # Since LightGBM can accept categorical features as indices after encoding
    # If you want to specify categorical features, you can use their indices or names
    # Here, we assume that the categorical features remain the same after encoding

    return data, label_encoders, original_categorical_features

def save_processed_data(data, output_path):
    """
    Save the processed data to a CSV file.
    """
    data.to_csv(output_path, index=False)
