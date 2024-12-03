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

def preprocess_data(user_df, item_df, interaction_df):
    """
    Merge datasets and perform feature engineering.
    """
    # Merge user and interaction data on 'user_id'
    data = interaction_df.merge(user_df, on='user_id', how='left')
    # Merge item data on 'item_id'
    data = data.merge(item_df, on='item_id', how='left')

    # Handle missing values if any
    data.fillna(-1, inplace=True)

    # Encode categorical features
    categorical_features = data.select_dtypes(include=['object']).columns.tolist()
    label_encoders = {}
    for col in categorical_features:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col].astype(str))
        label_encoders[col] = le

    return data, label_encoders, categorical_features

def save_processed_data(data, output_path):
    """
    Save the processed data to a CSV file.
    """
    data.to_csv(output_path, index=False)
