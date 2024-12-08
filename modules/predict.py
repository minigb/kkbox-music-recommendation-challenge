# predict.py

import lightgbm as lgb
import pandas as pd

from modules.preprocessing import preprocess_data
from utils import *


def predict_fn(model, data, id_column_name):
    """
    Generate predictions using the trained model and include test set IDs.
    """
    # Ensure the ID column is present in the data
    if id_column_name not in data.columns:
        raise ValueError(f"ID column '{id_column_name}' not found in the test data.")

    # Separate the IDs from the features
    ids = data[id_column_name]
    features = data.drop(columns=[id_column_name])

    # Generate predictions
    y_pred = model.predict(features, num_iteration=model.best_iteration)

    # Create a DataFrame with IDs and predictions
    predictions_df = pd.DataFrame({
        id_column_name: ids,
        'target': y_pred
    })

    return predictions_df

def run_inference(config):
    # Load data and model
    encoder = load_pkl(config.output.encoder_path)
    categorical_features = load_pkl(config.output.cat_features_path)
    model = lgb.Booster(model_file=config.output.model_path)

    # Encode categorical features using the saved OrdinalEncoder
    test_data = preprocess_data(config, is_train=False)
    test_data[categorical_features] = encoder.transform(test_data[categorical_features].astype(str))

    # Make predictions
    # Ensure that 'id' column exists in your test_df
    predictions_df = predict_fn(model, test_data, id_column_name='id')

    # Save predictions locally
    save_csv(predictions_df, config.output.submission_path)