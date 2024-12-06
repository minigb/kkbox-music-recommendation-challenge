# predict.py

import lightgbm as lgb
import pandas as pd
from pathlib import Path

from feature_engineering import merge_df

def load_model(model_path):
    """
    Load the trained model from a file.
    """
    model = lgb.Booster(model_file=model_path)
    return model

def preprocess_test_data(user_df, item_df, interaction_df, encoder, categorical_features):
    """
    Preprocess the test data using the same steps as training data.
    """
    test_df = merge_df(user_df, item_df, interaction_df)

    # Encode categorical features using the saved OrdinalEncoder
    test_df[categorical_features] = encoder.transform(test_df[categorical_features].astype(str))

    return test_df

def predict(model, data, id_column_name):
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

# TODO(minigb): Saver functions are redundant. Clean these.
def save_predictions(predictions_df, output_path):
    """
    Save the predictions to a CSV file.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    predictions_df.to_csv(output_path, index=False)
