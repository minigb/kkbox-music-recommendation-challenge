# predict.py

import lightgbm as lgb
import pandas as pd

from feature_engineering import clean_column_names

def load_model(model_path):
    """
    Load the trained model from a file.
    """
    model = lgb.Booster(model_file=model_path)
    return model

def preprocess_test_data(test_df, label_encoders):
    """
    Preprocess the test data using the same steps as training data.
    """
    # Clean column names
    test_df = clean_column_names(test_df)

    # Handle missing values
    test_df.fillna(-1, inplace=True)

    # Encode categorical features using saved label encoders
    for col, le in label_encoders.items():
        if col in test_df.columns:
            test_df[col] = le.transform(test_df[col].astype(str))
        else:
            raise ValueError(f"Column '{col}' not found in test data.")

    return test_df

def predict(model, data, id_column_name):
    """
    Generate predictions using the trained model and include test set IDs.

    Parameters:
    - model: Trained LightGBM model.
    - data: DataFrame containing test features.
    - id_column_name: String name of the ID column in the test data.

    Returns:
    - predictions_df: DataFrame containing IDs and their corresponding predictions.
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
        'prediction': y_pred
    })

    return predictions_df


def save_predictions(predictions, output_path):
    """
    Save the predictions to a CSV file.
    """
    pd.DataFrame({'target': predictions}).to_csv(output_path, index=False)
