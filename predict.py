# predict.py

import lightgbm as lgb
import pandas as pd

def load_model(model_path):
    """
    Load the trained model from a file.
    """
    model = lgb.Booster(model_file=model_path)
    return model

def predict(model, data):
    """
    Generate predictions using the trained model.
    """
    y_pred = model.predict(data, num_iteration=model.best_iteration)
    return y_pred

def save_predictions(predictions, output_path):
    """
    Save the predictions to a CSV file.
    """
    pd.DataFrame({'prediction': predictions}).to_csv(output_path, index=False)
