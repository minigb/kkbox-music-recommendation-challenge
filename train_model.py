# train_model.py

import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# TODO(minigb): Remove this. This makes code confusing.
def load_data(data_path):
    """
    Load processed data from CSV file.
    """
    data = pd.read_csv(data_path)
    return data

def train_model(data, target_column, categorical_features=None):
    """
    Train a LightGBM model.
    """
    X = data.drop(columns=[target_column])
    y = data[target_column]

    # Split the data
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Create LightGBM datasets
    train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=categorical_features)
    valid_data = lgb.Dataset(X_valid, label=y_valid, reference=train_data, categorical_feature=categorical_features)

    # Set parameters
    params = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': 'binary_logloss',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'verbose': -1,
        'seed': 42,
    }

    # Define early stopping callback
    callbacks = [lgb.early_stopping(stopping_rounds=10)]

    # Train the model
    model = lgb.train(
        params,
        train_data,
        num_boost_round=100,
        valid_sets=[train_data, valid_data],
        callbacks=callbacks
    )

    # Make predictions
    y_pred = model.predict(X_valid, num_iteration=model.best_iteration)

    # Evaluate the model
    y_pred_classes = (y_pred > 0.5).astype(int)
    accuracy = accuracy_score(y_valid, y_pred_classes)
    print(f"Validation Accuracy: {accuracy * 100:.2f}%")

    # Feature importance
    print("Feature Importance:")
    for feature, importance in zip(X.columns, model.feature_importance()):
        print(f"{feature}: {importance}")

    return model

def save_model(model, model_path):
    """
    Save the trained model to a file.
    """
    model.save_model(model_path)
