# train_model.py

import lightgbm as lgb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

def load_data(data_path):
    """
    Load processed data from CSV file.
    """
    data = pd.read_csv(data_path)
    return data

def train_model(data, target_column, categorical_features):
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
    lgb_train = lgb.Dataset(X_train, y_train, categorical_feature=categorical_features)
    lgb_eval = lgb.Dataset(X_valid, y_valid, reference=lgb_train, categorical_feature=categorical_features)

    # Set parameters
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'verbose': -1,
        'seed': 42,
    }

    # Train the model
    gbm = lgb.train(
        params,
        lgb_train,
        num_boost_round=1000,
        valid_sets=[lgb_train, lgb_eval],
        early_stopping_rounds=50,
        verbose_eval=100
    )

    # Evaluate the model
    y_pred = gbm.predict(X_valid, num_iteration=gbm.best_iteration)
    auc = roc_auc_score(y_valid, y_pred)
    print('Validation AUC:', auc)

    return gbm

def save_model(model, model_path):
    """
    Save the trained model to a file.
    """
    model.save_model(model_path)
