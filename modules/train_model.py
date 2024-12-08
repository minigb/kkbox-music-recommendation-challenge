# train_model.py

import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

def train_model(data, target_column, categorical_features=None, kwargs=None):
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
    params.update(kwargs or {})

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

    # Validate the model
    # Make predictions
    y_pred = model.predict(X_valid, num_iteration=model.best_iteration)

    # Evaluate the model using AUROC
    auroc = roc_auc_score(y_valid, y_pred)

    return model, auroc
