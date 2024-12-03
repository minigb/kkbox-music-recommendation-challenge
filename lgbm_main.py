# main.py

from feature_engineering import load_data, preprocess_data, save_processed_data
from train_model import load_data as load_processed_data, train_model, save_model
from predict import load_model, predict, save_predictions

# Step 1: Feature Engineering
user_df, item_df, interaction_df = load_data('user_data.csv', 'item_data.csv', 'interaction_data.csv')
processed_data, label_encoders, categorical_features = preprocess_data(user_df, item_df, interaction_df)
save_processed_data(processed_data, 'processed_data.csv')

# Step 2: Model Training
data = load_processed_data('processed_data.csv')
model = train_model(data, target_column='target', categorical_features=categorical_features)
save_model(model, 'lightgbm_model.txt')

# Step 3: Prediction
model = load_model('lightgbm_model.txt')
test_data = data.drop(columns=['target'])  # Replace with your actual test data
predictions = predict(model, test_data)
save_predictions(predictions, 'predictions.csv')
