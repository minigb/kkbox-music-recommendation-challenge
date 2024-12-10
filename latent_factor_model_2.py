import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
from datetime import datetime
from tqdm import tqdm

class LatentFactorModel:
    def __init__(self, n_factors=50, learning_rate=0.01, n_epochs=100, reg=0.01, default_value=0.5):
        self.default_value = default_value
        self.n_factors = n_factors
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.reg = reg

    def fit(self, train_data, user_factors_init=None, song_factors_init=None):
        self.user_ids = train_data['msno'].unique()
        self.song_ids = train_data['song_id'].unique()
        
        self.user_map = {user_id: idx for idx, user_id in enumerate(self.user_ids)}
        self.song_map = {song_id: idx for idx, song_id in enumerate(self.song_ids)}
        
        self.n_users = len(self.user_ids)
        self.n_songs = len(self.song_ids)
        
        if user_factors_init is not None and song_factors_init is not None:
            if user_factors_init.shape != (self.n_users, self.n_factors) or song_factors_init.shape != (self.n_songs, self.n_factors):
                raise ValueError("Initial factors have incorrect dimensions.")
            self.user_factors = user_factors_init
            self.song_factors = song_factors_init
        else:
            self.user_factors = np.random.normal(scale=1./self.n_factors, size=(self.n_users, self.n_factors))
            self.song_factors = np.random.normal(scale=1./self.n_factors, size=(self.n_songs, self.n_factors))
        
        for epoch in range(self.n_epochs):
            for row in tqdm(train_data.itertuples(index=False), desc=f"Training Epoch {epoch+1}/{self.n_epochs}"):
                user_idx = self.user_map[row.msno]
                song_idx = self.song_map[row.song_id]
                rating = row.target
                
                prediction = self.predict_single(user_idx, song_idx)
                error = rating - prediction
                
                # Update user factors
                user_update = self.learning_rate * (error * self.song_factors[song_idx, :] - self.reg * self.user_factors[user_idx, :])
                self.user_factors[user_idx, :] += user_update
                
                # Update song factors
                song_update = self.learning_rate * (error * self.user_factors[user_idx, :] - self.reg * self.song_factors[song_idx, :])
                self.song_factors[song_idx, :] += song_update
                
    def predict_single(self, user_idx, song_idx):
        return np.dot(self.user_factors[user_idx, :], self.song_factors[song_idx, :])
    
    def predict(self, test_data):
        predictions = []
        for row in tqdm(test_data.itertuples(index=False), desc="Predicting"):
            user_idx = self.user_map.get(row.msno)
            song_idx = self.song_map.get(row.song_id)
            if user_idx is not None and song_idx is not None:
                predictions.append(self.predict_single(user_idx, song_idx))
            else:
                predictions.append(self.default_value)  # default value for missing user or song
        return np.array(predictions)

# Define file paths
# train_file_path = 'dataset/train.csv'
# test_file_path = 'dataset/test.csv'
# predictions_file_path = f'output/predictions_LFM_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.csv'

# Define file paths for small dataset
train_file_path = 'dataset/train_small.csv'
test_file_path = 'dataset/test_small.csv'
predictions_file_path = f'output/predictions_small_LFM_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.csv'

# Load the training data
train_data = pd.read_csv(train_file_path)

# Create a user-song matrix
user_song_matrix = train_data.pivot(index='msno', columns='song_id', values='target').fillna(0.5)

# Apply SVD to get initial factors
svd = TruncatedSVD(n_components=50, random_state=42)
user_factors_init = svd.fit_transform(user_song_matrix)
song_factors_init = svd.components_.T

# Initialize and train the model with SVD initial factors
model = LatentFactorModel(n_factors=50, learning_rate=0.01, n_epochs=100, reg=0.01)
model.fit(train_data, user_factors_init=user_factors_init, song_factors_init=song_factors_init)

# Load the test data
test_data = pd.read_csv(test_file_path)

# Predict the target values
predictions = model.predict(test_data)

# Save the predictions
test_data['target'] = predictions
test_data.to_csv(predictions_file_path, index=False)