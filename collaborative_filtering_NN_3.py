import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from tqdm import tqdm
import datetime
from scipy.sparse import csr_matrix

# Define file paths
train_file_path = 'dataset/train.csv'
test_file_path = 'dataset/test.csv'
output_file_path = 'output/predictions'

# Define file paths for small dataset
# train_file_path = 'dataset/train_small.csv'
# test_file_path = 'dataset/test_small.csv'
# output_file_path = 'output/predictions_small'

# Load the train and test datasets
print("Loading train and test datasets...")
train = pd.read_csv(train_file_path)
test = pd.read_csv(test_file_path)
print("Datasets loaded.")

# Optimize data types
print("Optimizing data types...")
train['target'] = train['target'].astype(np.float32)
print("Data types optimized.")

"""
# Create a user-item matrix
print("Creating user-item matrix...")
user_item_matrix = train.pivot(index='msno', columns='song_id', values='target').fillna(0.5).astype(np.float32)
print("User-item matrix created.")

# Convert to sparse matrix
print("Converting to sparse matrix...")
user_item_sparse_matrix = csr_matrix(user_item_matrix.values)
print("Sparse matrix created.")
"""

# Create a user-item matrix using sparse matrix
print("Creating user-item matrix...")
user_map = {user_id: idx for idx, user_id in enumerate(train['msno'].unique())}
song_map = {song_id: idx for idx, song_id in enumerate(train['song_id'].unique())}

rows = train['msno'].map(user_map)
cols = train['song_id'].map(song_map)
data = train['target']

user_item_sparse_matrix = csr_matrix((data, (rows, cols)), shape=(len(user_map), len(song_map)))
print("User-item matrix created.")

# Apply SVD to reduce the dimensionality
print("Applying SVD...")
svd = TruncatedSVD(n_components=50, random_state=42)
reduced_matrix = svd.fit_transform(user_item_sparse_matrix)
print("SVD applied.")

# Compute user similarity
print("Computing user similarity...")
user_similarity = cosine_similarity(reduced_matrix)
print("User similarity computed.")

# Find top 100 neighbors for each user
print("Finding top 100 neighbors for each user...")
top_k = 100
neighbors = np.argpartition(-user_similarity, top_k, axis=1)[:, :top_k]
print("Top 100 neighbors found.")

""" 
# Estimate user preferences for the test set
predictions = np.zeros(len(test))
for i in tqdm(range(len(user_indices)), desc="Estimating preferences"):
    user_idx = user_indices.iloc[i]
    song_idx = song_indices.iloc[i]
    neighbor_idx = neighbors[user_idx]
    neighbor_sim = user_similarity[user_idx, neighbor_idx]
    neighbor_pref = user_item_matrix.values[neighbor_idx, song_idx]

    numerator = np.sum(neighbor_sim * neighbor_pref)
    denominator = np.sum(neighbor_sim)
    predictions[i] = numerator / denominator if denominator != 0 else 0.5
"""

"""
# Estimate user preferences for the test set using vectorized operations
print("Estimating user preferences for the test set...")
user_indices = test['msno'].apply(lambda x: user_item_matrix.index.get_loc(x) if x in user_item_matrix.index else -1)
song_indices = test['song_id'].apply(lambda x: user_item_matrix.columns.get_loc(x) if x in user_item_matrix.columns else -1)

valid_indices = (user_indices != -1) & (song_indices != -1)
user_indices = user_indices[valid_indices]
song_indices = song_indices[valid_indices]

neighbor_indices = neighbors[user_indices]
neighbor_similarities = user_similarity[user_indices, :][:, neighbor_indices]
neighbor_preferences = user_item_matrix.values[neighbor_indices, :][:, song_indices]

numerators = np.sum(neighbor_similarities * neighbor_preferences, axis=1)
denominators = np.sum(neighbor_similarities, axis=1)
predictions = np.where(denominators != 0, numerators / denominators, 0.5)
"""

# Estimate user preferences for the test set using batch processing
print("Estimating user preferences for the test set...")
batch_size = 1000
predictions = np.full(len(test), 0.5)  # Initialize with default value 0.5

# user_indices = test['msno'].apply(lambda x: user_item_matrix.index.get_loc(x) if x in user_item_matrix.index else -1)
# song_indices = test['song_id'].apply(lambda x: user_item_matrix.columns.get_loc(x) if x in user_item_matrix.columns else -1)
# Fill missing user and song indices with -1 to indicate invalid entries
user_indices = test['msno'].map(user_map).fillna(-1).astype(int)
song_indices = test['song_id'].map(song_map).fillna(-1).astype(int)

valid_indices = (user_indices != -1) & (song_indices != -1)
user_indices = user_indices[valid_indices]
song_indices = song_indices[valid_indices]

for start in tqdm(range(0, len(user_indices), batch_size), desc="Estimating preferences"):
    end = min(start + batch_size, len(user_indices))
    batch_user_indices = user_indices.iloc[start:end]
    batch_song_indices = song_indices.iloc[start:end]

    for i in range(len(batch_user_indices)):
        user_idx = batch_user_indices.iloc[i]
        song_idx = batch_song_indices.iloc[i]
        neighbor_idx = neighbors[user_idx]
        neighbor_sim = user_similarity[user_idx, neighbor_idx]
        neighbor_pref = user_item_sparse_matrix[neighbor_idx, song_idx].toarray().flatten()

        # Apply higher weights to 0 and 1 values
        weights = np.where(neighbor_pref == 0.5, 1, 10)
        weighted_neighbor_pref = neighbor_pref * weights

        numerator = np.sum(neighbor_sim * weighted_neighbor_pref)
        denominator = np.sum(neighbor_sim * weights)
        predictions[valid_indices.index[start + i]] = numerator / denominator if denominator != 0 else 0.5

test['target'] = predictions
print("User preferences estimated.")

# Save the results with the current timestamp in the filename
print("Saving the results...")
current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
output_file_path_with_time = f"{output_file_path}_{current_time}.csv"
test.to_csv(output_file_path_with_time, index=False)
print(f"Results saved to {output_file_path_with_time}.")