import pickle
from pathlib import Path

def load_pkl(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)
    
def save_pkl(obj, file_path):
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'wb') as f:
        pickle.dump(obj, f)