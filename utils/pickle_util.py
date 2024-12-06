import pickle

def load_pkl(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)
    
def save_pkl(obj, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(obj, f)