import pandas as pd
from pathlib import Path

def load_csv(file_path):
    return pd.read_csv(file_path)

def save_csv(df, file_path):
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(file_path, index=False)