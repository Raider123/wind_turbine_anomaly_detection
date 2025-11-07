# Imports
import pandas as pd


def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df

def load_train_test(train_path: str, test_path: str):
    return (load_csv(train_path), load_csv(test_path))
