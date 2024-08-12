'''
Data loading and saving utilities.
'''

import os
import pandas as pd

def load_file(path): 
    '''
    Given a .csv or .json or .jsonl or .txt file,
    load it into a dataframe or string.
    '''
    if not os.path.exists(path):
        raise ValueError(f"Path {path} does not exist.")
    if '.csv' in path:
        data = pd.read_csv(path)
    elif '.jsonl' in path:
        data = pd.read_json(path, lines=True)
    elif '.json' in path:
        data = pd.read_json(path)
    elif '.txt' in path:
        with open(path, 'r') as f:
            data = f.read()
    else: 
        raise ValueError(f"Provided path {path} is not a valid file.")
    return data

def save_file(df, path, mode='w'):
    '''
    Given a dataframe, save it to a .csv or .json or .jsonl file.
    '''
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if '.csv' in path:
        df.to_csv(path, index=False, mode=mode)
    elif '.jsonl' in path:
        df.to_json(path, orient='records', lines=True, mode=mode)
    elif '.json' in path:
        df.to_json(path, orient='records', mode=mode)
    else: 
        raise ValueError(f"Provided path {path} is not a .csv, .json or .jsonl file.")
    return df

