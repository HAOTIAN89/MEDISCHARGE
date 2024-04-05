import json
from tqdm import tqdm

def dataframe_to_jsonl(dataframe, attributes, keys, file_path):
    """
    Convert a DataFrame to a JSONL file with specified keys replacing the attributes.

    Parameters:
    dataframe (pd.DataFrame): The DataFrame to convert.
    attributes (list): The list of attributes from the DataFrame to include in the JSONL file.
    keys (list): The list of keys to replace the attributes in the JSONL file.
    file_path (str): The path to save the JSONL file.
    """
    if len(attributes) != len(keys):
        raise ValueError("The length of attributes and keys must be the same.")
    
    with open(file_path, 'w') as file:
        for idx, row in tqdm(dataframe.iterrows()):
            row_dict = {"idx": idx + 1}
            for attribute, key in zip(attributes, keys):
                if attribute in row:
                    row_dict[key] = row[attribute]
                else:
                    print(f"Warning: {attribute} is not a column in the DataFrame.")
                    return
            file.write(f"{json.dumps(row_dict)}\n")
            
