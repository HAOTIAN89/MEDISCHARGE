import pandas as pd
import numpy as np
import torch.nn as nn
from rouge import RougeL, Rouge2

# Load the data
train_df = pd.read_csv('../../data/results/train.csv')
valid_df = pd.read_csv('../../data/results/valid.csv')

# Merge datasets for combined analysis
all_df = pd.concat([train_df, valid_df], ignore_index=True)

# Function to process a DataFrame
def process_df(df, rouge_model):
    columns = df.columns.tolist()
    reference_columns = ["discharge_instructions", "brief_hospital_course"]
    compare_columns = [c for c in columns if c not in reference_columns]
    # reomve the hadm_id column from compare_columns
    compare_columns.remove('hadm_id')
    results = {}
    for ref_col in reference_columns:
        for comp_col in compare_columns:
            ref_texts = df[ref_col].fillna("").astype(str)
            comp_texts = df[comp_col].fillna("").astype(str)
            scores = rouge_model.forward(ref_texts, comp_texts)
            # compute the average of the scores for one ref_col and comp_col pair
            avg_score = np.mean(list(scores.values()))
            results[f"{ref_col[:3]}_{comp_col}_avg"] = avg_score
            print(f"Processed {ref_col[:3]}_{comp_col}")
    return results

# Instantiate the Rouge model, options are RougeL and Rouge2
rouge_model = Rouge2()

# Compute scores
print("Computing ROUGE scores...")
# train_rouge = process_df(train_df, rouge_model)
# print("Processed train data")
valid_rouge = process_df(valid_df, rouge_model)
print("Processed valid data")
# all_rouge = process_df(all_df, rouge_model)
# print("Processed all data")

# Save the results
result_df = pd.DataFrame({
    # "train_rouge": pd.Series(train_rouge),
    "valid_rouge": pd.Series(valid_rouge)
    # "all_rouge": pd.Series(all_rouge)
})

result_df.to_csv('../../data/results/rouge2_scores.csv', index=True)

print("ROUGE scores saved to 'rouge_scores.csv'")