from transformers import AutoTokenizer
import pandas as pd
from preprocessing import load_data, build_combined_discharge, remove_bhc_di
import matplotlib.pyplot as plt
import seaborn as sns

model = "meta-llama/Llama-2-13b-hf"
tokenizer = AutoTokenizer.from_pretrained(model, token = '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')

discharge_path_train = "/home/haotian/make-discharge-me/data/version1.4_ours/train/discharge.csv.gz"
discharge_target_path_train = "/home/haotian/make-discharge-me/data/version1.4_ours/train/discharge_target.csv.gz"
discharge_path_valid = "/home/haotian/make-discharge-me/data/version1.4_ours/valid/discharge.csv.gz"
discharge_target_path_valid = "/home/haotian/make-discharge-me/data/version1.4_ours/valid/discharge_target.csv.gz"
discharge_path_test = "/home/haotian/make-discharge-me/data/version1.4_ours/test_phase_2/discharge.csv.gz"
discharge_target_path_test = "/home/haotian/make-discharge-me/data/version1.4_ours/test_phase_2/discharge_target.csv.gz"


discharges_df_train = load_data(discharge_path_train)
discharges_target_df_train = load_data(discharge_target_path_train)
combined_discharges_train = build_combined_discharge(discharges_df_train, discharges_target_df_train)
# print the length of the combined_discharges_train
print("Length of combined_discharges_train: ", len(combined_discharges_train))

discharges_df_valid = load_data(discharge_path_valid)
discharges_target_df_valid = load_data(discharge_target_path_valid)
combined_discharges_valid = build_combined_discharge(discharges_df_valid, discharges_target_df_valid)
print("Length of combined_discharges_valid: ", len(combined_discharges_valid))

discharges_df_test = load_data(discharge_path_test)
discharges_target_df_test = load_data(discharge_target_path_test)
combined_discharges_test = build_combined_discharge(discharges_df_test, discharges_target_df_test)
print("Length of combined_discharges_test: ", len(combined_discharges_test))

# Concatenate the datasets
print("Concatenating datasets...")
combined_discharges = pd.concat([combined_discharges_train, combined_discharges_valid, combined_discharges_test], ignore_index=True)

# Tokenize and calculate token counts
def tokenize_and_count(text):
    return len(tokenizer.tokenize(text))

# Apply tokenization and count tokens
combined_discharges['text_token_count'] = combined_discharges['text'].progress_apply(tokenize_and_count)
combined_discharges['discharge_instructions_token_count'] = combined_discharges['discharge_instructions'].progress_apply(tokenize_and_count)
combined_discharges['brief_hospital_course_token_count'] = combined_discharges['brief_hospital_course'].progress_apply(tokenize_and_count)

# Calculate final token number
combined_discharges['final_token_count'] = combined_discharges['text_token_count'] - (
    combined_discharges['discharge_instructions_token_count'] + combined_discharges['brief_hospital_course_token_count']
)

# Plot the distribution of final token numbers
sns.set(style="whitegrid")
plt.figure(figsize=(12, 6))
sns.histplot(combined_discharges['final_token_count'], kde=True, color="blue", binwidth=100)
plt.title('Distribution of Full Text Input Token Counts')
plt.xlabel('Token Count')
plt.ylabel('Frequency')

# Save the histogram
plt.savefig('/home/haotian/make-discharge-me/data/results/input_token_count_histogram.png')

# train set 68785
# valid set 14719
# test set 10962


