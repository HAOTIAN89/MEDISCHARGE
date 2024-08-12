python ../utils/preprocessing.py \
    --discharge_path /home/haotian/make-discharge-me/data/version1.4_ours/valid/discharge.csv.gz \
    --discharge_target_path /home/haotian/make-discharge-me/data/version1.4_ours/valid/discharge_target.csv.gz \
    --output_path /pure-mlo-scratch/make_project/spring2024/data/raw/version11.0-6k/BHC_valid_dataset.jsonl  \
    --max_tokens 6000 \
    --mode BHC \
    --prompt_path /home/haotian/make-discharge-me/src/prompts/bhc_test_prompt.json \
    --truncation_strategy samples \
    --features_to_consider sex,service,chief_complaint,history_of_present_illness,pertinent_results,physical_exam,major_surgical_procedures,past_medical_history,allergies,social_history,family_history



# In the filtered dataset
# 4k bhc train 32510/39447
# 4k bhc valid 6961/8445
# 4k di train 39279/39447
# 4k di valid 8409/8445

# In the unfiltered dataset
# 4k bhc train 51687/68785
# 4k bhc valid 11056/14719
# 4k di train 68424/68785
# 4k di valid 14649/14719

# BHC_test_dataset_sub_3.jsonl output_max_length: 800
# DI_test_dataset_sub_3.jsonl output_max_length: 800

# BHC_test_dataset_sub_4.jsonl output_max_length: 1200
# DI_test_dataset_sub_4.jsonl output_max_length: 600

# BHC_test_dataset_sub_5.jsonl output_max_length: 1200
# DI_test_dataset_sub_5.jsonl output_max_length: 600


# in the version10.0-2k
# BHC_train_dataset.jsonl  13433/68785
# BHC_valid_dataset.jsonl  2947/14719
# DI_train_dataset.jsonl   33035/68785
# DI_valid_dataset.jsonl   7001/14719

# in the version11.0-6k
# BHC_train_dataset.jsonl  65842/68785
# BHC_valid_dataset.jsonl  14064/14719
# DI_train_dataset.jsonl   68776/68785
# DI_train_dataset.jsonl   14717/14719