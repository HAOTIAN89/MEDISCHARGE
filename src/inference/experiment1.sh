python ../utils/preprocessing.py \
    --discharge_path /home/haotian/make-discharge-me/data/version1.4_ours/test_phase_2/discharge.csv.gz \
    --discharge_target_path /home/haotian/make-discharge-me/data/version1.4_ours/test_phase_2/discharge_target.csv.gz \
    --output_path /home/haotian/make-discharge-me/data/preprocessed_test/BHC_test_dataset_sub_3.jsonl  \
    --max_tokens 3200 \
    --mode BHC \
    --prompt_path /home/haotian/make-discharge-me/src/prompts/bhc_test_prompt.json \
    --truncation_strategy sections \
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