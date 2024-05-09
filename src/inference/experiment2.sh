python ../utils/preprocessing.py \
    --discharge_path /home/haotian/make-discharge-me/data/version1.4_ours/test_phase_2/discharge.csv.gz \
    --discharge_target_path /home/haotian/make-discharge-me/data/version1.4_ours/test_phase_2/discharge_target.csv.gz \
    --output_path /home/haotian/make-discharge-me/data/preprocessed_test/DI_test_dataset_sub_3.jsonl  \
    --max_tokens 3200 \
    --mode DI \
    --prompt_path /home/haotian/make-discharge-me/src/prompts/di_test_prompt.json \
    --truncation_strategy sections \
    --features_to_consider sex,service,chief_complaint,history_of_present_illness,physical_exam,discharge_medications,discharge_diagnosis,discharge_disposition,discharge_condition,medication_on_admission
