python ../utils/preprocessing.py \
    --discharge_path /home/haotian/make-discharge-me/data/version1.4_ours/valid/discharge.csv.gz \
    --discharge_target_path /home/haotian/make-discharge-me/data/version1.4_ours/valid/discharge_target.csv.gz \
    --output_path /pure-mlo-scratch/make_project/spring2024/data/raw/version11.0-6k/DI_valid_dataset.jsonl  \
    --max_tokens 6000 \
    --mode DI \
    --prompt_path /home/haotian/make-discharge-me/src/prompts/di_test_prompt.json \
    --truncation_strategy samples \
    --features_to_consider sex,service,chief_complaint,history_of_present_illness,physical_exam,discharge_medications,discharge_diagnosis,discharge_disposition,discharge_condition,medication_on_admission
