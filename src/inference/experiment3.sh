python ../utils/preprocessing.py \
    --discharge_path /home/haotian/make-discharge-me/data/version1.4_ours/valid/discharge.csv.gz \
    --discharge_target_path /home/haotian/make-discharge-me/data/version1.4_ours/valid/discharge_target.csv.gz \
    --output_path /home/haotian/make-discharge-me/data/preprocessed_test/test.csv \
    --mode DI \
    --prompt_path /home/haotian/make-discharge-me/src/prompts/di_test_prompt.json \
    --truncation_strategy rouge \
    --features_to_consider sex,service,allergies,chief_complaint,major_surgical_procedures,history_of_present_illness,past_medical_history,social_history,family_history,physical_exam,pertinent_results,discharge_medications,discharge_diagnosis,discharge_disposition,discharge_condition,medication_on_admission,facility
