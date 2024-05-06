python ../utils/preprocessing.py \
    --discharge_path /home/haotian/make-discharge-me/data/test_phase_2/discharge.csv.gz \
    --discharge_target_path /home/haotian/make-discharge-me/data/test_phase_2/discharge_target.csv.gz \
    --output_path /home/haotian/make-discharge-me/data/preprocessed_test/DI_test_dataset_sub_1.jsonl \
    --max_tokens 5200 \
    --generated_bhc_path /home/haotian/make-discharge-me/data/infered/bhc_7b_infered_1.csv \
    --mode DI \
    --prompt_path /home/haotian/make-discharge-me/src/prompts/di_test_prompt.json \
    --truncation_strategy sections