python ../utils/preprocessing.py \
    --discharge_path /home/haotian/make-discharge-me/data/test_phase_2/discharge.csv.gz \
    --discharge_target_path /home/haotian/make-discharge-me/data/test_phase_2/discharge_target.csv.gz \
    --output_path /home/haotian/make-discharge-me/data/DI_test_dataset_sub_2.jsonl \
    --max_tokens 5200 \
    --mode DI \
    --prompt_path /home/haotian/make-discharge-me/src/prompts/di_test_prompt.json \
    --truncation_strategy sections \
    --use_generated_bhc "False" \
    # --generated_bhc_path /home/haotian/make-discharge-me/data/infered/bhc_7b_infered_1.csv \

    # /pure-mlo-scratch/make_project/spring2024/data/raw/version6.0/DI_valid_dataset_v6_6k_without_BHC.jsonl