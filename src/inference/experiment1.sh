python ../utils/preprocessing.py \
    --discharge_path /home/haotian/make-discharge-me/data/version1.4_filtered/train/discharge.csv.gz \
    --discharge_target_path /home/haotian/make-discharge-me/data/version1.4_filtered/train/discharge_target.csv.gz \
    --output_path /pure-mlo-scratch/make_project/spring2024/data/raw/version6.0/DI_train_dataset_v6_6k.jsonl \
    --max_tokens 6000 \
    --mode DI \
    --features_to_exclude “”\
    --prompt_path /home/haotian/make-discharge-me/src/prompts/di_test_prompt.json \
    --truncation_strategy samples
    # --generated_bhc_path "" \