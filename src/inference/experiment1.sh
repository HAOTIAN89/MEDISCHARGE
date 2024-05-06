python ../utils/preprocessing.py \
    --discharge_path /home/haotian/make-discharge-me/data/test_phase_2/discharge.csv.gz \
    --discharge_target_path /home/haotian/make-discharge-me/data/test_phase_2/discharge_target.csv.gz \
    --output_path  /home/haotian/make-discharge-me/data/preprocessed_test/BHC_test_dataset_sub_1.jsonl \
    --max_tokens 5200 \
    --mode BHC \
    --prompt_path /home/haotian/make-discharge-me/src/prompts/bhc_test_prompt.json \
    --truncation_strategy sections
    

# 4k bhc train 32188/39447
# 4k bhc valid 6887/8445
# 4k di train 39056/39447
# 4k di valid 8342/8445