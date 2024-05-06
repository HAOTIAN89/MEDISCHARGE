# !bin/sh

ROOT=/
MAKEPATH=$ROOT"pure-mlo-scratch/make_project/spring2024"
HOMEPATH=$ROOT"home/faure/make"
MODEL="llama3-8b"

python3 $HOMEPATH"/src/training/llama3.py" \
    --train_path $MAKEPATH"/data/raw/version6.0/BHC_train_dataset_v6_6k_filtered.jsonl" \
    --eval_path $MAKEPATH"/data/raw/version6.0/BHC_valid_dataset_v6_6k_filtered.jsonl" \
    --model_path $MAKEPATH"/trial-runs/"$MODEL \
    --output_dir $MAKEPATH"/trial-runs/"$MODEL