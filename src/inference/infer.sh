# Models: Meditron 7B, Meditron 13B
if [ "$1" == "meditron-7b" ] || [ "$1" == "all" ]; then
    python3 src/inference/infer.py \
        --model_name meditron-7b \
        --model_path /pure-mlo-scratch/trial-runs/meditron-7b/hf_checkpoints/raw/iter_14500/ \
        --input_path data/preprocessed/test/bhc_short.csv.gz \
        --prompt /home/Paul/make-discharge-me/src/prompts/bhc_test_pompt.json \
        --output_path /home/Paul/make-discharge-me/data/infered/bhc_short_infered.csv.gz \
        --num_samples 100 \
        --verbose 1
fi     

