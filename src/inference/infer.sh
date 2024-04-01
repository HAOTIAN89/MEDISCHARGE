# Models: Meditron 7B, Meditron 13B
if [ "$1" == "meditron-7b" ] || [ "$1" == "all" ]; then
    python3 src/inference/infer.py \
        --model_name meditron-7b \
        --model_path /pure-mlo-scratch/trial-runs/meditron-7b/hf_checkpoints/raw/iter_14500/ \
        --input_path data/test_phase_1/BHC_test_dataset.jsonl \
        --output_path data/infered/bhc_short_infered.csv \
        --num_samples 10 \
        --verbose 1
fi


if [ "$1" == "medischarge-7b-bhc" ] || [ "$1" == "all" ]; then
    python3 src/inference/infer.py \
        --model_name medischarge-7b-BHC \
        --model_path /pure-mlo-scratch/make_project/spring2024/trial-runs/meditron-7B-BHC/hf_checkpoint \
        --input_path data/test_phase_1/BHC_test_dataset.jsonl \
        --output_path data/infered/bhc_short_7b_infered.csv \
        --verbose 0
fi     

if [ "$1" == "medischarge-7b-di" ] || [ "$1" == "all" ]; then
    python3 src/inference/infer.py \
        --model_name medischarge-7b-DI \
        --model_path /pure-mlo-scratch/make_project/spring2024/trial-runs/meditron-7B-DI/hf_checkpoint \
        --input_path data/test_phase_1/BHC_test_dataset.jsonl \
        --output_path data/infered/di_short_7b_infered.csv \
        --verbose 0
fi 