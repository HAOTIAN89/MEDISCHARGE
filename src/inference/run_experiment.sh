#!/bin/sh

# EXPERIMENT_NAME="$1"
echo "Starting Experiments Progress..."


echo "Starting BHC v2 cpk 1000 2k"

# BHC v2 cpk 1000
python3 src/utils/preprocessing.py \
    --discharge_path data/v1.3/test_phase_1/discharge.csv.gz \
    --discharge_target_path data/v1.3/test_phase_1/discharge_target.csv.gz \
    --output_path data/experiments/7B_bhc_v2_cpk_1000/input/BHC.jsonl \
    --max_tokens 1548 \
    --mode BHC \
    --prompt_path src/prompts/bhc_test_prompt.json

python3 src/inference/infer.py \
    --model_name medischarge-7b-BHC \
    --model_path /pure-mlo-scratch/make_project/spring2024/trial-runs/meditron-7B-BHC-v2/hf_checkpoint_1000 \
    --input_path data/experiments/7B_bhc_v2_cpk_1000/input/BHC.jsonl \
    --output_path data/experiments/7B_bhc_v2_cpk_1000/output/BHC.csv \
    --idx_col 'idx' \
    --verbose 0

python3 src/scoring/scoring_task.py \
    --input_dir data/experiments/7B_bhc_v2_cpk_1000/output/BHC.csv \
    --score_dir data/experiments/7B_bhc_v2_cpk_1000/score/BHC \
    --metrics bleu rouge meteor bertscore

echo "Starting DI v2 cpk 1500 2k"

# DI v2 cpk 1500
python3 src/utils/preprocessing.py \
    --discharge_path data/v1.3/test_phase_1/discharge.csv.gz \
    --discharge_target_path data/v1.3/test_phase_1/discharge_target.csv.gz \
    --output_path data/experiments/7B_di_v2_cpk_1500/input/DI.jsonl \
    --max_tokens 1700 \
    --mode DI \
    --prompt_path src/prompts/di_test_prompt.json \
    --generated_bhc_path data/experiments/7B_di_v2_cpk_1500/output/BHC.csv \

python3 src/inference/infer.py \
    --model_name medischarge-7b-DI \
    --model_path /pure-mlo-scratch/make_project/spring2024/trial-runs/meditron-7B-DI-v2/hf_checkpoint_1500 \
    --input_path data/experiments/7B_di_v2_cpk_1500/input/DI.jsonl \
    --output_path data/experiments/7B_di_v2_cpk_1500/output/DI.csv \
    --idx_col 'idx' \
    --verbose 0

python3 src/scoring/scoring_task.py \
    --input_dir data/experiments/7B_di_v2_cpk_1500/output/DI.csv \
    --score_dir data/experiments/7B_di_v2_cpk_1500/score/DI \
    --metrics bleu rouge meteor bertscore

echo "Starting BHC v3 cpk 1000 6k"

# BHC v4 cpk 1200 extended 6k
python3 src/utils/preprocessing.py \
    --discharge_path data/v1.3/test_phase_1/discharge.csv.gz \
    --discharge_target_path data/v1.3/test_phase_1/discharge_target.csv.gz \
    --output_path data/experiments/7B_bhc_v4_cpk_1000_extended_6k_input_6k/input/BHC.jsonl \
    --max_tokens 5000 \
    --mode BHC \
    --prompt_path src/prompts/bhc_test_prompt.json

python3 src/inference/infer.py \
    --model_name medischarge-7b-BHC-extended \
    --model_path /pure-mlo-scratch/make_project/spring2024/trial-runs/meditron-7B-BHC-v4-extended-6k \
    --input_path data/experiments/7B_bhc_v4_cpk_1000_extended_6k_input_6k/input/BHC.jsonl \
    --output_path data/experiments/7B_bhc_v4_cpk_1000_extended_6k_input_6k/output/BHC.csv \
    --idx_col 'idx' \
    --verbose 0

python3 src/scoring/scoring_task.py \
    --input_dir data/experiments/7B_bhc_v4_cpk_1000_extended_6k_input_6k/output/BHC.csv \
    --score_dir data/experiments/7B_bhc_v4_cpk_1000_extended_6k_input_6k/score/BHC \
    --metrics bleu rouge meteor bertscore

echo "Starting DI v4 cpk 2000 6k"

# DI v4 cpk 2000 extended 6k
python3 src/utils/preprocessing.py \
    --discharge_path data/v1.3/test_phase_1/discharge.csv.gz \
    --discharge_target_path data/v1.3/test_phase_1/discharge_target.csv.gz \
    --output_path data/experiments/7B_di_v4_cpk_2000_extended_6k_input_6k/input/DI.jsonl \
    --max_tokens 5000 \
    --mode DI \
    --prompt_path src/prompts/di_test_prompt.json \
    --generated_bhc_path data/experiments/7B_di_v4_cpk_2000_extended_6k_input_6k/output/BHC.csv \

python3 src/inference/infer.py \
    --model_name medischarge-7b-DI-extended \
    --model_path /pure-mlo-scratch/make_project/spring2024/trial-runs/meditron-7B-DI-v4 \
    --input_path data/experiments/7B_di_v4_cpk_2000_extended_6k_input_6k/input/DI.jsonl \
    --output_path data/experiments/7B_di_v4_cpk_2000_extended_6k_input_6k/output/DI.csv \
    --idx_col 'idx' \
    --verbose 0

python3 src/scoring/scoring_task.py \
    --input_dir data/experiments/7B_di_v4_cpk_2000_extended_6k_input_6k/output/DI.csv \
    --score_dir data/experiments/7B_di_v4_cpk_2000_extended_6k_input_6k/score/DI \
    --metrics bleu rouge meteor bertscore

# BHC v4 cpk 1200 extended 6k phase 2
python3 src/utils/preprocessing.py \
    --discharge_path data/v1.3/test_phase_2/discharge.csv.gz \
    --discharge_target_path data/v1.3/test_phase_2/discharge_target.csv.gz \
    --output_path data/experiments/7B_bhc_v4_cpk_1000_extended_6k_input_6k_phase_2/input/BHC.jsonl \
    --max_tokens 5000 \
    --mode BHC \
    --prompt_path src/prompts/bhc_test_prompt.json

python3 src/inference/infer.py \
    --model_name medischarge-7b-BHC-extended \
    --model_path /pure-mlo-scratch/make_project/spring2024/trial-runs/meditron-7B-BHC-v4-extended-6k \
    --input_path data/experiments/7B_bhc_v4_cpk_1000_extended_6k_input_6k_phase_2/input/BHC.jsonl \
    --output_path data/experiments/7B_bhc_v4_cpk_1000_extended_6k_input_6k_phase_2/output/BHC.csv \
    --idx_col 'idx' \
    --verbose 0

python3 src/scoring/scoring_task.py \
    --input_dir data/experiments/7B_bhc_v4_cpk_1000_extended_6k_input_6k_phase_2/output/BHC.csv \
    --score_dir data/experiments/7B_bhc_v4_cpk_1000_extended_6k_input_6k_phase_2/score/BHC \
    --metrics bleu rouge meteor bertscore

echo "Starting DI v4 cpk 2000 6k phase 2"

# DI v4 cpk 2000 extended 6k phase 2
python3 src/utils/preprocessing.py \
    --discharge_path data/v1.3/test_phase_2/discharge.csv.gz \
    --discharge_target_path data/v1.3/test_phase_2/discharge_target.csv.gz \
    --output_path data/experiments/7B_di_v4_cpk_2000_extended_6k_input_6k_phase_2/input/DI.jsonl \
    --max_tokens 5000 \
    --mode DI \
    --prompt_path src/prompts/di_test_prompt.json \
    --generated_bhc_path data/experiments/7B_di_v4_cpk_2000_extended_6k_input_6k_phase_2/output/BHC.csv \

python3 src/inference/infer.py \
    --model_name medischarge-7b-DI-extended \
    --model_path /pure-mlo-scratch/make_project/spring2024/trial-runs/meditron-7B-DI-v4 \
    --input_path data/experiments/7B_di_v4_cpk_2000_extended_6k_input_6k_phase_2/input/DI.jsonl \
    --output_path data/experiments/7B_di_v4_cpk_2000_extended_6k_input_6k_phase_2/output/DI.csv \
    --idx_col 'idx' \
    --verbose 0

python3 src/scoring/scoring_task.py \
    --input_dir data/experiments/7B_di_v4_cpk_2000_extended_6k_input_6k_phase_2/output/DI.csv \
    --score_dir data/experiments/7B_di_v4_cpk_2000_extended_6k_input_6k_phase_2/score/DI \
    --metrics bleu rouge meteor bertscore