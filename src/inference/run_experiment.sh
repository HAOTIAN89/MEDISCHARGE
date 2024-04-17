$experiment_name = $1

python3 src/utils/preprocessing.py \
    --discharge_path data/test_phase_1/discharge.csv.gz \
    --discharge_target_path data/test_phase_1/discharge_target.csv.gz \
    --output_path data/experiments/{experiment_name}/input/BHC.csv.gz \
    --max_tokens 1545 \
    --mode BHC \
    --prompt_path src/utils/prompts/bhc_test_prompt.json \
    --features_to_exclude physical_exam

python3 infer.py \
    --model_name medischarge-7b-BHC \
    --model_path /pure-mlo-scratch/make_project/spring2024/trial-runs/meditron-7B-BHC-v2/hf_checkpoint_1000 \
    --input_path data/experiments/{experiment_name}/input/BHC.csv.gz \
    --output_path data/experiments/{experiment_name}/output/BHC.csv \
    --idx_col 'idx' \
    --verbose 0
    
python3 src/utils/preprocessing.py \
    --discharge_path data/test_phase_1/discharge.csv.gz \
    --discharge_target_path data/test_phase_1/discharge_target.csv.gz \
    --output_path data/experiments/{experiment_name}/input/DI.csv.gz \
    --max_tokens 1700 \
    --mode DI \
    --prompt_path src/utils/prompts/di_test_prompt.json \
    --generated_bhc_path data/experiments/{experiment_name}/output/BHC.csv \
    --features_to_exclude physical_exam

python3 infer.py \
    --model_name medischarge-7b-DI \
    --model_path /pure-mlo-scratch/make_project/spring2024/trial-runs/meditron-7B-DI-v2/hf_checkpoint_1500 \
    --input_path data/experiments/{experiment_name}/input/DI.csv.gz \
    --output_path data/experiments/{experiment_name}/output/DI.csv \
    --idx_col 'idx' \
    --verbose 0
