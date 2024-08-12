test_discharge=data/version1.4_ours/test_phase_2/discharge.csv.gz
test_target=data/version1.4_ours/test_phase_2/discharge_target.csv.gz
train_discharge=data/version1.4_ours/train/discharge.csv.gz
train_target=data/version1.4_ours/train/discharge_target.csv.gz
modes="bhc,di"
nb_samples=250
n_shots="1,2,3"
output_folder_path=data/openai_inputs
prompt_folder_path=src/prompts

python3 openai_construct.py \
    --Modes "$modes" \
    --test_discharge_dataset $test_discharge \
    --test_target_dataset $test_target \
    --train_discharge_dataset $train_discharge \
    --train_target_dataset $train_target \
    --nb_samples $nb_samples \
    --n_shots_list "$n_shots" \
    --prompt_folder_path $prompt_folder_path \
    --output_folder_path $output_folder_path