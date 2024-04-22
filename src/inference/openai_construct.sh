test_discharge=data/test_phase_1/discharge.csv.gz
test_target=data/test_phase_1/discharge_target.csv.gz
train_discharge=data/train/discharge.csv.gz
train_target=data/train/discharge_target.csv.gz
modes="bhc,di"
nb_samples=1000
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