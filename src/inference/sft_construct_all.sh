dataset_saved_folder_path=/home/haotian/make-discharge-me/data/train
discharge_dataset_path=/home/haotian/make-discharge-me/data/train/discharge.csv.gz
target_dataset_path=/home/haotian/make-discharge-me/data/train/discharge_target.csv.gz
max_length=2048

python3 sft_construct_all.py \
    --constructed_dataset_folder $dataset_saved_folder_path \
    --discharge_dataset $discharge_dataset_path \
    --target_dataset $target_dataset_path \
    --max_length $max_length 