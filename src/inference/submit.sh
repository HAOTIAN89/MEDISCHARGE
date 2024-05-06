if [ "$1" == "medischarge-7b-di" ] || [ "$1" == "all" ]; then
    python3 infer.py \
        --model_name medischarge-7b-DI \
        --model_path /pure-mlo-scratch/make_project/spring2024/trial-runs/meditron-7B-DI-v2/hf_checkpoint_1500 \
        --input_path ../../data/test_phase_1/DI_test_dataset_sub_10.jsonl \
        --output_path ../../data/infered/di_7b_infered_10.csv \
        --idx_col 'hadm_id' \
        --verbose 0
fi    

if [ "$1" == "medischarge-7b-bhc" ] || [ "$1" == "all" ]; then
    python3 infer.py \
        --model_name medischarge-7b-BHC \
        --model_path /pure-mlo-scratch/make_project/spring2024/trial-runs/meditron-7B-BHC-v2/hf_checkpoint_1000 \
        --input_path ../../data/test_phase_1/BHC_test_dataset_sub_10.jsonl \
        --output_path ../../data/infered/bhc_7b_infered_1.csv \
        --idx_col 'hadm_id' \
        --verbose 0
fi   

if [ "$1" == "medischarge-7B-BHC-v6-6k" ] || [ "$1" == "all" ]; then
    python3 infer.py \
        --model_name medischarge-7B-BHC-v6-6k \
        --model_path /pure-mlo-scratch/make_project/spring2024/trial-runs/meditron-7B-BHC-v6-6k/hf_checkpoint \
        --input_path ../../data/preprocessed_test/BHC_test_dataset_sub_1.jsonl \
        --output_path ../../data/infered/bhc_7b_infered_1.csv \
        --idx_col 'hadm_id' \
        --verbose 0
fi

if [ "$1" == "medischarge-7B-DI-v6-6k" ] || [ "$1" == "all" ]; then
    python3 infer.py \
        --model_name medischarge-7B-DI-v6-6k \
        --model_path /pure-mlo-scratch/make_project/spring2024/trial-runs/meditron-7B-DI-v6-6k/hf_checkpoint \
        --input_path ../../data/preprocessed_test/DI_test_dataset_sub_1.jsonl \
        --output_path ../../data/infered/di_7b_infered_1.csv \
        --idx_col 'hadm_id' \
        --verbose 0
fi