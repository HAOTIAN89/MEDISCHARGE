# Pertinent results
if [ "$1" == "medischarge-7b-bhc" ] || [ "$1" == "all" ]; then
    python3 src/inference/infer_without_section.py \
        --model_name medischarge-7b-BHC \
        --model_path /pure-mlo-scratch/make_project/spring2024/trial-runs/meditron-7B-BHC/hf_checkpoint \
        --input_path data/test_phase_1/BHC_test_dataset_v2.jsonl \
        --output_path data/infered/bhc_short_7b_test_infered_wo_pertinent_results.csv \
        --verbose 0 \
        --section_name "pertinent results"
fi     

if [ "$1" == "medischarge-7b-di" ] || [ "$1" == "all" ]; then
    python3 src/inference/infer_without_section.py \
        --model_name medischarge-7b-DI \
        --model_path /pure-mlo-scratch/make_project/spring2024/trial-runs/meditron-7B-DI/hf_checkpoint \
        --input_path data/test_phase_1/DI_test_dataset_v2.jsonl \
        --output_path data/infered/di_short_7b_test_infered_wo_pertinent_results.csv \
        --verbose 0 \
        --section_name "pertinent results"
fi 

# Physical exam
if [ "$1" == "medischarge-7b-bhc" ] || [ "$1" == "all" ]; then
    python3 src/inference/infer_without_section.py \
        --model_name medischarge-7b-BHC \
        --model_path /pure-mlo-scratch/make_project/spring2024/trial-runs/meditron-7B-BHC/hf_checkpoint \
        --input_path data/test_phase_1/BHC_test_dataset_v2.jsonl \
        --output_path data/infered/bhc_short_7b_test_infered_wo_physical_exam.csv \
        --verbose 0 \
        --section_name "physical exam"
fi     

if [ "$1" == "medischarge-7b-di" ] || [ "$1" == "all" ]; then
    python3 src/inference/infer_without_section.py \
        --model_name medischarge-7b-DI \
        --model_path /pure-mlo-scratch/make_project/spring2024/trial-runs/meditron-7B-DI/hf_checkpoint \
        --input_path data/test_phase_1/DI_test_dataset_v2.jsonl \
        --output_path data/infered/di_short_7b_test_infered_wo_physical_exam.csv \
        --verbose 0 \
        --section_name "physical exam"
fi 

# Past medical history
if [ "$1" == "medischarge-7b-bhc" ] || [ "$1" == "all" ]; then
    python3 src/inference/infer_without_section.py \
        --model_name medischarge-7b-BHC \
        --model_path /pure-mlo-scratch/make_project/spring2024/trial-runs/meditron-7B-BHC/hf_checkpoint \
        --input_path data/test_phase_1/BHC_test_dataset_v2.jsonl \
        --output_path data/infered/bhc_short_7b_test_infered_wo_past_medical_history.csv \
        --verbose 0 \
        --section_name "past medical history"
fi     

if [ "$1" == "medischarge-7b-di" ] || [ "$1" == "all" ]; then
    python3 src/inference/infer_without_section.py \
        --model_name medischarge-7b-DI \
        --model_path /pure-mlo-scratch/make_project/spring2024/trial-runs/meditron-7B-DI/hf_checkpoint \
        --input_path data/test_phase_1/DI_test_dataset_v2.jsonl \
        --output_path data/infered/di_short_7b_test_infered_wo_past_medical_history.csv \
        --verbose 0 \
        --section_name "past medical history"
fi 

# History of present illness
if [ "$1" == "medischarge-7b-bhc" ] || [ "$1" == "all" ]; then
    python3 src/inference/infer_without_section.py \
        --model_name medischarge-7b-BHC \
        --model_path /pure-mlo-scratch/make_project/spring2024/trial-runs/meditron-7B-BHC/hf_checkpoint \
        --input_path data/test_phase_1/BHC_test_dataset_v2.jsonl \
        --output_path data/infered/bhc_short_7b_test_infered_wo_present_illness.csv \
        --verbose 0 \
        --section_name "history of present illness"
fi     

if [ "$1" == "medischarge-7b-di" ] || [ "$1" == "all" ]; then
    python3 src/inference/infer_without_section.py \
        --model_name medischarge-7b-DI \
        --model_path /pure-mlo-scratch/make_project/spring2024/trial-runs/meditron-7B-DI/hf_checkpoint \
        --input_path data/test_phase_1/DI_test_dataset_v2.jsonl \
        --output_path data/infered/di_short_7b_test_infered_wo_present_illness.csv \
        --verbose 0 \
        --section_name "history of present illness"
fi 

