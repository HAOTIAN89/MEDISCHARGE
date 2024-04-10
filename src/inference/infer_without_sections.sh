## default
# v1

# python3 src/inference/infer.py \
#     --model_name medischarge-7b-BHC \
#     --model_path /pure-mlo-scratch/make_project/spring2024/trial-runs/meditron-7B-BHC/hf_checkpoint \
#     --input_path data/test_phase_1/BHC_test_dataset_v2.jsonl \
#     --output_path data/infered/bhc_short_7b_v1_test_infered_v2.csv \
#     --verbose 0

# python3 src/inference/infer.py \
#     --model_name medischarge-7b-DI \
#     --model_path /pure-mlo-scratch/make_project/spring2024/trial-runs/meditron-7B-DI/hf_checkpoint \
#     --input_path data/test_phase_1/DI_test_dataset_v2.jsonl \
#     --output_path data/infered/di_short_7b_v1_test_infered_v2.csv \
#     --verbose 0

# python3 src/inference/infer_without_section.py \
#         --model_name medischarge-7b-DI \
#         --model_path /pure-mlo-scratch/make_project/spring2024/trial-runs/meditron-7B-DI/hf_checkpoint \
#         --input_path data/test_phase_1/DI_test_dataset_v2.jsonl \
#         --output_path data/infered/di_short_7b_v1_test_infered_wo_present_illness_v2.csv \
#         --verbose 0 \
#         --section_name "history of present illness"
        
python3 src/inference/infer.py \
    --model_name medischarge-7b-BHC \
    --model_path /pure-mlo-scratch/make_project/spring2024/trial-runs/meditron-7B-BHC-v2/hf_checkpoint_500 \
    --input_path data/test_phase_1/BHC_test_dataset_v2.jsonl \
    --output_path data/infered/bhc_short_7b_v2_ckp500_test_infered_v2.csv \
    --verbose 0

python3 src/inference/infer.py \
    --model_name medischarge-7b-DI \
    --model_path /pure-mlo-scratch/make_project/spring2024/trial-runs/meditron-7B-DI-v2/hf_checkpoint_1500 \
    --input_path data/test_phase_1/DI_test_dataset_v2.jsonl \
    --output_path data/infered/di_short_7b_v2_ckp1500_test_infered_v2.csv \
    --verbose 0
    

## Pertinent results
# v1
# if [ "$1" == "medischarge-7b-bhc" ] || [ "$1" == "all" || [ "$1" == "v1"] ]; then
#     python3 src/inference/infer_without_section.py \
#         --model_name medischarge-7b-BHC \
#         --model_path /pure-mlo-scratch/make_project/spring2024/trial-runs/meditron-7B-BHC/hf_checkpoint \
#         --input_path data/test_phase_1/BHC_test_dataset_v2.jsonl \
#         --output_path data/infered/bhc_short_7b_v1_test_infered_wo_pertinent_results_v2.csv \
#         --verbose 0 \
#         --section_name "pertinent results"
# fi     

# if [ "$1" == "medischarge-7b-di" ] || [ "$1" == "all" || [ "$1" == "v1"]]; then
#     python3 src/inference/infer_without_section.py \
#         --model_name medischarge-7b-DI \
#         --model_path /pure-mlo-scratch/make_project/spring2024/trial-runs/meditron-7B-DI/hf_checkpoint \
#         --input_path data/test_phase_1/DI_test_dataset_v2.jsonl \
#         --output_path data/infered/di_short_7b_v1_test_infered_wo_pertinent_results_v2.csv \
#         --verbose 0 \
#         --section_name "pertinent results"
# fi 

# # v2
# if [ "$1" == "medischarge-7b-BHC" ] || [ "$1" == "all" || [ "$1" == "v2"] ]; then
#     python3 src/inference/infer_without_section.py \
#         --model_name medischarge-7b-BHC \
#         --model_path /pure-mlo-scratch/make_project/spring2024/trial-runs/meditron-7B-BHC-v2/hf_checkpoint_1000 \
#         --input_path data/test_phase_1/BHC_test_dataset_v2.jsonl \
#         --output_path data/infered/bhc_short_7b_v2_test_infered_wo_pertinent_results_v2.csv \
#         --verbose 0 \
#         --section_name "pertinent results"
# fi     

# if [ "$1" == "medischarge-7b-DI" ] || [ "$1" == "all" || [ "$1" == "v2"]]; then
#     python3 src/inference/infer_without_section.py \
#         --model_name medischarge-7b-DI \
#         --model_path /pure-mlo-scratch/make_project/spring2024/trial-runs/meditron-7B-DI-v2/hf_checkpoint_2000 \
#         --input_path data/test_phase_1/DI_test_dataset_v2.jsonl \
#         --output_path data/infered/di_short_7b_v2_test_infered_wo_pertinent_results_v2.csv \
#         --verbose 0 \
#         --section_name "pertinent results"
# fi 

# ## Physical exam
# # v1
# if [ "$1" == "medischarge-7b-bhc" ] || [ "$1" == "all" || [ "$1" == "v1"]]; then
#     python3 src/inference/infer_without_section.py \
#         --model_name medischarge-7b-BHC \
#         --model_path /pure-mlo-scratch/make_project/spring2024/trial-runs/meditron-7B-BHC/hf_checkpoint \
#         --input_path data/test_phase_1/BHC_test_dataset_v2.jsonl \
#         --output_path data/infered/bhc_short_7b_v1_test_infered_wo_physical_exam_v2.csv \
#         --verbose 0 \
#         --section_name "physical exam"
# fi     

# if [ "$1" == "medischarge-7b-di" ] || [ "$1" == "all" || [ "$1" == "v1"] ]; then
#     python3 src/inference/infer_without_section.py \
#         --model_name medischarge-7b-DI \
#         --model_path /pure-mlo-scratch/make_project/spring2024/trial-runs/meditron-7B-DI/hf_checkpoint \
#         --input_path data/test_phase_1/DI_test_dataset_v2.jsonl \
#         --output_path data/infered/di_short_7b_v1_test_infered_wo_physical_exam_v2.csv \
#         --verbose 0 \
#         --section_name "physical exam"
# fi 

# # v2
# if [ "$1" == "medischarge-7b-BHC" ] || [ "$1" == "all" || [ "$1" == "v2"]]; then
#     python3 src/inference/infer_without_section.py \
#         --model_name medischarge-7b-BHC \
#         --model_path /pure-mlo-scratch/make_project/spring2024/trial-runs/meditron-7B-BHC-v2/hf_checkpoint_1000 \
#         --input_path data/test_phase_1/BHC_test_dataset_v2.jsonl \
#         --output_path data/infered/bhc_short_7b_v2_test_infered_wo_physical_exam_v2.csv \
#         --verbose 0 \
#         --section_name "physical exam"
# fi     

# if [ "$1" == "medischarge-7b-DI" ] || [ "$1" == "all" || [ "$1" == "v2"] ]; then
#     python3 src/inference/infer_without_section.py \
#         --model_name medischarge-7b-DI \
#         --model_path /pure-mlo-scratch/make_project/spring2024/trial-runs/meditron-7B-DI-v2/hf_checkpoint_2000 \
#         --input_path data/test_phase_1/DI_test_dataset_v2.jsonl \
#         --output_path data/infered/di_short_7b_v2_test_infered_wo_physical_exam_v2.csv \
#         --verbose 0 \
#         --section_name "physical exam"
# fi 

# ## Past medical history
# # v1
# if [ "$1" == "medischarge-7b-bhc" ] || [ "$1" == "all" || [ "$1" == "v1"] ]; then
#     python3 src/inference/infer_without_section.py \
#         --model_name medischarge-7b-BHC \
#         --model_path /pure-mlo-scratch/make_project/spring2024/trial-runs/meditron-7B-BHC/hf_checkpoint \
#         --input_path data/test_phase_1/BHC_test_dataset_v2.jsonl \
#         --output_path data/infered/bhc_short_7b_v1_test_infered_wo_past_medical_history_v2.csv \
#         --verbose 0 \
#         --section_name "past medical history"
# fi     

# if [ "$1" == "medischarge-7b-di" ] || [ "$1" == "all" || [ "$1" == "v1"] ]; then
#     python3 src/inference/infer_without_section.py \
#         --model_name medischarge-7b-DI \
#         --model_path /pure-mlo-scratch/make_project/spring2024/trial-runs/meditron-7B-DI/hf_checkpoint \
#         --input_path data/test_phase_1/DI_test_dataset_v2.jsonl \
#         --output_path data/infered/di_short_7b_v1_test_infered_wo_past_medical_history_v2.csv \
#         --verbose 0 \
#         --section_name "past medical history"
# fi 
# # v2
# if [ "$1" == "medischarge-7b-BHC" ] || [ "$1" == "all" || [ "$1" == "v2"] ]; then
#     python3 src/inference/infer_without_section.py \
#         --model_name medischarge-7b-BHC \
#         --model_path /pure-mlo-scratch/make_project/spring2024/trial-runs/meditron-7B-BHC-v2/hf_checkpoint_1000 \
#         --input_path data/test_phase_1/BHC_test_dataset_v2.jsonl \
#         --output_path data/infered/bhc_short_7b_v2_test_infered_wo_past_medical_history_v2.csv \
#         --verbose 0 \
#         --section_name "past medical history"
# fi     

# if [ "$1" == "medischarge-7b-DI" ] || [ "$1" == "all" || [ "$1" == "v2"] ]; then
#     python3 src/inference/infer_without_section.py \
#         --model_name medischarge-7b-DI \
#         --model_path /pure-mlo-scratch/make_project/spring2024/trial-runs/meditron-7B-DI-v2/hf_checkpoint_2000 \
#         --input_path data/test_phase_1/DI_test_dataset_v2.jsonl \
#         --output_path data/infered/di_short_7b_v2_test_infered_wo_past_medical_history_v2.csv \
#         --verbose 0 \
#         --section_name "past medical history"
# fi 

# ## History of present illness
# # v1
# if [ "$1" == "medischarge-7b-bhc" ] || [ "$1" == "all" || [ "$1" == "v1"] ]; then
#     python3 src/inference/infer_without_section.py \
#         --model_name medischarge-7b-BHC \
#         --model_path /pure-mlo-scratch/make_project/spring2024/trial-runs/meditron-7B-BHC/hf_checkpoint \
#         --input_path data/test_phase_1/BHC_test_dataset_v2.jsonl \
#         --output_path data/infered/bhc_short_7b_v1_test_infered_wo_present_illness_v2.csv \
#         --verbose 0 \
#         --section_name "history of present illness"
# fi     

# if [ "$1" == "medischarge-7b-di" ] || [ "$1" == "all"]; then
#     python3 src/inference/infer_without_section.py \
#         --model_name medischarge-7b-DI \
#         --model_path /pure-mlo-scratch/make_project/spring2024/trial-runs/meditron-7B-DI/hf_checkpoint \
#         --input_path data/test_phase_1/DI_test_dataset_v2.jsonl \
#         --output_path data/infered/di_short_7b_v1_test_infered_wo_present_illness_v2.csv \
#         --verbose 0 \
#         --section_name "history of present illness"
# fi 

# v2
# if [ "$1" == "medischarge-7b-BHC" ] || [ "$1" == "all" || [ "$1" == "v2"] ]; then
#     python3 src/inference/infer_without_section.py \
#         --model_name medischarge-7b-BHC \
#         --model_path /pure-mlo-scratch/make_project/spring2024/trial-runs/meditron-7B-BHC-v2/hf_checkpoint_1000 \
#         --input_path data/test_phase_1/BHC_test_dataset_v2.jsonl \
#         --output_path data/infered/bhc_short_7b_v2_test_infered_wo_present_illness_v2.csv \
#         --verbose 0 \
#         --section_name "history of present illness"
# fi     

# if [ "$1" == "medischarge-7b-DI" ] || [ "$1" == "all" || [ "$1" == "v2"] ]; then
#     python3 src/inference/infer_without_section.py \
#         --model_name medischarge-7b-DI \
#         --model_path /pure-mlo-scratch/make_project/spring2024/trial-runs/meditron-7B-DI-v2/hf_checkpoint_2000 \
#         --input_path data/test_phase_1/DI_test_dataset_v2.jsonl \
#         --output_path data/infered/di_short_7b_v2_test_infered_wo_present_illness_v2.csv \
#         --verbose 0 \
#         --section_name "history of present illness"
# fi 