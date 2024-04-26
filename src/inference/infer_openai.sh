input_path=data/openai_inputs/openai_input_bhc_1_shots.jsonl
output_path=data/infered_openai/openai_output_bhc_1_shots.jsonl

python3 infer_openai.py \
    --input_path $input_path \
    --output_path $output_path \

