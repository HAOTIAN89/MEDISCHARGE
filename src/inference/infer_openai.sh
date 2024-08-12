input_paths=(
    "data/openai_inputs/openai_input_bhc_1_shots.jsonl"
    "data/openai_inputs/openai_input_bhc_2_shots.jsonl"
    "data/openai_inputs/openai_input_bhc_3_shots.jsonl"
    "data/openai_inputs/openai_input_di_1_shots.jsonl"
    "data/openai_inputs/openai_input_di_2_shots.jsonl"
    "data/openai_inputs/openai_input_di_3_shots.jsonl"
)

# List of output paths
output_paths=(
    "data/infered_openai/gpt4/openai_output_bhc_1_shots.jsonl"
    "data/infered_openai/gpt4/openai_output_bhc_2_shots.jsonl"
    "data/infered_openai/gpt4/openai_output_bhc_3_shots.jsonl"
    "data/infered_openai/gpt4/openai_output_di_1_shots.jsonl"
    "data/infered_openai/gpt4/openai_output_di_2_shots.jsonl"
    "data/infered_openai/gpt4/openai_output_di_3_shots.jsonl"
)

# Loop through each input path
for i in "${!input_paths[@]}"; do
    echo "Processing ${input_paths[$i]}..."
    echo "Output will be saved to ${output_paths[$i]}"
    python3 infer_openai.py \
        --input_path "${input_paths[$i]}" \
        --output_path "${output_paths[$i]}"
done