import json

input_path = "/pure-mlo-scratch/make_project/spring2024/data/raw/version6.0/BHC_valid_dataset_v6_6k.jsonl"
output_path = "/pure-mlo-scratch/make_project/spring2024/data/raw/version6.0/BHC_valid_dataset_v6_6k_filtered.jsonl"

def filter_jsonl(input_file, output_file):
    # Initialize a counter for filtered lines
    filter_count = 0
    # String to be matched
    target_string = "You are a medical assistant. Your task is to write the brief hospital course corresponding to the following hospital discharge:\n\nnan\n\nBrief Hospital Course:"
    
    # Open the input file and the output file
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        # Iterate through each line in the input file
        for line in infile:
            # Convert the line from JSONL format to a dictionary
            data = json.loads(line)
            # Check if the 'prompt' key matches the target string
            if data.get('prompt') != target_string:
                # If it doesn't match, write the line to the output file
                json.dump(data, outfile)
                outfile.write('\n')
            else:
                # Increment the filter counter if it matches
                filter_count += 1
    
    # Print the number of filtered lines
    print(f"{filter_count} lines were filtered out.")
    
# Call the function with the input and output file paths
if __name__ == "__main__":
    filter_jsonl(input_path, output_path)
    
# 1070 lines were filtered out for BHC train
# 252 lines were filtered out for BHC valid