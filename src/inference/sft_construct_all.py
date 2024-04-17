import sys
sys.path.append('../')
import argparse
import pandas as pd
from utils.preprocessing import (load_data,
                                save_data,
                                build_combined_discharge,
                                get_bhc_input,
                                extract_clean_inputs,
                                remove_unecessary_tokens)
from utils.token_count import get_token_list, get_token_count, plot_token_count
from utils.format_change import dataframe_to_jsonl
from tqdm import tqdm
tqdm.pandas()

selected_features = [
                        # 'sex',
                        # 'allergies',
                        # 'chief_complaint',
                        # 'major_surgical_procedures',
                        'history_of_present_illness',
                        # 'past_medical_history',
                        # 'physical_exam',
                        # 'pertinent_results',
                        'medication_on_admission',
                        'discharge_medications',
                        'discharge_disposition',
                        'discharge_diagnosis',
                        'discharge_condition',
                    ]

system_prompt_bhc = "You are a medical assistant. Your task is to write the brief hospital course corresponding to the following hospital discharge.\n\n"
system_prompt_di = "You are a medical assistant. Your task is to write the discharge instructions corresponding to the following hospital discharge.\n\n"

def construct_sft_dataset (discharge_dataset: str, target_dataset: str, constructed_dataset_folder: str, max_length: int):
    discharge_df = load_data(discharge_dataset)
    targets_df = load_data(target_dataset)
    combined_discharge_df = build_combined_discharge(discharge_df, targets_df)
    print("The number of the combined discharge dataset: ", len(combined_discharge_df))
    
    combined_discharge_df['input'] = extract_clean_inputs(combined_discharge_df,
                                              features_to_include=selected_features)
    
    combined_discharge_df['input_of_bhc'] = system_prompt_bhc + combined_discharge_df['input']
    combined_discharge_df['input_of_bhc'] = combined_discharge_df['input_of_bhc'].progress_apply(remove_unecessary_tokens)
    combined_discharge_df['input_of_bhc_tokens'] = combined_discharge_df['input_of_bhc'].progress_apply(get_token_count)
    combined_discharge_df['bhc_token_count'] = combined_discharge_df['brief_hospital_course'].progress_apply(get_token_count)
    
    combined_discharge_df['input_of_di'] = system_prompt_di + combined_discharge_df['input']
    combined_discharge_df['input_of_di'] = combined_discharge_df['input_of_di'].progress_apply(remove_unecessary_tokens)
    combined_discharge_df['input_of_di_tokens'] = combined_discharge_df['input_of_di'].progress_apply(get_token_count)
    combined_discharge_df['di_token_count'] = combined_discharge_df['discharge_instructions'].progress_apply(get_token_count)
    
    # filter out the data samples where the input_of_bhc_tokens + bhc_token_count is greater than 2048 and print out the percentage of the outliers
    bhc_df = combined_discharge_df[combined_discharge_df['input_of_bhc_tokens'] + combined_discharge_df['bhc_token_count'] < max_length]
    print("The percentage of the bhc test set outliers: ", 1 - len(bhc_df) / len(combined_discharge_df))
    
    # filter out the data samples where the input_of_di_tokens + di_token_count is greater than 2048 and print out the percentage of the outliers
    di_df = combined_discharge_df[combined_discharge_df['input_of_di_tokens'] + combined_discharge_df['di_token_count'] < max_length]
    print("The percentage of the di test set outliers: ", 1 - len(di_df) / len(combined_discharge_df))
    
    # save the constructed dataset
    constructed_dataset_bhc = constructed_dataset_folder + "/bhc_v2.1.jsonl" 
    bhc_df.set_index('hadm_id', inplace=True)
    dataframe_to_jsonl(bhc_df, attributes=['input_of_bhc', 'brief_hospital_course'], keys=['prompt', 'gold'], file_path=constructed_dataset_bhc)
    
    constructed_dataset_di = constructed_dataset_folder + "/di_v2.1.jsonl"
    di_df.set_index('hadm_id', inplace=True)
    dataframe_to_jsonl(di_df, attributes=['input_of_di', 'discharge_instructions'], keys=['prompt', 'gold'], file_path=constructed_dataset_di)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Construct the train and vaild set.')
    parser.add_argument('--constructed_dataset_folder', 
                        type=str, 
                        help='Path to the constructed dataset folder')
    parser.add_argument('--discharge_dataset', 
                        type=str, 
                        help='Path to the discharge.csv.gz')
    parser.add_argument('--target_dataset', 
                        type=str, 
                        help='Path to the discharge_target.csv.gz')
    parser.add_argument('--max_length',
                        type=int,
                        default=2048,
                        help='The maximum length of the input')
        
    args = parser.parse_args()
    
    construct_sft_dataset(args.discharge_dataset, args.target_dataset, args.constructed_dataset_folder, args.max_length)
        
        
    
    