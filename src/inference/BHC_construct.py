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

strategy = [
    ['sex', 'allergies', 'chief_complaint', 'major_surgical_procedures', 'history_of_present_illness', 'physical_exam', 'pertinent_results', 'past_medical_history'],
    ['sex', 'allergies', 'chief_complaint', 'major_surgical_procedures', 'history_of_present_illness', 'physical_exam', 'pertinent_results'],
    ['sex', 'allergies', 'chief_complaint', 'major_surgical_procedures', 'history_of_present_illness', 'physical_exam', 'past_medical_history'],
    ['sex', 'allergies', 'chief_complaint', 'major_surgical_procedures', 'history_of_present_illness', 'pertinent_results', 'past_medical_history'],
    ['sex', 'allergies', 'chief_complaint', 'major_surgical_procedures', 'history_of_present_illness', 'physical_exam'],
    ['sex', 'allergies', 'chief_complaint', 'major_surgical_procedures', 'history_of_present_illness', 'pertinent_results'],
    ['sex', 'allergies', 'chief_complaint', 'major_surgical_procedures', 'history_of_present_illness', 'past_medical_history'],
    ['sex', 'allergies', 'chief_complaint', 'major_surgical_procedures', 'history_of_present_illness'],
    ['sex', 'allergies', 'chief_complaint', 'history_of_present_illness'],
    ['sex', 'allergies', 'chief_complaint', 'major_surgical_procedures', 'physical_exam', 'pertinent_results', 'past_medical_history'],
    ['sex', 'allergies', 'chief_complaint', 'major_surgical_procedures', 'physical_exam', 'pertinent_results'],
    ['sex', 'allergies', 'chief_complaint', 'major_surgical_procedures', 'physical_exam', 'past_medical_history'],
    ['sex', 'allergies', 'chief_complaint', 'major_surgical_procedures', 'pertinent_results', 'past_medical_history'],
    ['sex', 'allergies', 'chief_complaint', 'major_surgical_procedures', 'physical_exam'],
    ['sex', 'allergies', 'chief_complaint', 'major_surgical_procedures', 'pertinent_results'],
    ['sex', 'allergies', 'chief_complaint', 'major_surgical_procedures', 'past_medical_history'],
    ['sex', 'allergies', 'chief_complaint', 'physical_exam', 'pertinent_results', 'past_medical_history'],
    ['sex', 'allergies', 'chief_complaint', 'physical_exam', 'pertinent_results'],
    ['sex', 'allergies', 'chief_complaint', 'physical_exam', 'past_medical_history'],
    ['sex', 'allergies', 'chief_complaint', 'pertinent_results', 'past_medical_history'],
    ['sex', 'allergies', 'chief_complaint', 'physical_exam'],
    ['sex', 'allergies', 'chief_complaint', 'pertinent_results'],
    ['sex', 'allergies', 'chief_complaint', 'past_medical_history'],
]

system_prompt = "You are a medical assistant. Your task is to write the brief hospital course corresponding to the following hospital discharge.\n\n"

def construct_BHC_test (discharge_dataset: str, target_dataset: str, constructed_bhc_test: str, select_strategy: list, max_length: int, cutting_length: int):
    test_discharge = load_data(discharge_dataset)
    test_targets = load_data(target_dataset)
    test_combined_discharge = build_combined_discharge(test_discharge, test_targets)
    
    test_combined_discharge['input_of_bhc'] = extract_clean_inputs(test_combined_discharge,
                                              features_to_include=[
                                                'sex',
                                                'allergies',
                                                'chief_complaint',
                                                'major_surgical_procedures',
                                                'history_of_present_illness',
                                                'past_medical_history',
                                                'physical_exam',
                                                'pertinent_results',
                                            ])
    
    test_combined_discharge['input_of_bhc'] = system_prompt + test_combined_discharge['input_of_bhc']
    test_combined_discharge['input_of_bhc'] = test_combined_discharge['input_of_bhc'].progress_apply(remove_unecessary_tokens)
    
    # compute the token count for every section in BHC dataset
    bhc_sections = ['sex', 'allergies', 'chief_complaint', 'major_surgical_procedures', 'history_of_present_illness', 'past_medical_history', 'physical_exam', 'pertinent_results']
    
    for section in bhc_sections:
        print("Now token count computing: " + section)
        test_combined_discharge[section] = extract_clean_inputs(test_combined_discharge, features_to_include=[section])
        test_combined_discharge[section] = test_combined_discharge[section].progress_apply(remove_unecessary_tokens)
        test_combined_discharge[section + "_tokens"] = test_combined_discharge[section].progress_apply(get_token_count) 
        
    # check whether there is select_strategy and its length should not be 0
    if select_strategy is None or len(select_strategy) == 0:
        raise ValueError("Select strategy is invalid.")
    
    test_combined_discharge['input_of_bhc_new'] = test_combined_discharge['input_of_bhc']
    
    for index, row in test_combined_discharge.iterrows():
        total_tokens = 0
        for select in select_strategy:
            total_tokens = 0
            for section in select:
                total_tokens += row[section + "_tokens"]
            if total_tokens < cutting_length: 
                final_select = select
                break
            if select == select_strategy[-1]:
                final_select = select_strategy[-1]
                print("no suitable strategy found")
        test_combined_discharge.at[index, 'input_of_bhc_new'] = extract_clean_inputs(test_combined_discharge.iloc[index], features_to_include=final_select)
    test_combined_discharge['input_of_bhc_new'] = system_prompt + test_combined_discharge['input_of_bhc_new']
    test_combined_discharge['input_of_bhc_new'] = test_combined_discharge['input_of_bhc_new'].progress_apply(remove_unecessary_tokens)
    test_combined_discharge['input_of_bhc_new_tokens'] = test_combined_discharge['input_of_bhc_new'].progress_apply(get_token_count)
    # check how many rows where its input_of_bhc_new_tokens is greater than max_length
    print("The percentage of the di test set outliers: ", len(test_combined_discharge[test_combined_discharge['input_of_bhc_new_tokens'] > max_length]))
    if len(test_combined_discharge[test_combined_discharge['input_of_bhc_new_tokens'] > max_length]) > 0:
        print("Show out all length of outliers")
        print(test_combined_discharge[test_combined_discharge['input_of_bhc_new_tokens'] > max_length]['input_of_bhc_new_tokens'])
        raise ValueError("The length of input_of_bhc_new_tokens is greater than max_length.")
        
    # save the constructed BHC test set
    test_combined_discharge.set_index('hadm_id', inplace=True)
    dataframe_to_jsonl(test_combined_discharge, attributes=['input_of_bhc_new', 'brief_hospital_course'], keys=['prompt', 'gold'], file_path=constructed_bhc_test)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Construct the BHC test set directly.')
    parser.add_argument('--constructed_bhc_test', 
                        type=str, 
                        help='Path to the constructed BHC test set')
    parser.add_argument('--select_strategy',
                        type=list,
                        default=strategy,
                        help='The strategy to select the sections in BHC dataset')
    parser.add_argument('--discharge_dataset', 
                        type=str, 
                        default="/home/haotian/make-discharge-me/data/test_phase_2/discharge.csv.gz",
                        help='Path to the discharge.csv.gz')
    parser.add_argument('--target_dataset', 
                        type=str, 
                        default="/home/haotian/make-discharge-me/data/test_phase_2/discharge_target.csv.gz",
                        help='Path to the discharge_target.csv.gz')
    parser.add_argument('--max_length',
                        type=int,
                        default=2048,
                        help='The maximum length of the input')
    parser.add_argument('--cutting_length',
                        type=int,
                        default=1548,
                        help='The length to cut the input')
        
    args = parser.parse_args()
    
    construct_BHC_test(args.discharge_dataset, args.target_dataset, args.constructed_bhc_test, args.select_strategy, args.max_length, args.cutting_length)
        
        
    
    