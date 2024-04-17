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

strategy = [["medication_on_admission", "discharge_medications", "discharge_disposition", "discharge_diagnosis", "discharge_condition"], 
            ["discharge_medications", "discharge_disposition", "discharge_diagnosis", "discharge_condition"],
            ["medication_on_admission", "discharge_disposition", "discharge_diagnosis", "discharge_condition"],
            ["discharge_medications"],
            ["discharge_disposition", "discharge_diagnosis", "discharge_condition"],
            ["discharge_diagnosis", "discharge_condition"],
            ['discharge_condition'],
            ['discharge_diagnosis'],
            ['medication_on_admission'],
            ['discharge_disposition'],
            ]

system_prompt = "You are a medical assistant. Your task is to write the discharge instructions corresponding to the following hospital discharge.\n\n"

def construct_DI_test (discharge_dataset: str, target_dataset: str, generated_bhc_test: str, constructed_di_test: str, select_strategy: list, max_length: int, cutting_length: int, bhc_max_length: int):
    test_discharge = load_data(discharge_dataset)
    test_targets = load_data(target_dataset)
    test_combined_discharge = build_combined_discharge(test_discharge, test_targets)
    
    test_combined_discharge['input_of_di'] = extract_clean_inputs(test_combined_discharge,
                        features_to_include=[
                                'medication_on_admission',
                                'discharge_medications',
                                'discharge_disposition',
                                'discharge_diagnosis',
                                'discharge_condition',
                            ])
    
    test_combined_discharge['input_of_di'] = system_prompt + test_combined_discharge['input_of_di']
    test_combined_discharge['input_of_di'] = test_combined_discharge['input_of_di'].progress_apply(remove_unecessary_tokens)
    
    # compute the token count for every section in DI dataset
    di_sections = ['medication_on_admission', 'discharge_medications', 'discharge_disposition', 'discharge_diagnosis', 'discharge_condition']

    for section in di_sections:
        print("Now token count computing: " + section)
        test_combined_discharge[section] = extract_clean_inputs(test_combined_discharge, features_to_include=[section])
        test_combined_discharge[section] = test_combined_discharge[section].progress_apply(remove_unecessary_tokens)
        test_combined_discharge[section + "_tokens"] = test_combined_discharge[section].progress_apply(get_token_count)
        
    # add the generated BHC into the test_combined_discharge
    generated_bhc_test_df = pd.read_csv(generated_bhc_test)
    test_combined_discharge_all = pd.merge(test_combined_discharge, generated_bhc_test_df[['hadm_id', 'generated']], on='hadm_id')
    test_combined_discharge_all['generated'] = test_combined_discharge_all['generated'].progress_apply(remove_unecessary_tokens)
    test_combined_discharge_all['generated_tokens'] = test_combined_discharge_all['generated'].progress_apply(get_token_count)
    
    # check whether there is select_strategy and its length should not be 0
    if select_strategy is None or len(select_strategy) == 0:
        raise ValueError("Select strategy is invalid.")
    
    test_combined_discharge_all['input_of_di_new'] = test_combined_discharge_all['input_of_di']
    for index, row in test_combined_discharge_all.iterrows():
        total_tokens = 0
        BHC_tokens = row['generated_tokens']
        if BHC_tokens > bhc_max_length:
            test_combined_discharge_all.at[index, 'generated'] = extract_clean_inputs(test_combined_discharge_all.iloc[index], features_to_include=['history_of_present_illness'])
        for select in select_strategy:
            total_tokens = 0
            for section in select:
                total_tokens += row[section + "_tokens"]
            if total_tokens < (cutting_length - BHC_tokens): 
                final_select = select
                break
            if select == select_strategy[-1]:
                final_select = select_strategy[-1]
                print("No suitable strategy found.")
        test_combined_discharge_all.at[index, 'input_of_di_new'] = extract_clean_inputs(test_combined_discharge_all.iloc[index], features_to_include=final_select)
    test_combined_discharge_all['input_of_di_new'] = system_prompt + "Brief Hospital Course:\n" + test_combined_discharge_all['generated'] + "\n\n" + test_combined_discharge_all['input_of_di_new']
    test_combined_discharge_all['input_of_di_new'] = test_combined_discharge_all['input_of_di_new'].progress_apply(remove_unecessary_tokens)
    test_combined_discharge_all['input_of_di_new_tokens'] = test_combined_discharge_all['input_of_di_new'].progress_apply(get_token_count)
    # check how many rows where its input_of_bhc_new_tokens is greater than max_length
    print("The percentage of the di test set outliers: ", len(test_combined_discharge_all[test_combined_discharge_all['input_of_di_new_tokens'] > max_length]))
    if len(test_combined_discharge_all[test_combined_discharge_all['input_of_di_new_tokens'] > max_length]) > 0:
        print("Show out all length of outliers") 
        print(test_combined_discharge_all[test_combined_discharge_all['input_of_di_new_tokens'] > max_length]['input_of_di_new_tokens'])
        raise ValueError("The length of input_of_di_new_tokens is greater than max_length.")
        
    print("the length of original test set: ", len(test_combined_discharge))
    print("the length of new test set: ", len(test_combined_discharge_all))
    
    # save the constructed DI test set
    test_combined_discharge_all.set_index('hadm_id', inplace=True)
    dataframe_to_jsonl(test_combined_discharge_all, attributes=['input_of_di_new', 'discharge_instructions'], keys=['prompt', 'gold'], file_path=constructed_di_test)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Construct the DI test set using the generated BHC.')
    parser.add_argument('--generated_bhc_test', 
                        type=str, 
                        help='Path to the generated BHC test set')
    parser.add_argument('--constructed_di_test',
                        type=str,
                        help='Path to the constructed DI test set')
    parser.add_argument('--select_strategy',
                        type=list,
                        default=strategy,
                        help='The strategy to select the sections in DI dataset')
    parser.add_argument('--discharge_dataset', 
                        type=str, 
                        default="/home/haotian/make-discharge-me/data/test_phase_1/discharge.csv.gz",
                        help='Path to the discharge.csv.gz')
    parser.add_argument('--target_dataset', 
                        type=str, 
                        default="/home/haotian/make-discharge-me/data/test_phase_1/discharge_target.csv.gz",
                        help='Path to the discharge_target.csv.gz')
    parser.add_argument('--max_length',
                        type=int,
                        default=2048,
                        help='The maximum length of the input')
    parser.add_argument('--cutting_length',
                        type=int,
                        default=1548,
                        help='The cutting length of the input')
    parser.add_argument('--bhc_max_length',
                        type=int,
                        default=1400,
                        help='The maximum length of the BHC input')

    args = parser.parse_args()

    construct_DI_test(args.discharge_dataset, args.target_dataset, args.generated_bhc_test, args.constructed_di_test, args.select_strategy, args.max_length, args.cutting_length, args.bhc_max_length)