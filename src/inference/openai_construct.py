import argparse
import os
import sys

root_path = os.path.abspath(os.path.join(os.curdir, os.pardir, os.pardir))
os.chdir(root_path)
sys.path.append(root_path)

from src.utils.preprocessing import (load_data,
                                save_data,
                                build_combined_discharge)
from tqdm import tqdm

tqdm.pandas()

def remove_bhc_di(raw_discharge: str, bhc:str, di:str, mode:str='bhc'):
    '''
    Remove the BHC or DI from the discharge
    input:
        raw_discharge: str
            The raw discharge
        bhc: str
            The BHC text to be removed
        di: str
            The DI text to be removed
        mode: str
            Either bhc or di

    output:
        clean_discharge: str
            The clean discharge
    '''
    bhc_title = 'Brief Hospital Course:'
    di_title = 'Discharge Instructions:'

    #throw error if the BHC or DI is not in the discharge


    if di not in raw_discharge:
        print(f"The DI: {di} is not in the discharge")
        print(f"The raw discharge is: {raw_discharge}")
        raise ValueError
    
    
    
    if di_title not in raw_discharge:
        print(f"The DI title: {di_title} is not in the discharge")
        print(f"The raw discharge is: {raw_discharge}")
        raise ValueError
    
    clean_discharge = raw_discharge.replace(di_title, '').replace(di, '')

    if mode == 'bhc':
        if bhc not in raw_discharge:
            print(f"The BHC: {bhc} is not in the discharge")
            print(f"The raw discharge is: {raw_discharge}")
            raise ValueError
        
        if bhc_title not in raw_discharge:
            print(f"The BHC title: {bhc_title} is not in the discharge")
            print(f"The raw discharge is: {raw_discharge}")
            raise ValueError

        clean_discharge = clean_discharge.replace(bhc_title, '').replace(bhc, '')

    return clean_discharge


def concat_n_examples(discharges: list, outputs: list, mode:str):
    '''
    Concatenate n examples together
    input:
        discharges: list
            The list of discharges
        outputs: list
            The list of outputs
    output:
        n_examples concatenated in one string
    '''

    examples_string = ''
    full_mode = 'Brief Hospital Course' if mode == 'bhc' else 'Discharge Instructions'
    for i, (discharge, output) in enumerate(zip(discharges, outputs)):
        examples_string += f"Example {i+1}:\nSTART OF DISCHARGE:\n{discharge}\nEND OF DISCHARGE\n\nSTART OF EXPECTED {full_mode} OUTPUT:\n{output}\nEND OF EXPECTED {full_mode} OUTPUT\n\n"
        
    return examples_string

def construct_n_shot_prompt(prompt: str, clean_discharge: str, n_shot: int, examples_string: str):
    '''
    Construct the n-shot prompt
    input:
        prompt: str (to beformstted with the examples, n, and the clean discharge)
            The prompt to be formatted
        clean_discharge: str
        n_shot: int
            The number of example
        train_df: pd.DataFrame
            The training dataframe (where to extract the example from)
    output:
        n_shot_prompt: str
    '''
    example = 'example' if n_shot == 1 else 'examples'
    n_shot_prompt = prompt.format(example, examples_string, clean_discharge, example)

    return n_shot_prompt

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Construct the BHC or DI test set for open ai Inference.')
    parser.add_argument('--Modes',
                        type=str,
                        help='The modes to construct the prompt for in ["bhc","di"]')
    parser.add_argument('--test_discharge_dataset', 
                        type=str,
                        help='Path to the discharge.csv.gz')
    parser.add_argument('--test_target_dataset',
                        type=str,
                        help='Path to the target.csv.gz')
    parser.add_argument('--train_discharge_dataset', 
                        type=str,
                        help='Path to the discharge.csv.gz')
    parser.add_argument('--train_target_dataset',
                        type=str,
                        help='Path to the target.csv.gz')
    parser.add_argument('--nb_samples',
                        type=int,
                        help='Number of samples to construct')
    parser.add_argument('--n_shots_list',
                        type=str,
                        help='Number of shots to construct')
    parser.add_argument('--prompt_folder_path',
                        type=str,
                        help='Path to the prompt file')
    parser.add_argument('--output_folder_path',
                        type=str,
                        help='Path to the output file')
        
    args = parser.parse_args()

    modes = args.Modes.split(',')
    n_shots_list = args.n_shots_list.split(',')


    test_discharges = load_data(args.test_discharge_dataset)

    # sample the test_discharges
    test_discharges = test_discharges.sample(args.nb_samples)

    # merge with the targets
    test_targets = load_data(args.test_target_dataset)
    test_discharges = build_combined_discharge(test_discharges, test_targets)

    del test_targets

    #Repeat the same process for the training set
    train_discharges = load_data(args.train_discharge_dataset)
    train_targets = load_data(args.train_target_dataset)    

    # load the prompt

    prompts = {}

    for mode in modes:
        with open(args.prompt_folder_path + f'/{mode}_openai_prompt.json', 'r') as f:
            prompts[mode] = f.read().replace('\\n', '\n')

   
    for mode in modes:
        prompt = prompts[mode]
        output_key = 'brief_hospital_course' if mode == 'bhc' else ('discharge_instructions' if mode == 'di' else None)

         # get the clean discharges
        test_discharges['clean_discharge'] = test_discharges[['text','brief_hospital_course','discharge_instructions']].progress_apply(axis = 1,
                                                                                                    func = lambda x: remove_bhc_di(x['text'], x['brief_hospital_course'], x['discharge_instructions'], mode))


        for n_shots in n_shots_list:
            n_shots = int(n_shots)
            print(f"Constructing the {mode} {n_shots}-shot prompts")
            
            train_discharges_sample = train_discharges.sample(n_shots * args.nb_samples)
            train_discharges_sample = build_combined_discharge(train_discharges_sample, train_targets)

            train_discharges_sample['clean_discharge'] = train_discharges_sample[['text','brief_hospital_course','discharge_instructions']].progress_apply(axis = 1, 
                                                                                                    func = lambda x: remove_bhc_di(x['text'], x['brief_hospital_course'], x['discharge_instructions'], mode))


            test_discharges['clean_discharge_examples'] = train_discharges_sample['clean_discharge'].groupby(train_discharges_sample.index // n_shots).apply(list)
            test_discharges['output_examples'] = train_discharges_sample[output_key].groupby(train_discharges_sample.index // n_shots).apply(list)
            test_discharges['examples_string'] = test_discharges.progress_apply(lambda x: concat_n_examples(x['clean_discharge_examples'], x['output_examples'], mode=mode), axis=1)

            # construct the n-shot prompt
            test_discharges['prompt'] = test_discharges[['clean_discharge','examples_string']].progress_apply(axis = 1, func = lambda x: construct_n_shot_prompt(prompt, x['clean_discharge'], n_shots, x['examples_string']))
            
            final_df = test_discharges[['hadm_id','prompt', output_key]].copy()
            final_df['idx'] = final_df.index
            final_df.rename(columns={output_key: 'gold'}, inplace=True)

            output_path = args.output_folder_path + f'/openai_input_{mode}_{n_shots}_shots.jsonl'

            save_data(final_df, output_path)




        
        
    
    