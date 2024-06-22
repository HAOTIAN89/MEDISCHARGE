import pandas as pd
import regex as re
import argparse
from tqdm import tqdm
import os
import sys
tqdm.pandas()
from itertools import combinations
import json

current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(os.path.dirname(current_file_path))
sys.path.append(parent_directory)
from utils.token_count import get_token_count


# ----------------------- Preprocessing utilities ----------------------- #

def load_data(file_path: str, type='gzip') -> pd.DataFrame:
    """
    Loads the data from the file_path.
        Args:
            file_path: str, the path to the file
            type: str, the type of the file
        Return:
            data: pd.DataFrame, the data loaded from the file
    """
    if type == 'gzip':
        return pd.read_csv(file_path, compression='gzip')
    elif type == 'csv':
        return pd.read_csv(file_path)
    elif type == 'json':
        return pd.read_json(file_path)
    elif type == 'jsonl':
        return pd.read_json(file_path, lines=True)
 
    raise ValueError(f"Type {type} not supported.")

def save_data(data: pd.DataFrame, file_path: str):
    """
    Saves the data to the file_path.
        Args:
            data: pd.DataFrame, the data to save
            file_path: str, the path to the file
    """
    if file_path.endswith('.csv'):
        data.to_csv(file_path, index=False)
    elif file_path.endswith('.json'):
        data.to_json(file_path, orient='records')
    elif file_path.endswith('.jsonl'):
        data.to_json(file_path, orient='records', lines=True)
    elif file_path.endswith('.gzip'):
        data.to_csv(file_path, index=False,compression='gzip')
    else:
        raise ValueError(f"File type not supported. Supported types: csv, json, jsonl, gzip")

def build_combined_discharge(discharges: pd.DataFrame, discharges_target: pd.DataFrame) -> pd.DataFrame:
    """
    Builds the combined discharge dataframe by merging 
    the discharges and discharges_target (bhc and di) dataframes on the hadm_id.
        Args:
            discharges: pd.DataFrame, the discharges dataframe
            discharges_target: pd.DataFrame, the discharges_target dataframe
        Return:
            combined_discharge_train_df: pd.DataFrame, the combined discharge dataframe

    """
    combined_discharge_train_df = pd.merge(discharges[['hadm_id', 'text']],
                                discharges_target[['hadm_id', 'discharge_instructions', 'brief_hospital_course']],
                                on='hadm_id',
                                how='inner')
    
    return combined_discharge_train_df

def remove_bhc_di(raw_discharge: str, bhc:str, di:str, mode:str='bhc'):
    '''
    Remove the BHC or DI from the discharge, as well as their heder
    Args:
        raw_discharge: str, the raw discharge
        bhc: str, the BHC text to be removed
        di: str, the DI text to be removed
        mode: str, either bhc or di

    Return:
        clean_discharge: str
            The clean discharge
    
    Note: The di (and its header) is always removed.
    The bhc is removed only if the mode is 'bhc'
    '''

    bhc_header = 'Brief Hospital Course:'
    di_header = 'Discharge Instructions:'

    if di not in raw_discharge:
        print(f"The DI: {di} is not in the discharge")
        print(f"The raw discharge is: {raw_discharge}")
        raise ValueError
    
    if di_header.lower() not in raw_discharge.lower():
        print(f"The DI title: {di_header} is not in the discharge")
        print(f"The raw discharge is: {raw_discharge}")
        raise ValueError
    
    clean_discharge = raw_discharge.replace(di_header, '').replace(di, '')

    if mode == 'bhc':
        if bhc not in raw_discharge:
            print(f"The BHC: {bhc} is not in the discharge")
            print(f"The raw discharge is: {raw_discharge}")
            raise ValueError
        
        if bhc_header.lower() not in raw_discharge.lower():
            print(f"The BHC title: {bhc_header} is not in the discharge")
            print(f"The raw discharge is: {raw_discharge}")
            raise ValueError

        clean_discharge = clean_discharge.replace(bhc_header, '').replace(bhc, '')

    return clean_discharge


def get_original_bhc_di_input(combined_discharges: pd.DataFrame, mode ='bhc') -> pd.DataFrame:
    """
    Get the original discharge input, removing the BHC or DI from all discharges of the given dataframe.
    Args:
        combined_discharges: pd.DataFrame, the combined discharge dataframe
        mode: str, either bhc or di
    Return:
        original_bhc_input: pd.DataFrame
            The original discharge input
    """
    original_bhc_input = combined_discharges.progress_apply(lambda x: remove_bhc_di(x['text'], x['brief_hospital_course'], x['discharge_instructions'], mode=mode), axis=1)
    
    return original_bhc_input


def generate_strategies(importance_order, removeable_groups):
    """
    Generate the strategies in order of relevence to the task.
    Args:
        importance_order: list, the order of importance of the sections
        removeable_groups: dict, the groups of sections that can be removed successively
    Return:
        strategies: list, the strategies to try
    """
    strategies = [importance_order] #best strategie is the complete one (with all sections)
    last_removed_trial = 0 
    
    for i in removeable_groups: #iterating over the groups of sections that can be removed, starting from the first group
        for n_removed in range(1, len(removeable_groups[i])+1): #iterating over the possible numbers of sections to remove
                                                                #in the group
                                                                #starting from lowest number
            
            for to_remove in combinations(list(reversed(removeable_groups[i])), n_removed): #iterating over the possible combinations,
                                                                                            #reversed so that strategies with more important sections appear first
                if i > 1 and n_removed <= last_removed_trial: #no group 0
                                                              #no need to try strategies that remove 
                                                              #the same number of sections as the last removed trial
                    if to_remove in combinations(list(reversed(removeable_groups[i-1])), n_removed):
                        #if the same sections are removed in the previous group, no need to
                        continue 
                
                strategies.append([x for x in importance_order if x not in to_remove])
        
        last_removed_trial = len(removeable_groups[i])
    
    return strategies


additonal_data_dir = 'src/utils/additional_data'
additional_data_paths = {
    "section_to_header" :           os.path.join(additonal_data_dir, 'section_to_header.json'),
    "section_to_next_sections" :    os.path.join(additonal_data_dir, 'section_to_next_sections.json'),
    "section_to_starts" :           os.path.join(additonal_data_dir, 'section_to_starts.json'),    
    "all_sections_basic_ordered" :  os.path.join(additonal_data_dir, 'all_sections_basic_ordered.json'),
    "bhc_importance_order" :        os.path.join(additonal_data_dir, 'bhc_importance_oder.json'),
    "di_importance_order" :         os.path.join(additonal_data_dir, 'di_importance_oder.json')   
}

additonal_data = {}

for name, path in additional_data_paths.items():
    with open (path, "r") as f:
        additonal_data[name] = json.load(f.read())

section_to_header =             additonal_data["section_to_header"]
section_to_next_sections =      additonal_data["section_to_next_sections"]
section_to_starts =             additonal_data["section_to_starts"]
all_sections_basic_ordered =    additonal_data["all_sections_basic_ordered"]
bhc_importance_order =          additonal_data["bhc_importance_order"]
di_importance_order =           additonal_data["di_importance_order"]

removeable_bhc = {}

removeable_bhc[1] = ['allergies',
                    'family_history',
                    'social_history',
                    'past_medical_history']

removeable_bhc[2] = ['major_surgical_procedures']           + removeable_bhc[1]
removeable_bhc[3] = ['pertinent_results', 'physical_exam']  + removeable_bhc[2]
removeable_bhc[4] = ['history_of_present_illness']          + removeable_bhc[3]

bhc_strategy = generate_strategies(bhc_importance_order, removeable_bhc)

removeable_di = {}
removeable_di[1] = ['history_of_present_illness',
                    'medication_on_admission']

removeable_di[2] = ['physical_exam']            + removeable_di[1]

removeable_di[3] = ['discharge_medications',
                    'discharge_diagnosis',
                    'discharge_disposition',
                    'discharge_condition']      + removeable_di[2]

di_strategy = generate_strategies(di_importance_order, removeable_di)

def extract_section(text: str, section: str, start_idx: int, start_alias: str) -> str:
    """
    Extracts the section from the discharge text.
    Args:

    
    Returns:
        (str) extracted section
        (int
    """
    if start_alias: #put the given start into a list
        if start_alias not in section_to_starts[section]:
            #The section does not start with the given alias and hence does not exist in the text
            return ("None", start_idx, start_alias)
        starts = [start_alias]
    else: #select all possible starts
        starts = section_to_starts[section]
    
    next_sections = section_to_next_sections[section]
    section_text = ''
    extracted_until = start_idx
    next_start_alias = ''

    for s in starts:
        for ns in next_sections:
            next_headers = section_to_starts[ns]
            shortest = ''
            next_start_alias_of_shortest = ''
            for e in next_headers:
                current_list = re.findall(rf'(?:\s{{2,}}|\n+){s}(.*?)\n+{e}', text[start_idx:], re.DOTALL)

                if current_list:
                    current = current_list[0]
                    if current == '':
                        current = '\n'
                    if len(current) < len(shortest) or not shortest:
                        shortest = current
                        next_start_alias_of_shortest = e
                    
            section_text = shortest
            next_start_alias = next_start_alias_of_shortest
            if section_text:
                break
        
        if section_text:
            if section_text.replace('_', '').strip():
                indx = text[start_idx:].find(section_text)
                if indx == -1:
                    raise ValueError(f"Section {section} not found back")
                extracted_until = start_idx + indx + len(section_text)
                break
    
    section_text = section_text.strip()

    if not section_text:
        section_text = 'None'
    
    return section_text.strip(), extracted_until, next_start_alias

def extract_one_all_input_sections(text: str) -> str:
    """
        Extracts the sections from the text and returns a clean input, blanking the sections not included.
        
        Args:
            text (str): input text
            
        Returns:
            extracted_sections dict: dictionary with the extracted sections
    """
    #sub_sections = {}
    extracted_sections = {section : '' for section in section_to_next_sections.keys()}
    start_idx = 0
    start_alias = None
    for section in section_to_next_sections.keys():
        extracted_sections[section], start_idx, start_alias = extract_section(text, section, start_idx, start_alias)

    return extracted_sections
            
def extract_all_input_sections(combined_discharges: pd.DataFrame) -> pd.DataFrame:
 
    if isinstance(combined_discharges['text'], str): 
        extracted_sections = extract_one_all_input_sections(remove_bhc_di(combined_discharges['text']))      
    else:
        print("Removing BHC and DI from the discharge")
        clean_discharges = get_original_bhc_di_input(combined_discharges)
        print("Extracting sections")
        extracted_sections = clean_discharges.progress_apply(extract_one_all_input_sections)

    return pd.DataFrame(extracted_sections.tolist())

def remove_underscores(text):
    """
    When there are more than 2 underscores, we remove the extra underscores and leave a single 1.
    """
    text = re.sub(r'(_{2,})', '_', text)
    return text

def remove_enumerations(text):
    """
    When there are line returns followed by a number and a dot, we remove those.
    """
    text = re.sub(r'^(\d+[a-z]*\.\s*|\d+\.\s*)', '', text, flags=re.MULTILINE)
    return text

def treat_weird_tokens(text) -> str:
    text = text.replace('(_)', '_')
    text = text.replace('"_"', '_')
    text = text.replace('@', 'at')
    text = re.sub(r'-+', '-', text)
    return text

def treat_equals(text) -> str:
    """
    When there are more than 2,3,4,5 or equals we simply remove them.
    When there 6+ equls replace by \n
    """
    pattern = r'(?<![=])={2,5}(?![=])'
    text = re.sub(pattern, '', text)

    text = re.sub(r'={6,}', '\n', text)

    return text

def lowercase_first_letter(text) -> str:
    """
    Lowercase the first letter of words.
    """
    pattern = r'\b([A-Za-z])([a-z]*)(?![A-Z]|\.)\b'
    
    processed_text = re.sub(pattern, lambda match: match.group(1).lower() + match.group(2) if len(match.group(0)) > 1 else match.group(0), text)
    
    return processed_text

def remove_unecessary_tokens(text):
    return treat_weird_tokens(treat_equals(remove_enumerations(remove_underscores(text))))



def format_section(text, section):
    """
    Format the section to be extracted.
    """

    test_clean_text = text[:]

    for to_remove in ["_"]:
        test_clean_text = test_clean_text.replace(to_remove, "")
    
    if not test_clean_text.strip():
        text = "None"
    
    return f"{section_to_header[section]}:\n{text}\n"
    
def extract_clean_sections_and_count_tokens(raw_combined_df, sections_to_consider):
    """
    Tokenizes into sections and counts the number of tokens in each section of the text.

    """

    for section in sections_to_consider:
        if section not in section_to_next_sections.keys():
            raise ValueError(f"section {section} cannot be extracted. Choose from {list(sections_to_consider.keys())}.") 

    all_sections = extract_all_input_sections(raw_combined_df)
    
    for section in sections_to_consider:
        print(f"Formating and cleaning {section} section")
        raw_combined_df[section] = all_sections[section].progress_apply(format_section, section = section).progress_apply(remove_unecessary_tokens)
        
        print(f"Counting tokens in {section} section")
        raw_combined_df[section + '_tokens'] = raw_combined_df[section].progress_apply(get_token_count)
            
    return raw_combined_df

def construct_final_input(row, sections_to_consider, sections_selected):

    return '\n'.join([
                row[section] if section in sections_selected else '\n' + section_to_header[section] + ':\nNone\n' for section in sections_to_consider
            ])

def select_strategy(combined_df_with_sections, mode, max_length, sections_to_consider = None, return_strats_counter = False):
    """
    Selects the strategy for the preprocessing based on the mode.
    
    Args:
        df (pd.DataFrame): dataframe with the text
        mode (str): mode to preprocess the data
    
    Returns:
        (list) list of sections to include in the preprocessing
    """
    
    
    
    if mode == 'BHC':
        strategies = bhc_strategy
        if not sections_to_consider :
            sections_to_consider = bhc_strategy[0]
    elif mode == 'DI':
        strategies = di_strategy
        if not sections_to_consider :
            sections_to_consider = di_strategy[0]
  
    else:
        raise ValueError("Mode must be either 'BHC' or 'DI'.")
    
    sections_to_consider = sections_to_consider
    
    if return_strats_counter:
        strats_counter = {i: 0 for i in range(len(strategies))}

    outputs = []
    too_long = 0
    for _, row in tqdm(combined_df_with_sections.iterrows(), desc='Selecting strategy', total=len(combined_df_with_sections)):
        total_tokens = 0
        for i, select in enumerate(strategies):
            total_tokens = 0
            for section in sections_to_consider:
                if section in select:
                    total_tokens += row[section + "_tokens"]
                else:
                    total_tokens += get_token_count('\n' + section_to_header[section] + ':\nNone\n')
            if total_tokens < max_length: 
                final_select = select
                break
            if select == strategies[-1]:
                final_select = strategies[-1]
                print("no suitable strategy found")
        
        if final_select == strategies[-1] and total_tokens >= max_length:
            output = construct_final_input(row, sections_to_consider, final_select)[0:max_length]
            too_long += 1
        else:
            output = construct_final_input(row, sections_to_consider, final_select)
        outputs.append(output)
        if return_strats_counter:
            strats_counter[i] += 1
        
    print(f'Number of rows that exceed the maximum length: {too_long}/{len(combined_df_with_sections)}')
    if return_strats_counter:
        return outputs, strats_counter
    else:
        return outputs
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Preprocess the data.')
    parser.add_argument('--discharge_path', type=str, help='Path to the discharge file.', required = True)
    parser.add_argument('--discharge_target_path', type=str, help='Path to the discharge target file.', required = True)
    parser.add_argument('--output_path', type=str, help='Path to save the preprocessed file.', required = True)
    parser.add_argument('--max_tokens', type=int, help='Maximum number of tokens in the input.', default = None, required=False)
    parser.add_argument('--mode', type=str, required = True, help='Whether to preprocess for BHC or DI generation')
    parser.add_argument('--prompt_path', type=str, help='Path to the prompt file.', required = True)
    parser.add_argument('--generated_bhc_path', type=str, help='Path to the generated BHC file.', required = False, default=None)
    parser.add_argument('--truncation_strategy', type=str, help='Truncation strategy to use.', required = True)
    parser.add_argument('--sections_to_consider', type=str, help='List of sections to consider for preprocessing', required = False, default=None)
    parser.add_argument('--ablation_strategies_path', type=str, help='psth to list of strategies to try for ablation study', required = False, default=None)
    parser.add_argument('--nb_samples', type=int, help='List of samples to use for ablation strategie', required = False, default=None)

    args = parser.parse_args()


    if args.mode not in ['BHC', 'DI']:
        raise ValueError("Mode must be either 'BHC' or 'DI'.")
    
    #print(f"{args.output_path}{args.mode.lower()}_strat_.jsonl")

    if args.truncation_strategy not in ['sections', 'samples', 'ablation', 'rouge']:
        raise ValueError("Truncation strategy must be either 'sections' or 'samples'.")

    discharges_df = load_data(args.discharge_path)
    discharges_target_df = load_data(args.discharge_target_path)

    combined_discharges = build_combined_discharge(discharges_df, discharges_target_df)    

    # if the output directory does not exist, create it (root directory is ../../) from this file
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
    output_path = os.path.join(root_dir, args.output_path)
    output_dir = os.path.dirname(output_path)
    print(f"Output directory: {output_dir}")
    if not os.path.exists(output_dir):
        print(f"Creating output directory at {output_dir}")
        os.makedirs(output_dir, exist_ok=True)
        
    if args.truncation_strategy == 'rouge':
        sections_to_consider = args.sections_to_consider.split(',')
        print(f"sections considered: {sections_to_consider}")
        all_sections = extract_all_input_sections(combined_discharges)
        for section in sections_to_consider:
            print(f"Formating and cleaning {section} section")
            combined_discharges[section] = all_sections[section].progress_apply(remove_unecessary_tokens)
        # remove the "text" column
        combined_discharges = combined_discharges.drop(columns=['text'])
        # save the combined_discharges dataframe
        save_data(combined_discharges, output_path)
        print(f"Saved the preprocessed data at {output_path}")
        # exit the program
        sys.exit(0)

    if args.truncation_strategy == 'ablation':
        ablation_strategies_df = pd.read_json(args.ablation_strategies_path, lines=True)
        ablation_strategies = dict(zip(ablation_strategies_df['strat_idx'], ablation_strategies_df['strat']))

        

    sections_to_consider = args.sections_to_consider.split(',') if args.truncation_strategy in ['sections', 'samples'] \
                else list(set([section for strat in list(ablation_strategies_df['strat']) for section in strat]))
        
    sections_to_consider = [section for section in all_sections_basic_ordered if section in sections_to_consider]

    print(f"sections considered: {sections_to_consider}")

    try:
        prompt = load_data(args.prompt_path, type='json')            
    except FileNotFoundError:
        raise FileNotFoundError(f"Prompt file not found at {args.prompt_path}.")

    assert len(prompt) == 1 and isinstance(prompt[0][0], str), "Prompt file must be a list of strings with a single element."

    if args.mode == 'BHC':

        combined_discharges_with_section_and_counts = extract_clean_sections_and_count_tokens(combined_discharges, sections_to_consider)
        
        if args.truncation_strategy == 'sections':
            processed_bhc_input = select_strategy(combined_discharges_with_section_and_counts, mode='BHC', max_length=args.max_tokens, sections_to_consider=sections_to_consider)
            filtered_combined_discharges = combined_discharges_with_section_and_counts        
        
        elif args.truncation_strategy in ['samples', 'ablation']:
            print("Constructing Final input")
            combined_discharges_with_section_and_counts['final_input'] = combined_discharges_with_section_and_counts\
                    .progress_apply(lambda x: construct_final_input(x, sections_to_consider, sections_to_consider), axis=1)
            
            combined_discharges_with_section_and_counts['final_input_tokens'] = combined_discharges_with_section_and_counts[[f"{section}_tokens" for section in sections_to_consider]].sum(axis=1) + len(sections_to_consider)

            if args.truncation_strategy == 'samples':  
                print("Counting token in brief hospital Course")
                combined_discharges_with_section_and_counts['brief_hospital_course_tokens'] = combined_discharges_with_section_and_counts['brief_hospital_course'].progress_apply(get_token_count)

                combined_discharges_with_section_and_counts['total_tokens']\
                        = combined_discharges_with_section_and_counts['final_input_tokens'] \
                            + combined_discharges_with_section_and_counts['brief_hospital_course_tokens']
                
                filtered_combined_discharges = combined_discharges_with_section_and_counts[combined_discharges_with_section_and_counts['total_tokens'] <= args.max_tokens].reset_index(drop=True)

                processed_bhc_input = filtered_combined_discharges['final_input']
                print(f"{processed_bhc_input.shape[0]} samples ramining after selecting samples with less than {args.max_tokens} tokens (input + outptut).")

            if args.truncation_strategy == 'ablation':
                
                filtered_combined_discharges = combined_discharges_with_section_and_counts[combined_discharges_with_section_and_counts['final_input_tokens'] <= args.max_tokens].reset_index(drop=True)
                print(f"{filtered_combined_discharges.shape[0]} samples ramining after selecting samples with less than {args.max_tokens} tokens (input only).")

                nb_samples = int(args.nb_samples)
                print(f"Selecting {nb_samples} samples for ablation study")
                filtered_combined_discharges = filtered_combined_discharges.sample(n=nb_samples, random_state=42).reset_index(drop=True)
                for id, strat in ablation_strategies.items():
                    print(f"Selecting strategy {id}: {strat}")
                    processed_bhc_input = filtered_combined_discharges.progress_apply(lambda x: construct_final_input(x, sections_to_consider, strat), axis=1)
                    in_out = pd.DataFrame()
                    in_out['idx'] = filtered_combined_discharges['hadm_id']
                    in_out['prompt'] = processed_bhc_input
                    in_out['prompt'] = in_out['prompt'].progress_apply(lambda x: prompt[0][0].format(x))
                    in_out['reference'] = filtered_combined_discharges['brief_hospital_course']
                    mode = args.mode.lower()
                    save_data(in_out, f"{output_path}{mode}_strat_{id}.jsonl")

        if args.truncation_strategy != 'ablation':
            processed_bhc_input = pd.Series(processed_bhc_input)
            in_out = pd.DataFrame()
            in_out['idx'] = filtered_combined_discharges['hadm_id']
            in_out['prompt'] = processed_bhc_input
            in_out['reference'] = filtered_combined_discharges['brief_hospital_course']

    elif args.mode == 'DI':
        
        combined_discharges_with_section_and_counts = extract_clean_sections_and_count_tokens(combined_discharges, [section for section in sections_to_consider if section != 'brief_hospital_course'])

        if 'brief_hospital_course' in sections_to_consider:
            if args.generated_bhc_path:
                print("Using generated BHC as part of input")
                generated_bhc = load_data(args.generated_bhc_path, type='csv')
                
                combined_discharges_with_section_and_counts['brief_hospital_course'] = 'Brief Hospital Course:\n' + generated_bhc['generated'] + '\n'
            else:
                print("Using original gold BHC as part of input")
                combined_discharges_with_section_and_counts['brief_hospital_course'] = 'Brief Hospital Course:\n' + combined_discharges['brief_hospital_course'] + '\n'
            
            print("Cleaning BHC as input")
            combined_discharges_with_section_and_counts['brief_hospital_course'] = combined_discharges_with_section_and_counts['brief_hospital_course'].progress_apply(remove_unecessary_tokens)

            print("Counting tokens in BHC")
            combined_discharges_with_section_and_counts['brief_hospital_course_tokens'] = combined_discharges_with_section_and_counts['brief_hospital_course'].progress_apply(get_token_count)
        


        if args.truncation_strategy == 'sections':

            processed_di_input = select_strategy(combined_discharges_with_section_and_counts, mode='DI', max_length=args.max_tokens, sections_to_consider=sections_to_consider)
            filtered_combined_discharges = combined_discharges_with_section_and_counts

        elif args.truncation_strategy in ['samples', 'ablation']:
            print("Constructing Final input")
            combined_discharges_with_section_and_counts['final_input'] = combined_discharges_with_section_and_counts\
                    .progress_apply(lambda x: construct_final_input(x, sections_to_consider, sections_to_consider), axis=1)

            combined_discharges_with_section_and_counts['final_input_tokens'] = combined_discharges_with_section_and_counts[[f"{section}_tokens" for section in sections_to_consider]].sum(axis=1)\
                                                             + len(section_to_header) + 1
            
            if args.truncation_strategy == 'samples':            
                
                print("Counting token in discharge instructions")
                combined_discharges_with_section_and_counts['discharge_instructions_tokens'] = combined_discharges_with_section_and_counts['discharge_instructions'].progress_apply(get_token_count)

                combined_discharges_with_section_and_counts['total_tokens']\
                        = combined_discharges_with_section_and_counts['final_input_tokens'] \
                            + combined_discharges_with_section_and_counts['discharge_instructions_tokens']
                
                filtered_combined_discharges = combined_discharges_with_section_and_counts[combined_discharges_with_section_and_counts['total_tokens'] <= args.max_tokens].reset_index(drop=True)

                processed_di_input = filtered_combined_discharges['final_input']
                print(f"{processed_di_input.shape[0]} samples ramining after selecting samples with less than {args.max_tokens} tokens (input + outptut).")

            if args.truncation_strategy == 'ablation':
                

                filtered_combined_discharges = combined_discharges_with_section_and_counts[combined_discharges_with_section_and_counts['final_input_tokens'] <= args.max_tokens].reset_index(drop=True)
                print(f"{filtered_combined_discharges.shape[0]} samples ramining after selecting samples with less than {args.max_tokens} tokens (input only).")

                nb_samples = int(args.nb_samples)
                print(f"Selecting {nb_samples} samples for ablation study")
                filtered_combined_discharges = filtered_combined_discharges.sample(n=nb_samples, random_state=42).reset_index(drop=True)
                for id, strat in ablation_strategies.items():
                    print(f"Selecting strategy {id}: {strat}")
                    processed_di_input = filtered_combined_discharges.progress_apply(lambda x: construct_final_input(x, sections_to_consider, strat), axis=1)
                    in_out = pd.DataFrame()
                    in_out['idx'] = filtered_combined_discharges['hadm_id']
                    in_out['prompt'] = processed_di_input
                    in_out['prompt'] = in_out['prompt'].progress_apply(lambda x: prompt[0][0].format(x))
                    in_out['reference'] = filtered_combined_discharges['discharge_instructions']
                    mode = args.mode.lower()
                    save_data(in_out, f"{output_path}{mode}_strat_{id}.jsonl")
        
        if args.truncation_strategy != 'ablation':
            processed_di_input = pd.Series(processed_di_input)
            in_out = pd.DataFrame()
            in_out['idx'] = filtered_combined_discharges['hadm_id']
            in_out['prompt'] = processed_di_input
            in_out['reference'] = filtered_combined_discharges['discharge_instructions']
    
    
    if args.truncation_strategy != 'ablation':

        in_out['prompt'] = in_out['prompt'].progress_apply(lambda x: prompt[0][0].format(x))
    
        print(f'Number of rows: {len(in_out)}')
        #print(f"Max tokens: {in_out['prompt'].progress_apply(get_token_count).max()}")
        save_data(in_out, output_path)