import pandas as pd
import regex as re
import argparse
from tqdm import tqdm
import os
import sys
tqdm.pandas()
from itertools import combinations

current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(os.path.dirname(current_file_path))
sys.path.append(parent_directory)
from utils.token_count import get_token_count


# ----------------------- Preprocessing utilities ----------------------- #

def load_data(file_path: str, type='gzip') -> pd.DataFrame:
    """Loads the data from the file_path."""
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
    """Saves the data to the file_path."""
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
    combined_discharge_train_df = pd.merge(discharges[['hadm_id', 'text']],
                                discharges_target[['hadm_id', 'discharge_instructions', 'brief_hospital_course']],
                                on='hadm_id',
                                how='inner')
    
    return combined_discharge_train_df

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
    
    
    
    if di_title.lower() not in raw_discharge.lower():
        print(f"The DI title: {di_title} is not in the discharge")
        print(f"The raw discharge is: {raw_discharge}")
        raise ValueError
    
    clean_discharge = raw_discharge.replace(di_title, '').replace(di, '')

    if mode == 'bhc':
        if bhc not in raw_discharge:
            print(f"The BHC: {bhc} is not in the discharge")
            print(f"The raw discharge is: {raw_discharge}")
            raise ValueError
        
        if bhc_title.lower() not in raw_discharge.lower():
            print(f"The BHC title: {bhc_title} is not in the discharge")
            print(f"The raw discharge is: {raw_discharge}")
            raise ValueError

        clean_discharge = clean_discharge.replace(bhc_title, '').replace(bhc, '')

    return clean_discharge

def get_original_bhc_di_input(combined_discharges: pd.DataFrame, mode ='bhc') -> pd.DataFrame:
    original_bhc_input = combined_discharges.progress_apply(lambda x: remove_bhc_di(x['text'], x['brief_hospital_course'], x['discharge_instructions'], mode=mode), axis=1)
    
    return original_bhc_input

feature_to_header = {
    'sex': 'Sex',
    'service': 'Service',
    'allergies': 'Allergies',
    'chief_complaint': 'Chief Complaint',
    'major_surgical_procedures': 'Major Surgical or Invasive Procedure',
    'history_of_present_illness': 'History of Present Illness',
    'past_medical_history': 'Past Medical History',
    'social_history': 'Social History',
    'family_history': 'Family History',
    'physical_exam': 'Physical Exam',
    'pertinent_results': 'Pertinent Results',
    'medication_on_admission': 'Medications on Admission',
    'discharge_medications': 'Discharge Medications',
    'discharge_disposition': 'Discharge Disposition',
    'facility': 'Facility',
    'discharge_diagnosis': 'Discharge Diagnosis',
    'discharge_condition': 'Discharge Condition',
    'brief_hospital_course': 'Brief Hospital Course'
}

section_to_starts = {
    'sex' : ['Sex:'],

    'service' : ['Service:'],

    'allergies' : ['Allergies:'],

    'chief_complaint' : ['Attending:.*?\n \nChief Complaint:',
                         'Attending:.*?\n \n___ Complaint:',
                         '___ Complaint:'],

    'major_surgical_procedures' : ['Major Surgical or Invasive Procedure:',
                                   '___ Surgical or Invasive Procedure:',
                                   'Major Surgical ___ Invasive Procedure:',
                                   'Major ___ or Invasive Procedure:'],

    'history_of_present_illness' : ['History of Present Illness:',
                                    'HISTORY OF PRESENT ILLNESS:',
                                    'HISTORY OF THE PRESENTING ILLNESS:',
                                    '___ of Present Illness:',
                                    'HPI:'],

    'past_medical_history' : ['Past Medical History:'],

    'social_history' : ['Social History:',
                        '___ History:'],

    'family_history' : ['Family history:',
                        'Family History:'],

    'physical_exam' : [ 'Physical ___:',
                        'Physical Exam:',
                        'Physical ___ exam:',
                        'Physical ___ Exam:',
                        '___ Exam:',
                        'Physical ___ PE ___:',
                        'Physical ___ physical exam\nPhysical exam:',
                        'Physical ___ Physical Exam\nExam:',
                        'Physical ___ Physical Exam:',
                        'Physical ___ physical exam:',
                        'Physical ___ PHYSICAL EXAM:',
                        'Physical ___ PE',
                        'Physical ___ PE:'
                        ],

    'pertinent_results' : ['Pertinent Results:'],

    'medication_on_admission' : ['Medications on Admission:',
                                'Medications on admission:',
                                '___ on admission:',
                                '___ on Admission:',
                                '___ on ___:'],

    'discharge_medications' : ['Discharge Medications:',
                                'Discharge medications:',
                                '___ Medications:',
                                '___ medications:'],

    'discharge_disposition' : ['Discharge Disposition:',
                               '___ Disposition:',
                               'Discharge disposition:',
                               '___ disposition:'],

    'facility' : ['Facility:'],

    'discharge_diagnosis' : ['Discharge Diagnosis:',
                             '___ Diagnosis:',
                             '___ischarge Diagnosis:',
                             '___:'],

    'discharge_condition' : ['Discharge Condition:',
                             '___ Condition:',
                             'Discharge ___:',
                             '___:',],

    'end' : ['Followup Instructions:',
             ' Followup Instructions:']
}

section_to_next_section = {
    'sex' : ['service','allergies'],
    
    'service' : ['allergies'],
    
    'allergies' : ['chief_complaint',
                   'major_surgical_procedures',
                   'history_of_present_illness',
                   'past_medical_history',
                    'social_history',
                    'family_history',
                    'physical_exam',
                   'pertinent_results'],
    
    'chief_complaint' : ['major_surgical_procedures',
                         'history_of_present_illness'],
    
    'major_surgical_procedures' : ['history_of_present_illness',
                                   'past_medical_history',
                                   'social_history',
                                   'family_history',
                                   'physical_exam',
                                   'pertinent_results',
                                   'discharge_medications',
                                   'discharge_disposition',],
    
    'history_of_present_illness' : ['past_medical_history',
                                    'social_history',
                                    'family_history',
                                    'physical_exam',
                                    'pertinent_results',
                                    'medication_on_admission',
                                    'discharge_medications'],
    
    'past_medical_history' : ['social_history',
                              'family_history',
                                'physical_exam',
                                'pertinent_results',
                                'medication_on_admission',
                                'discharge_medications',
                                'discharge_disposition',
                                'facility',
                                'discharge_diagnosis',
                                'discharge_condition'],
    
    
    'social_history' : ['family_history',
                        'physical_exam',
                        'pertinent_results',
                        'medication_on_admission',
                        'discharge_medications'],
    
    'family_history' : ['physical_exam',
                        'pertinent_results',
                        'medication_on_admission',
                        'discharge_medications',
                        'discharge_disposition'],
    
    'physical_exam' : ['pertinent_results',
                        'medication_on_admission',
                        'discharge_medications',
                        'discharge_disposition'],
    
    'pertinent_results' : ['medication_on_admission',
                            'discharge_medications',
                            'discharge_disposition',
                            'facility',
                            'discharge_diagnosis'],                    
    
    'medication_on_admission' : ['discharge_medications',
                                'discharge_disposition',
                                'discharge_diagnosis'],
    
    'discharge_medications' : ['discharge_disposition',
                                'facility',
                                'discharge_diagnosis',
                                'discharge_condition'],
    
    'discharge_disposition' : ['facility',
                                'discharge_diagnosis'],

    
    'facility' : ['discharge_diagnosis',
                'discharge_condition'],
    
    'discharge_diagnosis' : ['discharge_condition','end'],
    
    'discharge_condition' : ['end']
}

starts_to_section = {start: section for section, starts in section_to_starts.items() for start in starts}

#concat all starts and ends keys into one set
all_starts = [start for starts in section_to_starts.values() for start in starts]

def extract_section(text: str, section: str, start_idx: int, start_alias: str) -> str:
    """
    Extracts the section from the text.
    
    Args:
        text (str): input text
        section (str): section to extract
        start_idx (int): index where to start extracting
        start_alias (str): alias of the start of the section
    
    Returns:
        (str) extracted section
    """
    if start_alias:
        if start_alias not in section_to_starts[section]:
            return ("None", start_idx, start_alias)#, {})
        starts = [start_alias]
    else:
        starts = section_to_starts[section]
    
    next_sections = section_to_next_section[section]
    section_text = ''
    extracted_until = start_idx
    next_start_alias = ''
    #sub_sections = {}
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
    

        '''aliases_in_section = [alias_ for alias_ in all_starts if (alias_ in section_text and alias_ not in section_to_starts[section])]
        if aliases_in_section:
            start_alias_ = aliases_in_section[0]
            start_idx_ = section_text.find(start_alias_)
            for section_ in section_to_next_section.keys():
                    sub_section, start_idx_, start_alias_, _  = extract_section(section_text, section_, start_idx_, start_alias_)
                    if sub_section not in ['___', 'None', 'as above', 'as bellow', '']:
                        sub_sections[section_] = sub_section'''
        
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
    
    return section_text.strip(), extracted_until, next_start_alias #,sub_sections

def extract_one_all_input_features(text: str) -> str:
    """
        Extracts the features from the text and returns a clean input, blanking the features not included.
        
        Args:
            text (str): input text
            
        Returns:
            extracted_features dict: dictionary with the extracted features
    """
    #sub_sections = {}
    extracted_features = {section : '' for section in section_to_next_section.keys()}
    start_idx = 0
    start_alias = None
    for section in section_to_next_section.keys():
        extracted_features[section], start_idx, start_alias = extract_section(text, section, start_idx, start_alias)

    return extracted_features
            
def extract_all_input_features(combined_discharges: pd.DataFrame) -> pd.DataFrame:
 
    if isinstance(combined_discharges['text'], str): 
        extracted_features = extract_one_all_input_features(remove_bhc_di(combined_discharges['text']))      
    else:
        print("Removing BHC and DI from the discharge")
        clean_discharges = get_original_bhc_di_input(combined_discharges)
        print("Extracting features")
        extracted_features = clean_discharges.progress_apply(extract_one_all_input_features)

    return pd.DataFrame(extracted_features.tolist())

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

def generate_strategies(importance_order, removeable_groups):
    
    strategies = [importance_order]
    last_removed_trial = 0 
    for i in removeable_groups:
        for n_removed in range(1, len(removeable_groups[i])+1):
            for to_remove in combinations(list(reversed(removeable_groups[i])), n_removed):
                if i > 1 and n_removed <= last_removed_trial:
                    if to_remove in combinations(list(reversed(removeable_groups[i-1])), n_removed):
                        continue 
                
                strategies.append([x for x in importance_order if x not in to_remove])
        
        last_removed_trial = len(removeable_groups[i])
    
    return strategies

bhc_importance_order = ['sex',
                        'service',
                        'chief_complaint',
                        'history_of_present_illness',
                        'pertinent_results',
                        'physical_exam',
                        'major_surgical_procedures',
                        'past_medical_history',
                        'allergies',
                        'social_history',
                        'family_history']

removeable_bhc = {}

removeable_bhc[1] = bhc_importance_order[9:]

removeable_bhc[2] = ['major_surgical_procedures','past_medical_history','medication_on_admission'] + removeable_bhc[1] 
removeable_bhc[3] = ['pertinent_results', 'physical_exam'] + removeable_bhc[2]
removeable_bhc[4] = ['history_of_present_illness', 'chief_complaint'] + removeable_bhc[3]

bhc_strategy = generate_strategies(bhc_importance_order, removeable_bhc)
last_removed_trial = 0 

di_importance_order = ['brief_hospital_course',
                        'discharge_medications',
                        'discharge_diagnosis',
                        'discharge_disposition',
                        'discharge_condition',
                        'medication_on_admission']

removeable_di = {}
removeable_di[1] = di_importance_order[5:]

removeable_di[2] = ['discharge_medications',
                        'discharge_disposition',
                        'discharge_diagnosis',
                        'discharge_condition'] + removeable_di[1]

di_strategy = generate_strategies(di_importance_order, removeable_di)

def format_section(text, section):
    """
    Format the section to be extracted.
    """

    test_clean_text = text[:]

    for to_remove in ["_"]:
        test_clean_text = test_clean_text.replace(to_remove, "")
    
    if not test_clean_text.strip():
        text = "None"
    
    return f"{feature_to_header[section]}:\n{text}\n"
    
def extract_clean_sections_and_count_tokens(raw_combined_df, sections_to_consider):
    """
    Tokenizes into sections and counts the number of tokens in each section of the text.

    """

    for feature in sections_to_consider:
        if feature not in section_to_next_section.keys():
            raise ValueError(f"Feature {feature} cannot be extracted. Choose from {list(sections_to_consider.keys())}.") 

    all_features = extract_all_input_features(raw_combined_df)
    
    for section in sections_to_consider:
        print(f"Formating and cleaning {section} section")
        raw_combined_df[section] = all_features[section].progress_apply(format_section, section = section).progress_apply(remove_unecessary_tokens)
        
        print(f"Counting tokens in {section} section")
        raw_combined_df[section + '_tokens'] = raw_combined_df[section].progress_apply(get_token_count)
            
    return raw_combined_df

def construct_final_input(row, features_to_consider, features_selected):

    return '\n'.join([
                row[section] if section in features_selected else '\n' + feature_to_header[section] + ':\nNone\n' for section in features_to_consider
            ])

def select_strategy(combined_df_with_sections, mode, max_length, features_to_consider):
    """
    Selects the strategy for the preprocessing based on the mode.
    
    Args:
        df (pd.DataFrame): dataframe with the text
        mode (str): mode to preprocess the data
    
    Returns:
        (list) list of features to include in the preprocessing
    """
    
    sections_to_consider = features_to_consider
    
    if mode == 'BHC':
        strategies = bhc_strategy
    elif mode == 'DI':
        strategies = di_strategy
  
    else:
        raise ValueError("Mode must be either 'BHC' or 'DI'.")
    
    outputs = []
    too_long = 0
    for _, row in tqdm(combined_df_with_sections.iterrows(), desc='Selecting strategy', total=len(combined_df_with_sections)):
        total_tokens = 0
        for select in strategies:
            total_tokens = 0
            for section in sections_to_consider:
                if section in select:
                    total_tokens += row[section + "_tokens"]
                else:
                    total_tokens += get_token_count('\n' + feature_to_header[section] + ':\nNone\n')
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
        
    print(f'Number of rows that exceed the maximum length: {too_long}/{len(combined_df_with_sections)}')
    return outputs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Preprocess the data.')
    parser.add_argument('--discharge_path', type=str, help='Path to the discharge file.', required = True)
    parser.add_argument('--discharge_target_path', type=str, help='Path to the discharge target file.', required = True)
    parser.add_argument('--output_path', type=str, help='Path to save the preprocessed file.', required = True)
    parser.add_argument('--max_tokens', type=int, help='Maximum number of tokens in the input.', default = None, required=False)
    parser.add_argument('--mode', type=str, required = True, help='Whether to preprocess for BHC or DI generation')
    parser.add_argument('--features_to_exclude', type=str, help='Features to exclude from the preprocessing', required  = False, default='')
    parser.add_argument('--prompt_path', type=str, help='Path to the prompt file.', required = True)
    parser.add_argument('--generated_bhc_path', type=str, help='Path to the generated BHC file.', required = False, default=None)
    parser.add_argument('--truncation_strategy', type=str, help='Truncation strategy to use.', required = True)

    args = parser.parse_args()

    if args.mode not in ['BHC', 'DI']:
        raise ValueError("Mode must be either 'BHC' or 'DI'.")
    
    if args.truncation_strategy not in ['sections', 'samples']:
        raise ValueError("Truncation strategy must be either 'sections' or 'samples'.")

    discharges_df = load_data(args.discharge_path)
    discharges_target_df = load_data(args.discharge_target_path)

    combined_discharges = build_combined_discharge(discharges_df, discharges_target_df)

    in_out = pd.DataFrame()
    
    features_to_exclude = args.features_to_exclude.split(',') if args.features_to_exclude else []

    # if the output directory does not exist, create it (root directory is ../../) from this file
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
    output_path = os.path.join(root_dir, args.output_path)
    output_dir = os.path.dirname(output_path)
    print(f"Output directory: {output_dir}")
    if not os.path.exists(output_dir):
        print(f"Creating output directory at {output_dir}")
        os.makedirs(output_dir, exist_ok=True)

    if args.mode == 'BHC':
        features_to_consider = [
                'sex',
                'service',
                'allergies',
                'chief_complaint',
                'major_surgical_procedures',
                'history_of_present_illness',
                'past_medical_history',
                'social_history',
                'family_history',
                'pertinent_results',
                'physical_exam',
            ]
        
        combined_discharges_with_section_and_counts = extract_clean_sections_and_count_tokens(combined_discharges, features_to_consider)
        
        if args.truncation_strategy == 'sections':
            processed_bhc_input = select_strategy(combined_discharges_with_section_and_counts, mode='BHC', max_length=args.max_tokens, features_to_consider=features_to_consider)
            filtered_combined_discharges = combined_discharges_with_section_and_counts        
        
        elif args.truncation_strategy == 'samples':
            print("Constructing Final input")
            combined_discharges_with_section_and_counts['final_input'] = combined_discharges_with_section_and_counts\
                    .progress_apply(lambda x: construct_final_input(x, features_to_consider, features_to_consider), axis=1)
            
            print("Counting token in brief hospital Course")
            combined_discharges_with_section_and_counts['brief_hospital_course_tokens'] = combined_discharges_with_section_and_counts['brief_hospital_course'].progress_apply(get_token_count)

            combined_discharges_with_section_and_counts['final_input_tokens'] = combined_discharges_with_section_and_counts[[f"{section}_tokens" for section in features_to_consider]].sum(axis=1) + len(features_to_consider)

            
            combined_discharges_with_section_and_counts['total_tokens']\
                     = combined_discharges_with_section_and_counts['final_input_tokens'] \
                        + combined_discharges_with_section_and_counts['brief_hospital_course_tokens']
            
            filtered_combined_discharges = combined_discharges_with_section_and_counts[combined_discharges_with_section_and_counts['total_tokens'] <= args.max_tokens].reset_index(drop=True)

            processed_bhc_input = filtered_combined_discharges['final_input']
            print(f"{processed_bhc_input.shape[0]} samples ramining after selecting samples with less than {args.max_tokens} tokens (input + outptut).")

        processed_bhc_input = pd.Series(processed_bhc_input)
        in_out['idx'] = filtered_combined_discharges['hadm_id']
        in_out['prompt'] = processed_bhc_input
        in_out['reference'] = filtered_combined_discharges['brief_hospital_course']

    elif args.mode == 'DI':
        original_di_input = combined_discharges['text']
        features_to_consider = [
                'medication_on_admission',
                'discharge_medications',
                'discharge_disposition',
                'discharge_diagnosis',
                'discharge_condition',
            ]
        
        combined_discharges_with_section_and_counts = extract_clean_sections_and_count_tokens(combined_discharges, features_to_consider)

        features_to_consider = ['brief_hospital_course'] + features_to_consider

        if args.generated_bhc_path:
            print("Using generated BHC as part of input")
            generated_bhc = load_data(args.generated_bhc_path)
            
            combined_discharges_with_section_and_counts['brief_hospital_course'] = 'Brief Hospital Course:\n' + generated_bhc['generated'] + '\n'
        else:
            print("Using original gold BHC as part of input")
            combined_discharges_with_section_and_counts['brief_hospital_course'] = 'Brief Hospital Course:\n' + combined_discharges['brief_hospital_course'] + '\n'
        
        print("Cleaning BHC as input")
        combined_discharges_with_section_and_counts['brief_hospital_course'] = combined_discharges_with_section_and_counts['brief_hospital_course'].progress_apply(remove_unecessary_tokens)

        print("Counting tokens in BHC")
        combined_discharges_with_section_and_counts['brief_hospital_course_tokens'] = combined_discharges_with_section_and_counts['brief_hospital_course'].progress_apply(get_token_count)
        


        if args.truncation_strategy == 'sections':

            processed_di_input = select_strategy(combined_discharges_with_section_and_counts, mode='DI', max_length=args.max_tokens, features_to_consider=features_to_consider)
            filtered_combined_discharges = combined_discharges_with_section_and_counts

        elif args.truncation_strategy == 'samples':
            print("Constructing Final input")
            combined_discharges_with_section_and_counts['final_input'] = combined_discharges_with_section_and_counts\
                    .progress_apply(lambda x: construct_final_input(x, features_to_consider, features_to_consider), axis=1)
            
            print("Counting token in discharge instructions")
            combined_discharges_with_section_and_counts['discharge_instructions_tokens'] = combined_discharges_with_section_and_counts['discharge_instructions'].progress_apply(get_token_count)

            combined_discharges_with_section_and_counts['final_input_tokens'] = combined_discharges_with_section_and_counts[[f"{section}_tokens" for section in features_to_consider]].sum(axis=1)\
                                                             + len(feature_to_header) + 1
            
            combined_discharges_with_section_and_counts['total_tokens']\
                     = combined_discharges_with_section_and_counts['final_input_tokens'] \
                        + combined_discharges_with_section_and_counts['discharge_instructions_tokens']
            
            filtered_combined_discharges = combined_discharges_with_section_and_counts[combined_discharges_with_section_and_counts['total_tokens'] <= args.max_tokens].reset_index(drop=True)

            processed_di_input = filtered_combined_discharges['final_input']
            print(f"{processed_di_input.shape[0]} samples ramining after selecting samples with less than {args.max_tokens} tokens (input + outptut).")
        
        processed_di_input = pd.Series(processed_di_input)
        in_out['idx'] = filtered_combined_discharges['hadm_id']
        in_out['prompt'] = processed_di_input
        in_out['reference'] = filtered_combined_discharges['discharge_instructions']
    

    try:
        prompt = load_data(args.prompt_path, type='json')            
    except FileNotFoundError:
        raise FileNotFoundError(f"Prompt file not found at {args.prompt_path}.")

    assert len(prompt) == 1 and isinstance(prompt[0][0], str), "Prompt file must be a list of strings with a single element."
    
    in_out['prompt'] = in_out['prompt'].progress_apply(lambda x: prompt[0][0].format(x))
   
    print(f'Number of rows: {len(in_out)}')
    #print('Max tokens:', in_out['prompt'].progress_apply(get_token_count).max())

    save_data(in_out, output_path)