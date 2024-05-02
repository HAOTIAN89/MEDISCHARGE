import pandas as pd
import regex as re
import argparse
from tqdm import tqdm
import os
import sys
tqdm.pandas()


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


def get_bhc_input(combined_discharges: pd.DataFrame, mode ='bhc') -> pd.DataFrame:
    original_bhc_input = combined_discharges.progress_apply(lambda x: remove_bhc_di(x['text'], x['brief_hospital_course'], x['discharge_instructions'], mode=mode), axis=1)
    
    return original_bhc_input


def extract_sex(text):
    sex = re.findall(r'Sex:\s*\n{0,2}(.*?)\n(?:Service:|___:.*?\nAllergies:)', text, re.DOTALL)
    if not sex:
        print("HERE")
        print(text)
        raise ValueError("Sex not found")
    sex_text = "Sex:   " + ''.join(sex)
    return sex_text

def extract_service(text):
    service = re.findall(r'(?:Service:|Sex:.*?\n \n___:)\s*\n{0,2}(.*?)\nAllergies:', text, re.DOTALL)
    if not service:
        print(text)
        raise ValueError("Service not found")
    service_text = "\nService: " + ''.join(service)
    return service_text

def extract_allergies(text):
    allergies = re.findall(r'Allergies:\s*\n{0,2}(.*?)\n(?:Attending:|___.\n \nChief Complaint:)', text, re.DOTALL)
    if not allergies:
        print(text)
        raise ValueError("Allergies not found.")
    allergies_text = "\nAllergies: \n" + ''.join(allergies)
    return allergies_text

def extract_chief_complaint(text):
    if 'Complaint' in text:
        chief_complaint = re.findall(r'Complaint:\s*\n{0,2}(.*?)\n(?:Major Surgical or Invasive Procedure:|___ Surgical or Invasive Procedure:|Major ___ or Invasive Procedure:|___ or Invasive Procedure:)', text, re.DOTALL) 
        
        if not chief_complaint:
            if "chief complaint" in text.lower():
                print(text)
                raise ValueError("Chief Complaint not found.")
    else:
        chief_complaint = []
    #chief_complaint = re.findall(r'Chief Complaint:\s*\n{0,2}(.*?)(?=\n\s*\n{0,2}\s*[A-Z_]+[^\n:]+:\n)', text, re.DOTALL)
    chief_complaint_text = "\nChief Complaint:\n" + ''.join(chief_complaint)
    return chief_complaint_text

def extract_major_surgical_procedures(text):
    major_surgical_procedures = re.findall(r'(?:Major Surgical or Invasive Procedure:|___ Surgical or Invasive Procedure:)\s*\n{0,2}(.*?)\nHistory of Present Illness:', text, re.DOTALL)
    if not major_surgical_procedures:
        major_surgical_procedures = re.findall(r'Major Surgical or Invasive Procedure:\s*\n{0,2}(.*?)\n(?:Pertinent Results:|Brief Hospital Course:)', text, re.DOTALL)
        if not major_surgical_procedures:
            if "major surgical procedures" in text.lower():
                print(text)
                raise ValueError("Major Surgical Procedures not found.")
    major_surgical_procedures_text = "\nMajor Surgical or Invasive Procedure:\n" + ''.join(major_surgical_procedures)
    return major_surgical_procedures_text

def extract_history_of_present_illness(text): # TODO: changer le regex
    history_of_present_illness = re.findall(r'(?:History of Present Illness:|HISTORY OF PRESENT ILLNESS:)\s*\n{0,2}(.*?)\n(?:REVIEW OF SYSTEMS:|Review of sytems:|Social History:|Family History:|Physical Exam:|Brief Hospital Course:)', text, re.DOTALL)
    if not history_of_present_illness:
        history_of_present_illness = re.findall(r'History of Present Illness:\s*\n{0,2}(.*?)\nPast Medical History:', text, re.DOTALL)
    #if not found, try with capital letters
    if not history_of_present_illness:
        history_of_present_illness = re.findall(r'HISTORY OF THE PRESENTING ILLNESS:\s*\n{0,2}(.*?)\nREVIEW OF SYSTEMS:', text, re.DOTALL)
    
    if not history_of_present_illness:
        if "history of present illness" in text.lower() or "history of the presenting illness" in text.lower():
            print(text)
            raise ValueError("history of present_illness not found")

    history_of_present_illness_text = "\n\nHistory of Present Illness:\n" + ''.join(history_of_present_illness)
    return history_of_present_illness_text

def extract_review_of_systems(text):
    if 'REVIEW OF SYSTEMS' in text:
        review_of_systems = re.findall(r'REVIEW OF SYSTEMS:?\s*\n{0,2}(.*?)\n(?:Past Medical History:|Social History:)', text, re.DOTALL)

        if not review_of_systems:
            print(text)
            raise ValueError("Review of Systems not found.")
    elif 'review of systems:' in text.lower():
        review_of_systems = re.findall(r'Review of Systems:\s*\n{0,2}(.*?)\nPast Medical History:', text, re.DOTALL)  
        if not review_of_systems:
            review_of_systems = re.findall(r'Review of systems:\s*\n{0,2}(.*?)\nPast Medical History:', text, re.DOTALL)  
            if not review_of_systems:
                review_of_systems = re.findall(r'Review of sytems:\s*\n{0,2}(.*?)\nPast Medical History:', text, re.DOTALL)  
                if not review_of_systems:
                    print(text)
                    #raise ValueError("Review of Systems not found.")
    else:
        review_of_systems = []

    review_of_systems_text = "\n\nReview of Systems:\n" + ''.join(review_of_systems)
  
    return review_of_systems_text

def extract_past_medical_history(text): # TODO: changer le regex
    if 'past medical history:' in text.lower():
        past_medical_history = re.findall(r'Past Medical History:\s*\n{0,2}(.*?)\n(?:Social History:|___ History:\n___\nFamily History:|Family History:|Physical Exam:|Pertinent Results:|Brief Hospital Course:)', text, re.DOTALL)
        if not past_medical_history:
            print(text)
            raise ValueError("Past Medical History not found.")
    else:
        past_medical_history = []
    past_medical_history_text = "\n\nPast Medical History:\n" + ''.join(past_medical_history)
    return past_medical_history_text

# TODO: Past surgical history:

def extract_social_history(text):
    if 'social history' in text.lower():
        social_history = re.findall(r'Social History:\s*\n{0,2}(.*?)\n(?:Family History:|___ History:|Brief Hospital Course:)', text, re.DOTALL)
        if not social_history:
            social_history = re.findall(r'Social History:\s*\n{0,2}(.*?)\nPhysical Exam:', text, re.DOTALL)
            if not social_history:
                social_history = re.findall(r'SOCIAL HISTORY:\s*\n{0,2}(.*?)\n(?:FAMILY HISTORY:|___ History:|Brief Hospital Course:)', text, re.DOTALL)
                if not social_history:
                    print(text)
                    raise ValueError("Social History not found.")
    else:
        social_history = []
    social_history_text = "\n\nSocial History:\n" + ''.join(social_history)
    return social_history_text

def extract_family_history(text):
    if 'family history:' in text.lower():
        family_history = re.findall(r'Family History:\s*\n{0,2}(.*?)\n(?:Physical Exam:|Physical .*? PE .*?:|Physical ___:|Brief Hospital Course:)', text, re.DOTALL)
        if not family_history:
            family_history = re.findall(r'FAMILY HISTORY:\s*\n{0,2}(.*?)\n(?:Physical Exam:|Physical .*? PE .*?:|Physical ___:|Brief Hospital Course:)', text, re.DOTALL)
            if not family_history:
                print(text)
                raise ValueError("Family History not found.")
    else: 
        family_history = []
    family_history_text = "\n\nFamily History:\n" + ''.join(family_history)
    return family_history_text

def extract_physical_exam(text):
    physical_exam = re.findall(r'(?:Physical Exam:|Physical .*? PE .*?:|Physical ___:|Physical ___ exam:|Family History:.*?___ Exam:)\s*\n{0,2}(.*?)Pertinent Results:', text, re.DOTALL)
    if not physical_exam:
        physical_exam = re.findall(r'(?:Physical Exam:|Physical .*? PE .*?:)\s*\n{0,2}(.*?)Brief Hospital Course:', text, re.DOTALL)
        if not physical_exam:
            if 'physical exam ' in text.lower().split('brief hospital course:')[0] or 'physical exam:' in text.lower().split('brief hospital course:')[0]:
                print("==============")
                print(text)
                #raise ValueError("Physical Exam not found")
            
            physical_exam = []
    
    physical_exam_text = "\n\nPhysical Exam:\n" + ''.join(physical_exam)
    return physical_exam_text

def extract_pertinent_results(text):
    if 'pertinent results' in text.lower():
        pertinent_results = re.findall(r'Pertinent Results:\s*\n{0,2}(.*?)Brief Hospital Course:', text, re.DOTALL)
        if not pertinent_results:
            print(text)
            raise ValueError("Pertinent Results not found.")
    else:
        pertinent_results = []
    pertinent_results_text = "\n\nPertinent Results:\n" + ''.join(pertinent_results)
    return pertinent_results_text


def extract_medication_on_admission(text):
    medication_on_admission = re.findall(r'(?:Medications on Admission:|___ on ___:|___ on Admission:)\s*\n{0,2}(.*?)\n(?:Discharge Medications:|Discharge Disposition:)', text, re.DOTALL)

    if not medication_on_admission:
        if "medications on admission" in text.lower() or "___ on ___:" in text.lower() or "___ on admission" in text.lower() or "medications on ___" in text.lower():
            print(text)
            raise ValueError("Medications on Admission not found.")
    
    medication_on_admission_text = "Medications on Admission:\n" + ''.join(medication_on_admission)
    return medication_on_admission_text

def extract_discharge_medications(text): 
    if "discharge medications:" in text.lower():
        discharge_medications = re.findall(r'Discharge Medications:\s*\n{0,2}(.*?)\n(?:Discharge Disposition:|___ Disposition:|___:)', text, re.DOTALL)
        if not discharge_medications:
            print(text)
            raise ValueError("Discharge Medications not found...")
    else:
        discharge_medications = []
    discharge_medications_text = "Discharge Medications:\n" + ''.join(discharge_medications)
    return discharge_medications_text

def extract_discharge_disposition(text):
    discharge_disposition = re.findall(r'(?:Discharge Disposition:|Discharge Medications:.*?___:)\s*\n{0,2}(.*?)\n(?:Facility:|Discharge Diagnosis:)', text, re.DOTALL)
    if not discharge_disposition:
        if "discharge disposition:" in text.lower():
            print(text)
            raise ValueError("Discharge Disposition not found.")
    discharge_disposition_text = "Discharge Disposition:\n" + ''.join(discharge_disposition)
    return discharge_disposition_text

def extract_discharge_diagnosis(text):
    discharge_diagnosis = re.findall(r'(?:Discharge Diagnosis:|Facility:\n___\n \n___ Diagnosis:)\s*\n{0,2}(.*?)\n(?:Discharge Condition:|___ Condition:)', text, re.DOTALL)
    if not discharge_diagnosis:
        if 'discharge diagnosis' in text.lower():
            print(text)
            raise ValueError("Discharge Diagnosis not found.")
    discharge_diagnosis_text = "Discharge Diagnosis:\n" + ''.join(discharge_diagnosis)
    return discharge_diagnosis_text

def extract_discharge_condition(text):
    discharge_condition = re.findall(r'(?:Discharge Condition:|___ Condition:)\s*\n{0,2}(.*?)\nFollowup Instructions:', text, re.DOTALL)
    if not discharge_condition:
        print(text)
        raise ValueError("Discharge Condition not found.")
    discharge_condition_text = "Discharge Condition:\n" + ''.join(discharge_condition)
    return discharge_condition_text

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
    'discharge_diagnosis': 'Discharge Diagnosis',
    'discharge_condition': 'Discharge Condition'
}

feature_to_function = {
    'sex': extract_sex, #important
    'service': extract_service, 
    'allergies': extract_allergies, 
    'chief_complaint': extract_chief_complaint, #important
    'major_surgical_procedures': extract_major_surgical_procedures,
    'history_of_present_illness': extract_history_of_present_illness, #important
    'past_medical_history': extract_past_medical_history, 
    'social_history': extract_social_history, 
    'family_history': extract_family_history, 
    'physical_exam': extract_physical_exam, #maybe remove
    'pertinent_results': extract_pertinent_results,#maybe remove 
    'medication_on_admission': extract_medication_on_admission,
    'discharge_medications': extract_discharge_medications,
    'discharge_disposition': extract_discharge_disposition,
    'discharge_diagnosis': extract_discharge_diagnosis,
    'discharge_condition': extract_discharge_condition
}


def extract_one_clean_input(text: str, features_to_include: list) -> str:
    """
        Extracts the features from the text and returns a clean input, blanking the features not included.
        
        Args:
            text (str): input text
            features_to_include (list): list of features to include in the clean input
            
        Returns:
            (str) clean input
    """
    
    extracted_features = []
    for feature in feature_to_function.keys():
        if feature in features_to_include:
            extracted_features.append(feature_to_function[feature](text) + '\n')
            continue
        
        '''# BLANK SECTION with header if feature not included
            extracted_features.append('\n' + feature_to_header[feature] + ': \n')'''
        
    return ''.join(extracted_features)
            

def extract_clean_inputs(combined_discharges: pd.DataFrame, features_to_include: list):
    for feature in features_to_include:
        if feature not in feature_to_function:
            raise ValueError(f"Feature {feature} cannot be extracted. Choose from {list(feature_to_function.keys())}.")  
        
    if isinstance(combined_discharges['text'], str): 
        extracted_features = extract_one_clean_input(combined_discharges['text'], features_to_include=features_to_include)      
    else:
        extracted_features = combined_discharges['text'].progress_apply(extract_one_clean_input, features_to_include=features_to_include)

    return extracted_features


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
    return lowercase_first_letter(treat_weird_tokens(treat_equals(remove_enumerations(remove_underscores(text)))))


bhc_strategy = [
    ['sex', 'service', 'allergies', 'chief_complaint', 'major_surgical_procedures', 'history_of_present_illness', 'physical_exam', 'pertinent_results', 'past_medical_history'],
    ['sex', 'service', 'allergies', 'chief_complaint', 'major_surgical_procedures', 'history_of_present_illness', 'physical_exam', 'pertinent_results'],
    ['sex', 'service', 'allergies', 'chief_complaint', 'major_surgical_procedures', 'history_of_present_illness', 'physical_exam', 'past_medical_history'],
    ['sex', 'service', 'allergies', 'chief_complaint', 'major_surgical_procedures', 'history_of_present_illness', 'pertinent_results', 'past_medical_history'],
    ['sex', 'service', 'allergies', 'chief_complaint', 'major_surgical_procedures', 'history_of_present_illness', 'physical_exam'],
    ['sex', 'service', 'allergies', 'chief_complaint', 'major_surgical_procedures', 'history_of_present_illness', 'pertinent_results'],
    ['sex', 'service', 'allergies', 'chief_complaint', 'major_surgical_procedures', 'history_of_present_illness', 'past_medical_history'],
    ['sex', 'service', 'allergies', 'chief_complaint', 'major_surgical_procedures', 'history_of_present_illness'],
    ['sex', 'service', 'allergies', 'chief_complaint', 'history_of_present_illness'],
    ['sex', 'service', 'allergies', 'chief_complaint', 'major_surgical_procedures', 'physical_exam', 'pertinent_results', 'past_medical_history'],
    ['sex', 'service', 'allergies', 'chief_complaint', 'major_surgical_procedures', 'physical_exam', 'pertinent_results'],
    ['sex', 'service', 'allergies', 'chief_complaint', 'major_surgical_procedures', 'physical_exam', 'past_medical_history'],
    ['sex', 'service', 'allergies', 'chief_complaint', 'major_surgical_procedures', 'pertinent_results', 'past_medical_history'],
    ['sex', 'service', 'allergies', 'chief_complaint', 'major_surgical_procedures', 'physical_exam'],
    ['sex', 'service', 'allergies', 'chief_complaint', 'major_surgical_procedures', 'pertinent_results'],
    ['sex', 'service', 'allergies', 'chief_complaint', 'major_surgical_procedures', 'past_medical_history'],
    ['sex', 'service', 'allergies', 'chief_complaint', 'physical_exam', 'pertinent_results', 'past_medical_history'],
    ['sex', 'service', 'allergies', 'chief_complaint', 'physical_exam', 'pertinent_results'],
    ['sex', 'service', 'allergies', 'chief_complaint', 'physical_exam', 'past_medical_history'],
    ['sex', 'service', 'allergies', 'chief_complaint', 'pertinent_results', 'past_medical_history'],
    ['sex', 'service', 'allergies', 'chief_complaint', 'physical_exam'],
    ['sex', 'service', 'allergies', 'chief_complaint', 'pertinent_results'],
    ['sex', 'service', 'allergies', 'chief_complaint', 'past_medical_history'],
]

di_strategy = [["medication_on_admission", "discharge_medications", "discharge_disposition", "discharge_diagnosis", "discharge_condition", "brief_hospital_course"], 
            ["discharge_medications", "discharge_disposition", "discharge_diagnosis", "discharge_condition", "brief_hospital_course"],
            ["medication_on_admission", "discharge_disposition", "discharge_diagnosis", "discharge_condition", "brief_hospital_course"],
            ["discharge_medications", "brief_hospital_course"],
            ["discharge_disposition", "discharge_diagnosis", "discharge_condition", "brief_hospital_course"],
            ["discharge_diagnosis", "discharge_condition", "brief_hospital_course"],
            ['discharge_condition', "brief_hospital_course"],
            ['discharge_diagnosis', "brief_hospital_course"],
            ['medication_on_admission', "brief_hospital_course"],
            ['discharge_disposition', "brief_hospital_course"],
            ["brief_hospital_course"]
            ]

def count_tokens_per_section(df, sections):
    """
    Counts the number of tokens in each section of the text.
    
    Args:
        text (str): input text
        sections (list): list of sections to count tokens
    
    Returns:
        (dict) dictionary with the number of tokens per section
    """
    
    for section in sections:
        df[section] = extract_clean_inputs(df, features_to_include=[section]) \
            .progress_apply(remove_unecessary_tokens)
        
        df[section + '_tokens'] = df[section].progress_apply(get_token_count)
            
    return df

def select_strategy(df, mode, max_length=1548):
    """
    Selects the strategy for the preprocessing based on the mode.
    
    Args:
        df (pd.DataFrame): dataframe with the text
        mode (str): mode to preprocess the data
    
    Returns:
        (list) list of features to include in the preprocessing
    """
    
    if mode == 'BHC':
        strategies = bhc_strategy
    elif mode == 'DI':
        strategies = di_strategy
    else:
        raise ValueError("Mode must be either 'BHC' or 'DI'.")
    
    outputs = []
    too_long = 0
    for index, row in df.iterrows():
        total_tokens = 0
        for select in strategies:
            total_tokens = 0
            for section in feature_to_header:
                if section in select:
                    total_tokens += row[section + "_tokens"]
                else:
                    total_tokens += get_token_count('\n' + feature_to_header[section] + ': \n')
            if total_tokens < max_length: 
                final_select = select
                break
            if select == strategies[-1]:
                final_select = strategies[-1]
                print("no suitable strategy found")
        
        if final_select == strategies[-1] and total_tokens >= max_length:
            output = '\n'.join([
                row[section] if section in final_select else '\n' + feature_to_header[section] + ': \n' for section in feature_to_header
            ])[0:max_length]
            too_long += 1
        else:
            output = '\n'.join([
                row[section] if section in final_select else '\n' + feature_to_header[section] + ': \n' for section in feature_to_header
            ])
        outputs.append(output)
        
    print(f'Number of rows that exceed the maximum length: {too_long}/{len(df)}')
    return outputs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Preprocess the data.')
    parser.add_argument('--discharge_path', type=str, help='Path to the discharge file.')
    parser.add_argument('--discharge_target_path', type=str, help='Path to the discharge target file.')
    parser.add_argument('--output_path', type=str, help='Path to save the preprocessed file.')
    parser.add_argument('--max_tokens', type=int, help='Maximum number of tokens in the input.', default=None)
    parser.add_argument('--mode', type=str, help='Whether to preprocess for BHC or DI generation')
    parser.add_argument('--features_to_exclude', type=str, help='Features to exclude from the preprocessing', default='')
    parser.add_argument('--prompt_path', type=str, help='Path to the prompt file.', default=None)
    parser.add_argument('--generated_bhc_path', type=str, help='Path to the generated BHC file.', default=None)

    args = parser.parse_args()

    if args.mode not in ['BHC', 'DI']:
        raise ValueError("Mode must be either 'BHC' or 'DI'.")

    discharges_df = load_data(args.discharge_path)
    discharges_target_df = load_data(args.discharge_target_path)

    combined_discharges = build_combined_discharge(discharges_df, discharges_target_df)

    in_out = pd.DataFrame()
    
    in_out['idx'] = combined_discharges['hadm_id']
    
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
        original_bhc_input = get_bhc_input(combined_discharges)
        features_to_include = [
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
        combined_discharges = count_tokens_per_section(combined_discharges, sections= [
            feature for feature in features_to_include if feature not in features_to_exclude
        ])
        
        processed_bhc_input = select_strategy(combined_discharges, mode='BHC', max_length=args.max_tokens)
        processed_bhc_input = pd.Series(processed_bhc_input)

        clean_bhc_input = processed_bhc_input.progress_apply(remove_unecessary_tokens)
        in_out['prompt'] = clean_bhc_input
        in_out['reference'] = combined_discharges['brief_hospital_course']
    
    elif args.mode == 'DI':
        original_di_input = combined_discharges['text']
        features_to_include = [
                'medication_on_admission',
                'discharge_medications',
                'discharge_disposition',
                'discharge_diagnosis',
                'discharge_condition',
            ]
        
        if args.generated_bhc_path:
            generated_bhc = load_data(args.generated_bhc_path)
            
            combined_discharges['brief_hospital_course'] = 'brief hospital course: \n' + generated_bhc['generated'] + '\n'
        else:
            combined_discharges['brief_hospital_course'] = 'brief hospital course: \n' + combined_discharges['brief_hospital_course'] + '\n'
            
        combined_discharges['brief_hospital_course'] = combined_discharges['brief_hospital_course'].progress_apply(remove_unecessary_tokens)
        combined_discharges['brief_hospital_course_tokens'] = combined_discharges['brief_hospital_course'].progress_apply(get_token_count)
        
        combined_discharges = count_tokens_per_section(combined_discharges, sections=[
            feature for feature in features_to_include if feature not in features_to_exclude
        ])
            
        processed_di_input = select_strategy(combined_discharges, mode='DI', max_length=args.max_tokens)
        processed_di_input = pd.Series(processed_di_input)

        clean_di_input = processed_di_input.progress_apply(remove_unecessary_tokens)
        in_out['prompt'] = clean_di_input
        in_out['reference'] = combined_discharges['discharge_instructions']
    
    if args.prompt_path:
        try:
            prompt = load_data(args.prompt_path, type='json')            
        except FileNotFoundError:
            raise FileNotFoundError(f"Prompt file not found at {args.prompt_path}.")

        assert len(prompt) == 1 and isinstance(prompt[0][0], str), "Prompt file must be a list of strings with a single element."
        
        in_out['prompt'] = in_out['prompt'].progress_apply(lambda x: prompt[0][0].format(x))
   
    print(f'Number of rows: {len(in_out)}')
    print('Max tokens:', in_out['prompt'].progress_apply(get_token_count).max())

    save_data(in_out, output_path)