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
    data.to_csv(file_path, index=False,compression='gzip')


def build_combined_discharge(discharges: pd.DataFrame, discharges_target: pd.DataFrame) -> pd.DataFrame:
    combined_discharge_train_df = pd.merge(discharges[['hadm_id', 'text']],
                                discharges_target[['hadm_id', 'discharge_instructions', 'brief_hospital_course']],
                                on='hadm_id',
                                how='inner')
    
    return combined_discharge_train_df


def get_bhc_input(combined_discharges: pd.DataFrame) -> pd.DataFrame:
    original_bhc_input = combined_discharges['text'].progress_apply(
        lambda x: re.sub(r'Brief Hospital Course:\s*\n{0,2}(.*?)(?=\n\s*\n{0,2}\s*[A-Z_]+[^\n:]+:\n)',
                        '',
                        x,
                        flags=re.DOTALL))
    
    original_bhc_input = original_bhc_input.progress_apply(
            lambda x: re.sub(r'Discharge Instructions:\n(.*?)Followup Instruction',
                             '',
                             x,
                             flags=re.DOTALL))
    
    return original_bhc_input



def extract_sex(text):
    sex = re.findall(r'Sex:\s*\n{0,2}(.*?)\nService:', text, re.DOTALL)
    sex_text = "Sex: \n" + ''.join(sex)
    return sex_text

def extract_allergies(text):
    allergies = re.findall(r'Allergies:\s*\n{0,2}(.*?)\nAttending:', text, re.DOTALL)
    allergies_text = "\nAllergies: \n" + ''.join(allergies)
    return allergies_text

def extract_chief_complaint(text):
    chief_complaint = re.findall(r'Chief Complaint:\s*\n{0,2}(.*?)(?=\n\s*\n{0,2}\s*[A-Z_]+[^\n:]+:\n)', text, re.DOTALL)
    chief_complaint_text = "\nChief Complaint: \n" + ''.join(chief_complaint)
    return chief_complaint_text

def extract_major_surgical_procedures(text):
    major_surgical_procedures = re.findall(r'Major Surgical or Invasive Procedure:\s*\n{0,2}(.*?)(?=\n\s*\n{0,2}\s*[A-Z_]+[^\n:]+:\n)', text, re.DOTALL)
    major_surgical_procedures_text = "\n\nMajor Surgical or Invasive Procedure: \n" + ''.join(major_surgical_procedures)
    return major_surgical_procedures_text

def extract_history_of_present_illness(text):
    history_of_present_illness = re.findall(r'History of Present Illness:\s*\n{0,2}(.*?)(?=\n\s*\n{0,2}\s*[A-Z_]+[^\n:]+:\n)', text, re.DOTALL)
    history_of_present_illness_text = "\n\nHistory of Present Illness: \n" + ''.join(history_of_present_illness)
    return history_of_present_illness_text

def extract_past_medical_history(text):
    past_medical_history = re.findall(r'Past Medical History:\s*\n{0,2}(.*?)(?=\n\s*\n{0,2}\s*[A-Z_]+[^\n:]+:\n)', text, re.DOTALL)
    past_medical_history_text = "\n\nPast Medical History: \n" + ''.join(past_medical_history)
    return past_medical_history_text

def extract_social_history(text):
    social_history = re.findall(r'Social History:\s*\n{0,2}(.*?)(?=\n\s*\n{0,2}\s*[A-Z_]+[^\n:]+:\n)', text, re.DOTALL)
    social_history_text = "\n\nSocial History: \n" + ''.join(social_history)
    return social_history_text

def extract_family_history(text):
    family_history = re.findall(r'Family History:\s*\n{0,2}(.*?)(?=\n\s*\n{0,2}\s*[A-Z_]+[^\n:]+:\n)', text, re.DOTALL)
    family_history_text = "\n\nFamily History: \n" + ''.join(family_history)
    return family_history_text

def extract_physical_exam(text):
    physical_exam = re.findall(r'Physical Exam:\s*\n{0,2}(.*?)(?=\n\s*\n{0,2}\s*[A-Z_]+[^\n:]+:\n)', text, re.DOTALL)
    physical_exam_text = "\n\nPhysical Exam: \n" + ''.join(physical_exam)
    return physical_exam_text

def extract_pertinent_results(text):
    pertinent_results = re.findall(r'Pertinent Results:\s*\n{0,2}(.*?)(?=\n\s*\n{0,2}\s*[A-Z_]+[^\n:]+:\n)', text, re.DOTALL)
    pertinent_results_text = "\n\nPertinent Results: \n" + ''.join(pertinent_results)
    return pertinent_results_text


def extract_medication_on_admission(text):
    medications_on_admission = re.findall(r'Medications on Admission:\n(.*?)Discharge Medications', text, re.DOTALL)
    medications_on_admission_text = "Medications on Admission: \n" + ''.join(medications_on_admission)
    return medications_on_admission_text

def extract_discharge_medications(text):
    discharge_medications = re.findall(r'Discharge Medications:\n(.*?)Discharge Disposition:', text, re.DOTALL)
    discharge_medications_text = "Discharge Medications: \n" + ''.join(discharge_medications)
    return discharge_medications_text

def extract_discharge_disposition(text):
    discharge_disposition = re.findall(r'Discharge Disposition:\n(.*?)Discharge Diagnosis:', text, re.DOTALL)
    discharge_disposition_text = "Discharge Disposition: \n" + ''.join(discharge_disposition)
    return discharge_disposition_text

def extract_discharge_diagnosis(text):
    discharge_diagnosis = re.findall(r'Discharge Diagnosis:\n(.*?)Discharge Condition:', text, re.DOTALL)
    discharge_diagnosis_text = "Discharge Diagnosis: \n" + ''.join(discharge_diagnosis)
    return discharge_diagnosis_text

def extract_discharge_condition(text):
    discharge_condition = re.findall(r'Discharge Condition:\n(.*?)Discharge Instructions:', text, re.DOTALL)
    discharge_condition_text = "Discharge Condition: \n" + ''.join(discharge_condition)
    return discharge_condition_text

feature_to_header = {
    'sex': 'Sex',
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
        
        # BLANK SECTION with header if feature not included
        extracted_features.append('\n' + feature_to_header[feature] + ': \n')
        
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

    if args.mode == 'BHC':
        original_bhc_input = get_bhc_input(combined_discharges)
        features_to_include = [
                'sex',
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
        processed_bhc_input = extract_clean_inputs(combined_discharges, features_to_include = [
            feature for feature in features_to_include if feature not in features_to_exclude
        ])

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
        processed_di_input = extract_clean_inputs(combined_discharges, features_to_include = [
            feature for feature in features_to_include if feature not in features_to_exclude
        ])

        clean_di_input = processed_di_input.progress_apply(remove_unecessary_tokens)
        in_out['prompt'] = clean_di_input
        in_out['reference'] = combined_discharges['discharge_instructions']
    
        # Add the generated BHC to the input prompt
        if args.generated_bhc_path:
            generated_bhc = load_data(args.generated_bhc_path)
            
            # match by idx and add the generated bhc to the in_out dataframe
            in_out['prompt'] = in_out['prompt'].progress_apply(lambda x: x + 'brief hospital course: \n' + generated_bhc[generated_bhc['idx'] == x['idx']]['generated'].values[0] + '\n')
        
        # Else add the original BHC to the prompt
        else: 
            in_out['prompt'] = in_out['prompt'] + 'brief hospital course: \n' + combined_discharges['brief_hospital_course'] + '\n'
        
    if args.prompt_path:
        try:
            prompt = load_data(args.prompt_path, type='json')            
        except FileNotFoundError:
            raise FileNotFoundError(f"Prompt file not found at {args.prompt_path}.")

        assert len(prompt) == 1 and isinstance(prompt[0][0], str), "Prompt file must be a list of strings with a single element."
        
        in_out['prompt'] = in_out['prompt'].progress_apply(lambda x: prompt[0][0].format(x))
        
    if args.max_tokens:
        in_out['token_count'] = in_out['prompt'].progress_apply(get_token_count)
        in_out = in_out[in_out['token_count'] < args.max_tokens]
        in_out.drop(columns=['token_count'], inplace=True)

    save_data(in_out, args.output_path)

