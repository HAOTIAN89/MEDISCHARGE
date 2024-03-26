import pandas as pd
import regex as re

def load_data(file_path: str) -> pd.DataFrame:
    """Loads the data from the file_path."""
    return pd.read_csv(file_path,compression='gzip')

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

def extract_one_clean_input(text, features_to_include: list) -> str:
    return ''.join([feature_to_function[feature](text) for feature in features_to_include])

def extract_clean_inputs(combined_discharges: pd.DataFrame, features_to_include: list) -> pd.DataFrame:
    for feature in features_to_include:
        if feature not in feature_to_function:
            raise ValueError(f"Feature {feature} cannot be extracted. Choose from {list(feature_to_function.keys)}.")    
    extracted_features = combined_discharges['text'].apply(extract_one_clean_input, features_to_include=features_to_include)

    return extracted_features



def remove_underscores(text):
    """
    When there are more than 2 underscores, we remove the extra underscores.
    """
    text = re.sub(r'(_{2,})', '_', text)
    return text

def treat_equals(text):
    """
    When there are more than 2,3,4,5 or equals simply remove them.
    When there 6+ equls replace by \n
    """
    pattern = r'(?<![=])={2,5}(?![=])'
    text = re.sub(pattern, '', text)

    text = re.sub(r'={6,}', '\n', text)

    return text

