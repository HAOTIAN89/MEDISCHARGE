import pandas as pd
import argparse
import os

def format_generations(bhc_df, di_df):
    """
    Format the BHC and DI dataframes into generated and reference dataframes with the following columns:
    - hadm_id
    - discharge_instructions
    - brief_hospital_course
    
    Parameters:
    - bhc_df: (pd.DataFrame) Dataframe with the brief hospital course data
    - di_df: (pd.DataFrame) Dataframe with the discharge instructions data
    
    """

    bhc_df.rename(columns={"generated": "generated_bhc", 
                           "gold": "gold_bhc"}, inplace=True)
    di_df.rename(columns={"generated": "generated_di",
                            "gold": "gold_di"}, inplace=True) 
    
    # Merge the two dataframes
    merged_df = pd.merge(bhc_df, di_df, left_index=True, right_index=True)
    
    # drop index and name it hadm_id
    merged_df.reset_index(inplace=True)
    merged_df.rename(columns={"idx": "hadm_id"}, inplace=True)
    
    # Create the reference and generated dataframes
    reference = merged_df[["hadm_id", "gold_di", "gold_bhc"]]
    generated = merged_df[["hadm_id", "generated_di", "generated_bhc"]]
    
    # Rename the columns
    reference.rename(columns={"gold_di": "discharge_instructions",
                              "gold_bhc": "brief_hospital_course"}, inplace=True)
    
    generated.rename(columns={"generated_di": "discharge_instructions",
                                "generated_bhc": "brief_hospital_course"}, inplace=True)
    
    return reference, generated
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Preprocess the data.')
    parser.add_argument('--bhc_path', type=str, help='Path to the brief hospital course data')
    parser.add_argument('--di_path', type=str, help='Path to the discharge instructions data')
    parser.add_argument('--output_path', type=str, help='Path to save the formatted data')
    parser.add_argument('--output_name', type=str, help='Name of the output file')
    
    args = parser.parse_args()
    
    bhc_df = pd.read_csv(args.bhc_path, index_col=0)
    di_df = pd.read_csv(args.di_path, index_col=0)
    
    reference, generated = format_generations(bhc_df, di_df)
    
    reference.to_csv(os.path.join(args.output_path, args.output_name + "_reference.csv"), index=False)
    generated.to_csv(os.path.join(args.output_path, args.output_name + "_generated.csv"), index=False)
