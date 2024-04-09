import sys
sys.path.append('../')
import argparse
import pandas as pd
from utils.preprocessing import load_data

test_targets_path = '../../data/test_phase_1/discharge_target.csv.gz'

def combine_files(bhc_path, di_path, output_path):
    # Load the CSV files
    bhc_df = pd.read_csv(bhc_path)
    di_df = pd.read_csv(di_path)
    
    # Rename the 'generated' column to the specified names
    bhc_df = bhc_df.rename(columns={'generated': 'brief_hospital_course'})
    di_df = di_df.rename(columns={'generated': 'discharge_instructions'})
    
    # Merge the two dataframes on the 'idx' column
    combined_df = pd.merge(bhc_df[['hadm_id', 'brief_hospital_course']], di_df[['hadm_id', 'discharge_instructions']], on='hadm_id')   
    
    # print out the length of three dataframes
    print('Length of BHC dataframe:', len(bhc_df))
    print('Length of DI dataframe:', len(di_df))
    print('Length of combined dataframe:', len(combined_df))
    
    # load the original test_targets dataframe to order the hadm_id of combined_df
    test_targets = load_data(test_targets_path)
    combined_df_ordered = combined_df.set_index('hadm_id').reindex(test_targets['hadm_id']).reset_index()
    
    # Save the combined dataframe to a new CSV file
    combined_df_ordered.to_csv(output_path, index=False)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Combine two CSV files.')
    parser.add_argument('bhc_path', type=str, help='Path to the bhc_7b_inferred CSV file')
    parser.add_argument('di_path', type=str, help='Path to the di_7b_inferred CSV file')
    parser.add_argument('output_path', type=str, help='Path to the output CSV file')

    args = parser.parse_args()

    combine_files(args.bhc_path, args.di_path, args.output_path)

