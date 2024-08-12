import streamlit as st
import pandas as pd

discharges_df = pd.read_csv('../data/train/discharge.csv.gz', compression='gzip')
radiology_df = pd.read_csv('../data/train/radiology.csv.gz', compression='gzip')

st.set_page_config(layout="wide")
st.title('Discharge Data')

# select the discharge idx
idx = st.selectbox('Select a discharge idx', discharges_df['hadm_id'])

# two columns
col1, col2 = st.columns(2)



# independently scrollable columns
with col1:
    # display the 'text' of the discharge data
    st.write(discharges_df.loc[discharges_df['hadm_id'] == idx, 'text'].values[0])
    
with col2:
    # display the related radiology data
    radiology_idx = st.selectbox('Select a radiology idx', radiology_df[radiology_df['hadm_id'] == idx]['note_id'])

    col2.write(radiology_df.loc[radiology_df['note_id'] == radiology_idx].iloc[0].drop('text'))
    col2.write(radiology_df.loc[radiology_df['note_id'] == radiology_idx, 'text'].values[0])