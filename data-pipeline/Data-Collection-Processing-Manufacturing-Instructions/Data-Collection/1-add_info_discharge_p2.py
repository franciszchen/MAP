import pandas as pd
import re

# Define file paths
brain_diagnoses_file = "./MIMIC-iv/mimic-iv-2.2/hosp/diagnoses_icd.csv/diagnoses_icd_seq1_subj1_endoscope_v1.csv"

icd_reference_file = "./MIMIC-iv/mimic-iv-2.2/icu/icustays.csv/discharge.csv"

# Read the CSV files
brain_df = pd.read_csv(brain_diagnoses_file)
icd_ref_df = pd.read_csv(icd_reference_file)

def extract_physical_exam(text):
    if pd.isna(text):
        return ''
    
    # Match different variations of Physical Exam headers
    patterns = [
        r'PHYSICAL EXAMINATION:(.*?)(?:(?:PERTINENT RESULTS|LABORATORY DATA|HOSPITAL COURSE|MEDICATIONS|ALLERGIES|Discharge|$))',
        r'Physical Exam:(.*?)(?:(?:PERTINENT RESULTS|LABORATORY DATA|HOSPITAL COURSE|MEDICATIONS|ALLERGIES|Discharge|$))',
        r'Physical Examination:(.*?)(?:(?:PERTINENT RESULTS|LABORATORY DATA|HOSPITAL COURSE|MEDICATIONS|ALLERGIES|Discharge|$))'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
    
    return ''

# First merge rows with same subject_id and hadm_id
brain_df = brain_df.groupby(['subject_id', 'hadm_id']).first().reset_index()

# Group and merge text fields from discharge data
grouped_df = icd_ref_df.groupby(['subject_id', 'hadm_id'])['text'].apply(
    lambda x: ' '.join(filter(pd.notna, x))
).reset_index()

# Extract Physical Exam section
grouped_df['Physical Exam'] = grouped_df['text'].apply(extract_physical_exam)

# Merge with brain_df
merged_df = pd.merge(
    brain_df,
    grouped_df[['subject_id', 'hadm_id', 'Physical Exam']],
    on=['subject_id', 'hadm_id'],
    how='left'
)

# Save the updated dataframe
merged_df.to_csv(brain_diagnoses_file, index=False)

