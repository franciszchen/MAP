import pandas as pd
import re

# Define file paths
brain_diagnoses_file = "./MIMIC-iv/mimic-iv-2.2/hosp/diagnoses_icd.csv/diagnoses_icd_seq1_subj1_endoscope_v1.csv"

icd_reference_file = "./MIMIC-iv/mimic-iv-2.2/radiology.csv/radiology.csv"

# Read the CSV files
brain_df = pd.read_csv(brain_diagnoses_file)
icd_ref_df = pd.read_csv(icd_reference_file)

def extract_findings(text):
    if pd.isna(text):
        return ''
    
    # Remove other sections first
    sections_to_remove = [
        'EXAMINATION:', 'INDICATION:', 'TECHNIQUE:', 
        'COMPARISON:', 'IMPRESSION:', 'HISTORY:', 'CONCLUSION:'
    ]
    
    # Find FINDINGS section
    findings_pattern = r'FINDINGS:(.*?)(?:' + '|'.join(sections_to_remove) + '|$)'
    match = re.search(findings_pattern, text, re.DOTALL | re.IGNORECASE)
    
    if match:
        findings = match.group(1).strip()
        # Remove any nested sections
        for section in sections_to_remove:
            findings = re.sub(f"{section}.*", '', findings, flags=re.DOTALL | re.IGNORECASE)
        return findings
    return ''

# First merge rows with same subject_id and hadm_id
brain_df = brain_df.groupby(['subject_id', 'hadm_id']).first().reset_index()

# Group and merge text fields from radiology data
grouped_df = icd_ref_df.groupby(['subject_id', 'hadm_id'])['text'].apply(
    lambda x: ' '.join(filter(pd.notna, x))
).reset_index()

# Extract FINDINGS sections
grouped_df['text'] = grouped_df['text'].apply(extract_findings)

# Merge with brain_df
merged_df = pd.merge(
    brain_df,
    grouped_df[['subject_id', 'hadm_id', 'text']],
    on=['subject_id', 'hadm_id'],
    how='left'
)

# Save the updated dataframe with long titles
merged_df.to_csv(brain_diagnoses_file, index=False)

