'''
ICD-9: 530, 531, 532, 533, 534, 535, 536, 537, 538, 539, 555, 556, 557, 558, 562, 578
ICD-10: K20, K21, K22, K23, K24, K25, K26, K27, K28, K29, K30, K31, K50, K51, K52, K57, K80, K81, K82, K92

Reference:
[1] Arcadian et al, "List of ICD-9 codes", https://en.wikipedia.org/wiki/List_of_ICD-9_codes
[2] World Health Organization, "International Statistical Classification of Diseases and Related Health Problems 10th Revision", The global standard for diagnostic health information, https://icd.who.int/browse10/2019/en
'''

import pandas as pd

# Define the file paths
input_file = "./MIMIC-iv/mimic-iv-2.2/hosp/diagnoses_icd.csv/diagnoses_icd_seq1_subj1_brain_v4.csv"
output_file = "./MIMIC-iv/mimic-iv-2.2/hosp/diagnoses_icd.csv/diagnoses_icd_seq1_subj1_brain_v5.csv"

# Define the ICD codes we want to filter

# Read the CSV file
df = pd.read_csv(input_file)

icd9_53_mask = (df['icd_version'] == 9) & (df['icd_code'].str.startswith('53'))  #
icd9_555_mask = (df['icd_version'] == 9) & (df['icd_code'].str.startswith('555'))  # 
icd9_556_mask = (df['icd_version'] == 9) & (df['icd_code'].str.startswith('556'))  # 
icd9_557_mask = (df['icd_version'] == 9) & (df['icd_code'].str.startswith('557'))  # 
icd9_558_mask = (df['icd_version'] == 9) & (df['icd_code'].str.startswith('558'))  #
icd9_562_mask = (df['icd_version'] == 9) & (df['icd_code'].str.startswith('562'))  #
icd9_578_mask = (df['icd_version'] == 9) & (df['icd_code'].str.startswith('578'))  # 

icd10_k20_mask = (df['icd_version'] == 10) & (df['icd_code'].str.startswith('K20'))  # 
icd10_k30_mask = (df['icd_version'] == 10) & (df['icd_code'].str.startswith('K30'))  # 
icd10_k31_mask = (df['icd_version'] == 10) & (df['icd_code'].str.startswith('K31'))  # 
icd10_k50_mask = (df['icd_version'] == 10) & (df['icd_code'].str.startswith('K50'))  # 
icd10_k51_mask = (df['icd_version'] == 10) & (df['icd_code'].str.startswith('K51'))  # 
icd10_k52_mask = (df['icd_version'] == 10) & (df['icd_code'].str.startswith('K52'))  # 
icd10_k57_mask = (df['icd_version'] == 10) & (df['icd_code'].str.startswith('K57'))  # 
icd10_k80_mask = (df['icd_version'] == 10) & (df['icd_code'].str.startswith('K80'))  # 
icd10_k81_mask = (df['icd_version'] == 10) & (df['icd_code'].str.startswith('K81'))  # 
icd10_k82_mask = (df['icd_version'] == 10) & (df['icd_code'].str.startswith('K82'))  # 
icd10_k92_mask = (df['icd_version'] == 10) & (df['icd_code'].str.startswith('K92'))  # 


category1_mask = ((df['icd_version'] == 9) & (df['icd_code'].str.startswith('53'))) | \
                ((df['icd_version'] == 10) & (df['icd_code'].str.startswith(('K2', 'K30', 'K31'))))


category2_mask = ((df['icd_version'] == 9) & (df['icd_code'].str.startswith(('555', '556', '557', '558')))) | \
                ((df['icd_version'] == 10) & (df['icd_code'].str.startswith(('K50', 'K51', 'K52'))))


category3_mask = ((df['icd_version'] == 9) & (df['icd_code'].str.startswith('562'))) | \
                ((df['icd_version'] == 10) & (df['icd_code'].str.startswith('K57')))


category4_mask = (df['icd_version'] == 10) & (df['icd_code'].str.startswith(('K80', 'K81', 'K82')))


category5_mask = ((df['icd_version'] == 9) & (df['icd_code'].str.startswith('578'))) | \
                ((df['icd_version'] == 10) & (df['icd_code'].str.startswith('K92')))

# Instead of creating new dataframe, update labels in existing data
endoscope_diagnoses = df.copy()

# Only update null labels where category masks match
null_labels = endoscope_diagnoses['label'].isnull()
endoscope_diagnoses.loc[null_labels & category1_mask, 'label'] = 4  
endoscope_diagnoses.loc[null_labels & category2_mask, 'label'] = 5  
endoscope_diagnoses.loc[null_labels & category3_mask, 'label'] = 6  
endoscope_diagnoses.loc[null_labels & category4_mask, 'label'] = 7  
endoscope_diagnoses.loc[null_labels & category5_mask, 'label'] = 8  

# Print verification info
print(f"Total rows: {len(endoscope_diagnoses)}")
print(f"Rows with labels: {endoscope_diagnoses['label'].notna().sum()}")
print(f"Rows without labels: {endoscope_diagnoses['label'].isna().sum()}")

# Save updated data
endoscope_diagnoses.to_csv(output_file, index=False)