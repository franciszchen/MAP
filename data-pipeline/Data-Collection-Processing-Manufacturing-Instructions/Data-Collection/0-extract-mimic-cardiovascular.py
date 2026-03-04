import pandas as pd

'''
ICD-9: 410-414，401-405，428，440-448，451-459
ICD-10: I20-I25，I10-I15，I50，I70-I79，I80-I89

Reference:
[1] Arcadian et al, "List of ICD-9 codes", https://en.wikipedia.org/wiki/List_of_ICD-9_codes
[2] World Health Organization, "International Statistical Classification of Diseases and Related Health Problems 10th Revision", The global standard for diagnostic health information, https://icd.who.int/browse10/2019/en
'''

# Define file paths
input_file = "./MIMIC-iv/mimic-iv-2.2/hosp/diagnoses_icd.csv/diagnoses_icd_seq1_subj1.csv"
output_file = "./MIMIC-iv/mimic-iv-2.2/hosp/diagnoses_icd.csv/diagnoses_icd_seq1_subj1_cardiovascular.csv"

# Read the CSV file
df = pd.read_csv(input_file)

# Create masks for ICD-9 codes
icd9_410_414_mask = (df['icd_version'] == 9) & (df['icd_code'].str.startswith(('410', '411', '412', '413', '414')))
icd9_401_405_mask = (df['icd_version'] == 9) & (df['icd_code'].str.startswith(('401', '402', '403', '404', '405')))
icd9_428_mask = (df['icd_version'] == 9) & (df['icd_code'].str.startswith('428'))
icd9_440_448_mask = (df['icd_version'] == 9) & (df['icd_code'].str.startswith(('440', '441', '442', '443', '444', '445', '446', '447', '448')))
icd9_451_459_mask = (df['icd_version'] == 9) & (df['icd_code'].str.startswith(('451', '452', '453', '454', '455', '456', '457', '458', '459')))

# Create masks for ICD-10 codes
icd10_i20_i25_mask = (df['icd_version'] == 10) & (df['icd_code'].str.startswith(('I20', 'I21', 'I22', 'I23', 'I24', 'I25')))
icd10_i10_i15_mask = (df['icd_version'] == 10) & (df['icd_code'].str.startswith(('I10', 'I11', 'I12', 'I13', 'I14', 'I15')))
icd10_i50_mask = (df['icd_version'] == 10) & (df['icd_code'].str.startswith('I50'))
icd10_i70_i79_mask = (df['icd_version'] == 10) & (df['icd_code'].str.startswith(('I70', 'I71', 'I72', 'I73', 'I74', 'I75', 'I76', 'I77', 'I78', 'I79')))
icd10_i80_i89_mask = (df['icd_version'] == 10) & (df['icd_code'].str.startswith(('I80', 'I81', 'I82', 'I83', 'I84', 'I85', 'I86', 'I87', 'I88', 'I89')))

# Combine all masks
combined_mask = (
    icd9_410_414_mask | icd9_401_405_mask | icd9_428_mask | icd9_440_448_mask | icd9_451_459_mask |
    icd10_i20_i25_mask | icd10_i10_i15_mask | icd10_i50_mask | icd10_i70_i79_mask | icd10_i80_i89_mask
)

# Filter data and save to new file
cardiovascular_diagnoses = df[combined_mask]
cardiovascular_diagnoses.to_csv(output_file, index=False)

# Print statistics
print(f"Total input rows: {len(df)}")
print(f"Filtered rows: {len(cardiovascular_diagnoses)}")
