'''
ICD-9: 460-466, 490-496，480-486, 510-519, 162
ICD-10: J00-J06, J40-J44, J12-J18, J90-J94, C34

Reference:
[1] Arcadian et al, "List of ICD-9 codes", https://en.wikipedia.org/wiki/List_of_ICD-9_codes
[2] World Health Organization, "International Statistical Classification of Diseases and Related Health Problems 10th Revision", The global standard for diagnostic health information, https://icd.who.int/browse10/2019/en
'''


import pandas as pd

input_file = "./MIMIC-iv/mimic-iv-2.2/hosp/diagnoses_icd.csv/diagnoses_icd_seq1_subj1.csv"
output_file = "./MIMIC-iv/mimic-iv-2.2/hosp/diagnoses_icd.csv/diagnoses_respiratory.csv"

df = pd.read_csv(input_file)

# ICD-9 masks
icd9_460_466 = (df['icd_version'] == 9) & (df['icd_code'].str.startswith(tuple(str(i) for i in range(460, 467))))
icd9_490_496 = (df['icd_version'] == 9) & (df['icd_code'].str.startswith(tuple(str(i) for i in range(490, 497))))
icd9_480_486 = (df['icd_version'] == 9) & (df['icd_code'].str.startswith(tuple(str(i) for i in range(480, 487))))
icd9_510_519 = (df['icd_version'] == 9) & (df['icd_code'].str.startswith(tuple(str(i) for i in range(510, 520))))
icd9_162 = (df['icd_version'] == 9) & (df['icd_code'].str.startswith('162'))

# ICD-10 masks
icd10_j00_j06 = (df['icd_version'] == 10) & (df['icd_code'].str.startswith(tuple(f'J0{i}' for i in range(0, 7))))
icd10_j40_j44 = (df['icd_version'] == 10) & (df['icd_code'].str.startswith(tuple(f'J4{i}' for i in range(0, 5))))
icd10_j12_j18 = (df['icd_version'] == 10) & (df['icd_code'].str.startswith(tuple(f'J1{i}' for i in range(2, 9))))
icd10_j90_j94 = (df['icd_version'] == 10) & (df['icd_code'].str.startswith(tuple(f'J9{i}' for i in range(0, 5))))
icd10_c34 = (df['icd_version'] == 10) & (df['icd_code'].str.startswith('C34'))

mask = (
    icd9_460_466 | icd9_490_496 | icd9_480_486 | icd9_510_519 | icd9_162 |
    icd10_j00_j06 | icd10_j40_j44 | icd10_j12_j18 | icd10_j90_j94 | icd10_c34
)

df_out = df[mask]
df_out.to_csv(output_file, index=False)
