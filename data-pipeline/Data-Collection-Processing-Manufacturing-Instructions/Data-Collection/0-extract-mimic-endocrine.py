import pandas as pd
'''
ICD-9: 250，240-246, 255, 252, 270-279
ICD-10: E10-E14, E00-E07, E20-E35, E20-E21, E70-E88

Reference:
[1] Arcadian et al, "List of ICD-9 codes", https://en.wikipedia.org/wiki/List_of_ICD-9_codes
[2] World Health Organization, "International Statistical Classification of Diseases and Related Health Problems 10th Revision", The global standard for diagnostic health information, https://icd.who.int/browse10/2019/en
'''

input_file = "./MIMIC-iv/mimic-iv-2.2/hosp/diagnoses_icd.csv/diagnoses_icd_seq1_subj1.csv"
output_file = "./MIMIC-iv/mimic-iv-2.2/hosp/diagnoses_icd.csv/diagnoses_endocrine.csv"

df = pd.read_csv(input_file)

# ICD-9 masks
icd9_250 = (df['icd_version'] == 9) & (df['icd_code'].str.startswith('250'))
icd9_240_246 = (df['icd_version'] == 9) & (df['icd_code'].str.startswith(tuple(str(i) for i in range(240, 247))))
icd9_255 = (df['icd_version'] == 9) & (df['icd_code'].str.startswith('255'))
icd9_252 = (df['icd_version'] == 9) & (df['icd_code'].str.startswith('252'))
icd9_270_279 = (df['icd_version'] == 9) & (df['icd_code'].str.startswith(tuple(str(i) for i in range(270, 280))))

# ICD-10 masks
icd10_e10_e14 = (df['icd_version'] == 10) & (df['icd_code'].str.startswith(tuple(f'E1{i}' for i in range(0, 5))))
icd10_e00_e07 = (df['icd_version'] == 10) & (df['icd_code'].str.startswith(tuple(f'E0{i}' for i in range(0, 8))))
icd10_e20_e35 = (df['icd_version'] == 10) & (df['icd_code'].str.startswith(tuple(f'E2{i}' for i in range(0, 10))) | df['icd_code'].str.startswith(tuple(f'E3{i}' for i in range(0, 6))))
icd10_e70_e88 = (df['icd_version'] == 10) & (df['icd_code'].str.startswith(tuple(f'E{str(i)}' for i in range(70, 89))))

mask = (
    icd9_250 | icd9_240_246 | icd9_255 | icd9_252 | icd9_270_279 |
    icd10_e10_e14 | icd10_e00_e07 | icd10_e20_e35 | icd10_e70_e88
)

df_out = df[mask]
df_out.to_csv(output_file, index=False)
