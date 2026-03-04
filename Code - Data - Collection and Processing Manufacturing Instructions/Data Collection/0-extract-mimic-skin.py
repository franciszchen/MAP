import pandas as pd
'''
Reference:
[1] Arcadian et al, "List of ICD-9 codes", https://en.wikipedia.org/wiki/List_of_ICD-9_codes
[2] World Health Organization, "International Statistical Classification of Diseases and Related Health Problems 10th Revision", The global standard for diagnostic health information, https://icd.who.int/browse10/2019/en
'''

input_file = "D:/Study/0-Data/MIMIC-iv/mimic-iv-2.2/hosp/diagnoses_icd.csv/diagnoses_icd_seq1_subj1.csv"
output_file = "D:/Study/0-Data/MIMIC-iv/mimic-iv-2.2/hosp/diagnoses_icd.csv/diagnoses_skin.csv"

df = pd.read_csv(input_file)

# ICD-9 masks
icd9_680_686 = (df['icd_version'] == 9) & (df['icd_code'].str.startswith(tuple(str(i) for i in range(680, 687))))
icd9_690_693 = (df['icd_version'] == 9) & (df['icd_code'].str.startswith(tuple(str(i) for i in range(690, 694))))
icd9_172_173 = (df['icd_version'] == 9) & (df['icd_code'].str.startswith(('172', '173')))
icd9_707 = (df['icd_version'] == 9) & (df['icd_code'].str.startswith('707'))
icd9_700_709 = (df['icd_version'] == 9) & (df['icd_code'].str.startswith(tuple(str(i) for i in range(700, 710))))

# ICD-10 masks
icd10_l00_l08 = (df['icd_version'] == 10) & (df['icd_code'].str.startswith(tuple(f'L0{i}' for i in range(0, 9))))
icd10_l20_l30 = (df['icd_version'] == 10) & (df['icd_code'].str.startswith(tuple(f'L{str(i)}' for i in range(20, 31))))
icd10_c43_c44 = (df['icd_version'] == 10) & (df['icd_code'].str.startswith(('C43', 'C44')))
icd10_l89 = (df['icd_version'] == 10) & (df['icd_code'].str.startswith('L89'))
icd10_l80_l99 = (df['icd_version'] == 10) & (df['icd_code'].str.startswith(tuple(f'L{str(i)}' for i in range(80, 100))))

mask = (
    icd9_680_686 | icd9_690_693 | icd9_172_173 | icd9_707 | icd9_700_709 |
    icd10_l00_l08 | icd10_l20_l30 | icd10_c43_c44 | icd10_l89 | icd10_l80_l99
)

df_out = df[mask]
df_out.to_csv(output_file, index=False)
