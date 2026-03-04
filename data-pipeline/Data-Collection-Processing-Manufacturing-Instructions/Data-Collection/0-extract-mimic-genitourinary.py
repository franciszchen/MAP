import pandas as pd
'''
Reference:
[1] Arcadian et al, "List of ICD-9 codes", https://en.wikipedia.org/wiki/List_of_ICD-9_codes
[2] World Health Organization, "International Statistical Classification of Diseases and Related Health Problems 10th Revision", The global standard for diagnostic health information, https://icd.who.int/browse10/2019/en
'''

input_file = "./MIMIC-iv/mimic-iv-2.2/hosp/diagnoses_icd.csv/diagnoses_icd_seq1_subj1.csv"
output_file = "./MIMIC-iv/mimic-iv-2.2/hosp/diagnoses_icd.csv/diagnoses_genitourinary.csv"

df = pd.read_csv(input_file)

# ICD-9 masks
icd9_580_589 = (df['icd_version'] == 9) & (df['icd_code'].str.startswith(tuple(str(i) for i in range(580, 590))))
icd9_595_599 = (df['icd_version'] == 9) & (df['icd_code'].str.startswith(tuple(str(i) for i in range(595, 600))))
icd9_600_602 = (df['icd_version'] == 9) & (df['icd_code'].str.startswith(tuple(str(i) for i in range(600, 603))))
icd9_610_629 = (df['icd_version'] == 9) & (df['icd_code'].str.startswith(tuple(str(i) for i in range(610, 630))))
icd9_630_679 = (df['icd_version'] == 9) & (df['icd_code'].str.startswith(tuple(str(i) for i in range(630, 680))))

# ICD-10 masks
icd10_n00_n08 = (df['icd_version'] == 10) & (df['icd_code'].str.startswith(tuple(f'N0{i}' for i in range(0, 9))))
icd10_n17_n19 = (df['icd_version'] == 10) & (df['icd_code'].str.startswith(tuple(f'N1{i}' for i in range(7, 10))))
icd10_n30_n39 = (df['icd_version'] == 10) & (df['icd_code'].str.startswith(tuple(f'N3{i}' for i in range(0, 10))))
icd10_n40_n42 = (df['icd_version'] == 10) & (df['icd_code'].str.startswith(('N40', 'N41', 'N42')))
icd10_n60_n98 = (df['icd_version'] == 10) & (df['icd_code'].str.startswith(tuple(f'N{str(i).zfill(2)}' for i in range(60, 99))))
icd10_o00_o99 = (df['icd_version'] == 10) & (df['icd_code'].str.startswith(tuple(f'O{str(i).zfill(2)}' for i in range(0, 100))))

mask = (
    icd9_580_589 | icd9_595_599 | icd9_600_602 | icd9_610_629 | icd9_630_679 |
    icd10_n00_n08 | icd10_n17_n19 | icd10_n30_n39 | icd10_n40_n42 | icd10_n60_n98 | icd10_o00_o99
)

df_out = df[mask]
df_out.to_csv(output_file, index=False)