import pandas as pd
'''
ICD-9: 710-719，720-724，725-729，730-739，800-829
ICD-10: M00-M25，M40-M54，M60-M79，M80-M99，S02, S12, S22, S32, S42, S52, S62, S72, S82, S92

Reference:
[1] Arcadian et al, "List of ICD-9 codes", https://en.wikipedia.org/wiki/List_of_ICD-9_codes
[2] World Health Organization, "International Statistical Classification of Diseases and Related Health Problems 10th Revision", The global standard for diagnostic health information, https://icd.who.int/browse10/2019/en
'''

input_file = "./MIMIC-iv/mimic-iv-2.2/hosp/diagnoses_icd.csv/diagnoses_icd_seq1_subj1.csv"
output_file = "./MIMIC-iv/mimic-iv-2.2/hosp/diagnoses_icd.csv/diagnoses_musculoskeletal.csv"

df = pd.read_csv(input_file)

# ICD-9 masks
icd9_710_719 = (df['icd_version'] == 9) & (df['icd_code'].str.startswith(tuple(str(i) for i in range(710, 720))))
icd9_720_724 = (df['icd_version'] == 9) & (df['icd_code'].str.startswith(tuple(str(i) for i in range(720, 725))))
icd9_725_729 = (df['icd_version'] == 9) & (df['icd_code'].str.startswith(tuple(str(i) for i in range(725, 730))))
icd9_730_739 = (df['icd_version'] == 9) & (df['icd_code'].str.startswith(tuple(str(i) for i in range(730, 740))))
icd9_800_829 = (df['icd_version'] == 9) & (df['icd_code'].str.startswith(tuple(str(i) for i in range(800, 830))))

# ICD-10 masks
icd10_m00_m25 = (df['icd_version'] == 10) & (df['icd_code'].str.startswith(tuple(f'M{str(i).zfill(2)}' for i in range(0, 26))))
icd10_m40_m54 = (df['icd_version'] == 10) & (df['icd_code'].str.startswith(tuple(f'M{str(i).zfill(2)}' for i in range(40, 55))))
icd10_m60_m79 = (df['icd_version'] == 10) & (df['icd_code'].str.startswith(tuple(f'M{str(i).zfill(2)}' for i in range(60, 80))))
icd10_m80_m99 = (df['icd_version'] == 10) & (df['icd_code'].str.startswith(tuple(f'M{str(i).zfill(2)}' for i in range(80, 100))))
icd10_s_codes = (df['icd_version'] == 10) & (
    df['icd_code'].str.startswith(('S02', 'S12', 'S22', 'S32', 'S42', 'S52', 'S62', 'S72', 'S82', 'S92'))
)

mask = (
    icd9_710_719 | icd9_720_724 | icd9_725_729 | icd9_730_739 | icd9_800_829 |
    icd10_m00_m25 | icd10_m40_m54 | icd10_m60_m79 | icd10_m80_m99 | icd10_s_codes
)

df_out = df[mask]
df_out.to_csv(output_file, index=False)
