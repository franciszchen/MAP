import pandas as pd

# Read the files
diagnoses_file = "./MIMIC-iv/mimic-iv-2.2/hosp/diagnoses_icd.csv/diagnoses_icd.csv"
brain_v1_file = "./MIMIC-iv/mimic-iv-2.2/hosp/diagnoses_icd.csv/diagnoses_icd_seq1_subj1_brain_v1.csv"
output_file = "./MIMIC-iv/mimic-iv-2.2/hosp/diagnoses_icd.csv/diagnoses_icd_seq1_subj1_brain_v2.csv"

# Read data
df_diagnoses = pd.read_csv(diagnoses_file)
df_brain_v1 = pd.read_csv(brain_v1_file)

# Get unique subject_id and hadm_id pairs from brain_v1
id_pairs = df_brain_v1[['subject_id', 'hadm_id']].drop_duplicates()

# Filter diagnoses for matching pairs and seq_num 2 or 3
additional_rows = df_diagnoses[
    df_diagnoses['subject_id'].isin(id_pairs['subject_id']) & 
    df_diagnoses['hadm_id'].isin(id_pairs['hadm_id']) & 
    df_diagnoses['seq_num'].isin([2, 3])
]

# Combine original brain_v1 data with additional rows
df_combined = pd.concat([df_brain_v1, additional_rows], ignore_index=True)

# Sort by subject_id, hadm_id, and seq_num
df_combined = df_combined.sort_values(['subject_id', 'hadm_id', 'seq_num'])

# Save the combined data
df_combined.to_csv(output_file, index=False)

# Print some statistics
print(f"Original brain v1 rows: {len(df_brain_v1)}")
print(f"Additional rows added: {len(additional_rows)}")
print(f"Total rows in new file: {len(df_combined)}")
