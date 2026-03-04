import pandas as pd

# Define file paths
brain_file = "./MIMIC-iv/mimic-iv-2.2/hosp/diagnoses_icd.csv/diagnoses_icd_seq1_subj1_brain.csv"
icd_ref_file = "./MIMIC-iv/mimic-iv-2.2/hosp/d_icd_diagnoses.csv/d_icd_diagnoses.csv"
output_file = "./MIMIC-iv/mimic-iv-2.2/hosp/diagnoses_icd.csv/diagnoses_icd_seq1_subj1_brain_v1.csv"

# Read data
df_brain = pd.read_csv(brain_file)
df_icd_ref = pd.read_csv(icd_ref_file)

# Print initial null count
print("Initial null count in long_title:", df_brain['long_title'].isnull().sum())

# Update missing long_title values
df_updated = pd.merge(
    df_brain,
    df_icd_ref[['icd_code', 'icd_version', 'long_title']],
    on=['icd_code', 'icd_version'],
    how='left',
    suffixes=('', '_ref')
)

# Fill missing values
mask = df_updated['long_title'].isnull()
df_updated.loc[mask, 'long_title'] = df_updated.loc[mask, 'long_title_ref']

# Drop reference column
df_updated = df_updated.drop('long_title_ref', axis=1)

# Print final null count
print("Final null count in long_title:", df_updated['long_title'].isnull().sum())

# Save updated data to new file
df_updated.to_csv(output_file, index=False)

# Print comparison
print("\nFile information:")
print(f"Original file: {brain_file}")
print(f"Updated file: {output_file}")
