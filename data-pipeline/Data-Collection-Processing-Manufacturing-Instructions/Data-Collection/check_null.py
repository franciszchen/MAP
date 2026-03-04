import pandas as pd

# Read the file
file_path = "./MIMIC-iv/mimic-iv-2.2/hosp/diagnoses_icd.csv/diagnoses_icd_seq1_subj1_endoscope_v1.csv"
df = pd.read_csv(file_path)

# Print overall null statistics
print("=== Null Value Statistics ===")
print("\nNull counts per column:")
print(df.isnull().sum())
print("\nPercentage of null values per column:")
print((df.isnull().sum() / len(df) * 100).round(2))

# Display row indices with null values for each column
print("\n=== Rows with Null Values by Column ===")
for column in df.columns:
    null_indices = df[df[column].isnull()].index.tolist()
    if null_indices:
        print(f"\n{column}:")
        print(f"Number of nulls: {len(null_indices)}")
        print(f"Row indices: {null_indices}")