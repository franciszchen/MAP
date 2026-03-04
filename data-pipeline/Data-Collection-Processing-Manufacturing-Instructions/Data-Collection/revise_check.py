import pandas as pd

def check_nulls(df, step_name):
    print(f"\n=== Null Value Check for {step_name} ===")
    print("\nNull counts per column:")
    print(df.isnull().sum())
    print(f"\nPercentage of null values per column:")
    print((df.isnull().sum() / len(df) * 100).round(2))
    print(f"\nTotal rows: {len(df)}")
    if df.isnull().any().any():
        print("\nColumns with null values:", df.columns[df.isnull().any()].tolist())

# Read and analyze data
input_file = "./MIMIC-iv/mimic-iv-2.2/hosp/diagnoses_icd.csv/diagnoses_icd.csv"
df = pd.read_csv(input_file)

# Show statistics for each step
check_nulls(df, "Original Data")
df_seq1 = df[df['seq_num'] == 1]
check_nulls(df_seq1, "After seq_num filter")
df_seq1_unique = df_seq1.drop_duplicates(subset=['subject_id'], keep='first')
check_nulls(df_seq1_unique, "After unique subject_id filter")
