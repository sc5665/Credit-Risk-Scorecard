"""
Explore the Credit Risk dataset (ARFF format) from Downloads/dataset_.

ARFF files store metadata (@ATTRIBUTE lines) separately from the actual
data (@DATA section), so we parse the attribute names ourselves and hand
pandas just the CSV rows that follow @DATA.
"""

from pathlib import Path

import pandas as pd

DATASET_PATH = Path(__file__).parent / "dataset_"


def load_dataset(path: Path = DATASET_PATH) -> pd.DataFrame:
    columns = []
    data_start = None

    with open(path, "r") as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.upper().startswith("@ATTRIBUTE"):
            columns.append(stripped.split()[1])
        elif stripped.upper().startswith("@DATA"):
            data_start = i + 1
            break

    df = pd.read_csv(path, skiprows=data_start, header=None, names=columns)
    return df


if __name__ == "__main__":
    df = load_dataset()

    print(f"Shape: {df.shape}")
    print()
    print(df.dtypes)
    print()
    print(df.head())
    print()
    print(df.describe())
    print()
    print("Missing values per column:")
    print(df.isna().sum())
    print()
    print("loan_status value counts:")
    print(df["loan_status"].value_counts())

    
