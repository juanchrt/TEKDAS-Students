from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent

def find_dataset():
    candidates = [
        BASE_DIR / "dataset.csv",
        BASE_DIR / "Dataset.csv",
        BASE_DIR / "data.csv",
        BASE_DIR / "student_data.csv",
    ]
    for p in candidates:
        if p.exists():
            return p
    csvs = list(BASE_DIR.glob("*.csv"))
    return csvs[0] if csvs else None

path = find_dataset()

if path is None:
    raise FileNotFoundError(
        "Dataset CSV tidak ditemukan. Letakkan dataset.csv di folder StudentDOA."
    )

df = pd.read_csv(path)

print("=" * 60)
print("DATA PREVIEW - STUDENT DROPOUT ANALYSIS")
print("=" * 60)
print(f"File   : {path.name}")
print(f"Shape  : {df.shape}")
print("\nKolom:")
for i, col in enumerate(df.columns, 1):
    print(f"{i:>3}. {col}")

print("\n5 baris pertama:")
print(df.head().to_string())

print("\nMissing values:")
print(df.isna().sum().to_string())

print("\nTipe data:")
print(df.dtypes.to_string())

for target in ["Target", "target", "Status", "status", "Dropout", "dropout"]:
    if target in df.columns:
        print(f"\nDistribusi target ({target}):")
        print(df[target].value_counts(dropna=False).to_string())
        break
