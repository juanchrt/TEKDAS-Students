from pathlib import Path

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "dropout_model.pkl"

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

target_candidates = ["Target", "target", "Status", "status", "Dropout", "dropout"]
target_col = next((c for c in target_candidates if c in df.columns), None)

if target_col is None:
    raise ValueError(
        "Kolom target tidak ditemukan. Dataset Student Dropout biasanya "
        "menggunakan kolom 'Target'."
    )

df = df.dropna(subset=[target_col]).copy()

X = df.drop(columns=[target_col])
y = df[target_col].astype(str)

# ID/kolom konstan tidak membantu model dan dapat membuat input aplikasi membingungkan.
drop_cols = [c for c in X.columns if X[c].nunique(dropna=False) <= 1]
X = X.drop(columns=drop_cols)

numeric_features = X.select_dtypes(include="number").columns.tolist()
categorical_features = [c for c in X.columns if c not in numeric_features]

numeric_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipe, numeric_features),
    ("cat", categorical_pipe, categorical_features),
])

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=2000)),
])

stratify = y if y.value_counts().min() >= 2 else None

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=stratify,
)

model.fit(X_train, y_train)
pred = model.predict(X_test)

print("=" * 60)
print("STUDENT DROPOUT ML MODEL")
print("=" * 60)
print(f"Dataset : {path.name}")
print(f"Target  : {target_col}")
print(f"Train   : {len(X_train)}")
print(f"Test    : {len(X_test)}")
print(f"Accuracy: {accuracy_score(y_test, pred):.4f}")
print("\nClassification report:")
print(classification_report(y_test, pred, zero_division=0))

bundle = {
    "model": model,
    "features": X.columns.tolist(),
    "target": target_col,
    "classes": model.classes_.tolist(),
}

joblib.dump(bundle, MODEL_PATH)
print(f"\nModel tersimpan di: {MODEL_PATH}")
