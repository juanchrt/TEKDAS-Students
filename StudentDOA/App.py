import os
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "dropout_model.pkl"
DATA_PATH = BASE_DIR / "dataset.csv"

st.set_page_config(
    page_title="Student Dropout Analysis",
    page_icon="🎓",
    layout="wide",
)

@st.cache_data
def load_data(path):
    return pd.read_csv(path)

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

st.title("🎓 Student Dropout Analysis")
st.caption("Analisis data dan prediksi status mahasiswa berdasarkan dataset Student Dropout.")

dataset = find_dataset()

if dataset is None:
    st.error("Dataset CSV belum ditemukan. Letakkan dataset.csv di folder StudentDOA.")
    st.stop()

df = load_data(str(dataset))

st.sidebar.header("Menu")
page = st.sidebar.radio(
    "Pilih halaman",
    ["Dashboard", "Data Preview", "Visualisasi", "Prediksi"]
)

target_candidates = ["Target", "target", "Status", "status", "Dropout", "dropout"]
target_col = next((c for c in target_candidates if c in df.columns), None)

if page == "Dashboard":
    st.subheader("Ringkasan Dataset")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Jumlah Data", f"{len(df):,}")
    c2.metric("Jumlah Kolom", f"{len(df.columns):,}")
    c3.metric("Missing Values", f"{int(df.isna().sum().sum()):,}")
    c4.metric("Target", target_col or "Tidak ditemukan")

    st.divider()
    st.subheader("Distribusi Target")
    if target_col:
        counts = df[target_col].value_counts(dropna=False)
        st.bar_chart(counts)
        st.dataframe(
            counts.rename("Jumlah").reset_index().rename(columns={target_col: "Target"}),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.warning("Kolom target tidak ditemukan.")

elif page == "Data Preview":
    st.subheader("Data Preview")
    st.write(f"File: `{dataset.name}`")
    st.write(f"Shape: **{df.shape[0]} baris × {df.shape[1]} kolom**")

    n = st.slider("Jumlah baris yang ditampilkan", 5, min(100, len(df)), min(20, len(df)))
    st.dataframe(df.head(n), use_container_width=True)

    st.subheader("Tipe Data")
    info = pd.DataFrame({
        "Column": df.columns,
        "Data Type": [str(x) for x in df.dtypes],
        "Missing": [int(df[c].isna().sum()) for c in df.columns],
        "Unique": [int(df[c].nunique(dropna=True)) for c in df.columns],
    })
    st.dataframe(info, use_container_width=True, hide_index=True)

elif page == "Visualisasi":
    st.subheader("Visualisasi Student Dropout")

    if target_col:
        fig, ax = plt.subplots()
        df[target_col].value_counts().plot(kind="bar", ax=ax)
        ax.set_title("Distribusi Target")
        ax.set_xlabel(target_col)
        ax.set_ylabel("Jumlah")
        plt.xticks(rotation=0)
        st.pyplot(fig)
        plt.close(fig)

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if numeric_cols:
        col = st.selectbox("Pilih variabel numerik", numeric_cols)
        fig, ax = plt.subplots()
        df[col].dropna().plot(kind="hist", bins=20, ax=ax)
        ax.set_title(f"Distribusi {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("Frekuensi")
        st.pyplot(fig)
        plt.close(fig)

elif page == "Prediksi":
    st.subheader("Prediksi Student Dropout")

    if not MODEL_PATH.exists():
        st.warning(
            "Model belum tersedia. Jalankan `python Ml_model.py` terlebih dahulu "
            "dari folder StudentDOA."
        )
        st.stop()

    try:
        bundle = joblib.load(MODEL_PATH)
        model = bundle["model"]
        features = bundle["features"]
    except Exception as e:
        st.error(f"Model tidak dapat dibaca: {e}")
        st.stop()

    st.write("Masukkan nilai fitur berikut untuk mendapatkan prediksi.")

    input_data = {}
    cols = st.columns(2)

    for i, feature in enumerate(features):
        col = df[feature] if feature in df.columns else None
        with cols[i % 2]:
            if col is None:
                continue

            if pd.api.types.is_numeric_dtype(col):
                median = float(col.median()) if not col.dropna().empty else 0.0
                input_data[feature] = st.number_input(
                    feature,
                    value=median,
                    key=f"input_{i}",
                )
            else:
                values = col.dropna().astype(str).unique().tolist()
                if not values:
                    values = [""]
                input_data[feature] = st.selectbox(
                    feature,
                    values,
                    key=f"input_{i}",
                )

    if st.button("🔮 Prediksi", type="primary", use_container_width=True):
        X = pd.DataFrame([input_data], columns=features)
        try:
            prediction = model.predict(X)[0]
            st.success(f"Hasil prediksi: **{prediction}**")

            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(X)[0]
                classes = model.classes_
                prob_df = pd.DataFrame({
                    "Kelas": classes,
                    "Probabilitas": probabilities,
                })
                st.dataframe(prob_df, use_container_width=True, hide_index=True)
                st.bar_chart(prob_df.set_index("Kelas"))
        except Exception as e:
            st.error(
                "Prediksi gagal. Pastikan input dan struktur dataset sama dengan "
                f"dataset saat model dibuat.\n\nDetail: {e}"
            )

st.sidebar.markdown("---")
st.sidebar.caption("StudentDOA • Student Dropout Analysis")
