import streamlit as st
import joblib
import numpy as np
import pandas as pd
import time
from streamlit_option_menu import option_menu

# Konfigurasi Halaman (Harus di paling atas)
st.set_page_config(page_title="Sistem Deteksi Hipertensi", layout="wide")

# ==========================================
# 1. FUNGSI LOAD MODEL
# ==========================================
@st.cache_resource
def load_model():
    model = joblib.load('model_hipertensi_lr.pkl')
    scaler = joblib.load('scaler_hipertensi.pkl')
    return model, scaler
try:
    model, scaler = load_model()
except Exception as e:
    st.error(f"Error: File model/scaler tidak ditemukan! {e}")

# ==========================================
# 2. PENGATURAN SIDEBAR (MENU KIRI)
# ==========================================
with st.sidebar:
    st.image("pict/logo.png", width=300)
    st.divider()
    pilihan_menu = option_menu(
        menu_title="Menu", 
        options=["Dashboard Utama", "Prediksi Diagnosa", "Informasi Model"], 
        icons=["house", "clipboard2-pulse", "graph-up"], 
        menu_icon="cast", 
        default_index=0, 
        styles={
            "container": {"padding": "0!important", "background-color": "transparent"},
            "icon": {"color": "gray", "font-size": "18px"}, 
            "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
            "nav-link-selected": {"background-color": "#4cbcf8"}, 
        }
    )
st.sidebar.divider()
st.sidebar.caption("© 2026 - Riskina Salsabilla Bayzura")

# ==========================================
# 3. KONTEN HALAMAN UTAMA (TENGAH)
# ==========================================

# --- HALAMAN 1: DASHBOARD ---
if pilihan_menu == "Dashboard Utama":
    st.title("Selamat Datang di Sistem Deteksi Hipertensi 🩺")
    st.write("""
    Aplikasi ini merupakan hasil penelitian skripsi yang bertujuan untuk membantu tenaga medis di Puskesmas dalam memprediksi risiko penyakit hipertensi pada pasien secara cepat dan akurat.
    """)
    st.info("**Cara Penggunaan:**\n"
            "1. Buka menu **Prediksi Diagnosa** di sidebar sebelah kiri.\n"
            "2. Masukkan 5 data klinis pasien (Sistole, Diastole, Umur, Tinggi Badan, dan IMT).\n"
            "3. Klik tombol **Mulai Diagnosa** untuk melihat hasil dan rekomendasi medis.")
    
    st.image("pict/ilustrasi.jpg", caption="Ilustrasi Pemeriksaan Medis", use_container_width=True)

# --- HALAMAN 2: PREDIKSI DIAGNOSA ---
elif pilihan_menu == "Prediksi Diagnosa":
    st.title("🩺 Prediksi Risiko Hipertensi")
    st.write("Silakan masukkan data hasil pemeriksaan pasien pada kolom di bawah ini.")
    st.divider()
    
    col1, col2 = st.columns(2)
    with col1:
        sistole = st.number_input("Tekanan Sistole (mmHg)", min_value=70, max_value=250, value=120)
        diastole = st.number_input("Tekanan Diastole (mmHg)", min_value=40, max_value=150, value=80)
        umur = st.number_input("Usia (Tahun)", min_value=1, max_value=100, value=25)
    with col2:
        imt = st.number_input("Indeks Massa Tubuh (IMT)", min_value=10.0, max_value=60.0, value=22.0)
        tinggi = st.number_input("Tinggi Badan (cm)", min_value=100, max_value=250, value=160)
    
    st.divider()

    if st.button("Mulai Diagnosa", type="primary"):
        start_time = time.time()

        input_df = pd.DataFrame([[tinggi, imt, diastole, umur, sistole]], 
                                columns=['tinggi_cm', 'imt', 'diastole', 'umur', 'sistole'])
        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)
        proba = model.predict_proba(input_scaled)[0][1] 

        end_time = time.time()
        durasi = end_time - start_time
        
        st.divider()

        if prediction == 1:
            st.error("### 🔴 HASIL: TERDETEKSI HIPERTENSI")
            st.write(f"**Tingkat Probabilitas:** {proba:.2%}")
            st.warning("**Rekomendasi Tindakan:** Pasien terdeteksi risiko tinggi. Disarankan untuk segera melakukan konsultasi lanjutan dengan dokter, membatasi asupan garam, dan menjaga pola makan.")
        else:
            st.success("### 🟢 HASIL: NORMAL (BUKAN HIPERTENSI)")
            st.write(f"**Tingkat Probabilitas:** {proba:.2%}")
            st.write("**Rekomendasi Tindakan:** Kondisi pasien saat ini normal. Tetap pertahankan pola hidup sehat dan asupan nutrisi seimbang.")
        
        st.caption(f"⚡ *Waktu Proses Komputasi: {durasi:.4f} detik*")   

# --- HALAMAN 3: INFORMASI MODEL ---
elif pilihan_menu == "Informasi Model":
    st.title("📊 Informasi Performa Model")
    st.write("Sistem ini dibangun menggunakan algoritma *Logistic Regression* yang telah dioptimasi dengan seleksi fitur *Random Forest* (Skenario 4).")
    
    col_metrik1, col_metrik2, col_metrik3 = st.columns(3)
    col_metrik1.metric(label="Akurasi Model", value="91.74%")
    col_metrik2.metric(label="Presisi", value="92.85%")
    col_metrik3.metric(label="Recall", value="92.85%")
    
    st.divider()

    st.write("### Visualisasi Evaluasi Model")
    col_vis1, col_vis2 = st.columns(2)
    with col_vis1:
        st.write("**1. Confusion Matrix**")
        st.image("pict/confusion_matrix.png")

    with col_vis2:
        st.write("**2. Tingkat Kepentingan Fitur (*Feature Importance*)**")
        chart_data = pd.DataFrame(
            [0.35, 0.25, 0.20, 0.15, 0.05], 
            index=['Sistole', 'Usia', 'Diastole', 'Tinggi Badan', 'IMT'],
            columns=['Kontribusi Fitur']
        )
        st.bar_chart(chart_data)

    st.divider()

    st.write("### 🗂️ Cuplikan Dataset (*Preview*)")
    st.write("Model dilatih menggunakan dataset rekam medis pasien dengan 5 fitur klinis terpilih. Berikut adalah cuplikan 5 baris pertama:")
    
    dummy_data = pd.DataFrame({
        'sistole': [120, 140, 110, 160, 125],
        'umur': [25, 55, 22, 60, 30],
        'diastole': [80, 90, 70, 100, 85],
        'tinggi_cm': [160, 165, 155, 170, 162],
        'imt': [22.0, 27.5, 20.1, 30.2, 24.5],
        'label_hipertensi': ['Normal', 'Hipertensi', 'Normal', 'Hipertensi', 'Normal']
    })
    st.dataframe(dummy_data, use_container_width=True)
