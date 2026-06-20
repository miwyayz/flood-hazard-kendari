"""
app.py — Aplikasi Streamlit: Klasifikasi Zona Rawan Banjir
Kecamatan Kendari Barat (XGBoost)
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
import folium
from streamlit_folium import st_folium
import plotly.graph_objects as go
import plotly.express as px

# ============================================================
# KONFIGURASI HALAMAN
# ============================================================
st.set_page_config(
    page_title="Prediksi Zona Rawan Banjir — Kendari Barat",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# LOAD MODEL & ARTEFAK
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def load_artifacts():
    model     = joblib.load(os.path.join(BASE_DIR, "model", "xgb_banjir_model.joblib"))
    le_target = joblib.load(os.path.join(BASE_DIR, "model", "le_target.joblib"))
    le_soil   = joblib.load(os.path.join(BASE_DIR, "model", "le_soil.joblib"))
    with open(os.path.join(BASE_DIR, "model", "model_metadata.json"), encoding="utf-8") as f:
        meta = json.load(f)
    return model, le_target, le_soil, meta

model, le_target, le_soil, meta = load_artifacts()

# Ambil nama fitur langsung dari model — sumber kebenaran tunggal
FEATURE_NAMES = model.get_booster().feature_names
CLASS_NAMES   = meta["kelas_output"]

WARNA_ZONA = {
    "sangat rendah": "#2ecc71",
    "rendah"       : "#a8e6a3",
    "sedang"       : "#f39c12",
    "tinggi"       : "#e74c3c",
}
EMOJI_ZONA = {
    "sangat rendah": "🟢",
    "rendah"       : "🟡",
    "sedang"       : "🟠",
    "tinggi"       : "🔴",
}

# ============================================================
# HELPER — preprocessing input sebelum prediksi
# ============================================================
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df[FEATURE_NAMES].copy()
    if df["Soil_Type"].dtype == object:
        df["Soil_Type"] = le_soil.transform(df["Soil_Type"].astype(str))
    for col in FEATURE_NAMES:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
    return df

# ============================================================
# SIDEBAR — NAVIGASI
# ============================================================
with st.sidebar:
    st.markdown("""
    <div style="background:linear-gradient(135deg,#1a6fc4,#56ccf2);
         padding:20px; border-radius:10px; text-align:center; color:white; margin-bottom:10px;">
      <div style="font-size:48px;">🌊</div>
      <div style="font-size:18px; font-weight:700;">Flood Hazard</div>
      <div style="font-size:12px; opacity:0.85;">Kendari Barat</div>
    </div>
    """, unsafe_allow_html=True)
    st.title("🌊 Flood Hazard App")
    st.caption("Kecamatan Kendari Barat")
    st.divider()
    halaman = st.radio(
        "Navigasi",
        ["🏠 Beranda", "🔍 Prediksi Titik", "📊 Prediksi Batch (CSV)", "🗺️ Peta Interaktif", "📈 Info Model"],
        label_visibility="collapsed"
    )
    st.divider()
    st.caption(f"Model: **{meta['model_name']}**")
    st.caption(f"Versi: `{meta['versi']}`")
    st.caption(f"Dilatih: {meta['tanggal_latih']}")

# ============================================================
# HALAMAN 1: BERANDA
# ============================================================
if halaman == "🏠 Beranda":
    st.title("🌊 Sistem Klasifikasi Zona Rawan Banjir")
    st.subheader("Kecamatan Kendari Barat — Model XGBoost")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Algoritma", "XGBoost")
    col2.metric("Akurasi Test", f"{meta['test_accuracy']*100:.2f}%")
    col3.metric("F1-Score Test", f"{meta['test_f1']*100:.2f}%")
    col4.metric("CV F1 Mean", f"{meta['cv_f1_mean']*100:.2f}%")

    st.divider()

    col_a, col_b = st.columns([1.2, 1])
    with col_a:
        st.markdown("""
        ### Tentang Aplikasi
        Aplikasi ini menggunakan model **XGBoost** untuk mengklasifikasikan
        zona rawan banjir di Kecamatan Kendari Barat berdasarkan karakteristik
        fisik dan lingkungan suatu wilayah.

        **Kelas Zona Rawan:**
        - 🟢 **Sangat Rendah** — aman dari banjir
        - 🟡 **Rendah** — potensi banjir kecil
        - 🟠 **Sedang** — perlu kewaspadaan
        - 🔴 **Tinggi** — rawan banjir, perlu penanganan

        **Fitur yang Digunakan:**
        """)
        for f in FEATURE_NAMES:
            st.markdown(f"  - `{f}`")

    with col_b:
        st.markdown("### Performa Model")
        fig = go.Figure(go.Bar(
            x=["Accuracy", "F1-Score", "Precision", "Recall", "CV F1"],
            y=[
                meta["test_accuracy"],
                meta["test_f1"],
                meta.get("test_precision", meta["test_f1"]),
                meta.get("test_recall",    meta["test_f1"]),
                meta["cv_f1_mean"],
            ],
            marker_color=["#3498db","#2ecc71","#9b59b6","#e67e22","#1abc9c"],
            text=[f"{v*100:.1f}%" for v in [
                meta["test_accuracy"], meta["test_f1"],
                meta.get("test_precision", meta["test_f1"]),
                meta.get("test_recall",    meta["test_f1"]),
                meta["cv_f1_mean"],
            ]],
            textposition="outside",
        ))
        fig.update_layout(
            yaxis_range=[0, 1.1], height=320,
            margin=dict(t=20, b=20),
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True)

# ============================================================
# HALAMAN 2: PREDIKSI TITIK
# ============================================================
elif halaman == "🔍 Prediksi Titik":
    st.title("🔍 Prediksi Zona — Input Manual")
    st.markdown("Masukkan nilai parameter lokasi untuk mendapatkan prediksi zona rawan banjir.")

    with st.form("form_prediksi"):
        col1, col2 = st.columns(2)

        with col1:
            elevation = st.number_input(
                "Elevasi / Ketinggian (m dpl)", min_value=0.0, max_value=3000.0,
                value=15.0, step=0.5,
                help="Ketinggian lokasi dari permukaan laut"
            )
            rainfall = st.number_input(
                "Curah Hujan (mm/tahun)", min_value=0.0, max_value=10000.0,
                value=2500.0, step=50.0,
                help="Rata-rata curah hujan tahunan"
            )
            slope = st.number_input(
                "Kemiringan Lereng / Slope (°)", min_value=0.0, max_value=90.0,
                value=3.0, step=0.1,
                help="Sudut kemiringan permukaan tanah dalam derajat"
            )
            land_cover = st.number_input(
                "Land Cover (kode)", min_value=0, max_value=999,
                value=40, step=1,
                help="Kode tutupan lahan (sesuai dataset)"
            )

        with col2:
            soil_type = st.selectbox(
                "Jenis Tanah (Soil Type)",
                options=list(le_soil.classes_),
                help="Jenis tanah pada lokasi yang diprediksi"
            )
            drainage_length = st.number_input(
                "Panjang Drainase (m)", min_value=0.0, max_value=100000.0,
                value=300.0, step=10.0,
                help="Panjang saluran drainase terdekat"
            )

        submitted = st.form_submit_button("🔮 Prediksi Sekarang", use_container_width=True, type="primary")

    if submitted:
        input_values = {
            "Elevation"      : float(elevation),
            "LandCover"      : float(land_cover),
            "Rainfall"       : float(rainfall),
            "Slope"          : float(slope),
            "Soil_Type"      : float(le_soil.transform([soil_type])[0]),
            "Drainage_Length": float(drainage_length),
        }

        try:
            df_input = pd.DataFrame(
                [[input_values[f] for f in FEATURE_NAMES]],
                columns=FEATURE_NAMES
            )

            label_enc  = model.predict(df_input)[0]
            label_name = le_target.inverse_transform([label_enc])[0]
            proba      = model.predict_proba(df_input)[0]
            proba_dict = {le_target.inverse_transform([i])[0]: float(p) for i, p in enumerate(proba)}
            confidence = float(proba.max())

            st.divider()
            warna = WARNA_ZONA.get(label_name.lower(), "#95a5a6")
            emoji = EMOJI_ZONA.get(label_name.lower(), "⚪")

            col_res1, col_res2 = st.columns([1, 1.5])
            with col_res1:
                st.markdown(
                    f"""
                    <div style="background:{warna}22; border-left:6px solid {warna};
                         padding:24px; border-radius:12px; text-align:center;">
                      <div style="font-size:56px;">{emoji}</div>
                      <div style="font-size:28px; font-weight:700; color:{warna};">
                        {label_name.upper()}
                      </div>
                      <div style="font-size:16px; color:#555; margin-top:8px;">
                        Zona Rawan Banjir
                      </div>
                      <div style="font-size:22px; font-weight:600; margin-top:12px;">
                        Kepercayaan: {confidence*100:.1f}%
                      </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with col_res2:
                st.markdown("#### Probabilitas per Kelas")
                sorted_proba = sorted(proba_dict.items(), key=lambda x: -x[1])
                fig_proba = go.Figure(go.Bar(
                    x=[p*100 for _, p in sorted_proba],
                    y=[k.title() for k, _ in sorted_proba],
                    orientation="h",
                    marker_color=[WARNA_ZONA.get(k.lower(), "#95a5a6") for k, _ in sorted_proba],
                    text=[f"{p*100:.2f}%" for _, p in sorted_proba],
                    textposition="outside",
                ))
                fig_proba.update_layout(
                    xaxis_title="Probabilitas (%)", xaxis_range=[0, 110],
                    height=250, margin=dict(t=10, b=10),
                    plot_bgcolor="rgba(0,0,0,0)"
                )
                st.plotly_chart(fig_proba, use_container_width=True)

            st.info(
                f"📌 Lokasi ini diprediksi berada pada zona **{label_name}** "
                f"dengan tingkat kepercayaan model sebesar **{confidence*100:.1f}%**."
            )

        except Exception as e:
            st.error(f"❌ Gagal melakukan prediksi: {e}")
            st.info(f"Fitur model: {FEATURE_NAMES}")

# ============================================================
# HALAMAN 3: PREDIKSI BATCH
# ============================================================
elif halaman == "📊 Prediksi Batch (CSV)":
    st.title("📊 Prediksi Batch — Upload CSV")
    st.markdown(
        "Upload file CSV yang berisi beberapa lokasi sekaligus. "
        "CSV harus memiliki kolom: `" + "`, `".join(FEATURE_NAMES) + "`"
    )

    template_df = pd.DataFrame(columns=FEATURE_NAMES)
    template_df.loc[0] = [15.0, 40, 2500.0, 3.0, le_soil.classes_[0], 300.0]
    template_df.loc[1] = [5.0,  20, 3200.0, 1.2, le_soil.classes_[0], 100.0]

    st.download_button(
        "⬇️ Download Template CSV",
        data=template_df.to_csv(index=False),
        file_name="template_prediksi_banjir.csv",
        mime="text/csv"
    )

    uploaded = st.file_uploader("Upload CSV Anda", type=["csv"])

    if uploaded:
        df_up = pd.read_csv(uploaded)
        st.success(f"✅ File berhasil dibaca: {len(df_up)} baris, {len(df_up.columns)} kolom")
        st.dataframe(df_up.head(), use_container_width=True)

        missing_cols = [c for c in FEATURE_NAMES if c not in df_up.columns]
        if missing_cols:
            st.error(f"❌ Kolom berikut tidak ditemukan: {missing_cols}")
        else:
            try:
                df_proc = preprocess(df_up)

                pred_enc   = model.predict(df_proc)
                pred_label = le_target.inverse_transform(pred_enc)
                pred_proba = model.predict_proba(df_proc).max(axis=1)

                df_hasil = df_up.copy()
                df_hasil["Prediksi_Zona"] = pred_label
                df_hasil["Kepercayaan"]   = (pred_proba * 100).round(2)

                st.divider()
                st.subheader("Hasil Prediksi")

                dist = pd.Series(pred_label).value_counts()
                cols = st.columns(len(dist))
                for col, (kls, cnt) in zip(cols, dist.items()):
                    col.metric(f"{EMOJI_ZONA.get(kls.lower(),'⚪')} {kls.title()}", f"{cnt} lokasi")

                st.dataframe(df_hasil, use_container_width=True)

                fig_dist = px.pie(
                    values=dist.values, names=dist.index,
                    color=dist.index,
                    color_discrete_map=WARNA_ZONA,
                    title="Distribusi Prediksi Zona"
                )
                st.plotly_chart(fig_dist, use_container_width=True)

                st.download_button(
                    "⬇️ Download Hasil CSV",
                    data=df_hasil.to_csv(index=False, encoding="utf-8-sig"),
                    file_name="hasil_prediksi_banjir.csv",
                    mime="text/csv"
                )
            except Exception as e:
                st.error(f"❌ Gagal memproses data: {e}")
                st.info(f"Fitur model: {FEATURE_NAMES}")

# ============================================================
# HALAMAN 4: PETA INTERAKTIF
# ============================================================
elif halaman == "🗺️ Peta Interaktif":
    st.title("🗺️ Peta Zona Rawan Banjir")
    st.markdown(
        "Upload CSV **dengan kolom LATITUDE & LONGITUDE** untuk memvisualisasikan "
        "prediksi pada peta Kecamatan Kendari Barat."
    )

    uploaded_map = st.file_uploader("Upload CSV (harus ada LATITUDE, LONGITUDE + fitur model)", type=["csv"])

    if uploaded_map:
        df_map = pd.read_csv(uploaded_map)
        req_cols = FEATURE_NAMES + ["LATITUDE", "LONGITUDE"]
        missing_map = [c for c in req_cols if c not in df_map.columns]
        if missing_map:
            st.error(f"❌ Kolom berikut tidak ada: {missing_map}")
        else:
            try:
                df_proc = preprocess(df_map)

                pred_enc   = model.predict(df_proc)
                pred_label = le_target.inverse_transform(pred_enc)
                pred_proba = model.predict_proba(df_proc).max(axis=1)

                df_map["Pred_Label"] = pred_label
                df_map["Pred_Proba"] = (pred_proba * 100).round(1)

                center = [df_map["LATITUDE"].mean(), df_map["LONGITUDE"].mean()]
                m = folium.Map(location=center, zoom_start=13, tiles="CartoDB positron")

                COLOR_FOLIUM = {
                    "sangat rendah": "green",
                    "rendah"       : "lightgreen",
                    "sedang"       : "orange",
                    "tinggi"       : "red",
                }

                for _, row in df_map.iterrows():
                    kls = str(row["Pred_Label"]).lower()
                    folium.CircleMarker(
                        location=[row["LATITUDE"], row["LONGITUDE"]],
                        radius=5,
                        color=COLOR_FOLIUM.get(kls, "gray"),
                        fill=True,
                        fill_color=COLOR_FOLIUM.get(kls, "gray"),
                        fill_opacity=0.75,
                        tooltip=(
                            f"<b>Zona:</b> {row['Pred_Label']}<br>"
                            f"<b>Kepercayaan:</b> {row['Pred_Proba']}%<br>"
                            f"<b>Elevasi:</b> {row.get('Elevation','—')} m<br>"
                            f"<b>Rainfall:</b> {row.get('Rainfall','—')} mm"
                        ),
                    ).add_to(m)

                legend_html = """
                <div style="position:fixed;bottom:30px;left:30px;z-index:1000;
                     background:white;padding:12px 16px;border-radius:8px;
                     box-shadow:2px 2px 6px rgba(0,0,0,0.3);font-family:Arial;font-size:13px;">
                  <b>🌊 Zona Rawan Banjir</b><br>
                  <span style="color:green;">●</span> Sangat Rendah<br>
                  <span style="color:lightgreen;">●</span> Rendah<br>
                  <span style="color:orange;">●</span> Sedang<br>
                  <span style="color:red;">●</span> Tinggi
                </div>"""
                m.get_root().html.add_child(folium.Element(legend_html))

                col_map, col_stat = st.columns([2, 1])
                with col_map:
                    st_folium(m, width=700, height=500)
                with col_stat:
                    st.subheader("Statistik Prediksi")
                    dist2 = pd.Series(pred_label).value_counts()
                    for kls, cnt in dist2.items():
                        e = EMOJI_ZONA.get(kls.lower(), "⚪")
                        st.metric(f"{e} {kls.title()}", f"{cnt} titik ({cnt/len(df_map)*100:.1f}%)")

            except Exception as e:
                st.error(f"❌ Gagal memproses data: {e}")
                st.info(f"Fitur model: {FEATURE_NAMES}")
    else:
        m_default = folium.Map(
            location=[-3.983, 122.513], zoom_start=13, tiles="CartoDB positron"
        )
        folium.Marker([-3.983, 122.513], tooltip="Kendari Barat").add_to(m_default)
        st_folium(m_default, width=700, height=400)

# ============================================================
# HALAMAN 5: INFO MODEL
# ============================================================
elif halaman == "📈 Info Model":
    st.title("📈 Informasi & Performa Model")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Parameter XGBoost")
        params = meta["parameter_xgb"]
        df_params = pd.DataFrame(list(params.items()), columns=["Parameter", "Nilai"])
        st.table(df_params)

        st.subheader("Kelas Output")
        for kls in CLASS_NAMES:
            e = EMOJI_ZONA.get(kls.lower(), "⚪")
            st.markdown(f"{e} `{kls}`")

    with col2:
        st.subheader("Metrik Evaluasi")
        metrics = {
            "Test Accuracy" : meta["test_accuracy"],
            "Test F1-Score" : meta["test_f1"],
            "CV F1 Mean"    : meta["cv_f1_mean"],
            "CV F1 Std"     : meta["cv_f1_std"],
        }
        for k, v in metrics.items():
            st.metric(k, f"{v*100:.2f}%")

        st.subheader("Fitur Input Model")
        for i, f in enumerate(FEATURE_NAMES, 1):
            st.markdown(f"`{i}.` **{f}**")

    st.divider()
    st.subheader("Preprocessing")
    st.markdown("""
    - **SMOTE** diterapkan hanya pada data training untuk mengatasi ketidakseimbangan kelas
    - **LabelEncoder** digunakan untuk `Soil_Type` (kategorik → numerik) dan `Flood_Label` (target)
    - **Train/Test Split**: 80% train, 20% test (stratified)
    - **Cross Validation**: 5-Fold Stratified KFold
    """)
