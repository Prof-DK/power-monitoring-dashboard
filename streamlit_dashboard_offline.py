"""
DASHBOARD OFFLINE -- BACA LANGSUNG DARI DATABASE (TANPA MQTT/UPLOAD)
=========================================================
Berbeda dari versi sebelumnya (yang perlu mqtt_publisher.py jalan
terus-menerus), dashboard ini CUKUP membaca dari database SQLite
yang sudah diisi SEKALI lewat import_to_sqlite.py.

Tidak perlu proses tambahan yang harus tetap hidup -- begitu
di-deploy ke VPS, dashboard langsung bisa diakses siapa saja,
kapan saja, tanpa bergantung pada broker MQTT atau script lain
yang harus terus berjalan.

CARA JALANKAN LOKAL (untuk uji coba sebelum di-deploy ke VPS):
    pip install streamlit plotly pandas --break-system-packages
    streamlit run streamlit_dashboard_offline.py

PRASYARAT: file 'power_data.db' harus ada di folder yang sama
(hasil dari import_to_sqlite.py).

CARA DEPLOY KE VPS:
    1. Copy 'power_data.db' + 'streamlit_dashboard_offline.py' +
       'requirements.txt' ke VPS Anda.
    2. Install dependency: pip install -r requirements.txt
    3. Jalankan permanen pakai pm2 atau systemd, contoh dengan pm2:
       pm2 start "streamlit run streamlit_dashboard_offline.py
       --server.port 8501 --server.address 0.0.0.0" --name dashboard
    4. Arahkan domain/reverse proxy (nginx) ke port 8501 supaya
       bisa diakses publik lewat domain Anda.

ALTERNATIF (lebih simpel, tanpa perlu VPS sama sekali):
    Upload ke GitHub + deploy gratis lewat share.streamlit.io
    (Streamlit Community Cloud) -- tinggal sertakan power_data.db
    di repo yang sama.
"""

import json
import os
import sqlite3
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# =========================================================
# KONFIGURASI
# =========================================================
DB_PATH = "power_data.db"
TABLE_NAME = "readings"
CAPACITY_KVA = 2000

st.set_page_config(
    page_title="Power Monitoring",
    page_icon="⚡",
    layout="wide",
)

# =========================================================
# KONEKSI DATABASE (di-cache supaya tidak buka file berulang-ulang)
# =========================================================
@st.cache_resource
def get_connection():
    return sqlite3.connect(DB_PATH, check_same_thread=False)


@st.cache_data
def get_date_range():
    conn = get_connection()
    row = conn.execute(f'SELECT MIN("Date Time"), MAX("Date Time") FROM {TABLE_NAME}').fetchone()
    return pd.to_datetime(row[0]), pd.to_datetime(row[1])


@st.cache_data
def query_range(start_date, end_date):
    conn = get_connection()
    query = f"""
        SELECT * FROM {TABLE_NAME}
        WHERE "Date Time" >= ? AND "Date Time" <= ?
        ORDER BY "Date Time" ASC
    """
    df = pd.read_sql_query(query, conn, params=[str(start_date), str(end_date)])
    df['Date Time'] = pd.to_datetime(df['Date Time'])
    return df


@st.cache_data
def query_summary_stats(start_date, end_date):
    conn = get_connection()
    query = f"""
        SELECT
            AVG(kVA) as avg_kva, MIN(kVA) as min_kva, MAX(kVA) as max_kva,
            AVG(F) as avg_freq, MAX(kWh) - MIN(kWh) as energy_used,
            SUM(CASE WHEN status = 'OVERLOAD' THEN 1 ELSE 0 END) as overload_count,
            COUNT(*) as total_rows
        FROM {TABLE_NAME}
        WHERE "Date Time" >= ? AND "Date Time" <= ?
    """
    return conn.execute(query, [str(start_date), str(end_date)]).fetchone()


# =========================================================
# REVISI: MODEL PREDIKSI LSTM (Program A) -- ONE-STEP-AHEAD
# -----------------------------------------------------
# Prediksi setiap titik dihitung dari NILAI AKTUAL sebelumnya
# (bukan recursive), persis metodologi yang sama seperti
# evaluate() di programA_lookback1.py. Karena LOOKBACK=1, seluruh
# rentang tanggal bisa diprediksi SEKALIGUS (batch), bukan satu-satu.
#
# PRASYARAT FILE (taruh sejajar dengan dashboard ini):
#   - model_programA.h5
#   - scaler_params.json
# Kalau file ini tidak ada, fitur prediksi otomatis dinonaktifkan
# (dashboard tetap bisa dipakai tanpa fitur ini).
# =========================================================
MODEL_PATH = "model_programB_s4d.h5"
SCALER_PATH = "scaler_params_s4d.json"


# =========================================================
# CUSTOM LAYER: S4D (WAJIB SAMA PERSIS DENGAN SAAT TRAINING)
# -----------------------------------------------------
# Diperlukan karena model_programB_s4d.h5 memakai custom layer
# S4DLayer -- tanpa definisi ini, load_model() akan gagal.
# =========================================================
def _build_s4d_layer_class():
    import tensorflow as tf
    from tensorflow.keras import layers

    class S4DLayer(layers.Layer):
        def __init__(self, state_dim=64, **kwargs):
            super().__init__(**kwargs)
            self.state_dim = state_dim

        def build(self, input_shape):
            self.channels = input_shape[-1]
            N = self.state_dim
            C_ch = self.channels
            self.log_dt = self.add_weight(
                name='log_dt', shape=(C_ch,),
                initializer=tf.keras.initializers.RandomUniform(np.log(0.001), np.log(0.1)),
                trainable=True
            )
            A_re_init = np.zeros((C_ch, N))
            A_im_init = np.pi * np.tile(np.arange(N), (C_ch, 1))
            self.A_re_raw = self.add_weight(
                name='A_re_raw', shape=(C_ch, N),
                initializer=tf.keras.initializers.Constant(A_re_init), trainable=True
            )
            self.A_im = self.add_weight(
                name='A_im', shape=(C_ch, N),
                initializer=tf.keras.initializers.Constant(A_im_init), trainable=True
            )
            self.C_re = self.add_weight(name='C_re', shape=(C_ch, N), initializer='glorot_uniform', trainable=True)
            self.C_im = self.add_weight(name='C_im', shape=(C_ch, N), initializer='glorot_uniform', trainable=True)
            self.D = self.add_weight(name='D', shape=(C_ch,), initializer='ones', trainable=True)
            super().build(input_shape)

        def call(self, u):
            dt = tf.exp(self.log_dt)
            A_re = -tf.exp(self.A_re_raw)
            A = tf.complex(A_re, self.A_im)
            dt_complex = tf.cast(dt, tf.complex64)[:, None]
            dtA = dt_complex * A
            Abar = tf.exp(dtA)
            Bbar = (Abar - tf.complex(1.0, 0.0)) / A
            C = tf.complex(self.C_re, self.C_im)
            u_t_major = tf.transpose(u, [1, 0, 2])
            u_complex = tf.complex(u_t_major, tf.zeros_like(u_t_major))
            batch_size = tf.shape(u)[0]
            init_state = tf.zeros((batch_size, self.channels, self.state_dim), dtype=tf.complex64)

            def step_fn(prev_state, u_t):
                return Abar[None, :, :] * prev_state + Bbar[None, :, :] * u_t[:, :, None]

            all_states = tf.scan(step_fn, u_complex, initializer=init_state)
            y_complex = tf.reduce_sum(C[None, None, :, :] * all_states, axis=-1)
            y = 2.0 * tf.math.real(y_complex)
            y = y + self.D[None, None, :] * u_t_major
            return tf.transpose(y, [1, 0, 2])

        def compute_output_shape(self, input_shape):
            return input_shape

    return S4DLayer


@st.cache_resource
def load_prediction_model():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        return None, None, "not_found"
    try:
        from tensorflow.keras.models import load_model
        S4DLayer = _build_s4d_layer_class()
        model = load_model(MODEL_PATH, custom_objects={'S4DLayer': S4DLayer}, compile=False)
        with open(SCALER_PATH, "r") as f:
            scaler_params = json.load(f)
        return model, scaler_params, None
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        print(f"Gagal memuat model prediksi:\n{error_detail}")
        return None, None, str(e)


def predict_batch(df, model, scaler_params):
    """
    Hitung prediksi one-step-ahead untuk SELURUH baris di df sekaligus.
    Input tiap prediksi = nilai kVA AKTUAL 1 baris sebelumnya (bukan
    hasil prediksi feedback/recursive) -- konsisten dengan cara
    evaluasi Program A.
    """
    prev_kva = df['kVA'].shift(1)
    valid_mask = prev_kva.notna()

    if valid_mask.sum() == 0:
        return pd.Series([np.nan] * len(df), index=df.index)

    scaled_input = (prev_kva[valid_mask] - scaler_params['min']) / (scaler_params['max'] - scaler_params['min'])
    X = scaled_input.values.reshape(-1, 1, 1).astype('float32')

    pred_scaled = model.predict(X, verbose=0).flatten()
    pred_kva = pred_scaled * (scaler_params['max'] - scaler_params['min']) + scaler_params['min']

    result = pd.Series(np.nan, index=df.index)
    result[valid_mask] = pred_kva
    return result


def predict_single_next(last_kva, model, scaler_params):
    """Prediksi 1 nilai kVA berikutnya dari 1 nilai kVA saat ini."""
    scaled_input = (last_kva - scaler_params['min']) / (scaler_params['max'] - scaler_params['min'])
    X = np.array([[[scaled_input]]], dtype='float32')
    pred_scaled = model.predict(X, verbose=0)[0, 0]
    return pred_scaled * (scaler_params['max'] - scaler_params['min']) + scaler_params['min']


model, scaler_params, load_error = load_prediction_model()
prediction_available = model is not None


# =========================================================
# HEADER
# =========================================================
st.title("⚡ Power Monitoring")

min_date, max_date = get_date_range()

# =========================================================
# TOOLBAR: PILIH RENTANG TANGGAL
# =========================================================
col1, col2, col3 = st.columns([2, 2, 1])
with col1:
    start_date = st.date_input(
        "Tanggal Mulai", value=max_date.date() - pd.Timedelta(days=7),
        min_value=min_date.date(), max_value=max_date.date()
    )
with col2:
    end_date = st.date_input(
        "Tanggal Akhir", value=max_date.date(),
        min_value=min_date.date(), max_value=max_date.date()
    )
with col3:
    st.write("")
    st.write("")
    load_clicked = st.button("🔄 Muat Data", use_container_width=True)

if start_date > end_date:
    st.error("Tanggal mulai tidak boleh lebih besar dari tanggal akhir.")
    st.stop()

# query rentang yang dipilih (otomatis di-cache oleh Streamlit selama parameter sama)
df_range = query_range(start_date, str(pd.Timestamp(end_date) + pd.Timedelta(days=1)))
stats = query_summary_stats(start_date, str(pd.Timestamp(end_date) + pd.Timedelta(days=1)))

if len(df_range) == 0:
    st.warning("Tidak ada data pada rentang tanggal yang dipilih.")
    st.stop()

avg_kva, min_kva, max_kva, avg_freq, energy_used, overload_count, total_rows = stats

st.divider()

# =========================================================
# RINGKASAN
# =========================================================
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Rata-rata kVA", f"{avg_kva:.1f}")
m2.metric("Min kVA", f"{min_kva:.1f}")
m3.metric("Max kVA", f"{max_kva:.1f}")
m4.metric("Frekuensi Rata-rata", f"{avg_freq:.2f} Hz")
m5.metric("Titik Potensi Overload", f"{overload_count}", delta=None,
          delta_color="inverse" if overload_count > 0 else "normal")

st.caption(f"Menampilkan {total_rows:,} titik data dari {start_date} s/d {end_date}")

st.divider()

# =========================================================
# SNAPSHOT GAUGE -- MIRIP PANEL GRAFANA "LVMDB PLANT 3"
# -----------------------------------------------------
# Menampilkan gauge untuk 1 titik waktu tertentu (default: titik
# TERAKHIR dari rentang tanggal yang dipilih), meniru tampilan
# panel monitoring produksi Anda (gauge arus/tegangan per fasa).
# =========================================================
st.subheader("Parameter")

# pilih titik waktu yang ingin dilihat sebagai "snapshot" -- default
# ke titik terakhir dalam rentang yang dipilih
snapshot_options = df_range['Date Time'].dt.strftime('%Y-%m-%d %H:%M:%S').tolist()
selected_ts = st.selectbox(
    "Pilih titik waktu",
    options=snapshot_options,
    index=len(snapshot_options) - 1,  # default: titik terakhir
)
snapshot = df_range[df_range['Date Time'].dt.strftime('%Y-%m-%d %H:%M:%S') == selected_ts].iloc[0]


def make_gauge(value, title, unit, gauge_min, gauge_max, nominal_low, nominal_high):
    """Gauge gaya Grafana: hijau di rentang normal, merah di luar rentang."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number={'suffix': f" {unit}", 'font': {'size': 22}},
        gauge={
            'axis': {'range': [gauge_min, gauge_max], 'tickfont': {'size': 8}},
            'bar': {'color': "#34D399" if nominal_low <= value <= nominal_high else "#F87171"},
            'bgcolor': "#1D2536",
            'borderwidth': 1,
            'bordercolor': "#2A3448",
            'steps': [
                {'range': [gauge_min, nominal_low], 'color': 'rgba(248, 113, 113, 0.15)'},
                {'range': [nominal_low, nominal_high], 'color': 'rgba(52, 211, 153, 0.15)'},
                {'range': [nominal_high, gauge_max], 'color': 'rgba(248, 113, 113, 0.15)'},
            ],
        },
        title={'text': title, 'font': {'size': 12}},
    ))
    fig.update_layout(height=160, margin=dict(l=15, r=15, t=35, b=10),
                       paper_bgcolor="rgba(0,0,0,0)", font={'color': "#E8ECF4"})
    return fig


# REVISI: sesuaikan rentang nominal ini dengan spesifikasi trafo/kabel Anda
# yang sebenarnya kalau berbeda -- ini nilai wajar untuk sistem 3-fasa
# 400V/230V standar.
CURRENT_MAX = 2000
CURRENT_NOMINAL = (0, 1600)       # asumsi wajar hingga 80% kapasitas
VOLTAGE_LN_RANGE = (200, 240)
VOLTAGE_LN_NOMINAL = (218, 232)   # +-5% dari 230V nominal
VOLTAGE_LL_RANGE = (360, 420)
VOLTAGE_LL_NOMINAL = (390, 410)   # +-5% dari 400V nominal

st.caption(f"Menampilkan data pada: **{selected_ts}**")

g1, g2, g3, g4 = st.columns(4)

with g1:
    st.plotly_chart(make_gauge(snapshot['IR'], "Current R", "A", 0, CURRENT_MAX, *CURRENT_NOMINAL), use_container_width=True, key="g_ir")
    st.plotly_chart(make_gauge(snapshot['IS'], "Current S", "A", 0, CURRENT_MAX, *CURRENT_NOMINAL), use_container_width=True, key="g_is")
    st.plotly_chart(make_gauge(snapshot['IT'], "Current T", "A", 0, CURRENT_MAX, *CURRENT_NOMINAL), use_container_width=True, key="g_it")

with g2:
    st.plotly_chart(make_gauge(snapshot['VRN'], "Voltage R-N", "V", *VOLTAGE_LN_RANGE, *VOLTAGE_LN_NOMINAL), use_container_width=True, key="g_vrn")
    st.plotly_chart(make_gauge(snapshot['VSN'], "Voltage S-N", "V", *VOLTAGE_LN_RANGE, *VOLTAGE_LN_NOMINAL), use_container_width=True, key="g_vsn")
    st.plotly_chart(make_gauge(snapshot['VTN'], "Voltage T-N", "V", *VOLTAGE_LN_RANGE, *VOLTAGE_LN_NOMINAL), use_container_width=True, key="g_vtn")

with g3:
    st.plotly_chart(make_gauge(snapshot['VRS'], "Voltage R-S", "V", *VOLTAGE_LL_RANGE, *VOLTAGE_LL_NOMINAL), use_container_width=True, key="g_vrs")
    st.plotly_chart(make_gauge(snapshot['VST'], "Voltage S-T", "V", *VOLTAGE_LL_RANGE, *VOLTAGE_LL_NOMINAL), use_container_width=True, key="g_vst")
    st.plotly_chart(make_gauge(snapshot['VTR'], "Voltage T-R", "V", *VOLTAGE_LL_RANGE, *VOLTAGE_LL_NOMINAL), use_container_width=True, key="g_vtr")

with g4:
    st.metric("Current Average (A)", f"{snapshot['IAvg']:,.1f}")
    st.metric("Voltage L-N Average (V)", f"{snapshot['VNAvg']:,.1f}")
    st.metric("Voltage L-L Average (V)", f"{snapshot['VLLAvg']:,.1f}")
    st.metric("Frequency (Hz)", f"{snapshot['F']:,.2f}")
    st.metric("Power Total (kVA)", f"{snapshot['kVA']:,.1f}")
    st.metric("kWh", f"{snapshot['kWh']:,.0f}")

    if prediction_available:
        pred_next = predict_single_next(snapshot['kVA'], model, scaler_params)
        st.metric(
            "🔮 Prediksi 5 Menit Berikutnya",
            f"{pred_next:,.1f} kVA",
            delta=f"{pred_next - snapshot['kVA']:+.1f} kVA",
        )

st.divider()

# =========================================================
# GRAFIK UTAMA: kVA SEPANJANG WAKTU
# =========================================================
st.subheader("Grafik Daya Semu real dan prediksi (kVA)")

# REVISI: hitung prediksi (untuk overlay di grafik) -- MAPE/MAE tidak
# ditampilkan lagi di dashboard ini sesuai permintaan, jadi cukup hitung
# Predicted_kVA saja untuk keperluan grafik.
if prediction_available:
    df_range['Predicted_kVA'] = predict_batch(df_range, model, scaler_params)

if not prediction_available:
    if load_error == "not_found":
        st.info(
            f"ℹ️ Fitur prediksi nonaktif — file `{MODEL_PATH}` / `{SCALER_PATH}` "
            f"tidak ditemukan di folder ini. Copy kedua file itu (hasil dari "
            f"programB_s4d_lookback1.py) ke folder yang sama dengan dashboard ini "
            f"untuk mengaktifkan overlay prediksi."
        )
    else:
        st.error(
            f"⚠️ File model ditemukan, TAPI GAGAL DIMUAT. Error: `{load_error}`\n\n"
            f"Kemungkinan penyebab: model disimpan dengan versi kode LAMA "
            f"(sebelum fix Activation/Add), atau versi TensorFlow/Keras berbeda "
            f"dari saat training. Cek log terminal untuk detail traceback lengkap."
        )

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=df_range['Date Time'], y=df_range['kVA'],
    mode='lines', line=dict(color='#4FD1C5', width=1.5),
    fill='tozeroy', fillcolor='rgba(79, 209, 197, 0.08)',
    name='Aktual',
))

# REVISI: overlay prediksi S4D (Predicted_kVA sudah dihitung di atas)
if prediction_available:
    fig.add_trace(go.Scatter(
        x=df_range['Date Time'], y=df_range['Predicted_kVA'],
        mode='lines', line=dict(color='#FBBF24', width=1.5, dash='dot'),
        name='Prediksi S4D',
    ))

fig.add_hline(
    y=CAPACITY_KVA * 0.8, line_dash="dot", line_color="#F87171",
    annotation_text="Batas 80% Kapasitas (1600 kVA)", annotation_position="top left"
)

overload_points = df_range[df_range['status'] == 'OVERLOAD']
if len(overload_points) > 0:
    fig.add_trace(go.Scatter(
        x=overload_points['Date Time'], y=overload_points['kVA'],
        mode='markers', marker=dict(color='#F87171', size=6, symbol='x'),
        name='Potensi Overload',
    ))

fig.update_layout(
    height=380,
    margin=dict(l=20, r=20, t=20, b=20),
    xaxis_title="Waktu", yaxis_title="kVA",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    hovermode='x unified',
)
st.plotly_chart(fig, use_container_width=True)

# =========================================================
# GRAFIK ARUS & TEGANGAN (opsional, expandable)
# =========================================================
with st.expander("📊 Grafik Arus per Fasa (IR/IS/IT)"):
    fig_i = go.Figure()
    for col, color in [('IR', '#4FD1C5'), ('IS', '#FBBF24'), ('IT', '#F87171')]:
        fig_i.add_trace(go.Scatter(
            x=df_range['Date Time'], y=df_range[col],
            mode='lines', line=dict(width=1.2, color=color), name=col
        ))
    fig_i.update_layout(height=300, margin=dict(l=20, r=20, t=20, b=20), yaxis_title="Arus (A)")
    st.plotly_chart(fig_i, use_container_width=True)

with st.expander("📊 Grafik Tegangan Line-to-Line (VRS/VST/VTR)"):
    fig_v = go.Figure()
    for col, color in [('VRS', '#4FD1C5'), ('VST', '#FBBF24'), ('VTR', '#A855F7')]:
        fig_v.add_trace(go.Scatter(
            x=df_range['Date Time'], y=df_range[col],
            mode='lines', line=dict(width=1.2, color=color), name=col
        ))
    fig_v.update_layout(height=300, margin=dict(l=20, r=20, t=20, b=20), yaxis_title="Tegangan (V)")
    st.plotly_chart(fig_v, use_container_width=True)

st.divider()

# =========================================================
# TABEL DATA (SAMPLE, biar tidak berat kalau rentang besar)
# =========================================================
st.subheader("Tabel Data")
show_all = st.checkbox("Tampilkan semua baris")

display_df = df_range if show_all else df_range.tail(100)
st.dataframe(
    display_df[['Date Time', 'kVA', 'IR', 'IS', 'IT', 'VLLAvg', 'F', 'kWh', 'capacity(%)', 'status']],
    use_container_width=True,
    height=350,
)

