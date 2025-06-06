import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import datetime
import plotly.graph_objects as go
from PIL import Image

st.set_page_config(layout="wide")                                                                                            

st.markdown(
    """
    <h1 style='text-align: center;'>BREAVA</h1>
    <h3 style='text-align: center;'>Aplikasi prediksi kualitas udara Kota Yogyakarta</h3>
    """, 
    unsafe_allow_html=True
)

# --- Fungsi kategorisasi PM2.5 ---
def categorize_pm25(pm25):
    if pm25 <= 9.0:
        return "Good", "#0F980F"
    elif pm25 <= 35.4:
        return "Moderate", "#E3BC3A"
    elif pm25 <= 55.4:
        return "Unhealthy for Sensitive Groups", "#E7750A"
    elif pm25 <= 125.4:
        return "Unhealthy", "#FF6A6A"
    elif pm25 <= 225.4:
        return "Very Unhealthy", "#A07CC5"
    else:
        return "Hazardous", "#A05252"

# --- Load model dan scaler ---
try:
    model = load_model("model/air_quality_lstm_model.h5", compile=False)
    # Workaround for the time_major issue
    for layer in model.layers:
        if hasattr(layer, 'time_major'):
            delattr(layer, 'time_major')
except Exception as e:
    st.error(f"Gagal memuat model: {e}")
    st.stop()

scaler = MinMaxScaler()
try:
    scaler_data = np.load("model/scaler_pm25.npy", allow_pickle=True)
    scaler.min_, scaler.scale_ = scaler_data
except Exception as e:
    st.error(f"Gagal memuat scaler: {e}")
    st.stop()

# --- Load data ---
@st.cache_data
def load_data():
    df = pd.read_csv("data/hasil_prediksi_air_quality.csv")
    return df

df = load_data()

st.markdown("""
    <style>
    /* Buat konten berada di tengah dan maksimal lebarnya dibatasi */
    .main .block-container {
        max-width: 900px;
        margin: auto;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    /* Ubah ukuran teks tab */
    .stTabs [role="tab"] {
        font-size: 20px;
        font-weight: 600;
        padding: 12px 24px;
    }

    /* (Opsional) Mewarnai tab aktif */
    .stTabs [aria-selected="true"] {
        color: white;
        background: #2a2a2a;
        border-radius: 8px 8px 0 0;
    }

    /* Tambahkan margin agar terlihat lega */
    .stTabs {
        margin-bottom: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# --- Tabs ---
tab1, tab2, tab3, tab4 = st.tabs(["Ringkasan", "Prediksi", "Visualisasi", "Peta"])

# --- Ringkasan ---
with tab1:
    latest_pm25 = df['PM2.5'].iloc[1]
    kategori_now, warna_now = categorize_pm25(latest_pm25)
    jam_now = pd.to_datetime(df['datetime'].iloc[-1]) if 'datetime' in df.columns else datetime.datetime.now()

    st.markdown(f"""
    <div style="background-color: {warna_now}; padding: 20px 25px; border-radius: 12px; margin-bottom: 20px; text-align: center;">
        <div style="display: flex; align-items: center; justify-content: center; gap: 15px;">
            <svg xmlns="http://www.w3.org/2000/svg" width="55" height="55" fill="white" viewBox="0 0 24 24">
                <path d="M6.995 12H19a3 3 0 1 0-2.83-4H16a4 4 0 0 0-7.874.98A4.5 4.5 0 0 0 6.995 12Zm0 2a6.5 6.5 0 1 1 6.452-7.52A6.001 6.001 0 0 1 19 10a5.99 5.99 0 0 1-1.75 4.243A6.5 6.5 0 0 1 6.995 14Z"/>
            </svg>
            <div>
                <h3 style="margin: 0; color: black;">Kualitas Udara Saat Ini<br>({jam_now.strftime('%d %b %Y, %H:%M')})</h3>
                <p style="font-size: 24px; margin: 10px 0; color: black;">
                    <strong>{latest_pm25:.2f} µg/m³</strong> - {kategori_now}
                </p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.subheader("Kategori Kualitas Udara Berdasarkan AQI")
    st.markdown("""
<style>
.aqi-table {
    width: 100%;
    border-collapse: separate;
    border-spacing: 0 12px;
    text-align: center;
    font-size: 16px;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    color: white;
}
.aqi-table th {
    background-color: #333;
    padding: 14px 12px;
    border-radius: 12px 12px 0 0;
}
.aqi-table td {
    padding: 14px 12px;
    border-radius: 8px;
}
.aqi-table tr {
    transition: background-color 0.3s ease;
}
.aqi-table tr:hover td {
    background-color: #444;
}
.good {
    background-color: #0F980F !important;
    color: black !important;
    font-weight: 600;
}
.moderate {
    background-color: #E3BC3A !important;
    color: black !important;
    font-weight: 600;
}
.usg {
    background-color: #E7750A !important;
    color: black !important;
    font-weight: 600;
}
.unhealthy {
    background-color: #FF6A6A !important;
    color: black !important;
    font-weight: 600;
}
.very-unhealthy {
    background-color: #A07CC5 !important;
    color: white !important;
    font-weight: 600;
}
.hazardous {
    background-color: #A05252 !important;
    color: white !important;
    font-weight: 600;
}
</style>

<table class="aqi-table">
    <tr>
        <th>Kategori</th>
        <th>Rentang PM2.5 (µg/m³)</th>
        <th>Rekomendasi Kesehatan</th>
    </tr>
    <tr class="good"><td>Good</td><td>0 - 9.0</td><td>Kualitas udara baik, risiko minim atau tidak ada.</td></tr>
    <tr class="moderate"><td>Moderate</td><td>9.1 - 35.4</td><td>Individu sensitif sebaiknya mengurangi aktivitas di luar.</td></tr>
    <tr class="usg"><td>Unhealthy for Sensitive Groups</td><td>35.5 - 55.4</td><td>Berisiko iritasi/pernapasan untuk kelompok sensitif.</td></tr>
    <tr class="unhealthy"><td>Unhealthy</td><td>55.5 - 125.4</td><td>Efek buruk untuk umum, jantung/paru bisa terpengaruh.</td></tr>
    <tr class="very-unhealthy"><td>Very Unhealthy</td><td>125.5 - 225.4</td><td>Semua orang mulai terpengaruh, batasi aktivitas luar.</td></tr>
    <tr class="hazardous"><td>Hazardous</td><td>225.5+</td><td>Sangat berbahaya, semua orang hindari aktivitas luar.</td></tr>
</table>
""", unsafe_allow_html=True)


# --- Prediksi ---
with tab2:
    st.subheader("Hasil Prediksi PM2.5 untuk 3 Jam ke Depan")
    n_steps_in, n_steps_out = 6, 3
    fitur_model = ['PM10', 'PM2.5', 'CO', 'SO2', 'NO2', 'O3']

    def prepare_input(data, scaler):
        if len(data) < n_steps_in:
            return None
        scaled = scaler.transform(data)
        sequences = []
        for i in range(len(scaled) - n_steps_in + 1):
            sequences.append(scaled[i:i+n_steps_in])
        return np.array(sequences[-1:])

    latest_data = df[fitur_model].tail(n_steps_in)
    input_data = prepare_input(latest_data, scaler)

    if input_data is None:
        st.error("Data historis tidak cukup untuk prediksi.")
        st.stop()

    predicted = model.predict(input_data)
    predicted_inv = scaler.inverse_transform(predicted[0])
    pm25_values = predicted_inv[:, 1]

    now = datetime.datetime.now()
    time_labels = [(now + datetime.timedelta(hours=i+1)).strftime("%H:%M") for i in range(3)]

    for i in range(len(pm25_values)):
        kategori, warna = categorize_pm25(pm25_values[i])
        st.markdown(f"""
        <div style="display: flex; justify-content: space-between; align-items: center; background-color: #1E1E1E; padding: 10px 15px; border-radius: 8px; margin-bottom: 5px;">
            <div style="flex: 1; color: white;"><strong>Jam {time_labels[i]}</strong></div>
            <div style="width: 120px; text-align: center; background-color: {warna}; padding: 5px 10px; border-radius: 5px;"><strong>{pm25_values[i]:.2f} \u00b5g/m\u00b3</strong></div>
            <div style="flex: 1; text-align: right; color: white;">{kategori}</div>
        </div>
        """, unsafe_allow_html=True)

# --- Visualisasi ---
with tab3:
    st.subheader("Grafik Kualitas Udara Hari Ini")
    kategori_list = []
    warna_list = []
    for val in pm25_values:
        k, w = categorize_pm25(val)
        kategori_list.append(k)
        warna_list.append(w)

    fig = go.Figure()
    for i in range(len(pm25_values)-1):
        fig.add_trace(go.Scatter(
            x=[time_labels[i], time_labels[i+1]],
            y=[pm25_values[i], pm25_values[i+1]],
            mode='lines',
            line=dict(color=warna_list[i], width=3),
            showlegend=False
        ))
    fig.add_trace(go.Scatter(
        x=time_labels,
        y=pm25_values,
        mode='markers+text',
        marker=dict(color=warna_list, size=14, line=dict(width=1, color='white')),
        text=[f"{v:.2f}" for v in pm25_values],
        textposition='top center',
        customdata=kategori_list,
        hovertemplate='Jam: %{x}<br>PM2.5: %{y:.2f} \u00b5g/m\u00b3<br>Kategori: %{customdata}',
        showlegend=False
    ))
    fig.update_layout(
        title="Prediksi Kualitas Udara Dalam 3 Jam Kedepan",
        plot_bgcolor="#1E1E1E",
        paper_bgcolor="#1E1E1E",
        font=dict(color="white"),
        xaxis=dict(title="Waktu (Jam)", gridcolor='gray'),
        yaxis=dict(title="Konsentrasi PM2.5 (\u00b5g/m\u00b3)", gridcolor='gray'),
        hovermode="x unified",
        margin=dict(l=40, r=40, t=60, b=40)
    )
    st.plotly_chart(fig, use_container_width=True)
    
with tab4:
    st.subheader("Peta Lokasi Pemantauan Kualitas Udara di Yogyakarta")

    # Ambil nilai PM2.5 terbaru dan kategorinya
    current_pm25 = df['PM2.5'].iloc[1]
    kategori_pm25, warna_hex = categorize_pm25(current_pm25)

    # Konversi warna hex ke rgba untuk pydeck
    def hex_to_rgba(hex_color, alpha=160):
        hex_color = hex_color.lstrip('#')
        return [int(hex_color[i:i+2], 16) for i in (0, 2, 4)] + [alpha]

    warna_rgba = hex_to_rgba(warna_hex)

    # Data lokasi + info kualitas udara
    lokasi_yogyakarta = pd.DataFrame({
        'lat': [-7.7956],
        'lon': [110.3695],
        'pm25': [f"{current_pm25:.2f} µg/m³"],
        'kategori': [kategori_pm25],
    })

    # Tampilkan peta dengan indikator
    import pydeck as pdk
    layer = pdk.Layer(
        "ScatterplotLayer",
        data=lokasi_yogyakarta,
        get_position='[lon, lat]',
        get_color=warna_rgba,
        get_radius=3000,
        pickable=True
    )

    text_layer = pdk.Layer(
        "TextLayer",
        data=lokasi_yogyakarta,
        get_position='[lon, lat]',
        get_text='kategori',
        get_color=[255, 255, 255],
        get_size=16,
        get_alignment_baseline="'bottom'",
    )

    view_state = pdk.ViewState(
        latitude=-7.7956,
        longitude=110.3695,
        zoom=11,
        pitch=45,
    )

    st.pydeck_chart(pdk.Deck(
        map_style='mapbox://styles/mapbox/dark-v10',
        initial_view_state=view_state,
        layers=[layer, text_layer],
        tooltip={
            "html": "<b>Kategori:</b> {kategori}<br><b>PM2.5:</b> {pm25}",
            "style": {"color": "white"}
        }
    ))

    
st.markdown("---")

# Buat layout kolom, misalnya 1 kolom kecil untuk FAQ dan 1 kolom kosong untuk spasi
col_faq, col_space = st.columns([1.2, 3])

with col_faq:
    st.header("FAQ")
    with st.expander("Apa tingkat PM2.5 saat ini di Indonesia?"):
        st.write("Tingkat PM2.5 waktu nyata saat ini di Indonesia adalah 16 µg/m³ (Good). Ini terakhir diperbarui 2 minutes ago (Local Time).")

    with st.expander("Kapan tingkat PM2.5 terbaik di Indonesia dalam 24 jam terakhir?"):
        st.write("Tingkat PM2.5 terbaik adalah 13 µg/m³ (Good) pada 3:10 PM, Jun 3, 2025 (Local Time) dalam 24 jam terakhir.")

    with st.expander("Kapan tingkat PM2.5 terburuk di Indonesia dalam 24 jam terakhir?"):
        st.write("Tingkat PM2.5 terburuk adalah 20 µg/m³ (Good) pada 3:10 AM (Local Time) dalam 24 jam terakhir.")

    with st.expander("Apa tren tingkat PM2.5 saat ini di Indonesia selama 24 jam terakhir?"):
        st.write("Tingkat PM2.5 di Indonesia telah berfluktuasi sepanjang 24 jam terakhir. Tingkatnya meningkat paling tinggi 20 µg/m³ pada 3:10 AM (Local Time), terendah 13 µg/m³ pada 3:10 PM, Jun 3, 2025 (Local Time).")

    with st.expander("Tindakan apa yang direkomendasikan sesuai dengan tingkat PM2.5 saat ini di Indonesia?"):
        st.write("Kualitas udara memuaskan, dan polusi udara menimbulkan sedikit atau tidak ada risiko. Tidak ada langkah pencegahan khusus yang diperlukan; nikmati aktivitas luar ruangan dengan bebas.")
