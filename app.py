import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error
)

from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.exponential_smoothing.ets import ETSModel

# ==========================================
# CONFIG HALAMAN
# ==========================================

st.set_page_config(
    page_title="Forecasting Barang",
    page_icon="📦",
    layout="wide"
)

# ==========================================
# CUSTOM CSS
# ==========================================

st.markdown("""
<style>
.main { background-color: #f5f7fa; }
h1 { color: #1f4e79; text-align: center; font-weight: bold; }
h2, h3 { color: #1f4e79; }
.stButton>button {
    background-color: #1f77b4; color: white;
    border-radius: 10px; border: none;
    padding: 10px 20px; font-weight: bold;
}
.stDownloadButton>button {
    background-color: #28a745; color: white;
    border-radius: 10px; border: none;
    padding: 10px 20px; font-weight: bold;
}
[data-testid="stMetricValue"] { color: #1f77b4; font-size: 28px; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# CLUSTER-METHOD MAPPING
# ==========================================

CLUSTER_METHOD_MAP = {
    'Fast Moving': {
        'method':      'HW Multiplicative',
        'label':       'Holt-Winters Multiplicative',
        'icon':        '🔴',
        'rationale': (
            "**Holt-Winters Multiplicative** dipilih karena produk fast moving "
            "memiliki volume permintaan tinggi dengan variasi musiman yang "
            "*proporsional* terhadap level permintaan (seasonal swing membesar "
            "saat volume naik). Model multiplicative menangkap pola ini lebih "
            "akurat dibanding model additive."
        )
    },
    'Medium Moving': {
        'method':      'ETS',
        'label':       'ETS (Additive-Additive-Additive)',
        'icon':        '🟡',
        'rationale': (
            "**ETS (A,A,A)** dipilih karena secara struktur identik dengan "
            "Holt-Winters Additive, namun semua parameter dioptimasi secara "
            "otomatis lewat *Maximum Likelihood Estimation*. Hasilnya lebih "
            "presisi untuk produk medium moving dengan pola tren dan musiman "
            "yang sedang, sekaligus lebih tahan overfitting pada data 24 bulan."
        )
    },
    'Slow Moving': {
        'method':      'Croston',
        'label':       "Croston's Method",
        'icon':        '🟢',
        'rationale': (
            "**Croston's Method** dipilih karena dirancang khusus untuk "
            "*intermittent demand* — permintaan sporadis dengan banyak nilai nol. "
            "Cara kerjanya memisahkan **ukuran demand** dan **interval antar demand** "
            "lalu menerapkan exponential smoothing pada keduanya secara terpisah. "
            "ARIMA tidak cocok untuk data banyak nol karena bisa menghasilkan "
            "forecast negatif atau tidak stabil."
        )
    }
}

# ==========================================
# JUDUL
# ==========================================

st.title("📦 Clustering dan Forecasting Permintaan Barang")

st.markdown("""
### Sistem Analisis Barang Keluar
Aplikasi ini digunakan untuk:
- Clustering produk menggunakan K-Means
- Forecasting permintaan barang **berdasarkan metode yang sesuai per cluster**
- Perbandingan hasil forecast vs data aktual
""")

# ==========================================
# METODE PER CLUSTER — INFO BOX
# ==========================================

with st.expander("ℹ️ Panduan: Metode Forecasting per Cluster", expanded=False):
    st.markdown("""
    | Cluster | Metode Forecasting | Alasan Pemilihan |
    |---|---|---|
    | 🔴 Fast Moving | **Holt-Winters Multiplicative** | Volume tinggi, seasonal swing *proporsional* dengan level permintaan |
    | 🟡 Medium Moving | **ETS (A,A,A)** | Struktur seperti HW Additive namun parameter dioptimasi otomatis (MLE), lebih akurat & tahan overfitting |
    | 🟢 Slow Moving | **Croston's Method** | Dirancang khusus untuk data *intermittent* (banyak nol), memisahkan ukuran & interval demand |

    Metode ini dipilih secara otomatis sesuai cluster produk yang Anda pilih.
    Anda tetap dapat mengganti metode secara manual jika diperlukan.

    **Catatan Croston's Method:**
    Menghasilkan forecast *flat* (konstan) — ini normal dan merupakan karakteristik metode ini.
    Nilai forecast = rata-rata ukuran demand ÷ rata-rata interval demand.
    Parameter **alpha** mengontrol seberapa cepat model beradaptasi (nilai kecil = lebih stabil, nilai besar = lebih responsif).
    
    **Catatan Data 2 Tahun (24 Bulan):**
    Dengan data 24 bulan, seasonality yang digunakan adalah **6 bulan** (semi-tahunan),
    karena model memerlukan minimal 2 siklus penuh. Seasonal period 12 bulan baru aktif
    jika data training ≥ 24 bulan.
    """)

# ==========================================
# UPLOAD FILE
# ==========================================

uploaded_file = st.file_uploader("Upload File Excel", type=["xlsx"])

# ==========================================
# JIKA FILE SUDAH DIUPLOAD
# ==========================================

if uploaded_file is not None:

    df = pd.read_excel(uploaded_file)

    # ==========================================
    # [FIX 1] PEMBERSIHAN DATA
    # Data mentah memiliki NaN pada kolom keluar
    # (~25% baris) dan id_produk (beberapa baris).
    # Baris tersebut dibuang agar tidak mempengaruhi
    # pivot table dan metric dashboard.
    # ==========================================
    df = df.dropna(subset=['id_produk', 'keluar'])
    df['keluar'] = df['keluar'].astype(int)

    st.subheader("📄 Data Awal")
    st.dataframe(df.head())

    # DASHBOARD
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Jumlah Data", len(df))
    with col2:
        st.metric("Jumlah Produk", df['id_produk'].nunique())
    with col3:
        st.metric("Total Barang Keluar", int(df['keluar'].sum()))

    # FORMAT TANGGAL & BULAN
    df['tgl_input'] = pd.to_datetime(df['tgl_input'])
    df['Bulan'] = df['tgl_input'].dt.strftime('%b-%y')

    # ==========================================
    # DETEKSI OTOMATIS RENTANG BULAN DARI DATA
    # ==========================================

    all_months_sorted = sorted(
        df['Bulan'].unique(),
        key=lambda x: pd.to_datetime(x, format='%b-%y')
    )

    # PIVOT TABLE
    pivot_table = df.pivot_table(
        index='id_produk',
        columns='Bulan',
        values='keluar',
        aggfunc='sum',
        fill_value=0
    )

    pivot_table = pivot_table.reindex(columns=all_months_sorted, fill_value=0)

    # Simpan informasi periode untuk label forecast
    first_month = pd.to_datetime(all_months_sorted[0],  format='%b-%y')
    total_months = len(all_months_sorted)

    st.subheader("📊 Pivot Table Barang Keluar")
    st.dataframe(pivot_table)

    csv_data = pivot_table.to_csv().encode('utf-8')
    st.download_button(
        label="⬇️ Download Pivot Table",
        data=csv_data,
        file_name='data_barang_keluar.csv',
        mime='text/csv'
    )

    # ==========================================
    # CLUSTERING
    # ==========================================

    st.header("📌 Clustering Produk")

    pivot_table['Total'] = pivot_table.sum(axis=1)
    filtered_data = pivot_table[pivot_table['Total'] > 1].copy()

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(filtered_data.drop(columns=['Total']))

    # ELBOW METHOD
    inertia = []
    K = range(1, 10)
    for k in K:
        km = KMeans(n_clusters=k, random_state=42)
        km.fit(scaled_data)
        inertia.append(km.inertia_)

    fig1, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(K, inertia, marker='o')
    ax1.set_title('Metode Elbow')
    ax1.set_xlabel('Jumlah Cluster')
    ax1.set_ylabel('Inertia')
    ax1.grid(True, linestyle='--', alpha=0.5)
    st.pyplot(fig1)

    jumlah_cluster = st.slider("Pilih Jumlah Cluster", min_value=2, max_value=10, value=3)

    kmeans = KMeans(n_clusters=jumlah_cluster, random_state=42)
    cluster = kmeans.fit_predict(scaled_data)
    filtered_data['Cluster'] = cluster

    # FAST - MEDIUM - SLOW berdasarkan rata-rata Total
    cluster_avg = filtered_data.groupby('Cluster')['Total'].mean().sort_values(ascending=False)
    mapping_cluster = {}
    if len(cluster_avg) >= 3:
        mapping_cluster[cluster_avg.index[0]] = 'Fast Moving'
        mapping_cluster[cluster_avg.index[1]] = 'Medium Moving'
        mapping_cluster[cluster_avg.index[-1]] = 'Slow Moving'
    for idx in cluster_avg.index:
        if idx not in mapping_cluster:
            mapping_cluster[idx] = 'Medium Moving'

    filtered_data['Kategori'] = filtered_data['Cluster'].map(mapping_cluster)

    cluster_count = filtered_data['Kategori'].value_counts().reindex(
        ['Fast Moving', 'Medium Moving', 'Slow Moving']
    ).fillna(0).astype(int)

    tabel_cluster = pd.DataFrame({
        'Kategori':      cluster_count.index,
        'Jumlah Produk': cluster_count.values,
        'Metode Forecast': [
            CLUSTER_METHOD_MAP[k]['label'] for k in cluster_count.index
        ]
    })
    tabel_cluster.index = range(1, len(tabel_cluster) + 1)

    st.subheader("📊 Jumlah Produk per Cluster & Metode Forecasting")
    st.dataframe(tabel_cluster, use_container_width=True)

    for kat, info in CLUSTER_METHOD_MAP.items():
        if kat in cluster_count.index:
            st.info(f"{info['icon']} **{kat}** → {info['label']}\n\n{info['rationale']}")

    fig_cluster, ax_cluster = plt.subplots(figsize=(8, 5))
    colors = ['#e74c3c', '#f39c12', '#27ae60']
    bars = ax_cluster.bar(
        tabel_cluster['Kategori'],
        tabel_cluster['Jumlah Produk'],
        color=colors[:len(tabel_cluster)]
    )
    for bar, val in zip(bars, tabel_cluster['Jumlah Produk']):
        ax_cluster.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            str(val), ha='center', va='bottom', fontweight='bold'
        )
    ax_cluster.set_title('Distribusi Produk per Cluster')
    ax_cluster.set_xlabel('Kategori Cluster')
    ax_cluster.set_ylabel('Jumlah Produk')
    ax_cluster.grid(True, linestyle='--', alpha=0.5)
    st.pyplot(fig_cluster)

    pilih_cluster = st.selectbox(
        "Pilih Cluster",
        ['Fast Moving', 'Medium Moving', 'Slow Moving']
    )

    produk_cluster = filtered_data[
        filtered_data['Kategori'] == pilih_cluster
    ].index.tolist()

    st.subheader(f"📦 Produk dalam {pilih_cluster}")
    df_produk_cluster = pd.DataFrame({'Produk': produk_cluster})
    df_produk_cluster.index = range(1, len(df_produk_cluster) + 1)
    st.dataframe(df_produk_cluster, use_container_width=True)

    # ==========================================
    # FORECASTING
    # ==========================================

    st.header("📈 Forecasting Barang")

    produk = st.selectbox("Pilih Produk", produk_cluster)

    data_produk = filtered_data.loc[produk].copy()
    data_produk = data_produk.drop(['Cluster', 'Kategori', 'Total'])
    data_produk = pd.to_numeric(data_produk)

    data_produk.index = pd.date_range(
        start=first_month,
        periods=len(data_produk),
        freq='ME'
    )

    jumlah_bulan_aktif = (data_produk > 0).sum()
    if jumlah_bulan_aktif < 6:
        st.warning(
            "⚠️ Produk ini hanya aktif kurang dari 6 bulan sehingga "
            "hasil forecasting kurang reliabel."
        )

    # ==========================================
    # AUTO-SELECT METODE BERDASARKAN CLUSTER
    # ==========================================

    metode_rekomendasi = CLUSTER_METHOD_MAP[pilih_cluster]['method']
    label_rekomendasi  = CLUSTER_METHOD_MAP[pilih_cluster]['label']
    icon_cluster       = CLUSTER_METHOD_MAP[pilih_cluster]['icon']

    st.success(
        f"{icon_cluster} Produk ini termasuk **{pilih_cluster}** → "
        f"Metode rekomendasi: **{label_rekomendasi}**"
    )

    daftar_metode = [
        "HW Additive",
        "HW Multiplicative",
        "ETS",
        "ARIMA",
        "Croston",
        "Perbandingan Semua Metode"
    ]
    default_idx = daftar_metode.index(metode_rekomendasi)

    metode = st.selectbox(
        "Metode Forecasting (otomatis dipilih sesuai cluster, bisa diubah manual)",
        daftar_metode,
        index=default_idx
    )

    jumlah_forecast = st.slider("Jumlah Forecast Bulan", 1, 12, 6)

    croston_alpha = 0.1
    if metode == 'Croston' or metode == 'Perbandingan Semua Metode':
        croston_alpha = st.slider(
            "Alpha Croston (smoothing parameter) — semakin kecil = lebih stabil",
            min_value=0.05, max_value=0.50, value=0.10, step=0.05,
            help="Alpha mengontrol seberapa cepat Croston's Method beradaptasi terhadap perubahan demand. "
                 "Nilai 0.1–0.2 direkomendasikan untuk slow moving items."
        )

    # ==========================================
    # TRAIN-TEST SPLIT
    # ==========================================

    n = len(data_produk)
    train = data_produk.iloc[:n - jumlah_forecast]
    test  = data_produk.iloc[n - jumlah_forecast:]

    if len(train) < 6:
        st.error(
            f"Data train hanya {len(train)} bulan. "
            "Kurangi jumlah forecast agar data train minimal 6 bulan."
        )
        st.stop()

    st.info(
        f"📌 Train: **{len(train)} bulan**  |  Test: **{len(test)} bulan**  |  "
        "MAE & RMSE dihitung dari forecast vs data test."
    )

    # ==========================================
    # [FIX 2] SEASONAL PERIODS OTOMATIS
    #
    # PERUBAHAN dari kode sebelumnya:
    #   Lama: n_train >= 25 → sp=12, n_train >= 13 → sp=6
    #   Baru: n_train >= 24 → sp=12, n_train >= 12 → sp=6
    #
    # Alasan:
    # - sp=12 membutuhkan minimal 2×12=24 titik data train.
    #   Threshold 25 tidak akan pernah tercapai dengan data
    #   24 bulan (train maksimal = 23 saat forecast=1).
    #   Threshold 24 lebih tepat secara statistik.
    #
    # - sp=6 membutuhkan minimal 2×6=12 titik data train.
    #   Threshold lama (13) menyebabkan bug: ketika
    #   forecast=12, n_train=12 → sp=1 (tanpa seasonality!).
    #   Threshold baru (12) memperbaiki ini.
    # ==========================================

    def get_sp(n_train):
        if n_train >= 24:   # sp=12: butuh minimal 2×12=24 data  [DIUBAH dari 25]
            return 12
        elif n_train >= 12: # sp=6 : butuh minimal 2×6 =12 data  [DIUBAH dari 13]
            return 6
        else:
            return 1        # data terlalu sedikit → tanpa seasonality

    # ==========================================
    # GRAFIK DATA AKTUAL
    # ==========================================

    fig2, ax2 = plt.subplots(figsize=(12, 5))
    ax2.plot(
        data_produk.index, data_produk.values,
        marker='o', linewidth=2, label='Data Aktual'
    )
    ax2.axvline(
        x=test.index[0], color='red', linestyle='--',
        alpha=0.7, label='Awal Periode Test'
    )
    ax2.set_title(f'Data Aktual Produk {produk}')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.5)
    st.pyplot(fig2)

    # ==========================================
    # WARNA PER METODE
    # ==========================================

    warna_metode = {
        'HW Additive':       'green',
        'HW Multiplicative': 'red',
        'ETS':               'purple',
        'ARIMA':             'brown',
        'Croston':           'teal'
    }

    # ==========================================
    # HELPER: CROSTON'S METHOD
    # ==========================================

    def croston_forecast(series, steps, alpha=0.1):
        data = series.values.astype(float)
        n    = len(data)

        non_zero = np.where(data > 0)[0]

        if len(non_zero) == 0:
            idx = pd.date_range(series.index[-1], periods=steps + 1, freq='ME')[1:]
            return pd.Series(np.zeros(steps), index=idx)

        z = float(data[non_zero[0]])
        p = float(non_zero[0] + 1)
        q = 1

        for i in range(non_zero[0] + 1, n):
            if data[i] > 0:
                z = alpha * data[i] + (1 - alpha) * z
                p = alpha * q       + (1 - alpha) * p
                q = 1
            else:
                q += 1

        rate = max(0.0, z / p)

        idx = pd.date_range(series.index[-1], periods=steps + 1, freq='ME')[1:]
        return pd.Series(np.full(steps, round(rate, 4)), index=idx)

    # ==========================================
    # HELPER: FIT MODEL (TRAIN)
    # ==========================================

    def fit_model(nama_metode, train_data, croston_alpha=0.1):
        sp = get_sp(len(train_data))
        if nama_metode == 'HW Additive':
            if sp > 1:
                return ExponentialSmoothing(
                    train_data, trend='add', seasonal='add', seasonal_periods=sp
                ).fit()
            else:
                return ExponentialSmoothing(train_data, trend='add').fit()
        elif nama_metode == 'HW Multiplicative':
            train_nz = train_data.copy()
            train_nz[train_nz <= 0] = 1
            if sp > 1:
                return ExponentialSmoothing(
                    train_nz, trend='add', seasonal='mul', seasonal_periods=sp
                ).fit()
            else:
                return ExponentialSmoothing(train_nz, trend='add').fit()
        elif nama_metode == 'ETS':
            if sp > 1:
                return ETSModel(
                    train_data, error="add", trend="add",
                    seasonal="add", seasonal_periods=sp
                ).fit(disp=False)
            else:
                return ETSModel(
                    train_data, error="add", trend="add"
                ).fit(disp=False)
        elif nama_metode == 'ARIMA':
            return ARIMA(train_data, order=(1, 1, 1)).fit()
        elif nama_metode == 'Croston':
            return {'series': train_data, 'alpha': croston_alpha, 'type': 'croston'}
        else:
            raise ValueError(f"Metode tidak dikenal: {nama_metode}")

    def forecast_model(model, nama_metode, steps, croston_alpha=0.1):
        if nama_metode == 'Croston':
            return croston_forecast(model['series'], steps, model['alpha'])
        elif nama_metode == 'ARIMA':
            return model.forecast(steps=steps)
        else:
            return model.forecast(steps).clip(lower=0)

    # ==========================================
    # HELPER: RETRAIN DAN FORECAST KE DEPAN
    # ==========================================

    def retrain_full(nama_metode, full_data, steps, croston_alpha=0.1):
        model = fit_model(nama_metode, full_data, croston_alpha)
        return forecast_model(model, nama_metode, steps, croston_alpha)

    # ==========================================
    # FUNGSI TAMPILKAN HASIL (1 METODE)
    # ==========================================

    def tampilkan_hasil(nama_metode, forecast_eval, test_actual, is_rekomendasi=False):

        label_rekomendasi_txt = " ✅ (Rekomendasi Cluster)" if is_rekomendasi else ""

        mae  = mean_absolute_error(test_actual.values, forecast_eval.values)
        rmse = np.sqrt(mean_squared_error(test_actual.values, forecast_eval.values))

        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.metric("MAE", f"{mae:.2f}")
        with col_m2:
            st.metric("RMSE", f"{rmse:.2f}")

        eval_df = pd.DataFrame({
            'Periode':              forecast_eval.index.strftime('%b-%Y'),
            'Hasil Forecast':       np.round(forecast_eval.values, 2),
            'Data Aktual (Test)':   np.round(test_actual.values, 2)
        })
        eval_df.index = range(1, len(eval_df) + 1)

        st.subheader(f"📋 Hasil Forecast vs Aktual — {nama_metode}{label_rekomendasi_txt}")
        st.dataframe(eval_df, use_container_width=True)

        fig_eval, ax_eval = plt.subplots(figsize=(12, 5))
        ax_eval.plot(
            train.index, train.values,
            marker='o', color='steelblue', label='Data Train'
        )
        ax_eval.plot(
            test_actual.index, test_actual.values,
            marker='o', color='orange', label='Data Test (Aktual)'
        )
        ax_eval.plot(
            forecast_eval.index, forecast_eval.values,
            marker='o', linestyle='--',
            color=warna_metode.get(nama_metode, 'green'),
            label=f'Forecast {nama_metode}{label_rekomendasi_txt}'
        )
        ax_eval.axvline(
            x=test_actual.index[0], color='gray',
            linestyle=':', alpha=0.7, label='Awal Periode Test'
        )
        ax_eval.set_title(
            f'Evaluasi Model — {nama_metode} — Produk {produk}'
        )
        ax_eval.legend()
        ax_eval.grid(True, linestyle='--', alpha=0.5)
        st.pyplot(fig_eval)

        st.subheader(f"🔮 Forecast ke Depan — {nama_metode}{label_rekomendasi_txt}")
        st.info(
            f"Model di-retrain menggunakan seluruh **{len(data_produk)} bulan** data, "
            "lalu meramalkan bulan-bulan berikutnya."
        )

        future_forecast = retrain_full(nama_metode, data_produk, jumlah_forecast, croston_alpha)

        future_df = pd.DataFrame({
            'Periode':        future_forecast.index.strftime('%b-%Y'),
            'Hasil Forecast': np.round(future_forecast.values, 2)
        })
        future_df.index = range(1, len(future_df) + 1)
        st.dataframe(future_df, use_container_width=True)

        csv_future = future_df.to_csv().encode('utf-8')
        st.download_button(
            label=f"⬇️ Download Forecast ke Depan ({nama_metode})",
            data=csv_future,
            file_name=f'forecast_ke_depan_{produk}_{nama_metode.replace(" ", "_")}.csv',
            mime='text/csv',
            key=f'dl_{nama_metode}'
        )

        fig_future, ax_future = plt.subplots(figsize=(12, 5))
        ax_future.plot(
            data_produk.index, data_produk.values,
            marker='o', linewidth=2, color='steelblue',
            label=f'Data Aktual ({len(data_produk)} bulan)'
        )
        ax_future.plot(
            future_forecast.index, future_forecast.values,
            marker='o', linestyle='--',
            color=warna_metode.get(nama_metode, 'green'),
            linewidth=2,
            label=f'Forecast ke Depan ({nama_metode})'
        )
        ax_future.axvline(
            x=future_forecast.index[0], color='red',
            linestyle='--', alpha=0.7, label='Awal Periode Forecast'
        )
        ax_future.set_title(
            f'Forecast ke Depan — {nama_metode} — Produk {produk}'
        )
        ax_future.legend()
        ax_future.grid(True, linestyle='--', alpha=0.5)
        st.pyplot(fig_future)

        return mae, rmse

    # ==========================================
    # EKSEKUSI BERDASARKAN METODE TERPILIH
    # ==========================================

    is_rekomendasi = (metode == metode_rekomendasi)

    if metode != "Perbandingan Semua Metode":
        try:
            model_fit  = fit_model(metode, train, croston_alpha)
            fc_eval    = forecast_model(model_fit, metode, jumlah_forecast, croston_alpha)
            tampilkan_hasil(metode, fc_eval, test, is_rekomendasi=is_rekomendasi)
        except Exception as e:
            st.error(f"❌ Metode {metode} gagal: {e}")

    # ==========================================
    # PERBANDINGAN SEMUA METODE
    # ==========================================

    else:
        hasil_eval   = {}
        hasil_future = {}

        for nm in ['HW Additive', 'HW Multiplicative', 'ETS', 'ARIMA', 'Croston']:
            try:
                m   = fit_model(nm, train, croston_alpha)
                fc  = forecast_model(m, nm, jumlah_forecast, croston_alpha)
                hasil_eval[nm] = {
                    'forecast': fc,
                    'mae':  mean_absolute_error(test.values, fc.values),
                    'rmse': np.sqrt(mean_squared_error(test.values, fc.values))
                }
                hasil_future[nm] = retrain_full(nm, data_produk, jumlah_forecast, croston_alpha)
            except Exception as e:
                st.warning(f"⚠️ Metode {nm} dilewati karena error: {e}")

        if not hasil_eval:
            st.error("Semua metode gagal. Coba kurangi jumlah forecast.")
            st.stop()

        perbandingan = pd.DataFrame([
            {
                'Metode':             m,
                'MAE':                round(v['mae'],  2),
                'RMSE':               round(v['rmse'], 2),
                'Rekomendasi Cluster': '✅' if m == metode_rekomendasi else ''
            }
            for m, v in hasil_eval.items()
        ])
        perbandingan.index = range(1, len(perbandingan) + 1)

        st.subheader("📊 Perbandingan MAE & RMSE Semua Metode")
        st.dataframe(perbandingan, use_container_width=True)

        fig_mae, ax_mae = plt.subplots(figsize=(10, 5))
        bar_colors = [
            '#e74c3c' if m == metode_rekomendasi else warna_metode.get(m, 'gray')
            for m in perbandingan['Metode']
        ]
        bars = ax_mae.bar(perbandingan['Metode'], perbandingan['MAE'], color=bar_colors)
        ax_mae.bar_label(bars, fmt='%.2f', padding=3)
        ax_mae.set_title('Perbandingan Nilai MAE — Semua Metode\n(merah = rekomendasi cluster)')
        ax_mae.set_ylabel('MAE')
        ax_mae.grid(True, linestyle='--', alpha=0.5)
        st.pyplot(fig_mae)

        fig_rmse, ax_rmse = plt.subplots(figsize=(10, 5))
        bars2 = ax_rmse.bar(perbandingan['Metode'], perbandingan['RMSE'], color=bar_colors)
        ax_rmse.bar_label(bars2, fmt='%.2f', padding=3)
        ax_rmse.set_title('Perbandingan Nilai RMSE — Semua Metode\n(merah = rekomendasi cluster)')
        ax_rmse.set_ylabel('RMSE')
        ax_rmse.grid(True, linestyle='--', alpha=0.5)
        st.pyplot(fig_rmse)

        best_idx       = perbandingan['MAE'].idxmin()
        metode_terbaik = perbandingan.loc[best_idx]

        if metode_terbaik['Metode'] == metode_rekomendasi:
            st.success(
                f"✅ Metode terbaik adalah **{metode_terbaik['Metode']}** "
                f"(MAE={metode_terbaik['MAE']:.2f}, RMSE={metode_terbaik['RMSE']:.2f}) "
                f"— sesuai rekomendasi cluster **{pilih_cluster}**!"
            )
        else:
            st.warning(
                f"ℹ️ Metode terbaik secara metrik adalah **{metode_terbaik['Metode']}** "
                f"(MAE={metode_terbaik['MAE']:.2f}, RMSE={metode_terbaik['RMSE']:.2f}). "
                f"Rekomendasi cluster (**{metode_rekomendasi}**) berada di posisi lain."
            )

        st.subheader("📉 Grafik Evaluasi Gabungan (Train vs Test vs Forecast)")
        fig_eval, ax_eval = plt.subplots(figsize=(14, 6))
        ax_eval.plot(train.index, train.values, marker='o', linewidth=2, color='steelblue', label='Data Train')
        ax_eval.plot(test.index,  test.values,  marker='o', linewidth=2, color='orange',   label='Data Test (Aktual)')
        for nm, v in hasil_eval.items():
            lw = 2.5 if nm == metode_rekomendasi else 1.5
            ls = '-' if nm == metode_rekomendasi else '--'
            ax_eval.plot(
                v['forecast'].index, v['forecast'].values,
                linestyle=ls, marker='s', linewidth=lw,
                color=warna_metode[nm],
                label=f'Forecast {nm}{"  ← rekomendasi" if nm == metode_rekomendasi else ""}'
            )
        ax_eval.axvline(x=test.index[0], color='gray', linestyle=':', alpha=0.7, label='Awal Test')
        ax_eval.set_title(f'Evaluasi Gabungan Semua Metode — Produk {produk}')
        ax_eval.legend(loc='upper left')
        ax_eval.grid(True, linestyle='--', alpha=0.5)
        st.pyplot(fig_eval)

        st.subheader("🔮 Forecast ke Depan — Semua Metode")
        st.info(
            f"Semua model di-retrain menggunakan seluruh **{len(data_produk)} bulan** data, "
            "lalu meramalkan bulan-bulan berikutnya."
        )

        future_index = list(hasil_future.values())[0].index
        future_combined = pd.DataFrame({'Periode': future_index.strftime('%b-%Y')})
        for nm, fc in hasil_future.items():
            future_combined[nm] = np.round(fc.values, 2)
        future_combined.index = range(1, len(future_combined) + 1)
        st.dataframe(future_combined, use_container_width=True)

        csv_future_all = future_combined.to_csv().encode('utf-8')
        st.download_button(
            label="⬇️ Download Forecast ke Depan (Semua Metode)",
            data=csv_future_all,
            file_name=f'forecast_ke_depan_{produk}_semua_metode.csv',
            mime='text/csv',
            key='dl_semua'
        )

        fig_fut_all, ax_fut_all = plt.subplots(figsize=(14, 6))
        ax_fut_all.plot(
            data_produk.index, data_produk.values,
            marker='o', linewidth=2, color='steelblue',
            label=f'Data Aktual ({len(data_produk)} bulan)'
        )
        for nm, fc in hasil_future.items():
            lw = 2.5 if nm == metode_rekomendasi else 1.5
            ls = '-' if nm == metode_rekomendasi else '--'
            ax_fut_all.plot(
                fc.index, fc.values,
                linestyle=ls, marker='s', linewidth=lw,
                color=warna_metode[nm],
                label=f'Forecast — {nm}{"  ← rekomendasi" if nm == metode_rekomendasi else ""}'
            )
        ax_fut_all.axvline(
            x=list(hasil_future.values())[0].index[0],
            color='red', linestyle='--', alpha=0.7, label='Awal Forecast'
        )
        ax_fut_all.set_title(f'Forecast ke Depan Gabungan — Produk {produk}')
        ax_fut_all.legend(loc='upper left')
        ax_fut_all.grid(True, linestyle='--', alpha=0.5)
        st.pyplot(fig_fut_all)

        if metode_rekomendasi in hasil_future:
            st.subheader(
                f"🏷️ Forecast ke Depan — Metode Rekomendasi Cluster "
                f"({pilih_cluster}): {metode_rekomendasi}"
            )
            fc_rek = hasil_future[metode_rekomendasi]
            rek_df = pd.DataFrame({
                'Periode':        fc_rek.index.strftime('%b-%Y'),
                'Hasil Forecast': np.round(fc_rek.values, 2)
            })
            rek_df.index = range(1, len(rek_df) + 1)
            st.dataframe(rek_df, use_container_width=True)

            csv_rek = rek_df.to_csv().encode('utf-8')
            st.download_button(
                label=f"⬇️ Download Forecast Rekomendasi ({metode_rekomendasi})",
                data=csv_rek,
                file_name=f'forecast_rekomendasi_{produk}_{metode_rekomendasi.replace(" ", "_")}.csv',
                mime='text/csv',
                key='dl_rekomendasi'
            )

            fig_rek, ax_rek = plt.subplots(figsize=(12, 5))
            ax_rek.plot(
                data_produk.index, data_produk.values,
                marker='o', linewidth=2, color='steelblue',
                label=f'Data Aktual ({len(data_produk)} bulan)'
            )
            ax_rek.plot(
                fc_rek.index, fc_rek.values,
                marker='o', linestyle='-', linewidth=2.5,
                color=warna_metode.get(metode_rekomendasi, 'green'),
                label=f'Forecast — {metode_rekomendasi} (Rekomendasi {pilih_cluster})'
            )
            ax_rek.axvline(
                x=fc_rek.index[0], color='red',
                linestyle='--', alpha=0.7, label='Awal Forecast'
            )
            ax_rek.set_title(
                f'Forecast Rekomendasi Cluster — {metode_rekomendasi} — Produk {produk}'
            )
            ax_rek.legend()
            ax_rek.grid(True, linestyle='--', alpha=0.5)
            st.pyplot(fig_rek)
