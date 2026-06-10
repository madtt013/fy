import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import matplotlib.colors as mcolors

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
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==========================================
# CUSTOM CSS — PREMIUM DARK THEME
# ==========================================

st.markdown("""
<style>
/* ── IMPORT FONTS ── */
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=Syne:wght@700;800&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── ROOT VARIABLES ── */
:root {
    --bg-base:      #0d0f14;
    --bg-card:      #13161e;
    --bg-hover:     #1a1e2a;
    --accent-blue:  #4f9eff;
    --accent-cyan:  #00d4c8;
    --accent-amber: #ffb547;
    --accent-red:   #ff6b6b;
    --accent-green: #4ade80;
    --accent-purple:#a78bfa;
    --text-primary: #e8eaf0;
    --text-muted:   #8892a4;
    --border:       rgba(255,255,255,0.07);
    --glow-blue:    0 0 24px rgba(79,158,255,0.3);
    --glow-cyan:    0 0 24px rgba(0,212,200,0.3);
    --radius:       14px;
    --radius-sm:    8px;
}

/* ── GLOBAL RESET ── */
html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif !important;
    color: var(--text-primary) !important;
}

.main, .block-container {
    background: var(--bg-base) !important;
    padding: 2rem 2.5rem !important;
    max-width: 1400px;
}

/* ── HERO HEADER ── */
.hero-wrapper {
    background: linear-gradient(135deg, #0d0f14 0%, #111827 50%, #0d0f14 100%);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 3rem 3.5rem;
    margin-bottom: 2.5rem;
    position: relative;
    overflow: hidden;
}
.hero-wrapper::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 280px; height: 280px;
    background: radial-gradient(circle, rgba(79,158,255,0.12) 0%, transparent 70%);
    pointer-events: none;
}
.hero-wrapper::after {
    content: '';
    position: absolute;
    bottom: -80px; left: 40%;
    width: 320px; height: 200px;
    background: radial-gradient(circle, rgba(0,212,200,0.08) 0%, transparent 70%);
    pointer-events: none;
}
.hero-badge {
    display: inline-flex; align-items: center; gap: 6px;
    background: rgba(79,158,255,0.12);
    border: 1px solid rgba(79,158,255,0.3);
    color: var(--accent-blue);
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px; font-weight: 500;
    padding: 4px 12px;
    border-radius: 20px;
    letter-spacing: 0.08em;
    margin-bottom: 1rem;
}
.hero-title {
    font-family: 'Syne', sans-serif !important;
    font-size: 3rem !important;
    font-weight: 800 !important;
    color: #ffffff !important;
    line-height: 1.1 !important;
    margin: 0 0 1rem 0 !important;
    letter-spacing: -0.03em;
}
.hero-title span {
    background: linear-gradient(90deg, var(--accent-blue), var(--accent-cyan));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.hero-subtitle {
    color: var(--text-muted) !important;
    font-size: 1.05rem !important;
    line-height: 1.7 !important;
    max-width: 600px;
    margin: 0 !important;
}
.hero-pills {
    display: flex; gap: 10px; flex-wrap: wrap;
    margin-top: 1.8rem;
}
.hero-pill {
    display: inline-flex; align-items: center; gap: 6px;
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    color: var(--text-muted);
    font-size: 0.82rem;
    padding: 6px 14px;
    border-radius: 20px;
}
.hero-pill .dot {
    width: 6px; height: 6px;
    border-radius: 50%;
    display: inline-block;
}

/* ── SECTION HEADERS ── */
h1, h2, h3 {
    font-family: 'Syne', sans-serif !important;
    color: #ffffff !important;
}
.section-header {
    display: flex; align-items: center; gap: 12px;
    margin: 2.5rem 0 1.5rem 0;
}
.section-header .icon-box {
    width: 38px; height: 38px;
    border-radius: 10px;
    background: linear-gradient(135deg, var(--accent-blue), var(--accent-cyan));
    display: flex; align-items: center; justify-content: center;
    font-size: 18px; flex-shrink: 0;
}
.section-header h2 {
    font-family: 'Syne', sans-serif !important;
    font-size: 1.5rem !important;
    font-weight: 700 !important;
    color: #ffffff !important;
    margin: 0 !important;
}

/* ── STAT CARDS ── */
.stats-row {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 16px;
    margin: 1.5rem 0;
}
.stat-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.5rem 1.75rem;
    position: relative; overflow: hidden;
    transition: border-color 0.2s, transform 0.2s;
}
.stat-card:hover {
    border-color: rgba(79,158,255,0.3);
    transform: translateY(-2px);
}
.stat-card::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0;
    height: 2px;
    border-radius: var(--radius) var(--radius) 0 0;
}
.stat-card.blue::before  { background: linear-gradient(90deg, var(--accent-blue), var(--accent-cyan)); }
.stat-card.amber::before { background: linear-gradient(90deg, var(--accent-amber), #ff9f43); }
.stat-card.green::before { background: linear-gradient(90deg, var(--accent-green), var(--accent-cyan)); }
.stat-label {
    font-size: 0.75rem; font-weight: 500;
    color: var(--text-muted);
    text-transform: uppercase; letter-spacing: 0.1em;
    margin-bottom: 0.5rem;
}
.stat-value {
    font-family: 'Syne', sans-serif;
    font-size: 2.2rem; font-weight: 800;
    color: #ffffff; line-height: 1;
    margin-bottom: 0.25rem;
}
.stat-sub {
    font-size: 0.78rem;
    color: var(--text-muted);
}

/* ── CLUSTER BADGE CARDS ── */
.cluster-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.25rem 1.5rem;
    margin: 0.6rem 0;
    display: flex; align-items: flex-start; gap: 14px;
    transition: border-color 0.2s;
}
.cluster-card.fast  { border-left: 3px solid var(--accent-red); }
.cluster-card.medium{ border-left: 3px solid var(--accent-amber); }
.cluster-card.slow  { border-left: 3px solid var(--accent-green); }
.cluster-card:hover { border-color: rgba(79,158,255,0.4); }
.cluster-card .emo { font-size: 1.8rem; line-height: 1; }
.cluster-card-body { flex: 1; }
.cluster-card-title {
    font-family: 'Syne', sans-serif;
    font-size: 0.95rem; font-weight: 700;
    color: #ffffff; margin-bottom: 4px;
}
.cluster-card-method {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    color: var(--accent-cyan);
    background: rgba(0,212,200,0.08);
    border: 1px solid rgba(0,212,200,0.2);
    padding: 2px 8px; border-radius: 4px;
    display: inline-block; margin-bottom: 6px;
}
.cluster-card-desc {
    font-size: 0.82rem; color: var(--text-muted);
    line-height: 1.55;
}

/* ── INFO / METRIC BLOCKS ── */
.info-block {
    background: rgba(79,158,255,0.06);
    border: 1px solid rgba(79,158,255,0.2);
    border-radius: var(--radius-sm);
    padding: 0.9rem 1.2rem;
    font-size: 0.88rem; color: var(--text-primary);
    margin: 0.8rem 0;
}
.metric-row {
    display: flex; gap: 16px; margin: 1rem 0;
}
.metric-box {
    flex: 1;
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    padding: 1.1rem 1.4rem;
    text-align: center;
}
.metric-box-label {
    font-size: 0.72rem; font-weight: 500;
    text-transform: uppercase; letter-spacing: 0.1em;
    color: var(--text-muted); margin-bottom: 6px;
}
.metric-box-val {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.8rem; font-weight: 500;
    color: var(--accent-cyan);
}

/* ── DATAFRAME ── */
[data-testid="stDataFrame"] {
    background: var(--bg-card) !important;
    border-radius: var(--radius) !important;
    border: 1px solid var(--border) !important;
    overflow: hidden;
}

/* ── FILE UPLOADER ── */
[data-testid="stFileUploader"] {
    background: var(--bg-card) !important;
    border: 2px dashed rgba(79,158,255,0.3) !important;
    border-radius: var(--radius) !important;
    padding: 1rem !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: rgba(79,158,255,0.6) !important;
}

/* ── SELECTBOX & SLIDER ── */
[data-testid="stSelectbox"] > div,
[data-testid="stSlider"] > div {
    background: transparent !important;
}
.stSelectbox label, .stSlider label, .stFileUploader label {
    color: var(--text-muted) !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
}

/* ── BUTTONS ── */
.stButton>button {
    background: linear-gradient(135deg, var(--accent-blue), #3b82f6) !important;
    color: white !important;
    border: none !important;
    border-radius: var(--radius-sm) !important;
    padding: 10px 24px !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    letter-spacing: 0.02em !important;
    transition: opacity 0.2s, transform 0.1s !important;
    box-shadow: var(--glow-blue) !important;
}
.stButton>button:hover {
    opacity: 0.9 !important;
    transform: translateY(-1px) !important;
}
.stDownloadButton>button {
    background: linear-gradient(135deg, #16a34a, #15803d) !important;
    color: white !important;
    border: none !important;
    border-radius: var(--radius-sm) !important;
    padding: 10px 24px !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    box-shadow: 0 0 18px rgba(74,222,128,0.25) !important;
}

/* ── EXPANDER ── */
[data-testid="stExpander"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
}
[data-testid="stExpander"] summary {
    color: var(--text-primary) !important;
    font-weight: 500 !important;
}

/* ── SUCCESS / WARNING / ERROR ── */
[data-testid="stAlert"] {
    border-radius: var(--radius-sm) !important;
    border: none !important;
}
div[data-baseweb="notification"] {
    border-radius: var(--radius-sm) !important;
}

/* ── METRIC WIDGET ── */
[data-testid="stMetric"] {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    padding: 1rem 1.25rem !important;
}
[data-testid="stMetricValue"] {
    color: var(--accent-blue) !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 2rem !important;
}
[data-testid="stMetricLabel"] {
    color: var(--text-muted) !important;
    font-size: 0.75rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
}

/* ── DIVIDER ── */
hr { border-color: var(--border) !important; }

/* ── SCROLLBAR ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.1); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: rgba(255,255,255,0.2); }
</style>
""", unsafe_allow_html=True)

# ==========================================
# MATPLOTLIB DARK THEME
# ==========================================

def apply_chart_style():
    plt.rcParams.update({
        'figure.facecolor':  '#13161e',
        'axes.facecolor':    '#13161e',
        'axes.edgecolor':    '#2a2f3e',
        'axes.labelcolor':   '#8892a4',
        'axes.titlecolor':   '#e8eaf0',
        'axes.titlesize':    13,
        'axes.titleweight':  'bold',
        'axes.labelsize':    11,
        'axes.grid':         True,
        'axes.spines.top':   False,
        'axes.spines.right': False,
        'grid.color':        '#1e2435',
        'grid.linestyle':    '--',
        'grid.linewidth':    0.7,
        'xtick.color':       '#8892a4',
        'ytick.color':       '#8892a4',
        'xtick.labelsize':   9,
        'ytick.labelsize':   9,
        'legend.facecolor':  '#1a1e2a',
        'legend.edgecolor':  '#2a2f3e',
        'legend.labelcolor': '#e8eaf0',
        'legend.fontsize':   9,
        'text.color':        '#e8eaf0',
        'font.family':       'monospace',
        'lines.linewidth':   2,
        'lines.markersize':  6,
    })

apply_chart_style()

# ==========================================
# COLOUR PALETTE
# ==========================================

C = {
    'train':     '#4f9eff',
    'test':      '#ffb547',
    'hw_add':    '#4ade80',
    'hw_mul':    '#ff6b6b',
    'ets':       '#a78bfa',
    'arima':     '#f97316',
    'croston':   '#00d4c8',
    'vline':     '#e8eaf0',
}

warna_metode = {
    'HW Additive':       C['hw_add'],
    'HW Multiplicative': C['hw_mul'],
    'ETS':               C['ets'],
    'ARIMA':             C['arima'],
    'Croston':           C['croston'],
}

# ==========================================
# CLUSTER-METHOD MAPPING
# ==========================================

CLUSTER_METHOD_MAP = {
    'Fast Moving': {
        'method':  'HW Multiplicative',
        'label':   'Holt-Winters Multiplicative',
        'icon':    '🔴',
        'color':   C['hw_mul'],
        'css':     'fast',
        'rationale': (
            "Produk fast moving memiliki volume permintaan tinggi dengan variasi musiman "
            "yang *proporsional* terhadap level permintaan. Seasonal swing membesar saat "
            "volume naik — pola ini hanya dapat ditangkap oleh model multiplicative."
        )
    },
    'Medium Moving': {
        'method':  'ETS',
        'label':   'ETS (Additive-Additive-Additive)',
        'icon':    '🟡',
        'color':   C['ets'],
        'css':     'medium',
        'rationale': (
            "Secara struktur identik dengan Holt-Winters Additive, namun semua parameter "
            "dioptimasi otomatis via Maximum Likelihood Estimation — lebih presisi dan "
            "tahan overfitting untuk pola tren & musiman sedang pada data 24 bulan."
        )
    },
    'Slow Moving': {
        'method':  'Croston',
        'label':   "Croston's Method",
        'icon':    '🟢',
        'color':   C['croston'],
        'css':     'slow',
        'rationale': (
            "Dirancang khusus untuk intermittent demand — permintaan sporadis dengan "
            "banyak nilai nol. Memisahkan ukuran demand dan interval antar demand, lalu "
            "menerapkan exponential smoothing pada keduanya secara terpisah."
        )
    }
}

# ==========================================
# METRIK ERROR PER CLUSTER
# ==========================================

def calc_mape(actual, forecast):
    a, f = np.array(actual, float), np.array(forecast, float)
    mask = a != 0
    if mask.sum() == 0:
        return np.nan
    return np.mean(np.abs((a[mask] - f[mask]) / a[mask])) * 100

def calc_smape(actual, forecast):
    a, f = np.array(actual, float), np.array(forecast, float)
    denom = (np.abs(a) + np.abs(f)) / 2
    mask = denom != 0
    if mask.sum() == 0:
        return np.nan
    return np.mean(np.abs(a[mask] - f[mask]) / denom[mask]) * 100

def calc_mase(actual, forecast, train):
    a, f, tr = np.array(actual, float), np.array(forecast, float), np.array(train, float)
    naive_mae = np.mean(np.abs(np.diff(tr)))
    if naive_mae == 0:
        return np.nan
    return np.mean(np.abs(a - f)) / naive_mae

def calc_mfe(actual, forecast):
    """Bias: positif = over-forecast (stok numpuk), negatif = under-forecast (stok kurang)."""
    a, f = np.array(actual, float), np.array(forecast, float)
    return np.mean(f - a)

# Pemetaan cluster → metrik yang paling cocok
CLUSTER_ERROR_MAP = {
    'Fast Moving': {
        'primary_key':   'MAPE',
        'primary_unit':  '%',
        'secondary_key': 'RMSE',
        'secondary_unit':'',
        'rationale': (
            "MAPE dipilih karena volume Fast Moving tinggi dan jarang bernilai nol, "
            "sehingga pembagian aman. Hasilnya intuitif (persentase kesalahan). "
            "RMSE sebagai pelengkap untuk menangkap outlier besar. "
            "MFE/Bias penting agar tidak terjadi over-stok sistematis."
        ),
    },
    'Medium Moving': {
        'primary_key':   'sMAPE',
        'primary_unit':  '%',
        'secondary_key': 'RMSE',
        'secondary_unit':'',
        'rationale': (
            "sMAPE dipilih karena Medium Moving kadang memiliki nilai mendekati nol — "
            "sMAPE simetris dan tidak meledak saat aktual atau forecast kecil. "
            "RMSE tetap disertakan untuk sensitifitas terhadap kesalahan besar. "
            "MFE/Bias untuk memantau apakah model cenderung over/under-forecast."
        ),
    },
    'Slow Moving': {
        'primary_key':   'MASE',
        'primary_unit':  '',
        'secondary_key': 'sMAPE',
        'secondary_unit':'%',
        'rationale': (
            "MASE adalah gold standard untuk intermittent demand (Croston) — "
            "tidak terpengaruh nilai nol karena membandingkan error model vs naive forecast. "
            "MASE < 1 berarti model mengalahkan random walk. "
            "sMAPE sebagai pelengkap. MFE krusial untuk keputusan reorder point."
        ),
    },
}

def compute_cluster_metrics(actual, forecast, train, cluster_kat):
    """Hitung semua metrik relevan untuk cluster tertentu."""
    info  = CLUSTER_ERROR_MAP[cluster_kat]
    rmse  = float(np.sqrt(mean_squared_error(actual, forecast)))
    mfe   = calc_mfe(actual, forecast)

    if info['primary_key'] == 'MAPE':
        primary_val = calc_mape(actual, forecast)
    elif info['primary_key'] == 'sMAPE':
        primary_val = calc_smape(actual, forecast)
    else:  # MASE
        primary_val = calc_mase(actual, forecast, train)

    if info['secondary_key'] == 'RMSE':
        secondary_val = rmse
    else:  # sMAPE
        secondary_val = calc_smape(actual, forecast)

    return {
        'primary_key':    info['primary_key'],
        'primary_unit':   info['primary_unit'],
        'primary_val':    primary_val,
        'secondary_key':  info['secondary_key'],
        'secondary_unit': info['secondary_unit'],
        'secondary_val':  secondary_val,
        'mfe':            mfe,
        'rmse':           rmse,
    }

# ==========================================
# HERO SECTION
# ==========================================

st.markdown("""
<div class="hero-wrapper">
  <div class="hero-badge">⬡ v2.0 &nbsp;·&nbsp; INVENTORY INTELLIGENCE</div>
  <h1 class="hero-title">Clustering &<br><span>Demand Forecasting</span></h1>
  <p class="hero-subtitle">
    Platform analisis permintaan barang berbasis machine learning — dari segmentasi produk
    otomatis hingga proyeksi ke depan dengan metode terbaik per cluster.
  </p>
  <div class="hero-pills">
    <span class="hero-pill"><span class="dot" style="background:#4f9eff"></span> K-Means Clustering</span>
    <span class="hero-pill"><span class="dot" style="background:#ff6b6b"></span> Holt-Winters</span>
    <span class="hero-pill"><span class="dot" style="background:#a78bfa"></span> ETS Model</span>
    <span class="hero-pill"><span class="dot" style="background:#00d4c8"></span> Croston's Method</span>
    <span class="hero-pill"><span class="dot" style="background:#f97316"></span> ARIMA</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ==========================================
# PANDUAN METODE
# ==========================================

with st.expander("📖  Panduan Metode Forecasting per Cluster", expanded=False):
    for kat, info in CLUSTER_METHOD_MAP.items():
        st.markdown(f"""
<div class="cluster-card {info['css']}">
  <div class="emo">{info['icon']}</div>
  <div class="cluster-card-body">
    <div class="cluster-card-title">{kat}</div>
    <span class="cluster-card-method">{info['label']}</span>
    <div class="cluster-card-desc">{info['rationale']}</div>
  </div>
</div>
""", unsafe_allow_html=True)

    st.markdown("""
<div class="info-block">
  📌 <strong>Catatan Data 2 Tahun (24 Bulan):</strong><br>
  Seasonal period yang digunakan adalah <strong>6 bulan</strong> (semi-tahunan)
  karena model memerlukan minimal 2 siklus penuh. Period 12 bulan baru aktif jika
  data training ≥ 24 bulan. Untuk Croston: forecast <em>flat</em> adalah normal —
  nilai = rata-rata ukuran demand ÷ rata-rata interval demand.
</div>
""", unsafe_allow_html=True)

# ==========================================
# UPLOAD FILE
# ==========================================

st.markdown("""
<div class="section-header">
  <div class="icon-box">📂</div>
  <h2>Upload Data</h2>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload File Excel (.xlsx)", type=["xlsx"])

# ==========================================
# JIKA FILE SUDAH DIUPLOAD
# ==========================================

if uploaded_file is not None:

    df = pd.read_excel(uploaded_file)
    df = df.dropna(subset=['id_produk', 'keluar'])
    df['keluar'] = df['keluar'].astype(int)

    # ── DASHBOARD KPI ──
    st.markdown("""
<div class="section-header">
  <div class="icon-box">📊</div>
  <h2>Overview Data</h2>
</div>
""", unsafe_allow_html=True)

    st.markdown(f"""
<div class="stats-row">
  <div class="stat-card blue">
    <div class="stat-label">Total Data</div>
    <div class="stat-value">{len(df):,}</div>
    <div class="stat-sub">baris transaksi</div>
  </div>
  <div class="stat-card amber">
    <div class="stat-label">Jumlah Produk</div>
    <div class="stat-value">{df['id_produk'].nunique():,}</div>
    <div class="stat-sub">SKU unik</div>
  </div>
  <div class="stat-card green">
    <div class="stat-label">Total Keluar</div>
    <div class="stat-value">{int(df['keluar'].sum()):,}</div>
    <div class="stat-sub">unit barang</div>
  </div>
</div>
""", unsafe_allow_html=True)

    with st.expander("🗂  Lihat Data Awal (5 baris pertama)"):
        st.dataframe(df.head(), use_container_width=True)

    # FORMAT TANGGAL & BULAN
    df['tgl_input'] = pd.to_datetime(df['tgl_input'])
    df['Bulan'] = df['tgl_input'].dt.strftime('%b-%y')

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

    first_month  = pd.to_datetime(all_months_sorted[0], format='%b-%y')
    total_months = len(all_months_sorted)

    with st.expander(f"📋  Pivot Table Barang Keluar ({total_months} bulan × {len(pivot_table)} produk)"):
        st.dataframe(pivot_table, use_container_width=True)
        csv_data = pivot_table.to_csv().encode('utf-8')
        st.download_button(
            label="⬇️  Download Pivot Table",
            data=csv_data,
            file_name='pivot_barang_keluar.csv',
            mime='text/csv'
        )

    # ==========================================
    # CLUSTERING
    # ==========================================

    st.markdown("""
<div class="section-header">
  <div class="icon-box">🎯</div>
  <h2>Clustering Produk</h2>
</div>
""", unsafe_allow_html=True)

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

    fig_elbow, ax_elbow = plt.subplots(figsize=(9, 4.5))
    ax_elbow.plot(list(K), inertia, marker='o', color=C['train'],
                  linewidth=2.5, markersize=8,
                  markerfacecolor='#13161e', markeredgewidth=2.5)
    ax_elbow.fill_between(list(K), inertia, alpha=0.06, color=C['train'])
    for k_val, in_val in zip(K, inertia):
        ax_elbow.annotate(f'{in_val:.0f}',
            xy=(k_val, in_val), xytext=(0, 10),
            textcoords='offset points',
            ha='center', fontsize=8, color='#8892a4')
    ax_elbow.set_title('Elbow Method — Penentuan Jumlah Cluster Optimal', pad=14)
    ax_elbow.set_xlabel('Jumlah Cluster (k)')
    ax_elbow.set_ylabel('Inertia')
    st.pyplot(fig_elbow)
    plt.close()

    jumlah_cluster = st.slider("Pilih Jumlah Cluster", min_value=2, max_value=10, value=3)

    kmeans = KMeans(n_clusters=jumlah_cluster, random_state=42)
    cluster = kmeans.fit_predict(scaled_data)
    filtered_data = filtered_data.copy()
    filtered_data['Cluster'] = cluster

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
        'Kategori':           cluster_count.index,
        'Jumlah Produk':      cluster_count.values,
        'Metode Forecast':    [CLUSTER_METHOD_MAP[k]['label'] for k in cluster_count.index]
    })
    tabel_cluster.index = range(1, len(tabel_cluster) + 1)

    # Cluster summary cards
    cols_c = st.columns(3)
    cat_colors_css = {'Fast Moving': C['hw_mul'], 'Medium Moving': C['ets'], 'Slow Moving': C['croston']}
    for i, (kat, cnt) in enumerate(cluster_count.items()):
        with cols_c[i]:
            info = CLUSTER_METHOD_MAP[kat]
            color = cat_colors_css[kat]
            st.markdown(f"""
<div style="background:#13161e;border:1px solid rgba(255,255,255,0.07);
            border-top:3px solid {color};border-radius:12px;
            padding:1.25rem 1.5rem;text-align:center;">
  <div style="font-size:2rem;margin-bottom:6px">{info['icon']}</div>
  <div style="font-family:'Syne',sans-serif;font-size:1.1rem;font-weight:700;
              color:#fff;margin-bottom:4px">{kat}</div>
  <div style="font-family:'JetBrains Mono',monospace;font-size:2.4rem;
              font-weight:500;color:{color};line-height:1">{cnt}</div>
  <div style="font-size:0.72rem;color:#8892a4;margin-top:4px">produk</div>
  <div style="font-family:'JetBrains Mono',monospace;font-size:0.7rem;
              color:#8892a4;background:rgba(255,255,255,0.04);
              border-radius:4px;padding:4px 8px;margin-top:8px;
              display:inline-block">{info['label']}</div>
</div>
""", unsafe_allow_html=True)

    # Bar chart distribusi cluster
    fig_cl, ax_cl = plt.subplots(figsize=(9, 4.5))
    bar_colors_cl = [cat_colors_css.get(k, '#888') for k in tabel_cluster['Kategori']]
    bars_cl = ax_cl.bar(
        tabel_cluster['Kategori'], tabel_cluster['Jumlah Produk'],
        color=bar_colors_cl, width=0.5,
        edgecolor='#1e2435', linewidth=1.5
    )
    for bar, val in zip(bars_cl, tabel_cluster['Jumlah Produk']):
        ax_cl.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            str(val), ha='center', va='bottom',
            fontweight='bold', fontsize=13, color='#e8eaf0'
        )
    ax_cl.set_title('Distribusi Produk per Cluster')
    ax_cl.set_xlabel('Kategori')
    ax_cl.set_ylabel('Jumlah Produk')
    ax_cl.set_ylim(0, tabel_cluster['Jumlah Produk'].max() * 1.25)
    st.pyplot(fig_cl)
    plt.close()

    # ==========================================
    # PILIH CLUSTER & PRODUK
    # ==========================================

    st.markdown("""
<div class="section-header">
  <div class="icon-box">🔍</div>
  <h2>Pilih Cluster & Produk</h2>
</div>
""", unsafe_allow_html=True)

    pilih_cluster = st.selectbox("Pilih Cluster", ['Fast Moving', 'Medium Moving', 'Slow Moving'])

    produk_cluster = filtered_data[
        filtered_data['Kategori'] == pilih_cluster
    ].index.tolist()

    info_cluster = CLUSTER_METHOD_MAP[pilih_cluster]
    color_cluster = cat_colors_css[pilih_cluster]

    st.markdown(f"""
<div class="cluster-card {info_cluster['css']}" style="margin:0.8rem 0 1.2rem 0">
  <div class="emo">{info_cluster['icon']}</div>
  <div class="cluster-card-body">
    <div class="cluster-card-title">{pilih_cluster} — {len(produk_cluster)} produk</div>
    <span class="cluster-card-method">{info_cluster['label']}</span>
    <div class="cluster-card-desc">{info_cluster['rationale']}</div>
  </div>
</div>
""", unsafe_allow_html=True)

    with st.expander(f"📦  Daftar Produk — {pilih_cluster} ({len(produk_cluster)} produk)"):
        df_produk = pd.DataFrame({'Produk': produk_cluster})
        df_produk.index = range(1, len(df_produk) + 1)
        st.dataframe(df_produk, use_container_width=True)

    # ==========================================
    # FORECASTING
    # ==========================================

    st.markdown("""
<div class="section-header">
  <div class="icon-box">📈</div>
  <h2>Forecasting Permintaan</h2>
</div>
""", unsafe_allow_html=True)

    produk = st.selectbox("Pilih Produk", produk_cluster)

    data_produk = filtered_data.loc[produk].copy()
    data_produk = data_produk.drop(['Cluster', 'Kategori', 'Total'])
    data_produk = pd.to_numeric(data_produk)
    data_produk.index = pd.date_range(
        start=first_month, periods=len(data_produk), freq='ME'
    )

    jumlah_bulan_aktif = (data_produk > 0).sum()
    if jumlah_bulan_aktif < 6:
        st.warning("⚠️ Produk ini hanya aktif kurang dari 6 bulan — hasil forecasting kurang reliabel.")

    # AUTO-SELECT METODE
    metode_rekomendasi = CLUSTER_METHOD_MAP[pilih_cluster]['method']
    label_rekomendasi  = CLUSTER_METHOD_MAP[pilih_cluster]['label']
    icon_cluster       = CLUSTER_METHOD_MAP[pilih_cluster]['icon']

    st.markdown(f"""
<div style="background:rgba(79,158,255,0.06);border:1px solid rgba(79,158,255,0.25);
            border-radius:8px;padding:0.8rem 1.2rem;margin:0.5rem 0 1rem 0;
            font-size:0.9rem;">
  {icon_cluster} &nbsp;Produk <strong style="color:#fff">{produk}</strong> termasuk
  <strong style="color:{color_cluster}">{pilih_cluster}</strong>
  &nbsp;→&nbsp; Metode rekomendasi:
  <code style="background:rgba(79,158,255,0.15);padding:2px 8px;border-radius:4px;
               color:#4f9eff">{label_rekomendasi}</code>
</div>
""", unsafe_allow_html=True)

    daftar_metode = [
        "HW Additive", "HW Multiplicative", "ETS",
        "ARIMA", "Croston", "Perbandingan Semua Metode"
    ]
    default_idx = daftar_metode.index(metode_rekomendasi)

    col_sel1, col_sel2 = st.columns([2, 1])
    with col_sel1:
        metode = st.selectbox(
            "Metode Forecasting (otomatis → bisa diganti manual)",
            daftar_metode, index=default_idx
        )
    with col_sel2:
        jumlah_forecast = st.slider("Jumlah Forecast (bulan)", 1, 12, 6)

    croston_alpha = 0.1
    if metode in ('Croston', 'Perbandingan Semua Metode'):
        croston_alpha = st.slider(
            "Alpha Croston (smoothing parameter)",
            min_value=0.05, max_value=0.50, value=0.10, step=0.05,
            help="Nilai kecil = stabil; nilai besar = responsif. Rekomendasi 0.10–0.20."
        )

    # TRAIN-TEST SPLIT
    n = len(data_produk)
    train = data_produk.iloc[:n - jumlah_forecast]
    test  = data_produk.iloc[n - jumlah_forecast:]

    if len(train) < 6:
        st.error(
            f"Data train hanya {len(train)} bulan. "
            "Kurangi jumlah forecast agar minimal 6 bulan."
        )
        st.stop()

    st.markdown(f"""
<div class="info-block">
  📌 Split data &nbsp;·&nbsp;
  <strong style="color:#4f9eff">Train: {len(train)} bulan</strong>
  &nbsp;/&nbsp;
  <strong style="color:#ffb547">Test: {len(test)} bulan</strong>
  &nbsp;·&nbsp; MAE & RMSE dihitung dari forecast vs data test.
</div>
""", unsafe_allow_html=True)

    # ==========================================
    # SEASONAL PERIODS
    # ==========================================

    def get_sp(n_train):
        if n_train >= 24: return 12
        elif n_train >= 12: return 6
        else: return 1

    # ==========================================
    # GRAFIK DATA AKTUAL
    # ==========================================

    fig_act, ax_act = plt.subplots(figsize=(13, 4.5))
    ax_act.fill_between(data_produk.index, data_produk.values,
                        alpha=0.12, color=C['train'])
    ax_act.plot(data_produk.index, data_produk.values,
                marker='o', linewidth=2.5, color=C['train'],
                markerfacecolor='#13161e', markeredgewidth=2.5,
                label='Data Aktual')
    ax_act.axvline(x=test.index[0], color='#ff6b6b',
                   linestyle='--', alpha=0.7, linewidth=1.5,
                   label='Awal Periode Test')
    ax_act.set_title(f'Data Aktual — Produk {produk}', pad=14)
    ax_act.legend()
    st.pyplot(fig_act)
    plt.close()

    # ==========================================
    # HELPER FUNCTIONS
    # ==========================================

    def croston_forecast(series, steps, alpha=0.1):
        data = series.values.astype(float)
        non_zero = np.where(data > 0)[0]
        if len(non_zero) == 0:
            idx = pd.date_range(series.index[-1], periods=steps + 1, freq='ME')[1:]
            return pd.Series(np.zeros(steps), index=idx)
        z = float(data[non_zero[0]])
        p = float(non_zero[0] + 1)
        q = 1
        for i in range(non_zero[0] + 1, len(data)):
            if data[i] > 0:
                z = alpha * data[i] + (1 - alpha) * z
                p = alpha * q       + (1 - alpha) * p
                q = 1
            else:
                q += 1
        rate = max(0.0, z / p)
        idx = pd.date_range(series.index[-1], periods=steps + 1, freq='ME')[1:]
        return pd.Series(np.full(steps, round(rate, 4)), index=idx)

    def fit_model(nama, train_data, c_alpha=0.1):
        sp = get_sp(len(train_data))
        if nama == 'HW Additive':
            if sp > 1:
                return ExponentialSmoothing(
                    train_data, trend='add', seasonal='add', seasonal_periods=sp).fit()
            return ExponentialSmoothing(train_data, trend='add').fit()
        elif nama == 'HW Multiplicative':
            td = train_data.copy(); td[td <= 0] = 1
            if sp > 1:
                return ExponentialSmoothing(
                    td, trend='add', seasonal='mul', seasonal_periods=sp).fit()
            return ExponentialSmoothing(td, trend='add').fit()
        elif nama == 'ETS':
            if sp > 1:
                return ETSModel(
                    train_data, error='add', trend='add',
                    seasonal='add', seasonal_periods=sp).fit(disp=False)
            return ETSModel(train_data, error='add', trend='add').fit(disp=False)
        elif nama == 'ARIMA':
            return ARIMA(train_data, order=(1, 1, 1)).fit()
        elif nama == 'Croston':
            return {'series': train_data, 'alpha': c_alpha, 'type': 'croston'}
        raise ValueError(f"Metode tidak dikenal: {nama}")

    def forecast_model(model, nama, steps, c_alpha=0.1):
        if nama == 'Croston':
            return croston_forecast(model['series'], steps, model['alpha'])
        elif nama == 'ARIMA':
            return model.forecast(steps=steps)
        return model.forecast(steps).clip(lower=0)

    def retrain_full(nama, full_data, steps, c_alpha=0.1):
        m = fit_model(nama, full_data, c_alpha)
        return forecast_model(m, nama, steps, c_alpha)

    # ==========================================
    # TAMPILKAN HASIL (1 METODE)
    # ==========================================

    def tampilkan_hasil(nama_metode, fc_eval, test_actual, train_data,
                        cluster_kat='Fast Moving', is_rek=False):
        label_sfx = " ✅ (Rekomendasi)" if is_rek else ""

        # Hitung metrik sesuai cluster
        m = compute_cluster_metrics(
            test_actual.values, fc_eval.values,
            train_data.values, cluster_kat
        )

        # Rationale info-block
        err_info = CLUSTER_ERROR_MAP[cluster_kat]
        st.markdown(f"""
<div class="info-block" style="margin-bottom:0.8rem;">
  📐 <strong>Metrik Error untuk {cluster_kat}:</strong><br>
  <span style="font-size:0.82rem;color:var(--text-muted)">{err_info['rationale']}</span>
</div>""", unsafe_allow_html=True)

        # Tentukan warna MFE
        mfe_color = '#ff6b6b' if m['mfe'] > 0 else ('#4ade80' if m['mfe'] < 0 else '#8892a4')
        mfe_label = "Over-forecast ▲" if m['mfe'] > 0 else ("Under-forecast ▼" if m['mfe'] < 0 else "Tidak Bias")

        # Format primary value
        if m['primary_key'] == 'MASE':
            primary_fmt = f"{m['primary_val']:.3f}" if not np.isnan(m['primary_val']) else "N/A"
        else:
            primary_fmt = f"{m['primary_val']:.2f}{m['primary_unit']}" if not np.isnan(m['primary_val']) else "N/A"

        secondary_unit_str = m['secondary_unit'] if m['secondary_unit'] else ''
        secondary_fmt = f"{m['secondary_val']:.2f}{secondary_unit_str}"

        col_m1, col_m2, col_m3 = st.columns(3)
        with col_m1:
            st.markdown(f"""
<div class="metric-box">
  <div class="metric-box-label">⭐ {m['primary_key']} (Primary)</div>
  <div class="metric-box-val">{primary_fmt}</div>
</div>""", unsafe_allow_html=True)
        with col_m2:
            st.markdown(f"""
<div class="metric-box">
  <div class="metric-box-label">{m['secondary_key']} (Secondary)</div>
  <div class="metric-box-val">{secondary_fmt}</div>
</div>""", unsafe_allow_html=True)
        with col_m3:
            st.markdown(f"""
<div class="metric-box">
  <div class="metric-box-label">MFE / Bias</div>
  <div class="metric-box-val" style="font-size:1.4rem;color:{mfe_color}">
    {m['mfe']:+.2f}
  </div>
  <div style="font-size:0.7rem;color:{mfe_color};margin-top:4px">{mfe_label}</div>
</div>""", unsafe_allow_html=True)

        eval_df = pd.DataFrame({
            'Periode':            fc_eval.index.strftime('%b-%Y'),
            'Hasil Forecast':     np.round(fc_eval.values, 2),
            'Data Aktual (Test)': np.round(test_actual.values, 2)
        })
        eval_df.index = range(1, len(eval_df) + 1)

        st.markdown(f"#### 📋 Forecast vs Aktual — {nama_metode}{label_sfx}")
        st.dataframe(eval_df, use_container_width=True)

        # Eval chart
        fig_ev, ax_ev = plt.subplots(figsize=(13, 5))
        ax_ev.fill_between(train.index, train.values,
                           alpha=0.1, color=C['train'])
        ax_ev.plot(train.index, train.values,
                   marker='o', color=C['train'], linewidth=2.5,
                   markerfacecolor='#13161e', markeredgewidth=2,
                   label='Data Train')
        ax_ev.plot(test_actual.index, test_actual.values,
                   marker='o', color=C['test'], linewidth=2.5,
                   markerfacecolor='#13161e', markeredgewidth=2,
                   label='Data Test (Aktual)')
        col_fc = warna_metode.get(nama_metode, '#4f9eff')
        ax_ev.plot(fc_eval.index, fc_eval.values,
                   marker='s', linestyle='--', color=col_fc, linewidth=2.5,
                   markerfacecolor='#13161e', markeredgewidth=2,
                   label=f'Forecast {nama_metode}{label_sfx}')
        ax_ev.axvline(x=test_actual.index[0], color='#e8eaf0',
                      linestyle=':', alpha=0.5, linewidth=1.5,
                      label='Awal Test')
        ax_ev.set_title(f'Evaluasi Model — {nama_metode} — {produk}', pad=14)
        ax_ev.legend()
        st.pyplot(fig_ev)
        plt.close()

        # Future forecast
        st.markdown(f"#### 🔮 Forecast ke Depan — {nama_metode}{label_sfx}")
        st.markdown(f"""
<div class="info-block">
  Model di-retrain menggunakan seluruh <strong>{len(data_produk)} bulan</strong> data,
  lalu meramalkan {jumlah_forecast} bulan ke depan.
</div>""", unsafe_allow_html=True)

        future_fc = retrain_full(nama_metode, data_produk, jumlah_forecast, croston_alpha)
        future_df = pd.DataFrame({
            'Periode':        future_fc.index.strftime('%b-%Y'),
            'Hasil Forecast': np.round(future_fc.values, 2)
        })
        future_df.index = range(1, len(future_df) + 1)
        st.dataframe(future_df, use_container_width=True)

        csv_fut = future_df.to_csv().encode('utf-8')
        st.download_button(
            label=f"⬇️  Download Forecast ({nama_metode})",
            data=csv_fut,
            file_name=f'forecast_{produk}_{nama_metode.replace(" ","_")}.csv',
            mime='text/csv',
            key=f'dl_{nama_metode}'
        )

        fig_fut, ax_fut = plt.subplots(figsize=(13, 5))
        ax_fut.fill_between(data_produk.index, data_produk.values,
                            alpha=0.1, color=C['train'])
        ax_fut.plot(data_produk.index, data_produk.values,
                    marker='o', color=C['train'], linewidth=2.5,
                    markerfacecolor='#13161e', markeredgewidth=2,
                    label=f'Data Aktual ({len(data_produk)} bulan)')
        ax_fut.fill_between(future_fc.index, future_fc.values,
                            alpha=0.15, color=col_fc)
        ax_fut.plot(future_fc.index, future_fc.values,
                    marker='o', linestyle='-', color=col_fc, linewidth=2.5,
                    markerfacecolor='#13161e', markeredgewidth=2,
                    label=f'Forecast ke Depan ({nama_metode})')
        ax_fut.axvline(x=future_fc.index[0], color='#ff6b6b',
                       linestyle='--', alpha=0.7, linewidth=1.5,
                       label='Awal Forecast')
        ax_fut.set_title(f'Forecast ke Depan — {nama_metode} — {produk}', pad=14)
        ax_fut.legend()
        st.pyplot(fig_fut)
        plt.close()

        return m['primary_val'], m['secondary_val'], m['mfe']

    # ==========================================
    # EKSEKUSI
    # ==========================================

    is_rek = (metode == metode_rekomendasi)

    if metode != "Perbandingan Semua Metode":
        try:
            m_fit = fit_model(metode, train, croston_alpha)
            fc_ev = forecast_model(m_fit, metode, jumlah_forecast, croston_alpha)
            tampilkan_hasil(metode, fc_ev, test, train,
                            cluster_kat=pilih_cluster, is_rek=is_rek)
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
                m  = fit_model(nm, train, croston_alpha)
                fc = forecast_model(m, nm, jumlah_forecast, croston_alpha)
                metrics = compute_cluster_metrics(
                    test.values, fc.values, train.values, pilih_cluster
                )
                hasil_eval[nm] = {
                    'forecast':       fc,
                    'primary_key':    metrics['primary_key'],
                    'primary_unit':   metrics['primary_unit'],
                    'primary_val':    metrics['primary_val'],
                    'secondary_key':  metrics['secondary_key'],
                    'secondary_unit': metrics['secondary_unit'],
                    'secondary_val':  metrics['secondary_val'],
                    'mfe':            metrics['mfe'],
                    # tetap simpan mae & rmse untuk referensi
                    'mae':  float(mean_absolute_error(test.values, fc.values)),
                    'rmse': metrics['rmse'],
                }
                hasil_future[nm] = retrain_full(nm, data_produk, jumlah_forecast, croston_alpha)
            except Exception as e:
                st.warning(f"⚠️ Metode {nm} dilewati: {e}")

        if not hasil_eval:
            st.error("Semua metode gagal. Coba kurangi jumlah forecast.")
            st.stop()

        primary_key   = list(hasil_eval.values())[0]['primary_key']
        primary_unit  = list(hasil_eval.values())[0]['primary_unit']
        secondary_key = list(hasil_eval.values())[0]['secondary_key']
        secondary_unit = list(hasil_eval.values())[0]['secondary_unit']

        # Info metrik yang dipakai
        err_info = CLUSTER_ERROR_MAP[pilih_cluster]
        st.markdown(f"""
<div class="info-block" style="margin-bottom:1rem;">
  📐 <strong>Metrik yang digunakan untuk cluster {pilih_cluster}:</strong>
  Primary = <code style="background:rgba(79,158,255,0.15);padding:2px 8px;
  border-radius:4px;color:#4f9eff">{primary_key}</code>
  &nbsp;·&nbsp; Secondary = <code style="background:rgba(79,158,255,0.15);
  padding:2px 8px;border-radius:4px;color:#4f9eff">{secondary_key}</code>
  &nbsp;·&nbsp; MFE/Bias<br>
  <span style="font-size:0.8rem;color:var(--text-muted)">{err_info['rationale']}</span>
</div>""", unsafe_allow_html=True)

        def fmt_val(v, unit):
            if np.isnan(v): return "N/A"
            if unit == '%': return f"{v:.2f}%"
            if primary_key == 'MASE' and unit == '': return f"{v:.3f}"
            return f"{v:.2f}"

        perbandingan = pd.DataFrame([
            {
                'Metode':                    m,
                f'{primary_key} ⭐ (Primary)': fmt_val(v['primary_val'], primary_unit),
                f'{secondary_key} (Secondary)': fmt_val(v['secondary_val'], secondary_unit),
                'MFE / Bias':                f"{v['mfe']:+.2f}",
                'Rekomendasi Cluster':       '✅' if m == metode_rekomendasi else ''
            }
            for m, v in hasil_eval.items()
        ])
        perbandingan.index = range(1, len(perbandingan) + 1)

        st.markdown(f"#### 📊 Perbandingan {primary_key} & {secondary_key} Semua Metode")
        st.dataframe(perbandingan, use_container_width=True)

        # Siapkan nilai numerik untuk bar chart
        primary_vals_num   = [v['primary_val']   for v in hasil_eval.values()]
        secondary_vals_num = [v['secondary_val'] for v in hasil_eval.values()]
        metode_labels      = list(hasil_eval.keys())

        bar_c = [
            '#ff6b6b' if nm == metode_rekomendasi else warna_metode.get(nm, '#888')
            for nm in metode_labels
        ]

        # Bar chart perbandingan
        fig_cmp, axes_cmp = plt.subplots(1, 2, figsize=(14, 5))
        chart_data = [
            (primary_vals_num,   primary_key,   primary_unit),
            (secondary_vals_num, secondary_key, secondary_unit),
        ]
        for ax_i, (vals, lbl, unit) in zip(axes_cmp, chart_data):
            # Filter NaN untuk max aman
            valid_vals = [v for v in vals if not np.isnan(v)]
            if not valid_vals:
                ax_i.set_title(f'{lbl} — tidak tersedia')
                continue
            brs = ax_i.bar(metode_labels, vals, color=bar_c, width=0.5,
                           edgecolor='#1e2435', linewidth=1.5)
            for b, v in zip(brs, vals):
                if np.isnan(v):
                    continue
                label_str = f"{v:.3f}" if lbl == 'MASE' else f"{v:.2f}{unit}"
                ax_i.text(b.get_x() + b.get_width() / 2,
                          b.get_height() + max(valid_vals) * 0.02,
                          label_str, ha='center', va='bottom',
                          fontsize=9, color='#e8eaf0', fontweight='bold')
            ax_i.set_title(f'Perbandingan {lbl}{unit}\n(merah = rekomendasi cluster)')
            ax_i.set_ylabel(f'{lbl}{unit}')
            ax_i.set_ylim(0, max(valid_vals) * 1.25)
            ax_i.tick_params(axis='x', rotation=15)

        plt.tight_layout()
        st.pyplot(fig_cmp)
        plt.close()

        # Kesimpulan — ranking berdasarkan primary metric cluster
        primary_series = {nm: v['primary_val'] for nm, v in hasil_eval.items()}
        valid_primary  = {nm: pv for nm, pv in primary_series.items() if not np.isnan(pv)}
        if valid_primary:
            best_nm  = min(valid_primary, key=valid_primary.get)
            best_pv  = valid_primary[best_nm]
            best_sv  = hasil_eval[best_nm]['secondary_val']
            pv_str   = f"{best_pv:.3f}" if primary_key == 'MASE' else f"{best_pv:.2f}{primary_unit}"
            sv_str   = f"{best_sv:.2f}{secondary_unit}"
            rek_pv   = primary_series.get(metode_rekomendasi, float('nan'))
            rek_pv_str = (
                f"{rek_pv:.3f}" if primary_key == 'MASE'
                else (f"{rek_pv:.2f}{primary_unit}" if not np.isnan(rek_pv) else "N/A")
            )
            if best_nm == metode_rekomendasi:
                st.success(
                    f"✅ Metode terbaik: **{best_nm}** "
                    f"({primary_key}={pv_str}, {secondary_key}={sv_str}) "
                    f"— sesuai rekomendasi cluster **{pilih_cluster}**!"
                )
            else:
                st.warning(
                    f"ℹ️ Metode terbaik secara {primary_key}: **{best_nm}** "
                    f"({primary_key}={pv_str}, {secondary_key}={sv_str}). "
                    f"Rekomendasi cluster **{metode_rekomendasi}** "
                    f"({primary_key}={rek_pv_str}) berada di posisi lain."
                )

        # Grafik evaluasi gabungan
        st.markdown("#### 📉 Grafik Evaluasi Gabungan")
        fig_eg, ax_eg = plt.subplots(figsize=(14, 6))
        ax_eg.fill_between(train.index, train.values, alpha=0.08, color=C['train'])
        ax_eg.plot(train.index, train.values,
                   marker='o', linewidth=2.5, color=C['train'],
                   markerfacecolor='#13161e', markeredgewidth=2, label='Data Train')
        ax_eg.plot(test.index, test.values,
                   marker='o', linewidth=2.5, color=C['test'],
                   markerfacecolor='#13161e', markeredgewidth=2, label='Data Test (Aktual)')
        for nm, v in hasil_eval.items():
            lw = 2.5 if nm == metode_rekomendasi else 1.5
            ls = '-' if nm == metode_rekomendasi else '--'
            ax_eg.plot(v['forecast'].index, v['forecast'].values,
                       linestyle=ls, marker='s', linewidth=lw,
                       color=warna_metode[nm],
                       markerfacecolor='#13161e', markeredgewidth=1.5,
                       label=f'{nm}{"  ← rek." if nm == metode_rekomendasi else ""}')
        ax_eg.axvline(x=test.index[0], color='#e8eaf0',
                      linestyle=':', alpha=0.5, linewidth=1.5, label='Awal Test')
        ax_eg.set_title(f'Evaluasi Gabungan Semua Metode — {produk}', pad=14)
        ax_eg.legend(loc='upper left', framealpha=0.8)
        st.pyplot(fig_eg)
        plt.close()

        # Forecast ke depan gabungan
        st.markdown("#### 🔮 Forecast ke Depan — Semua Metode")
        st.markdown(f"""
<div class="info-block">
  Semua model di-retrain menggunakan seluruh <strong>{len(data_produk)} bulan</strong> data.
</div>""", unsafe_allow_html=True)

        future_index = list(hasil_future.values())[0].index
        future_combined = pd.DataFrame({'Periode': future_index.strftime('%b-%Y')})
        for nm, fc in hasil_future.items():
            future_combined[nm] = np.round(fc.values, 2)
        future_combined.index = range(1, len(future_combined) + 1)
        st.dataframe(future_combined, use_container_width=True)

        csv_all = future_combined.to_csv().encode('utf-8')
        st.download_button(
            label="⬇️  Download Forecast ke Depan (Semua Metode)",
            data=csv_all,
            file_name=f'forecast_{produk}_semua_metode.csv',
            mime='text/csv',
            key='dl_semua'
        )

        fig_fall, ax_fall = plt.subplots(figsize=(14, 6))
        ax_fall.fill_between(data_produk.index, data_produk.values,
                             alpha=0.08, color=C['train'])
        ax_fall.plot(data_produk.index, data_produk.values,
                     marker='o', linewidth=2.5, color=C['train'],
                     markerfacecolor='#13161e', markeredgewidth=2,
                     label=f'Data Aktual ({len(data_produk)} bulan)')
        for nm, fc in hasil_future.items():
            lw = 2.5 if nm == metode_rekomendasi else 1.5
            ls = '-' if nm == metode_rekomendasi else '--'
            ax_fall.plot(fc.index, fc.values,
                         linestyle=ls, marker='s', linewidth=lw,
                         color=warna_metode[nm],
                         markerfacecolor='#13161e', markeredgewidth=1.5,
                         label=f'{nm}{"  ← rek." if nm == metode_rekomendasi else ""}')
        ax_fall.axvline(x=list(hasil_future.values())[0].index[0],
                        color='#ff6b6b', linestyle='--', alpha=0.7, linewidth=1.5,
                        label='Awal Forecast')
        ax_fall.set_title(f'Forecast ke Depan Gabungan — {produk}', pad=14)
        ax_fall.legend(loc='upper left', framealpha=0.8)
        st.pyplot(fig_fall)
        plt.close()

        # Rekomendasi cluster saja
        if metode_rekomendasi in hasil_future:
            col_rek = cat_colors_css[pilih_cluster]
            st.markdown(f"""
#### 🏷️ Forecast Rekomendasi Cluster — {pilih_cluster}: {metode_rekomendasi}
""")
            fc_rek = hasil_future[metode_rekomendasi]
            rek_df = pd.DataFrame({
                'Periode':        fc_rek.index.strftime('%b-%Y'),
                'Hasil Forecast': np.round(fc_rek.values, 2)
            })
            rek_df.index = range(1, len(rek_df) + 1)
            st.dataframe(rek_df, use_container_width=True)

            csv_rek = rek_df.to_csv().encode('utf-8')
            st.download_button(
                label=f"⬇️  Download Forecast Rekomendasi ({metode_rekomendasi})",
                data=csv_rek,
                file_name=f'forecast_rek_{produk}_{metode_rekomendasi.replace(" ","_")}.csv',
                mime='text/csv',
                key='dl_rek'
            )

            fig_rek, ax_rek = plt.subplots(figsize=(13, 5))
            ax_rek.fill_between(data_produk.index, data_produk.values,
                                alpha=0.1, color=C['train'])
            ax_rek.plot(data_produk.index, data_produk.values,
                        marker='o', linewidth=2.5, color=C['train'],
                        markerfacecolor='#13161e', markeredgewidth=2,
                        label=f'Data Aktual ({len(data_produk)} bulan)')
            ax_rek.fill_between(fc_rek.index, fc_rek.values,
                                alpha=0.15, color=col_rek)
            ax_rek.plot(fc_rek.index, fc_rek.values,
                        marker='o', linestyle='-', linewidth=2.5,
                        color=col_rek,
                        markerfacecolor='#13161e', markeredgewidth=2,
                        label=f'Forecast — {metode_rekomendasi} (Rekomendasi)')
            ax_rek.axvline(x=fc_rek.index[0], color='#ff6b6b',
                           linestyle='--', alpha=0.7, linewidth=1.5,
                           label='Awal Forecast')
            ax_rek.set_title(
                f'Forecast Rekomendasi — {metode_rekomendasi} — {produk}', pad=14)
            ax_rek.legend()
            st.pyplot(fig_rek)
            plt.close()
