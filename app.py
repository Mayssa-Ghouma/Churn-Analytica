"""
CHURN ANALYTICA
Dashboard · Simulateur de Risque · Carte Géographique
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import folium
from streamlit_folium import st_folium
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════════════════
# CONFIG PAGE
# ══════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Churn Analytica",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ══════════════════════════════════════════════════════════════════════
# CSS — SOFT PASTEL THEME
# ══════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Inter:wght@300;400;500;600;700&display=swap');

:root {
    --primary:       #6366f1;
    --primary-light: #a5b4fc;
    --secondary:     #ec4899;
    --bg-main:       #f8fafc;
    --card-bg:       rgba(255, 255, 255, 0.7);
    --card-border:   rgba(226, 232, 240, 0.8);
    --text-main:     #1e293b;
    --text-muted:    #64748b;
    --accent-red:    #ef4444;
    --accent-green:  #22c55e;
    --accent-amber:  #f59e0b;
    --sidebar-bg:    linear-gradient(180deg, #f1f5f9 0%, #ffffff 100%);
}

/* Base Styles */
html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--bg-main) !important;
    color: var(--text-main) !important;
    font-family: 'Inter', sans-serif;
}

[data-testid="stAppViewContainer"] > .main {
    background: radial-gradient(circle at top right, #f5f3ff, #f8fafc 50%, #f1f5f9) !important;
}

[data-testid="stSidebar"] {
    background: var(--sidebar-bg) !important;
    border-right: 1px solid var(--card-border);
    backdrop-filter: blur(20px);
}

[data-testid="stSidebar"] * {
    color: var(--text-main) !important;
}

h1, h2, h3, .logo-text {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-weight: 800 !important;
    letter-spacing: -0.02em !important;
    color: #0f172a !important;
}

/* Glass Cards */
.metric-card, [data-testid="stMetric"], .action-card, .stInfo, .stSuccess, .stError, .stWarning {
    background: var(--card-bg) !important;
    backdrop-filter: blur(12px) !important;
    border: 1px solid var(--card-border) !important;
    border-radius: 20px !important;
    padding: 24px !important;
    box-shadow: 0 10px 25px -5px rgba(0,0,0,0.05), 0 8px 10px -6px rgba(0,0,0,0.05) !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    color: var(--text-main) !important;
}

.metric-card:hover {
    transform: translateY(-5px);
    border-color: var(--primary-light) !important;
    box-shadow: 0 20px 40px -15px rgba(99, 102, 241, 0.15) !important;
}

.metric-value {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 2.5rem;
    font-weight: 800;
    background: linear-gradient(135deg, #1e293b 0%, #64748b 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.metric-label {
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: var(--text-muted);
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(226, 232, 240, 0.5);
    border-radius: 16px;
    padding: 6px;
    gap: 8px;
    border: 1px solid var(--card-border);
}

.stTabs [data-baseweb="tab"] {
    height: 44px;
    background: transparent;
    border-radius: 12px;
    color: var(--text-muted) !important;
    padding: 0 24px;
    font-weight: 600;
    transition: all 0.2s;
}

.stTabs [aria-selected="true"] {
    background: white !important;
    color: var(--primary) !important;
    box-shadow: 0 4px 12px rgba(0,0,0,0.08);
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, var(--primary) 0%, #4f46e5 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 14px !important;
    font-weight: 700 !important;
    padding: 12px 28px !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    box-shadow: 0 8px 15px -3px rgba(79, 70, 229, 0.3) !important;
    transition: all 0.3s !important;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 12px 20px -3px rgba(79, 70, 229, 0.4) !important;
}

/* Form Inputs */
.stSlider, .stSelectbox, .stNumberInput {
    background: rgba(255, 255, 255, 0.5);
    border-radius: 12px;
    padding: 10px;
    border: 1px solid var(--card-border);
}

/* Logo */
.logo-text {
    font-size: 1.8rem;
    background: linear-gradient(135deg, #6366f1, #ec4899);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 2rem;
}

/* Custom Scrollbar */
::-webkit-scrollbar { width: 8px; }
::-webkit-scrollbar-track { background: var(--bg-main); }
::-webkit-scrollbar-thumb { background: #cbd5e1; border-radius: 10px; }
::-webkit-scrollbar-thumb:hover { background: #94a3b8; }

.section-title {
    font-size: 0.9rem;
    font-weight: 700;
    color: var(--primary);
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin: 2rem 0 1rem;
    display: flex;
    align-items: center;
    gap: 10px;
}

.section-title::after {
    content: "";
    height: 1px;
    flex-grow: 1;
    background: linear-gradient(90deg, #e2e8f0, transparent);
}

[data-testid="stDataFrame"] {
    border: 1px solid var(--card-border) !important;
    border-radius: 16px !important;
    overflow: hidden !important;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
# THEME & CSS
# ══════════════════════════════════════════════════════════════════════
PLOTLY_LAYOUT = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(family='Inter', color='#1e293b', size=12),
    margin=dict(l=20, r=20, t=44, b=20),
    legend=dict(bgcolor='rgba(255, 255, 255, 0.8)', bordercolor='rgba(0,0,0,0.05)',
                borderwidth=1, font=dict(color='#1e293b')),
    xaxis=dict(gridcolor='rgba(0,0,0,0.05)', linecolor='rgba(0,0,0,0.1)', tickfont=dict(color='#64748b')),
    yaxis=dict(gridcolor='rgba(0,0,0,0.05)', linecolor='rgba(0,0,0,0.1)', tickfont=dict(color='#64748b')),
)


# ══════════════════════════════════════════════════════════════════════
# FONCTIONS UTILITAIRES
# ══════════════════════════════════════════════════════════════════════
@st.cache_data
def generate_synthetic_data(n=1500, seed=42):
    rng = np.random.default_rng(seed)
    n_churn = int(n * 0.265)
    n_loyal = n - n_churn

    cities = [
        ('Los Angeles',   34.05, -118.25, 400),
        ('San Francisco', 37.77, -122.42, 200),
        ('San Diego',     32.72, -117.15, 180),
        ('Sacramento',    38.58, -121.49, 150),
        ('San Jose',      37.34, -121.89, 120),
        ('Fresno',        36.74, -119.77,  90),
        ('Long Beach',    33.77, -118.19,  80),
        ('Oakland',       37.80, -122.27,  70),
        ('Bakersfield',   35.37, -119.02,  60),
        ('Anaheim',       33.84, -117.91,  50),
    ]

    def sample_city(n_rows):
        weights = np.array([c[3] for c in cities], dtype=float)
        weights /= weights.sum()
        idx  = rng.choice(len(cities), size=n_rows, p=weights)
        lats = np.array([cities[i][1] for i in idx]) + rng.normal(0, 0.15, n_rows)
        lons = np.array([cities[i][2] for i in idx]) + rng.normal(0, 0.15, n_rows)
        return lats, lons, [cities[i][0] for i in idx]

    # ── Churned ──────────────────────────────────────────────────
    tenure_c   = rng.integers(1, 24,  n_churn).astype(float)
    monthly_c  = rng.uniform(60, 120, n_churn).round(2)
    contract_c = rng.choice(['Month-to-month','One year','Two year'], n_churn, p=[.75,.15,.10])
    internet_c = rng.choice(['Fiber optic','DSL','No'],               n_churn, p=[.65,.25,.10])
    security_c = rng.choice(['Yes','No'], n_churn, p=[.20,.80])
    support_c  = rng.choice(['Yes','No'], n_churn, p=[.15,.85])
    backup_c   = rng.choice(['Yes','No'], n_churn, p=[.20,.80])
    depend_c   = rng.choice(['Yes','No'], n_churn, p=[.10,.90])
    payment_c  = rng.choice(['Electronic check','Mailed check','Bank transfer','Credit card'],
                             n_churn, p=[.55,.15,.15,.15])
    cltv_c     = rng.integers(2000, 5000, n_churn).astype(float)
    lat_c, lon_c, city_c = sample_city(n_churn)

    # ── Loyal ─────────────────────────────────────────────────────
    tenure_l   = rng.integers(12, 72, n_loyal).astype(float)
    monthly_l  = rng.uniform(20, 80,  n_loyal).round(2)
    contract_l = rng.choice(['Month-to-month','One year','Two year'], n_loyal, p=[.30,.35,.35])
    internet_l = rng.choice(['Fiber optic','DSL','No'],               n_loyal, p=[.40,.40,.20])
    security_l = rng.choice(['Yes','No'], n_loyal, p=[.60,.40])
    support_l  = rng.choice(['Yes','No'], n_loyal, p=[.55,.45])
    backup_l   = rng.choice(['Yes','No'], n_loyal, p=[.60,.40])
    depend_l   = rng.choice(['Yes','No'], n_loyal, p=[.65,.35])
    payment_l  = rng.choice(['Electronic check','Mailed check','Bank transfer','Credit card'],
                             n_loyal, p=[.25,.20,.30,.25])
    cltv_l     = rng.integers(3000, 8500, n_loyal).astype(float)
    lat_l, lon_l, city_l = sample_city(n_loyal)

    df = pd.DataFrame({
        'CustomerID'      : [f'C{i:05d}' for i in range(n)],
        'Tenure Months'   : np.concatenate([tenure_c,  tenure_l]),
        'Monthly Charges' : np.concatenate([monthly_c, monthly_l]),
        'Total Charges'   : np.concatenate([monthly_c*tenure_c, monthly_l*tenure_l]).round(2),
        'Contract'        : np.concatenate([contract_c, contract_l]),
        'Internet Service': np.concatenate([internet_c, internet_l]),
        'Online Security' : np.concatenate([security_c, security_l]),
        'Tech Support'    : np.concatenate([support_c,  support_l]),
        'Online Backup'   : np.concatenate([backup_c,   backup_l]),
        'Dependents'      : np.concatenate([depend_c,   depend_l]),
        'Payment Method'  : np.concatenate([payment_c,  payment_l]),
        'CLTV'            : np.concatenate([cltv_c,     cltv_l]),
        'Churn Value'     : np.concatenate([np.ones(n_churn), np.zeros(n_loyal)]).astype(int),
        'Latitude'        : np.concatenate([lat_c, lat_l]),
        'Longitude'       : np.concatenate([lon_c, lon_l]),
        'City'            : np.concatenate([city_c, city_l]),
    })
    svc = ['Online Security', 'Tech Support']
    df['N Services']   = df[svc].apply(lambda r: (r == 'Yes').sum(), axis=1)
    df['Tenure Group'] = pd.cut(df['Tenure Months'], bins=[0,12,24,48,72],
                                labels=['0-1an','1-2ans','2-4ans','4-6ans'])
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)


def load_and_prepare(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()
    for col in ['Total Charges','Monthly Charges','CLTV','Tenure Months']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    if all(c in df.columns for c in ['Total Charges','Monthly Charges','Tenure Months']):
        mask = df['Total Charges'].isna()
        df.loc[mask, 'Total Charges'] = (df.loc[mask,'Monthly Charges']
                                         * df.loc[mask,'Tenure Months'])
    svc_candidates = ['Online Security','Online Backup','Device Protection',
                      'Tech Support','Streaming TV','Streaming Movies']
    svc_cols = [c for c in svc_candidates if c in df.columns]
    if svc_cols and 'N Services' not in df.columns:
        df['N Services'] = df[svc_cols].apply(lambda r: (r == 'Yes').sum(), axis=1)
    if 'N Services' not in df.columns:
        df['N Services'] = 0
    if 'Tenure Group' not in df.columns and 'Tenure Months' in df.columns:
        df['Tenure Group'] = pd.cut(df['Tenure Months'], bins=[0,12,24,48,72],
                                    labels=['0-1an','1-2ans','2-4ans','4-6ans'])
    # ── Garantir les colonnes requises par l'app ─────────────────────
    if 'Contract' not in df.columns:
        df['Contract'] = 'Month-to-month'
    if 'Tenure Months' not in df.columns:
        df['Tenure Months'] = 12
    if 'Churn Value' not in df.columns:
        df['Churn Value'] = 0
    if 'CustomerID' not in df.columns:
        df['CustomerID'] = [f'C{i:05d}' for i in range(len(df))]
    # Fallback pour colonnes manquantes
    for c in ['Dependents', 'Online Backup', 'Paperless Billing']:
        if c not in df.columns:
            df[c] = 'No'
    return df


@st.cache_resource
def train_model(df: pd.DataFrame):
    # Liste étendue pour matcher le notebook (Top Features)
    FEATURES = [
        'Tenure Months', 'Monthly Charges', 'Total Charges', 'CLTV',
        'N Services', 'Contract', 'Internet Service', 'Online Security',
        'Tech Support', 'Online Backup', 'Dependents', 'Payment Method'
    ]
    available = [c for c in FEATURES if c in df.columns]
    df_enc = df[available + ['Churn Value']].copy()
    le_map = {}
    for col in df_enc.select_dtypes('object').columns:
        le = LabelEncoder()
        df_enc[col] = le.fit_transform(df_enc[col].astype(str))
        le_map[col] = le
    
    X = df_enc.drop('Churn Value', axis=1).fillna(0)
    y = df_enc['Churn Value']
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25,
                                               random_state=42, stratify=y)
    
    # Paramètres synchronisés avec le notebook (Section 8.2)
    rf = RandomForestClassifier(
        n_estimators=200, 
        max_depth=8,
        class_weight='balanced', 
        random_state=42
    )
    rf.fit(X_tr, y_tr)
    try:
        probs = rf.predict_proba(X_te)
        if probs.shape[1] > 1:
            auc = roc_auc_score(y_te, probs[:, 1])
        else:
            auc = 0.5
    except:
        auc = 0.5
    return rf, le_map, list(X.columns), auc


def predict_client(model, le_map, feature_cols, client_dict: dict) -> float:
    row = {}
    for col in feature_cols:
        val = client_dict.get(col, 0)
        if col in le_map:
            try:
                val = le_map[col].transform([str(val)])[0]
            except ValueError:
                val = 0
        row[col] = [val]
    probs = model.predict_proba(pd.DataFrame(row))[0]
    if len(probs) > 1:
        return float(probs[1])
    else:
        return 1.0 if model.classes_[0] == 1 else 0.0


# ══════════════════════════════════════════════════════════════════════
# SIDEBAR — chargement données PUIS filtres
# ══════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown('<div class="logo-text">CHURN ANALYTICA</div>', unsafe_allow_html=True)
    st.markdown('<div style="color: var(--text-muted); font-size: 0.75rem; margin-top: -1.5rem; margin-bottom: 2rem;">Intelligence Prédictive & Rétention</div>', unsafe_allow_html=True)
    st.markdown("---")

    # ── Upload CSV réel ─────────────────────────────────────────────
    st.markdown('<div class="section-title">Importation</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Fichier CSV client (optionnel)",
        type=["csv"],
        help="Colonnes : Tenure Months, Monthly Charges, Contract, Churn Value…"
    )
    if uploaded_file is not None:
        try:
            df_raw = pd.read_csv(uploaded_file)
            df = load_and_prepare(df_raw)
            st.success(f"Données chargées ({len(df):,} clients)")
        except Exception as e:
            st.error(f"Erreur : {e}")
            df = generate_synthetic_data()
    else:
        df = generate_synthetic_data()
        st.caption("Données synthétiques (1 500 clients)")

    st.markdown("---")
    st.markdown('<div class="section-title">Filtres Stratégiques</div>', unsafe_allow_html=True)

    contracts = st.multiselect(
        "Contrats",
        options=df['Contract'].unique().tolist(),
        default=df['Contract'].unique().tolist()
    )
    tenure_range = st.slider(
        "Ancienneté (mois)",
        min_value=0,
        max_value=int(df['Tenure Months'].max()),
        value=(0, int(df['Tenure Months'].max()))
    )


# ── Filtre appliqué ───────────────────────────────────────────────────
df_f = df[
    df['Contract'].isin(contracts) &
    df['Tenure Months'].between(tenure_range[0], tenure_range[1])
].copy()

# ── Modèle ────────────────────────────────────────────────────────────
model, le_map, feature_cols, model_auc = train_model(df)


# ══════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════
st.markdown("# CHURN ANALYSE")
st.markdown('<p style="color: var(--text-muted); font-size: 1.1rem;">Système de monitoring prédictif et d\'aide à la décision pour la fidélisation client.</p>', unsafe_allow_html=True)
st.markdown("---")

# ══════════════════════════════════════════════════════════════════════
# ONGLETS
# ══════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs([
    "Dashboard",
    "Simulateur de Risque",
    "Carte Géographique",
    "Recommandation d'Offres",
])


# ╔══════════════════════════════════════════════════════════════════╗
# ║  TAB 1 — DASHBOARD                                              ║
# ╚══════════════════════════════════════════════════════════════════╝
with tab1:

    churn_rate   = df_f['Churn Value'].mean() * 100
    n_churned    = int(df_f['Churn Value'].sum())
    n_total      = len(df_f)
    revenue_loss = df_f[df_f['Churn Value'] == 1]['Monthly Charges'].sum()

    c1, c2, c4 = st.columns(3)
    with c1:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Taux de Churn</div>
            <div class="metric-value" style="background: linear-gradient(135deg, #f87171, #ef4444); -webkit-background-clip: text;">{churn_rate:.1f}%</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Clients perdus</div>
            <div class="metric-value" style="background: linear-gradient(135deg, #818cf8, #6366f1); -webkit-background-clip: text;">{n_churned:,}</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Perte mensuelle</div>
            <div class="metric-value" style="background: linear-gradient(135deg, #fca5a5, #f87171); -webkit-background-clip: text;">${revenue_loss:,.0f}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── Ligne 1 : Churn par contrat + ancienneté ──────────────────
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown('<div class="section-title">Churn par type de contrat</div>', unsafe_allow_html=True)
        ct = df_f.groupby('Contract')['Churn Value'].agg(['sum','count']).reset_index()
        ct['rate'] = ct['sum'] / ct['count'] * 100
        fig = go.Figure(go.Bar(
            x=ct['Contract'], y=ct['rate'],
            marker=dict(color=['#f9a8d4','#fcd34d','#6ee7b7'],
                        line=dict(color='#ffffff', width=2)),
            text=ct['rate'].round(1).astype(str) + '%',
            textposition='outside', textfont=dict(color='#3b3557', size=12)
        ))
        fig.update_layout(**PLOTLY_LAYOUT, title='Taux de churn (%)')
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.markdown('<div class="section-title">Churn par ancienneté</div>', unsafe_allow_html=True)
        if 'Tenure Group' in df_f.columns:
            tg = (df_f.groupby('Tenure Group', observed=True)['Churn Value']
                      .mean().reset_index())
            tg['rate'] = tg['Churn Value'] * 100
            fig2 = go.Figure(go.Bar(
                x=tg['Tenure Group'].astype(str), y=tg['rate'],
                marker=dict(color=tg['rate'],
                            colorscale=[[0,'#6ee7b7'],[0.5,'#fcd34d'],[1,'#f9a8d4']],
                            showscale=False),
                text=tg['rate'].round(1).astype(str) + '%',
                textposition='outside', textfont=dict(color='#3b3557', size=12)
            ))
            fig2.update_layout(**PLOTLY_LAYOUT, title='Taux de churn (%)')
            st.plotly_chart(fig2, use_container_width=True)

    # ── Ligne 2 : Histogramme charges + Pie internet ──────────────
    col_c, col_d = st.columns(2)
    with col_c:
        st.markdown('<div class="section-title">Distribution des charges mensuelles</div>', unsafe_allow_html=True)
        fig3 = go.Figure()
        for val, label, color in [(0,'Fidèles','#818cf8'),(1,'Churned','#f9a8d4')]:
            d = pd.to_numeric(df_f[df_f['Churn Value']==val]['Monthly Charges'],
                              errors='coerce').dropna()
            fig3.add_trace(go.Histogram(x=d, name=label, marker_color=color,
                                        opacity=0.75, nbinsx=30))
        fig3.update_layout(**PLOTLY_LAYOUT, barmode='overlay',
                           title='Monthly Charges — Fidèles vs Churned')
        st.plotly_chart(fig3, use_container_width=True)

    with col_d:
        st.markdown('<div class="section-title">Churn par service internet</div>', unsafe_allow_html=True)
        if 'Internet Service' in df_f.columns:
            isp = (df_f.groupby('Internet Service')['Churn Value']
                       .agg(['sum','count']).reset_index())
            isp['rate'] = isp['sum'] / isp['count'] * 100
            fig4 = go.Figure(go.Pie(
                labels=isp['Internet Service'], values=isp['rate'], hole=0.55,
                marker=dict(colors=['#818cf8','#f9a8d4','#6ee7b7'],
                            line=dict(color='#ffffff', width=3)),
                textfont=dict(color='#3b3557')
            ))
            fig4.update_layout(**PLOTLY_LAYOUT, title='% churn par service')
            st.plotly_chart(fig4, use_container_width=True)

    st.markdown(
        f"<div style='color:#9e91c0;font-size:.8rem;text-align:right'>"
        f"Modèle AUC-ROC : <b style='color:#818cf8'>{model_auc:.4f}</b></div>",
        unsafe_allow_html=True
    )


# ╔══════════════════════════════════════════════════════════════════╗
# ║  TAB 2 — SIMULATEUR DE RISQUE                                   ║
# ╚══════════════════════════════════════════════════════════════════╝
with tab2:

    st.markdown("### Simulateur de Risque Client")
    st.markdown("Entrez les caractéristiques d'un client pour obtenir sa probabilité de churn.")
    st.markdown("---")

    col_form, col_result = st.columns(2)

    with col_form:
        st.markdown('<div class="section-title">Configuration du Profil</div>', unsafe_allow_html=True)
        # Slider and Selectbox are already styled via global CSS
        tenure   = st.slider("Ancienneté (mois)",     1,   72,  12)
        monthly  = st.slider("Charges mensuelles ($)", 18, 120,  75)
        cltv     = st.number_input("CLTV ($)", 500, 10000, 3500, step=100)
        contract = st.selectbox("Type de contrat",
                                ['Month-to-month','One year','Two year'])
        internet = st.selectbox("Service internet",  ['Fiber optic','DSL','No'])
        security = st.selectbox("Sécurité en ligne", ['No','Yes'])
        support  = st.selectbox("Support technique", ['No','Yes'])
        payment  = st.selectbox("Mode de paiement",  [
            'Electronic check','Mailed check',
            'Bank transfer (automatic)','Credit card (automatic)'
        ])

        client_dict = {
            'Tenure Months'   : tenure,
            'Monthly Charges' : monthly,
            'Total Charges'   : round(monthly * tenure, 2),
            'CLTV'            : cltv,
            'N Services'      : int(security == 'Yes') + int(support == 'Yes'),
            'Contract'        : contract,
            'Internet Service': internet,
            'Online Security' : security,
            'Tech Support'    : support,
            'Payment Method'  : payment,
        }
        run = st.button("ANALYSER CE CLIENT")

    with col_result:
        st.markdown("<div class='section-title'>Résultat de l'analyse</div>", unsafe_allow_html=True)

        if run:
            prob = predict_client(model, le_map, feature_cols, client_dict)
            pct  = prob * 100

            if pct >= 65:
                badge_cls, badge_lbl, color_gauge = 'badge-high',   'RISQUE ELEVE',  '#f87171'
            elif pct >= 35:
                badge_cls, badge_lbl, color_gauge = 'badge-medium', 'RISQUE MOYEN',  '#fbbf24'
            else:
                badge_cls, badge_lbl, color_gauge = 'badge-low',    'RISQUE FAIBLE', '#6ee7b7'

            fig_gauge = go.Figure(go.Indicator(
                mode='gauge+number',
                value=pct,
                number=dict(suffix='%', font=dict(size=48, color='#1e293b', family='Plus Jakarta Sans')),
                gauge=dict(
                    axis=dict(range=[0, 100], tickcolor='rgba(0,0,0,0.2)'),
                    bar=dict(color=color_gauge, thickness=0.3),
                    bgcolor='rgba(0,0,0,0.05)',
                    bordercolor='rgba(0,0,0,0.1)',
                    steps=[
                        dict(range=[0,  35], color='rgba(34, 197, 94, 0.1)'),
                        dict(range=[35, 65], color='rgba(245, 158, 11, 0.1)'),
                        dict(range=[65,100], color='rgba(239, 68, 68, 0.1)'),
                    ],
                )
            ))
            fig_gauge.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#1e293b'),
                height=280, margin=dict(l=30, r=30, t=40, b=20)
            )
            st.plotly_chart(fig_gauge, use_container_width=True)
            st.markdown(
                f'<div style="text-align:center; margin-top: -2rem;">'
                f'<span style="background: {color_gauge}22; color: {color_gauge}; border: 1px solid {color_gauge}55; padding: 8px 24px; border-radius: 99px; font-weight: 800; font-size: 0.9rem;">{badge_lbl}</span></div>',
                unsafe_allow_html=True
            )

            st.markdown("---")
            st.markdown('<div class="section-title">Actions recommandées</div>', unsafe_allow_html=True)

            actions = []
            if contract == 'Month-to-month':
                actions.append((" Migration contrat",  "Offrir -15% sur contrat annuel ou -25% sur 2 ans"))
            if tenure < 12:
                actions.append((" Programme accueil",  "Déclencher welcome program : appel J+30, J+90, J+180"))
            if internet == 'Fiber optic' and security == 'No':
                actions.append((" Bundle sécurité",    "Offrir Sécurité + Backup gratuit 3 mois"))
            if payment == 'Electronic check':
                actions.append((" Mode de paiement",   "Incentive prélèvement auto : 1 mois offert"))
            if monthly > 80:
                actions.append((" Révision tarifaire", "Proposer plan personnalisé adapté à l'usage réel"))
            if support == 'No' and tenure < 24:
                actions.append((" Support technique",  "Activer accompagnement proactif (appel mensuel)"))
            if not actions:
                actions.append((" Client stable",       "Maintenir qualité de service — programme fidélité"))

            for title, desc in actions:
                st.markdown(
                    f'''<div style="background: rgba(255,255,255,0.03); border: 1px solid var(--card-border); 
                                    border-radius: 14px; padding: 16px; margin-bottom: 10px; border-left: 4px solid var(--primary);">
                        <div style="color: var(--primary-light); font-weight: 700; font-size: 0.9rem;">{title}</div>
                        <div style="color: var(--text-muted); font-size: 0.8rem;">{desc}</div>
                    </div>''',
                    unsafe_allow_html=True
                )

            st.markdown("---")
            st.markdown('<div class="section-title">Profil synthétique</div>', unsafe_allow_html=True)
            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("Ancienneté",   f"{tenure} mois")
            mc2.metric("Charges/mois", f"${monthly}")
            mc3.metric("CLTV",         f"${cltv:,}")

        else:
            st.markdown("""
            <div style="display:flex; flex-direction:column; align-items:center; justify-content:center; height:400px; 
                        background: rgba(0,0,0,0.02); border: 2px dashed var(--card-border); border-radius: 24px; color: var(--text-muted);">
                <div style="font-size: 1rem; font-weight: 600;">En attente de configuration</div>
                <div style="font-size: 0.8rem;">Remplissez le profil et cliquez sur ANALYSER</div>
            </div>""", unsafe_allow_html=True)


# ╔══════════════════════════════════════════════════════════════════╗
# ║  TAB 3 — CARTE GÉOGRAPHIQUE                                     ║
# ╚══════════════════════════════════════════════════════════════════╝
with tab3:

    st.markdown("### Carte Géographique du Churn")
    st.markdown("Visualisez la distribution spatiale des clients et les zones à fort risque.")
    st.markdown("---")

    if 'Latitude' not in df_f.columns or 'Longitude' not in df_f.columns:
        st.warning("Colonnes Latitude/Longitude absentes. Affichage des données synthétiques.")
        df_map = generate_synthetic_data()
    else:
        df_map = df_f.copy()

    ctl1, ctl2, ctl3 = st.columns(3)
    with ctl1:
        map_mode    = st.selectbox("Mode d'affichage",
                                   ['Points individuels','Heatmap','Clusters par ville'])
    with ctl2:
        show_filter = st.selectbox("Afficher",
                                   ['Tous les clients','Churned seulement','Fidèles seulement'])
    with ctl3:
        max_pts = st.slider("Points max", 100, min(1000, len(df_map)), 500)

    if show_filter == 'Churned seulement':
        df_map_show = df_map[df_map['Churn Value'] == 1]
    elif show_filter == 'Fidèles seulement':
        df_map_show = df_map[df_map['Churn Value'] == 0]
    else:
        df_map_show = df_map

    df_map_show = (df_map_show
                   .dropna(subset=['Latitude','Longitude'])
                   .sample(min(max_pts, len(df_map_show)), random_state=42))

    k1, k2, k3 = st.columns(3)
    k1.metric("Points affichés",   f"{len(df_map_show):,}")
    k2.metric("Churned affichés",  f"{int(df_map_show['Churn Value'].sum()):,}")
    
    if 'City' in df_map_show.columns and df_map_show['Churn Value'].sum() > 0:
        top_city = (df_map_show[df_map_show['Churn Value'] == 1]
                    .groupby('City').size().idxmax())
        k3.metric("Ville la plus critique", top_city)
    else:
        k3.metric("Taux churn affiché", f"{df_map_show['Churn Value'].mean()*100:.1f}%")

    m = folium.Map(
        location=[df_map_show['Latitude'].mean(), df_map_show['Longitude'].mean()],
        zoom_start=8,
        tiles='CartoDB positron'
    )

    if map_mode == 'Points individuels':
        for _, row in df_map_show.iterrows():
            color  = '#f87171' if row['Churn Value'] == 1 else '#818cf8'
            radius = 5 if row['Churn Value'] == 1 else 3
            popup_html = (
                f"<div style='font-family:Nunito,sans-serif;font-size:12px;min-width:160px'>"
                f"<b style='color:{color}'>{'CHURNED' if row['Churn Value']==1 else 'FIDELE'}</b><br>"
                f"Ancienneté : {row.get('Tenure Months','—')} mois<br>"
                f"Charges/mois : ${row.get('Monthly Charges','—')}<br>"
                f"Contrat : {row.get('Contract','—')}<br>"
                f"Ville : {row.get('City','—')}</div>"
            )
            folium.CircleMarker(
                location=[row['Latitude'], row['Longitude']],
                radius=radius, color=color, fill=True,
                fill_color=color, fill_opacity=0.7, weight=1,
                popup=folium.Popup(popup_html, max_width=200)
            ).add_to(m)
        m.get_root().html.add_child(folium.Element("""
        <div style='position:fixed; bottom:30px; left:30px; z-index:9999;
                    background: rgba(255, 255, 255, 0.9); backdrop-filter: blur(10px);
                    border: 1px solid rgba(0,0,0,0.1); border-radius: 16px;
                    padding: 16px; font-family: Inter, sans-serif;
                    font-size: 12px; color: #1e293b;
                    box-shadow: 0 10px 30px rgba(0,0,0,0.1)'>
            <div style="margin-bottom: 8px; font-weight: 700; color: var(--primary);">LÉGENDE</div>
            <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 4px;">
                <div style='width: 12px; height: 12px; border-radius: 50%; background: #ef4444;'></div> Churned
            </div>
            <div style="display: flex; align-items: center; gap: 8px;">
                <div style='width: 12px; height: 12px; border-radius: 50%; background: #6366f1;'></div> Fidèle
            </div>
        </div>"""))

    elif map_mode == 'Heatmap':
        from folium.plugins import HeatMap
        heat_data = (df_map_show[df_map_show['Churn Value'] == 1]
                     [['Latitude','Longitude']].values.tolist())
        HeatMap(heat_data, radius=18, blur=15,
                gradient={0.2:'#c4b5fd', 0.5:'#f9a8d4', 0.8:'#f87171'}).add_to(m)

    elif map_mode == 'Clusters par ville':
        if 'City' in df_map_show.columns:
            city_stats = (df_map_show.groupby('City')
                          .agg(lat=('Latitude','mean'), lon=('Longitude','mean'),
                               n_total=('Churn Value','count'),
                               n_churn=('Churn Value','sum'))
                          .reset_index())
            city_stats['rate'] = city_stats['n_churn'] / city_stats['n_total'] * 100
            for _, row in city_stats.iterrows():
                rate   = row['rate']
                color  = '#f87171' if rate >= 30 else '#fbbf24' if rate >= 20 else '#6ee7b7'
                radius = max(10, min(35, row['n_total'] / 15))
                folium.CircleMarker(
                    location=[row['lat'], row['lon']],
                    radius=radius, color=color, fill=True,
                    fill_color=color, fill_opacity=0.6, weight=2,
                    popup=folium.Popup(
                        f"<b>{row['City']}</b><br>"
                        f"Clients : {int(row['n_total'])}<br>"
                        f"Churned : {int(row['n_churn'])}<br>"
                        f"Taux : {rate:.1f}%",
                        max_width=150
                    )
                ).add_to(m)
                folium.Marker(
                    location=[row['lat'], row['lon']],
                    icon=folium.DivIcon(
                        html=f"<div style='color:#3b3557;font-size:9px;"
                             f"font-weight:bold;text-shadow:1px 1px 2px white'>"
                             f"{rate:.0f}%</div>"
                    )
                ).add_to(m)

    st_folium(m, width=None, height=520, returned_objects=[])






# ╔══════════════════════════════════════════════════════════════════╗
# ║  TAB 4 — RECOMMANDATION D'OFFRES                                ║
# ╚══════════════════════════════════════════════════════════════════╝
with tab4:

    st.markdown("### Recommandation d'Offres Personnalisées")
    st.markdown("Identifiez les clients à risque de churn et générez automatiquement les 3 meilleures offres de rétention.")
    st.markdown("---")

    # ── Catalogue d'offres ────────────────────────────────────────────
    OFFERS = [
        {
            "id": "O1",
            "name": "Contrat Fidélité 1 An",
            "category": "Contrat",
            "description": "Migration vers un contrat annuel avec remise immédiate de 15% sur les charges mensuelles.",
            "discount_pct": 15,
            "monthly_saving": None,
            "target_contract": ["Month-to-month"],
            "target_tenure_max": 36,
            "target_internet": None,
            "min_monthly": 0,
            "color": "#818cf8",
            "icon": "C",
            "score_boost": 0.30,
        },
        {
            "id": "O2",
            "name": "Contrat Premium 2 Ans",
            "category": "Contrat",
            "description": "Engagement 2 ans avec -25% sur les charges + téléphone offert le 1er mois.",
            "discount_pct": 25,
            "monthly_saving": None,
            "target_contract": ["Month-to-month", "One year"],
            "target_tenure_max": 60,
            "target_internet": None,
            "min_monthly": 50,
            "color": "#a78bfa",
            "icon": "C",
            "score_boost": 0.40,
        },
        {
            "id": "O3",
            "name": "Bundle Sécurité Offert",
            "category": "Service",
            "description": "Activation gratuite pendant 6 mois de Online Security + Device Protection.",
            "discount_pct": 100,
            "monthly_saving": 15,
            "target_contract": None,
            "target_tenure_max": 999,
            "target_internet": ["Fiber optic", "DSL"],
            "min_monthly": 0,
            "color": "#6ee7b7",
            "icon": "S",
            "score_boost": 0.20,
        },
        {
            "id": "O4",
            "name": "Support Premium Gratuit",
            "category": "Service",
            "description": "Tech Support prioritaire 24/7 offert 3 mois + accompagnement dédié mensuel.",
            "discount_pct": 100,
            "monthly_saving": 10,
            "target_contract": None,
            "target_tenure_max": 24,
            "target_internet": None,
            "min_monthly": 0,
            "color": "#fbbf24",
            "icon": "S",
            "score_boost": 0.18,
        },
        {
            "id": "O5",
            "name": "Remise Fidélité Immédiate",
            "category": "Tarif",
            "description": "Crédit de 3 mois offert sur la prochaine facture + -10% permanent sur votre forfait.",
            "discount_pct": 10,
            "monthly_saving": None,
            "target_contract": None,
            "target_tenure_max": 999,
            "target_internet": None,
            "min_monthly": 70,
            "color": "#f87171",
            "icon": "T",
            "score_boost": 0.22,
        },
        {
            "id": "O6",
            "name": "Streaming Offert 6 Mois",
            "category": "Service",
            "description": "Streaming TV + Streaming Movies activés gratuitement pendant 6 mois.",
            "discount_pct": 100,
            "monthly_saving": 20,
            "target_contract": None,
            "target_tenure_max": 999,
            "target_internet": ["Fiber optic", "DSL"],
            "min_monthly": 60,
            "color": "#fb7185",
            "icon": "S",
            "score_boost": 0.15,
        },
        {
            "id": "O7",
            "name": "Migration Prélèvement Auto",
            "category": "Paiement",
            "description": "Passage au prélèvement automatique avec 1 mois de facture offert + -5% permanent.",
            "discount_pct": 5,
            "monthly_saving": None,
            "target_contract": None,
            "target_tenure_max": 999,
            "target_internet": None,
            "min_monthly": 0,
            "color": "#38bdf8",
            "icon": "P",
            "score_boost": 0.12,
        },
        {
            "id": "O8",
            "name": "Pack Bienvenue Renforcé",
            "category": "Programme",
            "description": "Pour les nouveaux clients : appel personnel J+30/J+90, accès Early Adopter + kit onboarding.",
            "discount_pct": 0,
            "monthly_saving": 0,
            "target_contract": None,
            "target_tenure_max": 12,
            "target_internet": None,
            "min_monthly": 0,
            "color": "#34d399",
            "icon": "P",
            "score_boost": 0.25,
        },
        {
            "id": "O9",
            "name": "Plan Sur-Mesure",
            "category": "Tarif",
            "description": "Audit de l'usage réel + proposition d'un plan 100% adapté avec économie garantie de 20%.",
            "discount_pct": 20,
            "monthly_saving": None,
            "target_contract": None,
            "target_tenure_max": 999,
            "target_internet": None,
            "min_monthly": 80,
            "color": "#c084fc",
            "icon": "T",
            "score_boost": 0.28,
        },
    ]

    def score_offer(offer, client):
        """Score de pertinence d'une offre pour un client (0–1)."""
        score = offer["score_boost"]

        # Contrat
        if offer["target_contract"] and client.get("Contract") in offer["target_contract"]:
            score += 0.25
        elif offer["target_contract"] and client.get("Contract") not in offer["target_contract"]:
            score -= 0.20

        # Ancienneté
        tenure = client.get("Tenure Months", 12)
        if tenure <= offer["target_tenure_max"]:
            score += 0.10

        # Internet
        if offer["target_internet"]:
            if client.get("Internet Service") in offer["target_internet"]:
                score += 0.15
            else:
                score -= 0.15

        # Budget
        monthly = client.get("Monthly Charges", 50)
        if monthly >= offer["min_monthly"]:
            score += 0.08

        # Paiement manuel → offre migration
        if offer["id"] == "O7" and client.get("Payment Method") in ["Electronic check", "Mailed check"]:
            score += 0.30

        # Nouveaux clients → Pack Bienvenue
        if offer["id"] == "O8" and tenure <= 12:
            score += 0.35

        # Charges élevées → Plan sur-mesure
        if offer["id"] == "O9" and monthly >= 80:
            score += 0.20

        return min(max(score, 0), 1)

    def get_top3_offers(client, churn_prob):
        scored = [(o, score_offer(o, client)) for o in OFFERS]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:3]

    def estimated_saving(offer, monthly):
        if offer["monthly_saving"]:
            return offer["monthly_saving"]
        if offer["discount_pct"] > 0:
            return round(monthly * offer["discount_pct"] / 100, 2)
        return 0

    def retention_probability(churn_prob, top3):
        """Estimation de la proba de rétention si toutes les offres sont acceptées."""
        reduction = sum(o["score_boost"] * 0.5 for o, _ in top3)
        retained = min(churn_prob + reduction, 0.95)
        return retained

    # ── SECTION A : Analyse de tous les clients à risque ────────────
    st.markdown('<div class="section-title">Liste des Clients à Risque Détectés</div>', unsafe_allow_html=True)

    # Prédire churn score pour tous les clients filtrés (ceux dans df_f)
    def predict_all(df_input, _model, _le_map, _feature_cols):
        scores = []
        for _, row in df_input.iterrows():
            client = row.to_dict()
            prob = predict_client(_model, _le_map, _feature_cols, client)
            scores.append(round(prob * 100, 1))
        return scores

    with st.spinner("Calcul des scores de churn pour tous les clients..."):
        # On utilise TOUS les clients du segment filtré
        df_risk_calc = df_f.copy()
        churn_scores = predict_all(
            df_risk_calc.reset_index(drop=True),
            model, le_map, feature_cols
        )
        df_risk_calc["Churn Score Predicted"] = churn_scores

    df_at_risk = df_risk_calc[df_risk_calc["Churn Score Predicted"] >= 50].copy()
    df_at_risk = df_at_risk.sort_values("Churn Score Predicted", ascending=False)

    # KPIs masse
    k1c, k2c, k3c, k4c = st.columns(4)
    k1c.metric("Clients analysés",   f"{len(df_risk_calc):,}")
    k2c.metric("Clients à risque",   f"{len(df_at_risk):,}",
               delta=f"{len(df_at_risk)/len(df_risk_calc)*100:.0f}% du segment")
    k3c.metric("Risque élevé (>=65%)", f"{len(df_at_risk[df_at_risk['Churn Score Predicted']>=65]):,}")
    avg_saving_est = df_at_risk["Monthly Charges"].mean() * 0.17 if len(df_at_risk) > 0 else 0
    k4c.metric("Eco. moyenne/mois", f"${avg_saving_est:.0f}" if len(df_at_risk) > 0 else "—")

    st.markdown("---")

    # ── SECTION B : Sélection et Recommandation ───────────────────────
    st.markdown('<div class="section-title">Recommandations Personnalisées par Client</div>', unsafe_allow_html=True)

    if len(df_at_risk) > 0:
        # Création d'une liste pour le selectbox
        client_options = df_at_risk.apply(lambda r: f"{r['CustomerID']} - {r['Churn Score Predicted']}% risque", axis=1).tolist()
        selected_option = st.selectbox("Sélectionnez un client à risque pour voir les offres :", client_options)
        
        # Récupération des données du client sélectionné
        selected_id = selected_option.split(" - ")[0]
        client_data = df_at_risk[df_at_risk['CustomerID'] == selected_id].iloc[0]
        
        # Pré-remplissage des variables pour la recommandation
        r_tenure = client_data['Tenure Months']
        r_monthly = client_data['Monthly Charges']
        r_cltv = client_data['CLTV']
        r_contract = client_data['Contract']
        r_internet = client_data['Internet Service']
        r_security = client_data['Online Security']
        r_support = client_data['Tech Support']
        r_payment = client_data['Payment Method']
    else:
        st.info("Aucun client à risque pour le moment.")
        r_tenure, r_monthly, r_cltv = 12, 70, 4000
        r_contract, r_internet, r_security, r_support, r_payment = 'Month-to-month', 'Fiber optic', 'No', 'No', 'Electronic check'

    col_left, col_right = st.columns([1, 2])

    with col_left:
        st.markdown("**Profil du client sélectionné :**")
        
        # On affiche les infos en lecture seule ou sliders pré-remplis
        st.info(f"""
        **ID:** {selected_id if len(df_at_risk)>0 else 'N/A'}  
        **Ancienneté:** {r_tenure} mois  
        **Charges:** ${r_monthly}  
        **Contrat:** {r_contract}  
        **Internet:** {r_internet}
        """)
        
        client_rec = {
            'Tenure Months'   : r_tenure,
            'Monthly Charges' : r_monthly,
            'Total Charges'   : round(r_monthly * r_tenure, 2),
            'CLTV'            : r_cltv,
            'N Services'      : int(r_security == 'Yes') + int(r_support == 'Yes'),
            'Contract'        : r_contract,
            'Internet Service': r_internet,
            'Online Security' : r_security,
            'Tech Support'    : r_support,
            'Payment Method'  : r_payment,
        }
        
        # Bouton pour forcer la regénération si besoin (bien que ce soit auto avec le selectbox)
        run_rec = st.button("GENERER LES OFFRES", use_container_width=True)

    with col_right:
        if run_rec:
            churn_prob = predict_client(model, le_map, feature_cols, client_rec)
            churn_pct  = churn_prob * 100
            top3       = get_top3_offers(client_rec, churn_prob)

            # Risque header
            if churn_pct >= 65:
                risk_color = "#f87171"; risk_label = "RISQUE ELEVE"
            elif churn_pct >= 35:
                risk_color = "#fbbf24"; risk_label = "RISQUE MOYEN"
            else:
                risk_color = "#6ee7b7"; risk_label = "RISQUE FAIBLE"

            st.markdown(f"""
            <div style="background: rgba(255, 255, 255, 0.6); border: 1px solid {risk_color}33; border-radius: 20px;
                        padding: 24px; margin-bottom: 24px; border-left: 8px solid {risk_color};
                        box-shadow: 0 8px 16px rgba(0,0,0,0.03)">
                <div style="display:flex; justify-content:space-between; align-items:center">
                    <div>
                        <div style="font-size: 0.8rem; color: var(--text-muted); font-weight: 600; text-transform: uppercase;">
                            Indice de Churn
                        </div>
                        <div style="font-size: 3rem; font-weight: 800; color: {risk_color}; line-height: 1">
                            {churn_pct:.0f}%
                        </div>
                    </div>
                    <div style="text-align: right">
                        <div style="font-size: 1.2rem; font-weight: 700; color: {risk_color}">{risk_label}</div>
                        <div style="font-size: 0.8rem; color: var(--text-muted)">Priorité Haute</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # 3 cartes d'offres
            st.markdown("**Top 3 Offres Recommandées :**")

            for rank, (offer, relevance) in enumerate(top3, 1):
                saving = estimated_saving(offer, r_monthly)
                saving_str = f"${saving:.0f}/mois économisés" if saving > 0 else "Valeur ajoutée services"
                relevance_pct = int(relevance * 100)

                rank_label = f"Top {rank}"

                st.markdown(f"""
                <div style="background: rgba(255,255,255,0.7); border: 1px solid var(--card-border);
                            border-radius: 16px; padding: 20px; margin-bottom: 16px;
                            border-left: 4px solid {offer['color']};
                            box-shadow: 0 4px 10px rgba(0,0,0,0.02)">
                    <div style="display:flex; justify-content:space-between; align-items:flex-start">
                        <div style="flex:1">
                            <div style="font-size: 1.1rem; font-weight: 700; color: var(--text-main); margin-bottom: 4px">
                                {rank_label} | {offer['name']}
                            </div>
                            <div style="font-size: 0.9rem; color: var(--text-muted); margin-bottom: 12px">
                                {offer['description']}
                            </div>
                            <div style="display:flex; gap:20px">
                                <div style="font-size: 0.85rem; font-weight: 600; color: {offer['color']}">
                                    ECONOMIE: {saving_str}
                                </div>
                                <div style="font-size: 0.85rem; font-weight: 600; color: var(--primary)">
                                    PERTINENCE: {relevance_pct}%
                                </div>
                            </div>
                        </div>
                        <div style="background: {offer['color']}11; color: {offer['color']}; 
                                    padding: 8px 12px; border-radius: 12px; font-weight: 800; border: 1px solid {offer['color']}22">
                            {offer['icon']}
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # Impact rétention estimé
            retained_prob = retention_probability(churn_prob, top3)
            delta_retention = (retained_prob - churn_prob) * 100

            total_saving_month = sum(estimated_saving(o, r_monthly) for o, _ in top3)
            annual_saving = total_saving_month * 12
            cltv_retained = round(r_cltv * retained_prob, 0)

            st.markdown("---")
            st.markdown("**Impact Estimé si Offres Acceptées :**")
            ia1, ia2, ia3 = st.columns(3)
            ia1.metric("Rétention estimée",
                       f"{retained_prob*100:.0f}%",
                       delta=f"+{delta_retention:.0f}pts vs sans offre")
            ia2.metric("Économie client/an",
                       f"${annual_saving:.0f}",
                       delta="en remises cumulées")
            ia3.metric("CLTV préservée",
                       f"${cltv_retained:,.0f}",
                       delta=f"sur ${r_cltv:,} total")

        else:
            st.markdown("""
            <div style="display:flex;flex-direction:column;align-items:center;
                        justify-content:center;height:320px;
                        background:linear-gradient(135deg,#f5f3ff,#fce7f3);
                        border-radius:16px;border:2px dashed #e9e2ff">
                <div style="font-size:3.5rem;margin-bottom:12px">Offres</div>
                <div style="font-size:1rem;color:#9e91c0;font-weight:600;text-align:center">
                    Sélectionnez un client à gauche<br>et cliquez sur <b style="color:#818cf8">GENERER LES OFFRES</b>
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # ── SECTION C : Vue d'ensemble offres les plus efficaces ─────────
    st.markdown('<div class="section-title">Efficacité du Catalogue d\'Offres</div>', unsafe_allow_html=True)

    offer_names  = [o["name"] for o in OFFERS]
    offer_boosts = [round(o["score_boost"] * 100, 0) for o in OFFERS]
    offer_colors = [o["color"] for o in OFFERS]

    fig_offers = go.Figure(go.Bar(
        x=offer_boosts,
        y=offer_names,
        orientation='h',
        marker=dict(color=offer_colors, line=dict(width=0)),
        text=[f"{b:.0f}%" for b in offer_boosts],
        textposition='outside',
    ))
    fig_offers.update_layout(
        **{**{
            "paper_bgcolor": "rgba(255,255,255,0.85)",
            "plot_bgcolor":  "rgba(255,255,255,0)",
            "font":          dict(family="Nunito, sans-serif", color="#3b3557"),
            "margin":        dict(l=20, r=40, t=30, b=20),
            "showlegend":    False,
        }},
        title="Impact sur la Rétention par Offre (score boost %)",
        height=360,
        xaxis=dict(title="Score boost (%)", gridcolor="#f0ebff"),
        yaxis=dict(tickfont=dict(size=11)),
    )
    st.plotly_chart(fig_offers, use_container_width=True)

    # ── SECTION D : Tableau récapitulatif clients à risque ───────────
    st.markdown('<div class="section-title">Tableau Récapitulatif : Clients à Risque</div>',
                unsafe_allow_html=True)

    if len(df_at_risk) == 0:
        st.info("Aucun client à risque détecté dans le segment filtré.")
    else:
        # Build recommendation table - On affiche tout le monde (ou top 100 pour performance)
        rec_rows = []
        for _, row in df_at_risk.iterrows():
            client_dict_row = row.to_dict()
            prob_row = row["Churn Score Predicted"] / 100
            top3_row = get_top3_offers(client_dict_row, prob_row)
            rec_rows.append({
                "ID Client":        row.get("CustomerID", "—"),
                "Ancienneté":       f"{int(row.get('Tenure Months', 0))} mois",
                "Charges/mois":     f"${row.get('Monthly Charges', 0):.0f}",
                "Contrat":          row.get("Contract", "—"),
                "Churn Score":      f"{row['Churn Score Predicted']:.0f}%",
                "Offre #1":         top3_row[0][0]["name"] if top3_row else "—",
                "Offre #2":         top3_row[1][0]["name"] if len(top3_row) > 1 else "—",
                "Offre #3":         top3_row[2][0]["name"] if len(top3_row) > 2 else "—",
            })

        df_recs = pd.DataFrame(rec_rows)

        def color_score(val):
            v = float(val.replace("%", ""))
            if v >= 65:
                return "background-color:rgba(239, 68, 68, 0.1); color:#ef4444; font-weight:700"
            elif v >= 35:
                return "background-color:rgba(245, 158, 11, 0.1); color:#f59e0b; font-weight:700"
            return "background-color:rgba(34, 197, 94, 0.1); color:#22c55e; font-weight:700"

        styled = df_recs.style.map(color_score, subset=["Churn Score"])
        st.dataframe(styled, use_container_width=True, hide_index=True)

        # Download button
        csv_export = df_recs.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Exporter ce tableau (CSV)",
            data=csv_export,
            file_name="clients_at_risk_offers.csv",
            mime="text/csv",
        )