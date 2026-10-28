"""
Delhi Air Pollution Regimes - Streamlit dashboard
Run:  streamlit run app.py
Folder layout (copy these from your project):
    app.py
    data/features/  delhi_scaled.csv, delhi_features.csv, delhi_context.csv
    data/results/   kmeans_labels.csv, gmm_labels.csv, hdbscan_mcs20_ms10_labels.csv
                    gmm_probabilities.csv (optional), method_comparison.csv (optional),
                    method_agreement.csv (optional)
    data/raw/       city_day.csv   (needed for the map)
"""
import base64
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from sklearn.cluster import KMeans, HDBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.neighbors import NearestNeighbors

st.set_page_config(page_title="Delhi Air Regimes", page_icon="🌫️", layout="wide")

BASE = Path(__file__).parent
FEAT_DIR, RES_DIR, RAW_DIR = BASE / "data" / "features", BASE / "data" / "results", BASE / "data" / "raw"
WIND = ["wind_dir_sin", "wind_dir_cos"]
BLUE, DARK = "#2196F3", "#0D47A1"
FONT = "Roboto, sans-serif"

# ----------------------------------------------------------------------------- styling
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap');
html, body, .stApp, [class*="css"] {{ font-family: {FONT}; }}
.stApp {{ background:#ffffff; }}
header[data-testid="stHeader"] {{ background:transparent !important; }}
[data-testid="stToolbar"], [data-testid="stDecoration"], [data-testid="stStatusWidget"] {{ display:none !important; }}
.block-container {{ padding:3.2rem 2.2rem 3rem 2.2rem; max-width:100%; }}
[data-testid="stSidebar"] {{
    background:
      radial-gradient(circle at 78% 22%, rgba(150,205,235,.55) 0%, rgba(150,205,235,0) 42%),
      radial-gradient(circle at 15% 52%, rgba(70,130,225,.60) 0%, rgba(70,130,225,0) 48%),
      radial-gradient(circle at 85% 78%, rgba(110,185,215,.65) 0%, rgba(110,185,215,0) 45%),
      radial-gradient(circle at 20% 95%, rgba(50,100,190,.55) 0%, rgba(50,100,190,0) 50%),
      linear-gradient(180deg,#3A78C4 0%,#4A88C6 50%,#3468B6 100%); }}
[data-testid="stSidebar"] > div:first-child {{ background:transparent !important; }}
[data-testid="stSidebar"] label, [data-testid="stSidebar"] label p,
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3,
[data-testid="stSidebar"] .stMarkdown p {{ color:#fff !important; }}
[data-testid="stSidebar"] div[data-baseweb="select"] > div,
[data-testid="stSidebar"] div[data-baseweb="select"] > div > div {{ background-color:#ffffff !important; border-radius:8px; }}
[data-testid="stSidebar"] div[data-baseweb="select"] * {{ color:#000000 !important; }}
[data-testid="stSidebar"] div[data-baseweb="select"] svg {{ fill:#000000 !important; }}
div[data-baseweb="select"] div, div[data-baseweb="select"] input, div[data-baseweb="select"] span {{
    background-color:#ffffff !important; color:#000000 !important; }}
div[data-baseweb="select"] > div {{ border:1px solid #CFD8DC !important; border-radius:8px; }}
div[data-baseweb="select"] svg {{ fill:#000000 !important; }}
div[data-baseweb="input"], div[data-baseweb="input"] > div, div[data-baseweb="base-input"], div[data-baseweb="base-input"] input {{
    background-color:#ffffff !important; color:#000000 !important; }}
ul[role="listbox"], ul[role="listbox"] li, div[data-baseweb="menu"] {{ background-color:#ffffff !important; color:#000000 !important; }}
ul[role="listbox"] li:hover, li[aria-selected="true"] {{ background-color:#E3F2FD !important; }}
.tbl-wrap {{ overflow:auto; max-height:540px; border-radius:6px; box-shadow:0 1px 6px rgba(0,0,0,.15); background:#fff; margin-bottom:14px; }}
table.clean-table {{ border-collapse:collapse; width:100%; font-size:.9rem; background:#fff; }}
.clean-table th {{ background:#E3F2FD; color:#0D47A1 !important; text-align:left; padding:9px 12px; font-weight:500; position:sticky; top:0; }}
.clean-table tbody th {{ background:#F4F8FD; color:#000 !important; position:static; }}
.clean-table td {{ padding:8px 12px; border-top:1px solid #ECEFF1; color:#000 !important; background:#fff; }}
.clean-table tbody tr:nth-child(even) td {{ background:#F8FAFD; }}
[data-testid="stSelectbox"] *:not(label):not(label *):not([data-testid="stWidgetLabel"]):not([data-testid="stWidgetLabel"] *),
[data-testid="stDateInput"] *:not(label):not(label *):not([data-testid="stWidgetLabel"]):not([data-testid="stWidgetLabel"] *),
[data-testid="stNumberInput"] *:not(label):not(label *):not([data-testid="stWidgetLabel"]):not([data-testid="stWidgetLabel"] *) {{
    background-color:#ffffff !important; color:#000000 !important; }}
[data-testid="stSelectbox"] svg, [data-testid="stDateInput"] svg {{ fill:#000000 !important; color:#000000 !important; }}
[data-testid="stSelectbox"] > div:last-child, [data-testid="stDateInput"] > div:last-child {{
    border:1px solid #CFD8DC; border-radius:8px; overflow:hidden; background:#ffffff; }}
[role="listbox"], [role="listbox"] *, [role="option"] {{ background-color:#ffffff !important; color:#000000 !important; }}
[role="option"]:hover, [role="option"][aria-selected="true"] {{ background-color:#E3F2FD !important; }}
.stApp {{ color:#263238; }}
.block-container p, .block-container li, .block-container label, .block-container h1,
.block-container h2, .block-container h3, .block-container h4,
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p {{ color:#263238 !important; }}
.card, .card p, .card li, .card h4, .kpi .t {{ color:#263238 !important; }}
.page-title {{ color:#1a237e !important; }}
.badge {{ color:#ffffff !important; }}
[data-testid="stSidebar"] hr {{ border-color:rgba(255,255,255,.25); }}
.logo-center {{ display:flex; justify-content:center; padding:.6rem 0 1rem 0; }}
.logo-center img {{ width:190px; max-width:90%; height:auto; filter:drop-shadow(0 3px 8px rgba(10,40,90,.35)); }}
.brand {{ font-size:1.75rem !important; font-weight:700; color:#fff !important; padding:.2rem 0 .6rem 0; line-height:1.2; }}
.side-stat {{ color:#BBDEFB; font-size:.85rem; margin-top:.9rem; }}
.side-num {{ color:#fff; font-size:1.9rem; font-weight:700; line-height:1.1; }}
h1, h2, h3, h4 {{ color:#263238; font-weight:500; }}
.page-title {{ font-family:Roboto,sans-serif !important; font-size:3.2rem !important; font-weight:700 !important; color:#1a237e !important; margin:0 0 1rem 0 !important; line-height:1.15 !important; }}
.page-sub {{ font-family:Roboto,sans-serif !important; font-size:1.3rem !important; font-weight:500; color:#263238 !important;
             margin:.5rem 0 1.1rem 0; line-height:1.4; }}
.block-container p, .block-container li, .block-container label, .block-container h1, .block-container h2,
.block-container h3, .block-container h4 {{ font-family:Roboto,sans-serif; }}
.aqi-strip {{ display:flex; border-radius:6px; overflow:hidden; margin:.4rem 0 1.2rem 0; font-size:.75rem; color:#fff; }}
.aqi-strip div {{ flex:1; text-align:center; padding:5px 0; }}
.kpi {{ background:#fff; border-radius:6px; padding:14px 20px; box-shadow:0 1px 6px rgba(0,0,0,.15); }}
.kpi .t {{ font-size:1.05rem; font-weight:500; color:#263238; }}
.kpi .t small {{ font-size:.7rem; color:#78909C; margin-left:6px; }}
.kpi .v {{ font-size:1.9rem; font-weight:700; margin-top:6px; }}
.kpi .v small {{ font-size:.72rem; color:#78909C; font-weight:400; margin-left:6px; }}
.card {{ background:#fff; border-radius:6px; padding:16px 22px; box-shadow:0 1px 6px rgba(0,0,0,.15); margin-bottom:14px; }}
.card h4 {{ margin:0 0 .4rem 0; }}
.card ul {{ margin:.2rem 0 0 1.1rem; padding:0; }}
.badge {{ display:inline-block; padding:3px 12px; border-radius:14px; font-size:.8rem; font-weight:500; color:#fff; }}
[data-testid="stPlotlyChart"] {{ background:#fff; border-radius:6px; box-shadow:0 1px 6px rgba(0,0,0,.15); padding:6px; }}
button[data-baseweb="tab"] p {{ font-size:.98rem; font-weight:500; }}
button[data-baseweb="tab"][aria-selected="true"] p {{ color:{BLUE}; }}
div[data-baseweb="tab-highlight"] {{ background-color:{BLUE}; }}
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------------- helpers
_NEW = tuple(int(p) for p in st.__version__.split(".")[:2]) >= (1, 50)
_k = [0]


def show(fig, h=440):
    fig.update_layout(template="plotly_white", height=h, paper_bgcolor="#ffffff", plot_bgcolor="#ffffff",
                      font=dict(family=FONT, color="#263238"), legend=dict(font=dict(color="#263238")),
                      margin=dict(l=40, r=20, t=55, b=40), title_font=dict(size=17, color="#263238"))
    _k[0] += 1
    if _NEW:
        st.plotly_chart(fig, width="stretch", key=f"p{_k[0]}", theme=None)
    else:
        st.plotly_chart(fig, use_container_width=True, key=f"p{_k[0]}", theme=None)


def table(df, hide_index=False, **kw):
    """White table with black text (st.dataframe follows the dark theme, so we draw our own)."""
    fmt = lambda v: f"{v:.3f}".rstrip("0").rstrip(".")
    html = df.to_html(classes="clean-table", index=not hide_index, border=0, na_rep="-", float_format=fmt)
    st.markdown(f"<div class='tbl-wrap'>{html}</div>", unsafe_allow_html=True)


def card(title, body, accent=BLUE):
    st.markdown(f"<div class='card' style='border-left:5px solid {accent}'><h4>{title}</h4>{body}</div>",
                unsafe_allow_html=True)


def kpi(col, title, value, sub="", color="#263238"):
    col.markdown(f"<div class='kpi'><div class='t'>{title}</div>"
                 f"<div class='v' style='color:{color}'>{value}<small>{sub}</small></div></div>",
                 unsafe_allow_html=True)


def aqi_cat(v):
    """CPCB AQI bands -> (label, colour, health advice)."""
    bands = [(50, "Good", "#2E7D32", "Air is clean. Great day for outdoor activity."),
             (100, "Satisfactory", "#7CB342", "Minor discomfort possible for very sensitive people."),
             (200, "Moderate", "#F9A825", "People with asthma or heart disease should limit long outdoor exertion."),
             (300, "Poor", "#EF6C00", "Prolonged outdoor exposure can cause discomfort. Consider a mask."),
             (400, "Very Poor", "#D32F2F", "Respiratory illness on prolonged exposure. Avoid outdoor exercise."),
             (10_000, "Severe", "#7B1FA2", "Health emergency. Stay indoors and use a purifier if you can.")]
    for hi, name, colour, advice in bands:
        if v <= hi:
            return name, colour, advice


# ----------------------------------------------------------------------------- data loading
REQUIRED = [FEAT_DIR / "delhi_scaled.csv", FEAT_DIR / "delhi_features.csv", FEAT_DIR / "delhi_context.csv",
            RES_DIR / "kmeans_labels.csv", RES_DIR / "gmm_labels.csv"]
missing = [str(p.relative_to(BASE)) for p in REQUIRED if not p.exists()]
if missing:
    st.error("Some project files are missing. Copy your `data/` folder next to `app.py`.\n\n" +
             "\n".join(f"- `{m}`" for m in missing))
    st.stop()


def _read(path):
    return pd.read_csv(path, parse_dates=["Date"], index_col="Date")


@st.cache_data(show_spinner="Loading Delhi data…")
def load_all():
    X = _read(FEAT_DIR / "delhi_scaled.csv")
    feat, ctx = _read(FEAT_DIR / "delhi_features.csv"), _read(FEAT_DIR / "delhi_context.csv")
    Xn = X.drop(columns=WIND)
    Xs = Xn.drop(columns=["SO2"])

    def lab(name):
        return _read(RES_DIR / name).iloc[:, 0].reindex(Xn.index)

    km, gmm = lab("kmeans_labels.csv"), lab("gmm_labels.csv")
    try:
        hdb = lab("hdbscan_mcs20_ms10_labels.csv")
    except FileNotFoundError:
        hdb = pd.Series(HDBSCAN(min_cluster_size=20, min_samples=10, copy=True).fit_predict(Xn), index=Xn.index)
    proba = _read(RES_DIR / "gmm_probabilities.csv") if (RES_DIR / "gmm_probabilities.csv").exists() else None
    pca = PCA(n_components=5).fit(Xn)
    coords = pd.DataFrame(pca.transform(Xn), index=Xn.index, columns=[f"PC{i}" for i in range(1, 6)])
    return dict(X=X, Xn=Xn, Xs=Xs, feat=feat, ctx=ctx, km=km, gmm=gmm, hdb=hdb, proba=proba,
                coords=coords, pca_var=pca.explained_variance_ratio_)


D = load_all()
ctx, feat = D["ctx"], D["feat"]
RAW_COLS = [c for c in feat.columns if c not in WIND]

# ----------------------------------------------------------------------------- algorithm metadata
ALGOS = {
    "KMeans": dict(
        labels=D["km"], space=D["Xn"], final=True, icon="🎯",
        names={0: "Monsoon washout", 1: "Moderate winter smog", 2: "Pre-monsoon dust",
               3: "Severe stagnant winter smog", 4: "Warm, cleaner days"},
        colors={0: "#1E88E5", 1: "#EF6C00", 2: "#C0A020", 3: "#C62828", 4: "#2E7D32"},
        what="KMeans splits the days into k groups. It places k centre points and assigns every day to the "
             "nearest centre, then moves the centres until the groups stop changing.",
        used=["k = 5, 13 features (wind direction left out), 20 random starts",
              "k chosen with inertia, silhouette, Davies-Bouldin and stability across random seeds",
              "k = 6 was rejected: one cluster was almost entirely 2017 days, a data-era artefact"],
        pros=["Stable: ARI 0.974 across random seeds", "Every day gets a regime (0% noise)",
              "Regimes are easy to name and explain"],
        cons=["Silhouette is only about 0.20, so the groups overlap", "Assumes round, similar-sized groups",
              "Does not isolate Diwali spikes"],
        verdict="Final choice. Most stable and most interpretable of the three. Lockdown days move from the dust "
                "regime to the cleaner regime in 2020 (88%), which supports the regimes.",
    ),
    "HDBSCAN": dict(
        labels=D["hdb"], space=D["Xn"], final=False, icon="🌌",
        names={-1: "Noise (unclassified)", 0: "Winter pocket", 1: "Moderate mix", 2: "Monsoon washout"},
        colors={-1: "#B0BEC5", 0: "#7E57C2", 1: "#EF6C00", 2: "#1E88E5"},
        what="HDBSCAN looks for dense crowds of similar days. Groups that stay together across many density "
             "levels are kept. Days in sparse areas are labelled noise instead of being forced into a group.",
        used=["min_cluster_size = 20, min_samples = 10, 13 features",
              "Result: 3 small clusters and about 75% of days marked as noise",
              "No clear elbow in the nearest-neighbour distance curve, so there is no natural density cut-off"],
        pros=["Finds irregular shapes", "Honest about days that fit nowhere", "No need to choose k"],
        cons=["75% noise makes it unusable as a regime map", "Only the monsoon washout is clearly separated",
              "17 of 21 Diwali days are noise"],
        verdict="Not chosen, but useful. It shows Delhi's air does not form sharp islands. Conditions change "
                "gradually, so regimes are tendencies, not hard categories.",
    ),
    "Gaussian Mixture": dict(
        labels=D["gmm"], space=D["Xs"], final=False, icon="🔔",
        names={0: "Hot, windy, coarse dust", 1: "Cold winter with rain", 2: "Cold calm smog",
               3: "Monsoon washout", 4: "Mixed moderate"},
        colors={0: "#C0A020", 1: "#7E57C2", 2: "#C62828", 3: "#1E88E5", 4: "#26A69A"},
        what="A Gaussian Mixture Model describes each regime as a bell-shaped cloud with its own centre and "
             "shape, and gives every day a probability of belonging to each regime.",
        used=["k = 5, full covariance, SO2 removed (it caused 2017-only components)",
              "BIC kept falling partly because a component fitted the 2017 SO2 anomaly",
              "Mean max membership probability 0.987, so it acted like hard clustering"],
        pros=["Soft, probabilistic membership", "Handles elongated and tilted clusters",
              "Finds a cold-with-rain winter group KMeans misses"],
        cons=["Stability 0.62, lower than KMeans", "Merges severe and moderate winter smog at k = 5",
              "Agreement with KMeans is low (ARI 0.25)"],
        verdict="Not chosen. It does not outperform KMeans here, but it agrees on the monsoon washout "
                "(82% overlap) and adds a useful view of rainy winter days.",
    ),
}

# Hard-coded numbers from your notebooks (used only if the saved CSVs are not present)
FALLBACK_CMP = pd.DataFrame({
    "method": ["KMeans k=5 (13 features)", "KMeans k=5 (no SO2)", "GMM k=5 (no SO2)", "HDBSCAN mcs20/ms10"],
    "clusters": [5, 5, 5, 3], "noise_%": [0.0, 0.0, 0.0, 74.7], "largest_cluster_%": [25.0, 24.0, 33.0, 79.8],
    "silhouette": [0.203, 0.208, 0.083, 0.107], "davies_bouldin": [1.585, 1.567, 2.847, 1.359],
    "stability": [0.974, 0.900, 0.620, np.nan]})
FALLBACK_ARI = pd.DataFrame(
    [[1, .84, .25, .13], [.84, 1, .28, .11], [.25, .28, 1, .20], [.13, .11, .20, 1]],
    index=["KMeans", "KMeans noSO2", "GMM", "HDBSCAN"], columns=["KMeans", "KMeans noSO2", "GMM", "HDBSCAN"])
KM_STAB = {2: .993, 3: .876, 4: .774, 5: .974, 6: .691, 7: .655, 8: .680}
GMM_BIC = {"full": [17782.8, 16920.8, 16579.6, 16746.4, 14936.9, 14947.2, 14907.9, 14530.6, 15196.1],
           "diag": [31941.2, 29490.0, 26757.4, 26509.3, 24933.0, 24640.9, 24415.2, 24248.2, 23875.2]}
GMM_STAB = {"full": [1.0, .499, .968, .992, .579, .593, .664], "diag": [.983, .842, .874, 1.0, .596, .647, .602]}

# ----------------------------------------------------------------------------- sidebar
with st.sidebar:
    LOGO = BASE / "assets" / "logo.png"
    if LOGO.exists():
        b64 = base64.b64encode(LOGO.read_bytes()).decode()
        st.markdown(f"<div class='logo-center'><img src='data:image/png;base64,{b64}'></div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='brand'>AirRegime.AI</div>", unsafe_allow_html=True)
    st.markdown("---")
    algo = st.selectbox("Select algorithm", list(ALGOS), help="Every tab updates for the algorithm you pick.")
    A = ALGOS[algo]
    st.markdown("---")
    pcs = [f"PC{i}" for i in range(1, 6)]
    x_ax = st.selectbox("X axis", pcs, index=0)
    y_ax = st.selectbox("Y axis", pcs, index=1)
    z_ax = st.selectbox("Z axis (3D)", pcs, index=2)
    psize = st.slider("Point size", 2, 10, 4)
    st.markdown("---")
    labels = A["labels"]
    n_clusters = labels[labels != -1].nunique()
    noise_pct = (labels == -1).mean() * 100
    st.markdown(f"<div class='side-stat'>Days displayed</div><div class='side-num'>{len(labels):,}</div>"
                f"<div class='side-stat'>Regimes found</div><div class='side-num'>{n_clusters}</div>"
                f"<div class='side-stat'>Unclassified (noise)</div><div class='side-num'>{noise_pct:.0f}%</div>",
                unsafe_allow_html=True)

# ----------------------------------------------------------------------------- header + KPIs
st.markdown("<p class='page-title'>AirRegime.AI</p>", unsafe_allow_html=True)
strip = [("Good 0-50", "#2E7D32"), ("Satisfactory 51-100", "#7CB342"), ("Moderate 101-200", "#F9A825"),
         ("Poor 201-300", "#EF6C00"), ("Very poor 301-400", "#D32F2F"), ("Severe 401+", "#7B1FA2")]
st.markdown("<div class='aqi-strip'>" + "".join(f"<div style='background:{c}'>{t}</div>" for t, c in strip) +
            "</div>", unsafe_allow_html=True)

aqi = ctx["AQI"].dropna()
c1, c2, c3, c4 = st.columns(4)
kpi(c1, "Days analysed", f"{len(labels):,}", "after cleaning")
kpi(c2, "Avg AQI", f"{aqi.mean():.0f}", aqi_cat(aqi.mean())[0], aqi_cat(aqi.mean())[1])
kpi(c3, "Max AQI", f"{aqi.max():.0f}", f"on {aqi.idxmax():%Y-%m-%d}", aqi_cat(aqi.max())[1])
kpi(c4, "Min AQI", f"{aqi.min():.0f}", f"on {aqi.idxmin():%Y-%m-%d}", aqi_cat(aqi.min())[1])
st.write("")


def plot_df():
    d = D["coords"].copy()
    d["Regime"] = labels.map(A["names"])
    d["AQI"] = ctx["AQI"]
    d["PM2.5"] = feat["PM2.5"]
    d["Date"] = d.index
    return d


cmap = {A["names"][k]: v for k, v in A["colors"].items()}
order = [A["names"][k] for k in sorted(A["names"])]

tabs = st.tabs(["🌐 3D Visualization", "📈 2D & Timeline", "🧠 About Algorithm", "⚖️ Algorithm Comparison",
                "📊 Cluster Statistics", "🗺️ Air Quality Map", "🗂️ Dataset Overview"])

# ============================================================================= TAB 1: 3D
with tabs[0]:
    st.subheader(f"3D Cluster Visualization: {algo}")
    pdf = plot_df()
    fig = px.scatter_3d(pdf, x=x_ax, y=y_ax, z=z_ax, color="Regime", color_discrete_map=cmap,
                        category_orders={"Regime": order}, hover_data={"AQI": ":.0f", "PM2.5": ":.0f", "Date": "|%d %b %Y"},
                        opacity=.8, title=f"{algo} clustering in 3D PCA space")
    fig.update_traces(marker=dict(size=psize, line=dict(width=0)))
    fig.update_layout(scene=dict(xaxis_title=x_ax, yaxis_title=y_ax, zaxis_title=z_ax), legend_title="Regime")
    show(fig, 640)
    ev = D["pca_var"]
    st.caption(f"PCA is only used to draw the picture (the models were fitted on the 13 scaled features). "
               f"PC1 = overall pollution load ({ev[0]*100:.0f}% of variance), PC2 = warm/windy/dusty vs cold/fine-particle "
               f"days ({ev[1]*100:.0f}%). The days form a continuous cloud, not separate islands.")

# ============================================================================= TAB 2: 2D + timeline
with tabs[1]:
    st.subheader(f"2D View and Timeline: {algo}")
    pdf = plot_df()
    fig = px.scatter(pdf, x=x_ax, y=y_ax, color="Regime", color_discrete_map=cmap,
                     category_orders={"Regime": order}, hover_data={"AQI": ":.0f", "Date": "|%d %b %Y"},
                     opacity=.75, title=f"{algo} clustering: {x_ax} vs {y_ax}")
    fig.update_traces(marker=dict(size=psize + 3))
    show(fig, 520)
    fig = px.scatter(pdf, x="Date", y="PM2.5", color="Regime", color_discrete_map=cmap,
                     category_orders={"Regime": order}, opacity=.8, title="Daily PM2.5 coloured by regime")
    fig.update_traces(marker=dict(size=psize + 1))
    fig.update_yaxes(title="PM2.5 (µg/m³)")
    show(fig, 400)
    m = pd.crosstab(labels.map(A["names"]), ctx["month"], normalize="index") * 100
    m = m.reindex(order).dropna(how="all")
    m.columns = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"][:len(m.columns)] \
        if len(m.columns) == 12 else m.columns
    fig = px.imshow(m, color_continuous_scale="Blues", aspect="auto", text_auto=".0f",
                    title="When does each regime happen? (% of the regime's days per month)")
    fig.update_layout(coloraxis_colorbar_title="%")
    show(fig, 360)

# ============================================================================= TAB 3: About
with tabs[2]:
    st.subheader(f"About {algo}")
    badge = ("<span class='badge' style='background:#2E7D32'>Final model</span>" if A["final"]
             else "<span class='badge' style='background:#78909C'>Compared, not chosen</span>")
    st.markdown(badge, unsafe_allow_html=True)
    st.write("")
    a, b = st.columns(2)
    with a:
        card("What is it?", f"<p>{A['what']}</p>")
        card("Settings used in this project", "<ul>" + "".join(f"<li>{u}</li>" for u in A["used"]) + "</ul>")
    with b:
        card("Strengths here", "<ul>" + "".join(f"<li>{u}</li>" for u in A["pros"]) + "</ul>", "#2E7D32")
        card("Limitations here", "<ul>" + "".join(f"<li>{u}</li>" for u in A["cons"]) + "</ul>", "#EF6C00")
    card("Did we go ahead with it?", f"<p>{A['verdict']}</p>", "#2E7D32" if A["final"] else "#78909C")

    st.markdown("#### How the settings were chosen")
    if algo == "KMeans":
        @st.cache_data(show_spinner="Scoring k = 2 to 10…")
        def km_scan(Xa, Xb):
            rows = []
            for name, data in [("all 15 features", Xa), ("no wind direction", Xb)]:
                for k in range(2, 11):
                    m_ = KMeans(n_clusters=k, n_init=20, random_state=42).fit(data)
                    rows.append(dict(variant=name, k=k, inertia=m_.inertia_,
                                     silhouette=silhouette_score(data, m_.labels_),
                                     davies_bouldin=davies_bouldin_score(data, m_.labels_)))
            return pd.DataFrame(rows)
        res = km_scan(D["X"], D["Xn"])
        c1, c2, c3 = st.columns(3)
        for col, metric, ttl in [(c1, "inertia", "Inertia (elbow)"), (c2, "silhouette", "Silhouette (higher is better)"),
                                 (c3, "davies_bouldin", "Davies-Bouldin (lower is better)")]:
            with col:
                show(px.line(res, x="k", y=metric, color="variant", markers=True, title=ttl), 330)
        stab = pd.DataFrame({"k": list(KM_STAB), "stability (ARI)": list(KM_STAB.values())})
        fig = px.bar(stab, x="k", y="stability (ARI)", title="Stability across random starts (no wind direction)",
                     color_discrete_sequence=[BLUE])
        fig.add_hline(y=.9, line_dash="dash", line_color="grey")
        show(fig, 330)
        st.info("Silhouette is low everywhere (0.16 to 0.29): the regimes overlap. k = 5 without wind direction is "
                "the best mix of stability (0.97) and interpretability.")
    elif algo == "HDBSCAN":
        @st.cache_data(show_spinner="Scanning HDBSCAN settings…")
        def hdb_scan(Xn):
            d = np.sort(NearestNeighbors(n_neighbors=10).fit(Xn).kneighbors(Xn)[0][:, -1])
            rows = []
            for mcs in [10, 20, 30, 50, 80]:
                for ms in [5, 10, 20]:
                    lab = HDBSCAN(min_cluster_size=mcs, min_samples=ms, copy=True).fit_predict(Xn)
                    keep = lab != -1
                    ncl = len(set(lab)) - (1 if -1 in lab else 0)
                    rows.append({"min_cluster_size": mcs, "min_samples": ms, "clusters": ncl,
                                 "noise_%": round((~keep).mean() * 100, 1),
                                 "silhouette_no_noise": round(silhouette_score(Xn[keep], lab[keep]), 3) if ncl >= 2 else np.nan})
            return d, pd.DataFrame(rows)
        dist, grid = hdb_scan(D["Xn"])
        c1, c2 = st.columns(2)
        with c1:
            fig = px.line(y=dist, title="Distance to the 10th nearest day (sorted)", labels={"y": "distance", "index": "days"})
            fig.update_traces(line_color=BLUE)
            show(fig, 360)
        with c2:
            g = grid.assign(setting=lambda t: "min_samples=" + t["min_samples"].astype(str))
            show(px.line(g, x="min_cluster_size", y="noise_%", color="setting", markers=True,
                         title="Noise share for each setting"), 360)
        table(grid, hide_index=True)
        st.info("A smooth curve with no elbow means there is no natural density threshold, which hints at a continuum.")
    else:
        ks = list(range(2, 11))
        c1, c2 = st.columns(2)
        with c1:
            bic = pd.DataFrame({"components": ks * 2, "BIC": GMM_BIC["full"] + GMM_BIC["diag"],
                                "covariance": ["full"] * 9 + ["diag"] * 9})
            show(px.line(bic, x="components", y="BIC", color="covariance", markers=True,
                         title="BIC (lower is better)"), 360)
        with c2:
            ks2 = list(range(2, 9))
            st_ = pd.DataFrame({"components": ks2 * 2, "stability": GMM_STAB["full"] + GMM_STAB["diag"],
                                "covariance": ["full"] * 7 + ["diag"] * 7})
            show(px.line(st_, x="components", y="stability", color="covariance", markers=True,
                         title="Stability across random starts"), 360)
        st.info("BIC flattens after k = 6 but assumes independent days, which is not true for a time series. "
                "k = 5 (full covariance) is stable (0.99), k = 6 is not (0.58).")

# ============================================================================= TAB 4: Comparison
with tabs[3]:
    st.subheader("Algorithm Comparison")
    cmp_path, ari_path = RES_DIR / "method_comparison.csv", RES_DIR / "method_agreement.csv"
    cmp_df = pd.read_csv(cmp_path) if cmp_path.exists() else FALLBACK_CMP
    ari = pd.read_csv(ari_path, index_col=0) if ari_path.exists() else FALLBACK_ARI
    card("Decision: we went ahead with KMeans (k = 5, wind direction excluded)",
         "<ul><li>Most stable across random starts (0.974 vs 0.62 for GMM)</li>"
         "<li>Labels every day, while HDBSCAN leaves 75% as noise</li>"
         "<li>Regimes match real seasons: monsoon washout, winter smog, pre-monsoon dust, cleaner warm days</li>"
         "<li>Honest caveat: silhouette is about 0.20 in every method, so regimes are tendencies, not sharp categories</li></ul>",
         "#2E7D32")
    table(cmp_df, hide_index=True)
    st.caption("Silhouette and Davies-Bouldin are computed in each method's own feature space, and HDBSCAN's only on "
               "non-noise days, so they are indicative, not strictly comparable.")
    pal = ["#1565C0", "#42A5F5", "#26A69A", "#EF6C00"]
    c1, c2 = st.columns(2)
    for col, metric, ttl in [(c1, "silhouette", "Silhouette (higher is better)"),
                             (c2, "davies_bouldin", "Davies-Bouldin (lower is better)"),
                             (c1, "stability", "Stability across seeds (higher is better)"),
                             (c2, "noise_%", "Share of days labelled noise")]:
        with col:
            show(px.bar(cmp_df, x="method", y=metric, color="method", color_discrete_sequence=pal, title=ttl)
                 .update_layout(showlegend=False, xaxis_title=None), 360)
    fig = px.imshow(ari.astype(float), text_auto=".2f", color_continuous_scale="Blues", zmin=0, zmax=1,
                    title="How much do the methods agree? (Adjusted Rand Index)")
    show(fig, 430)
    st.caption("KMeans with and without SO2 agree (0.84), but KMeans, GMM and HDBSCAN carve the same days quite "
               "differently (0.11 to 0.28). Only the monsoon washout is found by all three.")

# ============================================================================= TAB 5: Cluster stats
with tabs[4]:
    st.subheader(f"Cluster Statistics: {algo}")
    prof = feat[RAW_COLS].groupby(labels).mean().round(2)
    prof["days"] = labels.value_counts().sort_index()
    prof["AQI"] = ctx["AQI"].groupby(labels).mean().round(0)
    prof.index = [A["names"][i] for i in prof.index]
    table(prof.T)
    c1, c2 = st.columns(2)
    with c1:
        cnt = labels.map(A["names"]).value_counts().reindex(order).dropna().reset_index()
        cnt.columns = ["Regime", "Days"]
        show(px.bar(cnt, x="Regime", y="Days", color="Regime", color_discrete_map=cmap, title="Days per regime")
             .update_layout(showlegend=False, xaxis_title=None), 400)
    with c2:
        bx = pd.DataFrame({"Regime": labels.map(A["names"]), "AQI": ctx["AQI"]}).dropna()
        show(px.box(bx, x="Regime", y="AQI", color="Regime", color_discrete_map=cmap, category_orders={"Regime": order},
                    title="AQI distribution by regime").update_layout(showlegend=False, xaxis_title=None), 400)
    c1, c2 = st.columns(2)
    with c1:
        sp = A["space"]
        z = sp.groupby(labels).mean()
        fig = go.Figure()
        for k_, row in z.iterrows():
            fig.add_trace(go.Scatterpolar(r=list(row.values) + [row.values[0]], theta=list(z.columns) + [z.columns[0]],
                                          name=A["names"][k_], line_color=A["colors"][k_], fill="toself", opacity=.45))
        fig.update_layout(title="Regime fingerprint (scaled feature means)")
        show(fig, 460)
    with c2:
        yr = pd.crosstab(ctx["year"], labels.map(A["names"]))
        yr = yr[[c for c in order if c in yr.columns]]
        show(px.bar(yr, barmode="stack", color_discrete_map=cmap, title="Regimes by year (2020 is Jan to Jul only)",
                    labels={"value": "days", "year": "year"}), 460)
    if algo == "Gaussian Mixture" and D["proba"] is not None:
        conf = D["proba"].max(axis=1)
        fig = px.histogram(conf, nbins=30, title=f"GMM confidence: mean max probability {conf.mean():.3f}, "
                           f"{int((conf < .7).sum())} days below 0.7", color_discrete_sequence=[BLUE])
        show(fig.update_layout(showlegend=False, xaxis_title="highest membership probability"), 330)

    st.markdown("#### 🔍 Day explorer")
    d0 = st.date_input("Pick a date", value=pd.Timestamp("2018-11-08"), min_value=labels.index.min(),
                       max_value=labels.index.max())
    idx = labels.index.get_indexer([pd.Timestamp(d0)], method="nearest")[0]
    day = labels.index[idx]
    lab_day = labels.iloc[idx]
    aq = ctx["AQI"].iloc[idx]
    cat = aqi_cat(aq) if pd.notna(aq) else ("n/a", "#78909C", "No AQI reported for this day.")
    e1, e2, e3, e4 = st.columns(4)
    kpi(e1, "Date used", f"{day:%d %b %Y}")
    kpi(e2, "Regime", A["names"][lab_day], "", A["colors"][lab_day])
    kpi(e3, "AQI", "n/a" if pd.isna(aq) else f"{aq:.0f}", cat[0], cat[1])
    kpi(e4, "PM2.5", f"{feat['PM2.5'].iloc[idx]:.0f}", "µg/m³")
    st.write("")
    card("Health note", f"<p>{cat[2]}</p><p>Temp {feat['temp'].iloc[idx]:.1f} °C, wind {feat['wind_speed'].iloc[idx]:.0f} km/h, "
         f"rain {feat['rain'].iloc[idx]:.1f} mm, PM2.5/PM10 ratio {feat['PM_ratio'].iloc[idx]:.2f}.</p>", cat[1])

# ============================================================================= TAB 6: Map
with tabs[5]:
    st.subheader("Air Quality Map of Indian Cities")
    COORDS = {"Ahmedabad": (23.03, 72.58), "Aizawl": (23.73, 92.72), "Amaravati": (16.51, 80.52),
              "Amritsar": (31.63, 74.87), "Bengaluru": (12.97, 77.59), "Bhopal": (23.26, 77.41),
              "Brajrajnagar": (21.82, 83.92), "Chandigarh": (30.73, 76.78), "Chennai": (13.08, 80.27),
              "Coimbatore": (11.02, 76.96), "Delhi": (28.61, 77.21), "Ernakulam": (9.98, 76.28),
              "Gurugram": (28.46, 77.03), "Guwahati": (26.14, 91.74), "Hyderabad": (17.39, 78.49),
              "Jaipur": (26.91, 75.79), "Jorapokhar": (23.71, 86.41), "Kochi": (9.93, 76.27),
              "Kolkata": (22.57, 88.36), "Lucknow": (26.85, 80.95), "Mumbai": (19.08, 72.88),
              "Patna": (25.59, 85.14), "Shillong": (25.58, 91.89), "Talcher": (20.95, 85.22),
              "Thiruvananthapuram": (8.52, 76.94), "Visakhapatnam": (17.69, 83.22)}
    city_path = RAW_DIR / "city_day.csv"
    if not city_path.exists():
        st.warning("Put the Kaggle file `city_day.csv` in `data/raw/` to enable the map.")
    else:
        @st.cache_data
        def load_city():
            return pd.read_csv(city_path, parse_dates=["Date"])
        cd = load_city()
        f1, f2, f3, f4 = st.columns(4)
        metric = f1.selectbox("Measure", ["AQI (average)", "AQI (worst day)", "PM2.5", "PM10", "NO2"])
        yr = f2.selectbox("Year", ["All years"] + sorted(cd["Date"].dt.year.unique().tolist()))
        season = f3.selectbox("Season", ["All seasons", "Winter (Dec-Feb)", "Pre-monsoon (Mar-May)",
                                         "Monsoon (Jun-Sep)", "Post-monsoon (Oct-Nov)"])
        show_what = f4.selectbox("Show cities", ["All cities", "Bad air only (Poor or worse)",
                                                 "Good air only (Good/Satisfactory)"])
        q = cd.copy()
        if yr != "All years":
            q = q[q["Date"].dt.year == yr]
        months = {"Winter (Dec-Feb)": [12, 1, 2], "Pre-monsoon (Mar-May)": [3, 4, 5],
                  "Monsoon (Jun-Sep)": [6, 7, 8, 9], "Post-monsoon (Oct-Nov)": [10, 11]}.get(season)
        if months:
            q = q[q["Date"].dt.month.isin(months)]
        col = {"AQI (average)": "AQI", "AQI (worst day)": "AQI"}.get(metric, metric)
        agg = "max" if metric == "AQI (worst day)" else "mean"
        g = q.groupby("City").agg(value=(col, agg), avg_aqi=("AQI", "mean"), days=(col, "count")).dropna(subset=["value"])
        g = g[g["days"] > 0]
        g["lat"] = [COORDS.get(c, (np.nan, np.nan))[0] for c in g.index]
        g["lon"] = [COORDS.get(c, (np.nan, np.nan))[1] for c in g.index]
        g = g.dropna(subset=["lat"]).reset_index()
        g["Category"] = g["avg_aqi"].apply(lambda v: aqi_cat(v)[0] if pd.notna(v) else "n/a")
        if show_what.startswith("Bad"):
            g = g[g["avg_aqi"] > 200]
        elif show_what.startswith("Good"):
            g = g[g["avg_aqi"] <= 100]
        if g.empty:
            st.info("No cities match this selection. Try another year, season or filter.")
        else:
            aqi_scale = [[0, "#2E7D32"], [.1, "#2E7D32"], [.2, "#7CB342"], [.4, "#F9A825"], [.6, "#EF6C00"],
                         [.8, "#D32F2F"], [1, "#7B1FA2"]]
            is_aqi = metric.startswith("AQI")
            kw = dict(lat="lat", lon="lon", color="value", size="value", size_max=34, hover_name="City",
                      hover_data={"lat": False, "lon": False, "value": ":.0f", "Category": True, "days": True},
                      color_continuous_scale=aqi_scale if is_aqi else "RdYlGn_r",
                      range_color=(0, 500) if is_aqi else None, zoom=3.6, center=dict(lat=22.5, lon=79),
                      labels={"value": metric})
            if hasattr(px, "scatter_map"):
                fig = px.scatter_map(g, map_style="carto-positron", **kw)
            else:
                fig = px.scatter_mapbox(g, mapbox_style="carto-positron", **kw)
            fig.update_layout(title=f"{metric} by city: green is cleaner air, red is worse air",
                              coloraxis_colorbar_title=metric)
            show(fig, 640)
            r1, r2 = st.columns(2)
            top = g.sort_values("value", ascending=False)
            with r1:
                show(px.bar(top.head(8), x="value", y="City", orientation="h", color="value",
                            color_continuous_scale=aqi_scale if is_aqi else "RdYlGn_r",
                            title="Worst cities in this view").update_layout(yaxis=dict(autorange="reversed"),
                                                                           coloraxis_showscale=False), 380)
            with r2:
                show(px.bar(top.tail(8).iloc[::-1], x="value", y="City", orientation="h", color="value",
                            color_continuous_scale=aqi_scale if is_aqi else "RdYlGn_r",
                            title="Cleanest cities in this view").update_layout(yaxis=dict(autorange="reversed"),
                                                                              coloraxis_showscale=False), 380)
            st.caption("Averages use only days with a reading. Some cities have far fewer days, see the hover card.")

# ============================================================================= TAB 7: Dataset overview
with tabs[6]:
    st.subheader("Dataset Overview")
    o1, o2, o3, o4 = st.columns(4)
    kpi(o1, "Raw Delhi days", "2,009", "2015-01 to 2020-07")
    kpi(o2, "After cleaning", f"{len(feat):,}", f"from {feat.index.min():%Y-%m-%d}")
    kpi(o3, "Engineered features", f"{feat.shape[1]}", "13 used for clustering")
    kpi(o4, "Sources", "2", "CPCB + Open-Meteo")
    st.write("")
    p, q_ = st.columns(2)
    with p:
        card("Project pipeline", "<ul><li><b>01 Collection:</b> CPCB city_day.csv (Delhi) merged with Open-Meteo weather</li>"
             "<li><b>02 EDA:</b> peaks October to January, PM10 high but PM2.5 low in April to June (dust)</li>"
             "<li><b>03 Cleaning:</b> kept data from Dec 2016, zeros treated as missing, 2017 outage removed, 4 impossible days dropped</li>"
             "<li><b>04 Features:</b> 9 pollutants, PM2.5/PM10 ratio, temperature, rain, wind</li>"
             "<li><b>05 Scaling and PCA:</b> log transform, standardise, PCA for visualisation</li>"
             "<li><b>06 to 08:</b> KMeans, HDBSCAN, Gaussian Mixture</li><li><b>09:</b> comparison and final choice</li></ul>")
    with q_:
        card("Cleaning decisions", "<ul><li>Dropped Xylene (39% missing), City and AQI_Bucket</li>"
             "<li>Removed data before Dec 2016 because O3 and CO shift levels</li>"
             "<li>Removed 8 Jun to 25 Sep 2017 sensor outage</li><li>Interpolated gaps of up to 3 days only</li>"
             "<li>Excluded NOx and Toluene (inconsistent readings) from clustering; kept for validation</li>"
             "<li>Wind direction stored as sine and cosine, left out of final models</li></ul>", "#EF6C00")
    mdf = feat.groupby(feat.index.month).agg(pm25=("PM2.5", "mean"), ratio=("PM_ratio", "mean"))
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_bar(x=["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], y=mdf["pm25"],
                name="PM2.5 (µg/m³)", marker_color=BLUE)
    fig.add_scatter(x=["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], y=mdf["ratio"],
                    name="PM2.5/PM10 ratio", mode="lines+markers", line_color="#C62828", secondary_y=True)
    fig.update_layout(title="Delhi's pollution season: PM2.5 and fine-particle share by month")
    show(fig, 400)
    c1, c2 = st.columns(2)
    with c1:
        yrs = feat.groupby(feat.index.year)[["PM2.5", "PM10"]].mean().round(1)
        show(px.bar(yrs, barmode="group", title="Yearly mean PM2.5 and PM10 (2016 is Dec only, 2020 is Jan to Jul)",
                    color_discrete_sequence=[BLUE, "#90CAF9"], labels={"value": "µg/m³", "index": "year"}), 400)
    with c2:
        show(px.imshow(feat.corr().round(2), text_auto=".1f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                       title="Correlation between features"), 400)
    c1, c2 = st.columns(2)
    full = PCA().fit(D["X"])
    with c1:
        cum = np.cumsum(full.explained_variance_ratio_) * 100
        fig = px.line(x=list(range(1, len(cum) + 1)), y=cum, markers=True, title="PCA: cumulative variance explained",
                      labels={"x": "components", "y": "% variance"})
        fig.add_hline(y=90, line_dash="dash", line_color="grey")
        fig.update_traces(line_color=BLUE)
        show(fig, 380)
    with c2:
        load = pd.DataFrame(full.components_[:3].T, index=D["X"].columns, columns=["PC1", "PC2", "PC3"])
        show(px.imshow(load.round(2), text_auto=".2f", color_continuous_scale="RdBu_r", color_continuous_midpoint=0, aspect="auto",
                       title="What do the first 3 PCs mean? (loadings)"), 380)
    desc = pd.DataFrame([
        ("PM2.5, PM10", "Particles", "Fine and coarse dust, smoke (µg/m³)", "Yes"),
        ("NO, NO2", "Gases", "Traffic and combustion", "Yes"),
        ("NH3, SO2", "Gases", "Agriculture / industry and fuel burning", "Yes (SO2 not in GMM)"),
        ("O3, CO, Benzene", "Gases", "Ozone, incomplete combustion, fuel vapour", "Yes"),
        ("PM_ratio", "Derived", "PM2.5 / PM10, capped at 1.0", "Yes"),
        ("temp, rain, wind_speed", "Weather", "Open-Meteo daily values for Delhi", "Yes"),
        ("wind_dir_sin, wind_dir_cos", "Weather", "Wind direction as two coordinates", "Tested, then dropped"),
        ("AQI, NOx, Toluene, calendar flags", "Context", "Diwali, stubble season, monsoon, lockdown", "Validation only")],
        columns=["Feature", "Group", "Meaning", "Used for clustering"])
    st.markdown("#### Feature guide")
    table(desc, hide_index=True)
    with st.expander("Preview the engineered features"):
        table(feat.round(2).head(50))
