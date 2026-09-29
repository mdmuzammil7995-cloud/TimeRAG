"""
app.py
TimeRAG - a premium, multi-page forecasting dashboard.

This file owns UI/UX only. All forecasting, RAG retrieval, and LLM
explanation logic lives in pipeline.py / rag_utils.py / llm_utils.py /
train.py, which are used exactly as they exist - nothing here re-implements
or duplicates their logic, and nothing displayed is fabricated: every
number on screen traces back to a real call into one of those modules, or
an honest empty/error state when that data isn't available.

Pages: Overview, Dataset, Forecast, Historical Patterns, AI Explanation,
Model Performance, Reports - navigated via session state, not a full page
reload, so uploaded data and generated results persist as you move around.
"""

import json
import os
import subprocess
import sys

import pandas as pd
import streamlit as st

st.set_page_config(page_title="TimeRAG", page_icon="\u25C8", layout="wide")

# ============================================================== THEME =====
BG = "#080B18"
PANEL = "#0B0F20"
CARD = "#111834"
CARD_GRADIENT = "linear-gradient(135deg, #1C2B6B 0%, #0A1130 100%)"
BORDER = "#20294F"
TEXT = "#F2F4FC"
MUTED = "#8892B0"
ACCENT = "#3D5CFF"
POSITIVE = "#3ECF8E"
NEGATIVE = "#FF5C5C"

# ============================================================== THEME =====
BG = "#050712"
PANEL = "#080C18"
CARD = "#0C1224"
CARD_GRADIENT = "linear-gradient(135deg, rgba(35,52,125,.72) 0%, rgba(7,12,31,.88) 55%, rgba(11,18,43,.72) 100%)"
BORDER = "rgba(111,132,255,.20)"
TEXT = "#F4F7FF"
MUTED = "#8C98B8"
ACCENT = "#5875FF"
POSITIVE = "#45E0A1"
NEGATIVE = "#FF667A"

# UI-only visual enhancement. Forecasting/RAG/LLM logic below is unchanged.
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Space+Grotesk:wght@500;600;700&display=swap');

:root {{
    --tr-bg:{BG};
    --tr-panel:{PANEL};
    --tr-card:{CARD};
    --tr-text:{TEXT};
    --tr-muted:{MUTED};
    --tr-accent:{ACCENT};
    --tr-border:{BORDER};
}}

html, body, .stApp {{
    background:
      radial-gradient(circle at 12% 12%, rgba(72,96,255,.14), transparent 28%),
      radial-gradient(circle at 88% 22%, rgba(122,72,255,.12), transparent 30%),
      radial-gradient(circle at 55% 90%, rgba(0,209,255,.07), transparent 32%),
      {BG};
    color:var(--tr-text);
    font-family:'Inter',sans-serif;
}}

.stApp {{
    min-height:100vh;
    background-attachment:fixed;
}}

.stApp::before {{
    content:"";
    position:fixed;
    inset:-35%;
    z-index:-2;
    pointer-events:none;
    background:
      radial-gradient(circle at 25% 35%, rgba(61,92,255,.16), transparent 22%),
      radial-gradient(circle at 75% 60%, rgba(126,75,255,.13), transparent 25%);
    filter:blur(45px);
    animation:tr-orbit 18s ease-in-out infinite alternate;
}}

.stApp::after {{
    content:"";
    position:fixed;
    inset:0;
    z-index:-1;
    pointer-events:none;
    opacity:.24;
    background-image:
      radial-gradient(circle, rgba(255,255,255,.42) 0 1px, transparent 1.5px),
      radial-gradient(circle, rgba(88,117,255,.35) 0 1px, transparent 1.5px);
    background-size:83px 83px, 137px 137px;
    background-position:0 0, 42px 67px;
    animation:tr-stars 28s linear infinite;
    mask-image:linear-gradient(to bottom, transparent, black 18%, black 82%, transparent);
}}

@keyframes tr-orbit {{
    0% {{ transform:translate3d(-3%,-2%,0) scale(1); }}
    50% {{ transform:translate3d(4%,3%,0) scale(1.06); }}
    100% {{ transform:translate3d(-1%,5%,0) scale(1.02); }}
}}
@keyframes tr-stars {{ to {{ transform:translate3d(-80px,-120px,0); }} }}
@keyframes tr-rise {{
    from {{ opacity:0; transform:translateY(16px) scale(.985); filter:blur(3px); }}
    to {{ opacity:1; transform:none; filter:none; }}
}}
@keyframes tr-pulse {{
    0%,100% {{ box-shadow:0 0 0 0 rgba(69,224,161,.0), 0 0 10px rgba(69,224,161,.12); }}
    50% {{ box-shadow:0 0 0 5px rgba(69,224,161,.04), 0 0 20px rgba(69,224,161,.30); }}
}}
@keyframes tr-shimmer {{
    0% {{ transform:translateX(-130%) skewX(-18deg); }}
    100% {{ transform:translateX(230%) skewX(-18deg); }}
}}

.block-container {{
    padding-top:2rem;
    padding-bottom:4rem;
    max-width:1440px;
    animation:tr-rise .65s ease both;
}}

section[data-testid="stSidebar"] {{
    background:
      linear-gradient(180deg, rgba(9,14,31,.96), rgba(5,8,18,.98));
    border-right:1px solid rgba(105,125,255,.16);
    box-shadow:20px 0 70px rgba(0,0,0,.24);
    backdrop-filter:blur(22px);
    -webkit-backdrop-filter:blur(22px);
}}

section[data-testid="stSidebar"] > div {{
    background:transparent;
}}

h1, h2, h3, h4 {{
    color:var(--tr-text) !important;
    font-family:'Space Grotesk','Inter',sans-serif;
    font-weight:700;
    letter-spacing:-.025em;
}}
h1 {{
    font-size:clamp(2rem,4vw,3.4rem) !important;
    line-height:1.02 !important;
    background:linear-gradient(110deg,#fff 15%,#aebaff 48%,#6f8cff 80%);
    -webkit-background-clip:text;
    background-clip:text;
    -webkit-text-fill-color:transparent;
}}
p, span, label {{ color:var(--tr-text); }}

.tr-brand {{
    font-family:'Space Grotesk','Inter',sans-serif;
    font-size:1.35rem;
    font-weight:800;
    letter-spacing:-.03em;
    padding:5px 2px;
}}
.tr-brand .dot {{ color:var(--tr-accent); }}
.tr-sub {{
    color:var(--tr-muted);
    font-size:.72rem;
    letter-spacing:.14em;
    text-transform:uppercase;
}}

.nav-item {{
    padding:8px 10px;
    border-radius:9px;
    font-size:.88rem;
    color:var(--tr-muted);
}}

.status-row {{
    display:flex !important;
    justify-content:space-between !important;
    align-items:center;
    font-size:.78rem;
    padding:6px 0;
    color:var(--tr-muted);
    width:100%;
}}
.status-dot {{
    display:inline-block;
    width:7px;
    height:7px;
    border-radius:50%;
    margin-right:6px;
    animation:tr-pulse 2.4s infinite;
}}

.kpi-card {{
    position:relative;
    overflow:hidden;
    background:{CARD_GRADIENT};
    border:1px solid rgba(115,135,255,.22);
    border-radius:16px;
    padding:17px 18px;
    box-shadow:0 16px 45px rgba(0,0,0,.24), inset 0 1px 0 rgba(255,255,255,.045);
    backdrop-filter:blur(18px);
    -webkit-backdrop-filter:blur(18px);
    transition:transform .3s ease, border-color .3s ease, box-shadow .3s ease;
    animation:tr-rise .55s ease both;
}}
.kpi-card::before {{
    content:"";
    position:absolute;
    top:0;
    left:-120%;
    width:55%;
    height:100%;
    background:linear-gradient(90deg,transparent,rgba(255,255,255,.07),transparent);
    transform:skewX(-18deg);
    transition:none;
}}
.kpi-card:hover {{
    transform:translateY(-5px);
    border-color:rgba(104,128,255,.48);
    box-shadow:0 22px 55px rgba(35,55,160,.24), 0 0 30px rgba(75,100,255,.09);
}}
.kpi-card:hover::before {{ animation:tr-shimmer 1s ease; }}
.kpi-label {{
    color:var(--tr-muted);
    font-size:.69rem;
    text-transform:uppercase;
    letter-spacing:.12em;
}}
.kpi-value {{
    font-variant-numeric:tabular-nums;
    font-size:1.55rem;
    font-weight:700;
    margin-top:4px;
    color:var(--tr-text);
}}

.section-card, .feature-card, .empty-state {{
    position:relative;
    overflow:hidden;
    background:rgba(12,18,36,.70);
    border:1px solid rgba(105,126,255,.18);
    border-radius:16px;
    box-shadow:0 18px 55px rgba(0,0,0,.20), inset 0 1px 0 rgba(255,255,255,.035);
    backdrop-filter:blur(20px);
    -webkit-backdrop-filter:blur(20px);
}}
.section-card {{
    padding:22px 25px;
    animation:tr-rise .55s ease both;
}}
.feature-card {{
    padding:19px;
    height:100%;
    transition:transform .3s ease,border-color .3s ease,background .3s ease;
}}
.feature-card:hover {{
    transform:translateY(-7px);
    border-color:rgba(100,125,255,.42);
    background:linear-gradient(145deg,rgba(29,43,91,.65),rgba(9,14,30,.75));
}}
.feature-card h4 {{ margin:0 0 7px 0; font-size:.98rem; }}
.feature-card p {{ color:var(--tr-muted); font-size:.82rem; margin:0; line-height:1.6; }}

.empty-state {{
    border-style:dashed;
    padding:30px;
    color:var(--tr-muted);
}}
.empty-state h4 {{ color:var(--tr-text); margin-top:0; }}

.pill {{
    display:inline-block;
    padding:3px 10px;
    border-radius:999px;
    font-size:.7rem;
    background:rgba(88,117,255,.13);
    border:1px solid rgba(88,117,255,.22);
    color:var(--tr-text);
}}

div[data-testid="stDataFrame"], div[data-testid="stExpander"] {{
    background:rgba(10,15,31,.62);
    border:1px solid rgba(105,126,255,.16);
    border-radius:14px;
    overflow:hidden;
    backdrop-filter:blur(16px);
}}

div[data-testid="stFileUploader"] {{
    background:rgba(10,15,31,.52);
    border:1px dashed rgba(104,127,255,.34);
    border-radius:16px;
    transition:border-color .3s ease,box-shadow .3s ease,transform .3s ease;
}}
div[data-testid="stFileUploader"]:hover {{
    border-color:rgba(105,130,255,.75);
    box-shadow:0 0 35px rgba(74,98,255,.12);
    transform:translateY(-2px);
}}

.stButton>button, .stDownloadButton>button,
button[kind="primary"], button[kind="primaryFormSubmit"] {{
    position:relative;
    overflow:hidden;
    background:linear-gradient(135deg,#5875ff,#765cff) !important;
    color:#fff !important;
    border:1px solid rgba(155,170,255,.35) !important;
    border-radius:11px !important;
    font-weight:700 !important;
    letter-spacing:.01em;
    min-height:42px;
    box-shadow:0 9px 25px rgba(72,88,220,.30), inset 0 1px 0 rgba(255,255,255,.17) !important;
    transition:transform .22s ease,box-shadow .22s ease,filter .22s ease !important;
}}
.stButton>button::after, .stDownloadButton>button::after {{
    content:"";
    position:absolute;
    top:-50%;
    left:-120%;
    width:55%;
    height:200%;
    background:linear-gradient(90deg,transparent,rgba(255,255,255,.20),transparent);
    transform:skewX(-20deg);
}}
.stButton>button:hover, .stDownloadButton>button:hover,
button[kind="primary"]:hover {{
    transform:translateY(-3px) !important;
    filter:brightness(1.08) !important;
    box-shadow:0 14px 35px rgba(72,88,220,.40),0 0 24px rgba(88,117,255,.15) !important;
}}
.stButton>button:hover::after, .stDownloadButton>button:hover::after {{
    animation:tr-shimmer .75s ease;
}}

button[kind="secondary"] {{
    background:rgba(255,255,255,.025) !important;
    border:1px solid rgba(110,130,255,.17) !important;
    color:var(--tr-muted) !important;
    box-shadow:none !important;
    border-radius:10px !important;
    transition:all .22s ease !important;
}}
button[kind="secondary"]:hover {{
    color:#fff !important;
    border-color:rgba(110,130,255,.5) !important;
    background:rgba(88,117,255,.10) !important;
    transform:translateX(3px);
}}

.stRadio > div {{
    background:rgba(9,14,30,.55);
    border:1px solid rgba(105,126,255,.14);
    padding:6px 10px;
    border-radius:13px;
}}
.stSelectbox > div > div, .stTextInput > div > div {{
    background:rgba(8,13,27,.70) !important;
    border-color:rgba(105,126,255,.18) !important;
    border-radius:11px !important;
}}

[data-testid="stMetricValue"] {{
    color:var(--tr-text);
    font-weight:700;
}}

.stAlert {{
    border-radius:13px !important;
    backdrop-filter:blur(12px);
}}

div[data-testid="stPlotlyChart"] {{
    border:1px solid rgba(105,126,255,.16);
    border-radius:16px;
    overflow:hidden;
    background:rgba(7,11,24,.54);
    box-shadow:0 18px 55px rgba(0,0,0,.22), inset 0 1px 0 rgba(255,255,255,.03);
    padding:5px;
    animation:tr-rise .6s ease both;
}}

hr {{
    border-color:rgba(105,126,255,.13) !important;
}}

::-webkit-scrollbar {{ width:8px; height:8px; }}
::-webkit-scrollbar-track {{ background:rgba(0,0,0,.18); }}
::-webkit-scrollbar-thumb {{
    background:linear-gradient(180deg,#334aa8,#5c68a7);
    border-radius:999px;
}}
::-webkit-scrollbar-thumb:hover {{ background:#6378dd; }}

@media (max-width: 900px) {{
    .block-container {{ padding-left:1rem; padding-right:1rem; }}
    h1 {{ font-size:2.25rem !important; }}
}}
@media (prefers-reduced-motion: reduce) {{
    *, *::before, *::after {{
        animation-duration:.01ms !important;
        animation-iteration-count:1 !important;
        scroll-behavior:auto !important;
        transition-duration:.01ms !important;
    }}
}}


/* ===== TimeRAG product UI layer ===== */
.tr-hero {{
    position:relative; overflow:hidden; padding:28px 30px 26px; border-radius:24px;
    border:1px solid rgba(112,135,255,.24);
    background:linear-gradient(135deg,rgba(17,27,62,.78),rgba(7,11,25,.70));
    box-shadow:0 30px 90px rgba(0,0,0,.30), inset 0 1px 0 rgba(255,255,255,.06);
    backdrop-filter:blur(24px); -webkit-backdrop-filter:blur(24px); animation:tr-rise .7s ease both;
}}
.tr-hero::before {{ content:""; position:absolute; width:280px; height:280px; right:-80px; top:-120px; border-radius:50%; background:rgba(88,117,255,.20); filter:blur(45px); animation:tr-orbit 12s ease-in-out infinite alternate; }}
.tr-hero::after {{ content:""; position:absolute; inset:0; pointer-events:none; opacity:.12; background:linear-gradient(180deg,rgba(255,255,255,.12),transparent 25%,transparent 75%,rgba(88,117,255,.08)); }}
.tr-eyebrow {{ position:relative; z-index:1; color:#8fa1ff; font-size:.68rem; text-transform:uppercase; letter-spacing:.18em; font-weight:700; }}
.tr-hero-title {{ position:relative; z-index:1; font-family:'Space Grotesk','Inter',sans-serif; font-size:clamp(2.3rem,5vw,4.7rem); line-height:.96; font-weight:800; letter-spacing:-.055em; margin:8px 0 12px; background:linear-gradient(110deg,#fff,#b9c4ff 42%,#6f8cff); -webkit-background-clip:text; background-clip:text; -webkit-text-fill-color:transparent; }}
.tr-hero-copy {{ position:relative; z-index:1; max-width:720px; color:var(--tr-muted); font-size:.96rem; line-height:1.65; }}
.tr-status-cluster {{ position:relative; z-index:1; display:flex; flex-wrap:wrap; gap:8px; margin-top:18px; }}
.tr-status-chip {{ display:inline-flex; align-items:center; gap:7px; padding:7px 11px; border:1px solid rgba(112,135,255,.18); border-radius:999px; background:rgba(6,10,23,.46); color:#cdd5ef; font-size:.72rem; letter-spacing:.03em; }}
.tr-live-dot {{ width:7px; height:7px; border-radius:50%; background:#45e0a1; box-shadow:0 0 0 0 rgba(69,224,161,.5); animation:tr-live 1.8s infinite; }}
@keyframes tr-live {{ 0%{{box-shadow:0 0 0 0 rgba(69,224,161,.45)}} 70%{{box-shadow:0 0 0 8px rgba(69,224,161,0)}} 100%{{box-shadow:0 0 0 0 rgba(69,224,161,0)}} }}
.tr-commandbar {{ display:flex; align-items:center; justify-content:space-between; gap:12px; margin:0 0 18px; padding:10px 14px; border:1px solid rgba(112,135,255,.15); border-radius:14px; background:rgba(7,11,24,.52); backdrop-filter:blur(18px); }}
.tr-command-title {{ color:#aeb9dc; font-size:.72rem; text-transform:uppercase; letter-spacing:.13em; font-weight:700; }}
.tr-pipeline {{ display:flex; align-items:center; gap:0; overflow-x:auto; padding:18px 12px; scrollbar-width:none; }}
.tr-pipeline::-webkit-scrollbar {{ display:none; }}
.tr-pipe-step {{ flex:1 0 120px; min-width:120px; text-align:center; position:relative; }}
.tr-pipe-step:not(:last-child)::after {{ content:""; position:absolute; top:17px; left:calc(50% + 22px); right:-50%; height:1px; background:linear-gradient(90deg,rgba(88,117,255,.65),rgba(88,117,255,.08)); }}
.tr-pipe-node {{ position:relative; z-index:2; width:34px; height:34px; margin:0 auto 8px; display:grid; place-items:center; border-radius:50%; border:1px solid rgba(110,133,255,.32); background:#0a1023; color:#9eafff; box-shadow:0 0 22px rgba(88,117,255,.10); }}
.tr-pipe-label {{ font-size:.66rem; color:#9ba6c4; line-height:1.3; }}
.tr-kpi-grid {{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:12px; }}
.tr-mini-kpi {{ min-height:112px; }}
.tr-spark {{ margin-top:10px; height:3px; border-radius:99px; background:linear-gradient(90deg,rgba(88,117,255,.15),rgba(88,117,255,.8),rgba(69,224,161,.55)); background-size:200% 100%; animation:tr-spark 3s linear infinite; }}
@keyframes tr-spark {{ to {{ background-position:-200% 0; }} }}
.tr-timeline {{ padding:18px 8px 6px; }}
.tr-timeline-line {{ position:relative; height:4px; border-radius:99px; background:linear-gradient(90deg,rgba(150,160,190,.18) 0 48%,rgba(88,117,255,.9) 48% 100%); box-shadow:0 0 22px rgba(88,117,255,.15); }}
.tr-timeline-line::before {{ content:""; position:absolute; left:48%; top:50%; width:12px; height:12px; transform:translate(-50%,-50%); border-radius:50%; background:#fff; box-shadow:0 0 0 5px rgba(88,117,255,.18),0 0 22px rgba(88,117,255,.75); }}
.tr-timeline-labels {{ display:flex; justify-content:space-between; color:#7f8aa8; font-size:.66rem; margin-top:10px; text-transform:uppercase; letter-spacing:.1em; }}
.tr-sim-card {{ position:relative; overflow:hidden; padding:17px; border:1px solid rgba(112,135,255,.17); border-radius:16px; background:linear-gradient(145deg,rgba(18,28,64,.72),rgba(7,11,25,.72)); margin-bottom:10px; transition:transform .25s ease,border-color .25s ease; animation:tr-rise .5s ease both; }}
.tr-sim-card:hover {{ transform:translateY(-3px); border-color:rgba(112,135,255,.42); }}
.tr-sim-head {{ display:flex; justify-content:space-between; gap:12px; align-items:center; }}
.tr-sim-rank {{ font-family:'Space Grotesk'; font-weight:700; color:#fff; }}
.tr-sim-score {{ color:#9eafff; font-weight:700; font-size:.8rem; }}
.tr-progress {{ height:5px; border-radius:99px; overflow:hidden; background:rgba(255,255,255,.06); margin:11px 0; }}
.tr-progress > span {{ display:block; height:100%; border-radius:inherit; background:linear-gradient(90deg,#4e68ff,#78a1ff); box-shadow:0 0 16px rgba(88,117,255,.4); }}
.tr-ai-card {{ position:relative; padding:24px; border-radius:20px; border:1px solid rgba(88,117,255,.24); background:linear-gradient(135deg,rgba(20,30,68,.78),rgba(7,11,25,.78)); box-shadow:0 25px 70px rgba(0,0,0,.24); }}
.tr-ai-card::before {{ content:"✦"; position:absolute; right:22px; top:14px; font-size:42px; color:rgba(126,148,255,.12); animation:tr-pulse 3s infinite; }}
.tr-ai-label {{ color:#8fa1ff; text-transform:uppercase; letter-spacing:.16em; font-size:.68rem; font-weight:700; margin-bottom:10px; }}
.tr-transparent {{ padding:18px; border-radius:16px; border:1px solid rgba(112,135,255,.16); background:rgba(8,13,28,.55); }}
.tr-transparent-step {{ display:flex; align-items:center; gap:10px; padding:9px 0; color:#aeb8d3; font-size:.78rem; border-bottom:1px solid rgba(255,255,255,.045); }}
.tr-transparent-step:last-child {{ border-bottom:0; }}
.tr-step-dot {{ width:8px; height:8px; border-radius:50%; background:#5875ff; box-shadow:0 0 12px rgba(88,117,255,.7); }}
.tr-dropzone {{ padding:20px; margin-bottom:12px; border:1px dashed rgba(112,135,255,.30); border-radius:18px; background:radial-gradient(circle at 50% 0,rgba(88,117,255,.10),transparent 55%),rgba(7,11,25,.42); text-align:center; }}
.tr-drop-icon {{ font-size:1.6rem; color:#8ea0ff; animation:tr-float 2.8s ease-in-out infinite; }}
@keyframes tr-float {{ 0%,100%{{transform:translateY(0)}}50%{{transform:translateY(-5px)}} }}
.tr-presentation section[data-testid="stSidebar"] {{ display:none !important; }}
.tr-presentation .block-container {{ max-width:1600px; padding-top:1.5rem; }}
.tr-presentation [data-testid="stHeader"] {{ display:none; }}
@media(max-width:900px){{ .tr-kpi-grid{{grid-template-columns:repeat(2,minmax(0,1fr));}}.tr-hero{{padding:22px}}.tr-commandbar{{flex-wrap:wrap;}} }}
@media(max-width:560px){{ .tr-kpi-grid{{grid-template-columns:1fr;}} .tr-hero-title{{font-size:2.4rem;}} }}
@media(prefers-reduced-motion:reduce){{ *,*::before,*::after{{animation-duration:.01ms !important;animation-iteration-count:1 !important;scroll-behavior:auto !important;transition-duration:.01ms !important;}} }}

</style>
""", unsafe_allow_html=True)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pipeline import FEATURE_COLUMNS, add_features, infer_step, load_and_clean, recursive_forecast, trend_at  # noqa: E402
from rag_utils import describe_window, load_index, retrieve_top_k  # noqa: E402
from llm_utils import generate_explanation  # noqa: E402

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

MODEL_PATH = os.path.join("models_saved", "model_xgb_v1.pkl")
METRICS_PATH = os.path.join("models_saved", "metrics.json")
VECTOR_STORE_DIR = os.path.join("rag", "vector_store")
PAGES = ["Overview", "Dataset", "Forecast", "Historical Patterns", "AI Explanation", "Model Performance", "Reports"]


@st.cache_resource
def load_model():
    import joblib
    if not os.path.exists(MODEL_PATH):
        return None, None
    try:
        bundle = joblib.load(MODEL_PATH)
    except Exception as exc:  # noqa: BLE001
        return None, str(exc)
    if isinstance(bundle, dict) and "model" in bundle:
        return bundle, None
    return {"model": bundle, "trend_slope": 0.0, "trend_intercept": 0.0}, None


@st.cache_resource
def load_metrics_cached():
    if not os.path.exists(METRICS_PATH):
        return None
    try:
        with open(METRICS_PATH) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


@st.cache_resource
def load_rag_cached():
    try:
        return load_index(VECTOR_STORE_DIR)
    except Exception:  # noqa: BLE001
        return None


def llm_configured() -> bool:
    provider = os.getenv("LLM_PROVIDER", "gemini").lower()
    key = "OPENAI_API_KEY" if provider == "openai" else "GEMINI_API_KEY"
    return bool(os.getenv(key))


def detect_date_value_columns(df: pd.DataFrame) -> tuple[str | None, str | None]:
    date_col, best_ratio = None, 0.0
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            date_col, best_ratio = col, 1.0
            break
        parsed = pd.to_datetime(df[col], errors="coerce")
        ratio = parsed.notna().mean()
        if ratio >= 0.7 and ratio > best_ratio:
            date_col, best_ratio = col, ratio

    target_col, best_var = None, -1.0
    for col in df.columns:
        if col == date_col:
            continue
        series = pd.to_numeric(df[col].astype(str).str.replace(",", "", regex=False), errors="coerce")
        if series.notna().mean() >= 0.8:
            var = series.var()
            if pd.notna(var) and var > best_var:
                target_col, best_var = col, var

    return date_col, target_col


def init_state():
    defaults = {
        "page": "Overview",
        "dataset_key": None,
        "clean_df": None,
        "raw_stats": None,
        "date_label": "Date",
        "value_label": "Value",
        "horizon": 30,
        "forecast_df": None,
        "retrieved": [],
        "explanation": None,
        "explanation_err": None,
        "presentation_mode": False,
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)


def reset_results():
    st.session_state["forecast_df"] = None
    st.session_state["retrieved"] = []
    st.session_state["explanation"] = None
    st.session_state["explanation_err"] = None


def render_topbar(model_bundle, metrics, rag):
    """UI-only command center status bar."""
    model_ok = model_bundle is not None
    rag_ok = rag is not None
    llm_ok = llm_configured()
    dataset_ok = st.session_state.get("clean_df") is not None
    chips = [
        ("MODEL", model_ok), ("RAG", rag_ok), ("LLM", llm_ok), ("DATA", dataset_ok)
    ]
    chip_html = "".join(
        f'<span class="tr-status-chip"><span class="tr-live-dot" style="background:{POSITIVE if ok else MUTED};"></span>{label} {"READY" if ok else "WAITING"}</span>'
        for label, ok in chips
    )
    st.markdown(
        f'<div class="tr-commandbar"><div><span class="tr-command-title">TimeRAG Command Center</span></div>'
        f'<div class="tr-status-cluster" style="margin-top:0">{chip_html}</div></div>',
        unsafe_allow_html=True,
    )


def render_pipeline():
    steps = ["DATASET", "FEATURES", "XGBOOST", "FORECAST", "FAISS", "CONTEXT", "LLM", "INSIGHT"]
    icons = ["↑", "⚙", "◆", "↗", "⌁", "≋", "✦", "✓"]
    html = '<div class="section-card"><div class="tr-pipeline">'
    for i, (step, icon) in enumerate(zip(steps, icons)):
        html += f'<div class="tr-pipe-step"><div class="tr-pipe-node">{icon}</div><div class="tr-pipe-label">{step}</div></div>'
    html += '</div></div>'
    st.markdown(html, unsafe_allow_html=True)


def render_transparency():
    steps = [
        ("Dataset", "Input time-series data"),
        ("Feature Engineering", "Features prepared by the existing pipeline"),
        ("XGBoost Forecast", "Existing trained forecasting model"),
        ("Historical Retrieval", "FAISS retrieves similar windows"),
        ("Relevant Context", "Retrieved evidence supplied to the explanation layer"),
        ("LLM Explanation", "Existing LLM integration explains the forecast"),
    ]
    html = '<div class="tr-transparent">'
    for title, desc in steps:
        html += f'<div class="tr-transparent-step"><span class="tr-step-dot"></span><div><strong style="color:#eef2ff">{title}</strong><br><span style="color:#7f8aa8;font-size:.7rem">{desc}</span></div></div>'
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)


def render_forecast_timeline():
    st.markdown('<div class="tr-timeline"><div class="tr-timeline-line"></div><div class="tr-timeline-labels"><span>Historical</span><span>Forecast Start</span><span>Future</span></div></div>', unsafe_allow_html=True)


def render_presentation_toggle():
    current = st.session_state.get("presentation_mode", False)
    label = "Exit Presentation" if current else "Presentation Mode"
    if st.button(label, key="presentation_toggle", use_container_width=True):
        st.session_state["presentation_mode"] = not current
        st.rerun()
    if st.session_state.get("presentation_mode", False):
        st.markdown('<style>section[data-testid="stSidebar"]{display:none!important}.block-container{max-width:1600px!important}</style>', unsafe_allow_html=True)


def render_sidebar(model_bundle, model_err, metrics, rag):
    with st.sidebar:
        st.markdown('<div class="tr-brand">&#9672; Time<span style="color:#5875FF;">RAG</span></div>', unsafe_allow_html=True)
        st.markdown('<div class="tr-sub">AI Forecast Intelligence</div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        for page in PAGES:
            active = "active" if st.session_state["page"] == page else ""
            if st.button(page, key=f"nav_{page}", use_container_width=True,
                         type="primary" if active else "secondary"):
                st.session_state["page"] = page
                st.rerun()

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("---")

        model_ok = model_bundle is not None
        rag_ok = rag is not None
        llm_ok = llm_configured()

        def status_line(label, ok, ok_text, bad_text):
            color = POSITIVE if ok else MUTED
            text = ok_text if ok else bad_text
            st.markdown(
                f'<div class="status-row"><span>{label}</span>'
                f'<span><span class="status-dot" style="background-color:{color};"></span>{text}</span></div>',
                unsafe_allow_html=True,
            )

        status_line("MODEL", model_ok, "Ready", "Not trained")
        status_line("RAG", rag_ok, "Available", "Unavailable")
        status_line("LLM", llm_ok, "Configured", "Not configured")


def render_overview(model_bundle, rag):
    model_ok = model_bundle is not None
    rag_ok = rag is not None
    llm_ok = llm_configured()
    data_ok = st.session_state.get("clean_df") is not None
    st.markdown(
        '<div class="tr-hero"><div class="tr-eyebrow">TIME-SERIES INTELLIGENCE PLATFORM</div>'
        '<div class="tr-hero-title">Forecast the Future.<br>Understand Why.</div>'
        '<div class="tr-hero-copy">TimeRAG combines forecasting, historical pattern retrieval, and grounded AI explanation into one cinematic analytics workspace.</div>'
        f'<div class="tr-status-cluster">'
        f'<span class="tr-status-chip"><span class="tr-live-dot"></span>SYSTEM ONLINE</span>'
        f'<span class="tr-status-chip">XGBOOST {"✓" if model_ok else "—"}</span>'
        f'<span class="tr-status-chip">FAISS {"✓" if rag_ok else "—"}</span>'
        f'<span class="tr-status-chip">LLM {"✓" if llm_ok else "—"}</span>'
        f'<span class="tr-status-chip">DATASET {"✓" if data_ok else "—"}</span></div></div>',
        unsafe_allow_html=True,
    )
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Start Forecasting  →", type="primary", key="overview_start"):
        st.session_state["page"] = "Dataset"
        st.rerun()

    st.markdown("### AI Pipeline")
    render_pipeline()
    st.markdown("<br>", unsafe_allow_html=True)

    cols = st.columns(4)
    features = [
        ("01", "AI Forecasting", "XGBoost-powered time-series prediction."),
        ("02", "Historical Retrieval", "FAISS finds similar patterns from the series' own history."),
        ("03", "Explainable AI", "LLM explains forecasts using retrieved historical evidence."),
        ("04", "Model Evaluation", "MAE, RMSE, MAPE and R² quantify model performance."),
    ]
    for col, (num, title, desc) in zip(cols, features):
        col.markdown(f'<div class="feature-card"><div class="tr-eyebrow">{num}</div><h4>{title}</h4><p>{desc}</p></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### How the prediction is generated")
    render_transparency()

def _cinematic_plot(fig, height=420):
    """UI-only Plotly styling. Does not alter the underlying chart data."""
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(5,8,18,0.72)",
        font=dict(color=MUTED, family="Inter, sans-serif", size=12),
        height=height,
        margin=dict(l=18, r=18, t=38, b=18),
        hovermode="x unified",
        hoverlabel=dict(
            bgcolor="#0B1020",
            bordercolor="rgba(100,125,255,.35)",
            font=dict(color=TEXT, family="Inter, sans-serif"),
        ),
        legend=dict(
            orientation="h", y=1.05, x=0,
            bgcolor="rgba(0,0,0,0)",
            font=dict(color=MUTED),
        ),
        transition=dict(duration=500, easing="cubic-in-out"),
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            color=MUTED,
            linecolor="rgba(105,126,255,.12)",
            rangeslider=dict(visible=False),
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="rgba(105,126,255,.10)",
            zeroline=False,
            color=MUTED,
        ),
    )
    return fig


def render_dataset():
    st.markdown("## Dataset")
    st.markdown("Upload any time-series CSV. Pick which column is the date/time and which "
                "column to forecast below - a good guess is pre-selected automatically, but "
                "you can always change it, even if auto-detection didn't find anything.")

    st.markdown('<div class="tr-dropzone"><div class="tr-drop-icon">↑</div><strong>DROP YOUR DATASET</strong><br><span style="color:#7f8aa8;font-size:.72rem">CSV time-series data • automatic column detection • validation</span></div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload CSV", type=["csv"], label_visibility="collapsed")

    if uploaded is None:
        st.markdown(
            '<div class="empty-state"><h4>No dataset loaded</h4>'
            "Upload any CSV. As long as it has at least one numeric column, you'll be able to "
            "forecast it - a real date column is used if one is found, otherwise row order is "
            "used as a stand-in time axis.</div>",
            unsafe_allow_html=True,
        )
        return

    file_key = f"{uploaded.name}:{uploaded.size}"
    if st.session_state["dataset_key"] != file_key:
        st.session_state["dataset_key"] = file_key
        st.session_state["clean_df"] = None
        st.session_state["raw_stats"] = None
        reset_results()

    try:
        raw = pd.read_csv(uploaded)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Couldn't read this file: {exc}")
        return

    if raw.shape[1] == 0 or len(raw) == 0:
        st.markdown(
            f'<div class="empty-state" style="border-color:{NEGATIVE};"><h4>Empty file</h4>'
            "This CSV has no rows or no columns to work with.</div>",
            unsafe_allow_html=True,
        )
        return

    auto_date, auto_target = None, None
    if {"date", "value"}.issubset(raw.columns):
        auto_date, auto_target = "date", "value"
    else:
        auto_date, auto_target = detect_date_value_columns(raw)

    cols = list(raw.columns)
    date_options = ["(use row order - no real dates)"] + cols
    date_default_idx = date_options.index(auto_date) if auto_date in cols else 0
    target_default_idx = cols.index(auto_target) if auto_target in cols else 0

    st.markdown("**Columns to use**")
    dc1, dc2 = st.columns(2)
    date_choice = dc1.selectbox("Date / time column", date_options, index=date_default_idx)
    target_choice = dc2.selectbox("Value to forecast", cols, index=target_default_idx)

    using_synthetic_dates = date_choice == date_options[0]

    value_numeric = pd.to_numeric(
        raw[target_choice].astype(str).str.replace(",", "", regex=False), errors="coerce"
    )
    valid_ratio = value_numeric.notna().mean()
    if valid_ratio < 0.5:
        st.markdown(
            f'<div class="empty-state" style="border-color:{NEGATIVE};"><h4>"{target_choice}" isn\'t forecastable</h4>'
            f"Only {valid_ratio:.0%} of its values look numeric - it's mostly text, so there's nothing "
            "to forecast here. Pick a different column above.</div>",
            unsafe_allow_html=True,
        )
        return

    work = pd.DataFrame()
    if using_synthetic_dates:
        work["date"] = pd.date_range("2000-01-01", periods=len(raw), freq="D")
    else:
        work["date"] = raw[date_choice]
    work["value"] = value_numeric

    date_label = "Row order" if using_synthetic_dates else date_choice
    target_label = target_choice
    st.session_state["date_label"] = date_label
    st.session_state["value_label"] = target_label

    raw_missing = int(value_numeric.isna().sum())
    raw_dupes = 0 if using_synthetic_dates else int(pd.to_datetime(raw[date_choice], errors="coerce").duplicated().sum())

    try:
        clean_df = load_and_clean(work)
    except ValueError as exc:
        st.markdown(
            f'<div class="empty-state" style="border-color:{NEGATIVE};"><h4>Dataset validation failed</h4>{exc}</div>',
            unsafe_allow_html=True,
        )
        return

    os.makedirs("data/raw", exist_ok=True)
    normalized_name = f"normalized_{os.path.splitext(uploaded.name)[0]}.csv"
    normalized_path = os.path.join("data/raw", normalized_name)
    clean_df[["date", "value"]].to_csv(normalized_path, index=False)
    st.session_state["normalized_path"] = normalized_path

    st.session_state["clean_df"] = clean_df
    if using_synthetic_dates:
        freq_label = "Sequential (no real dates)"
    else:
        step = infer_step(clean_df["date"])
        freq_label = "Daily" if step.days == 1 else ("Weekly" if step.days == 7 else f"{step.days}-day")
    date_range = f"{clean_df['date'].min().year}\u2013{clean_df['date'].max().year}" if not using_synthetic_dates else "n/a"
    st.session_state["raw_stats"] = {
        "rows": len(clean_df), "date_range": date_range, "freq": freq_label,
        "missing": raw_missing, "duplicates": raw_dupes,
    }

    if using_synthetic_dates:
        st.info("No real date column selected - using sequential row order as the time axis. "
                 "Forecast dates shown later won't be real calendar dates.")

    st.success(f"Loaded **{uploaded.name}** \u2014 {len(clean_df)} valid rows.")

    c1, c2, c3, c4, c5 = st.columns(5)
    for col, label, val in zip(
        [c1, c2, c3, c4, c5],
        ["Rows", "Date range", "Frequency", "Missing", "Duplicate dates"],
        [len(clean_df), date_range, freq_label, raw_missing, raw_dupes],
    ):
        col.markdown(f'<div class="kpi-card"><div class="kpi-label">{label}</div>'
                      f'<div class="kpi-value">{val}</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("Preview data", expanded=False):
        preview = clean_df[["date", "value"]].rename(columns={"date": date_label, "value": target_label})
        st.dataframe(preview.head(20), use_container_width=True)

    import plotly.graph_objects as go
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=clean_df["date"], y=clean_df["value"], mode="lines",
                              line=dict(color=ACCENT, width=2.4, shape="spline"), name=target_label))
    _cinematic_plot(fig, height=400)
    fig.update_layout(
        xaxis=dict(showgrid=False, rangeslider=dict(visible=True), title=date_label),
        yaxis=dict(showgrid=True, gridcolor="rgba(105,126,255,.10)", title=target_label),
    )
    st.plotly_chart(fig, use_container_width=True)

    if st.button("Continue to Forecast \u2192", type="primary"):
        st.session_state["page"] = "Forecast"
        st.rerun()


def render_forecast(model_bundle, model_err, rag):
    st.markdown("# Forecast Intelligence")

    if st.session_state["clean_df"] is None:
        st.markdown(
            '<div class="empty-state"><h4>No dataset loaded</h4>'
            "Upload a dataset on the Dataset page first.</div>",
            unsafe_allow_html=True,
        )
        return

    if model_bundle is None:
        msg = model_err or "Train the model before generating a forecast."
        st.markdown(f'<div class="empty-state"><h4>Model not trained</h4>{msg}</div>', unsafe_allow_html=True)
        with st.expander("Train now"):
            render_train_widget()
        return

    clean_df = st.session_state["clean_df"]
    date_label = st.session_state["date_label"]
    value_label = st.session_state["value_label"]

    st.markdown("**Forecast horizon**")
    horizon = st.radio("Forecast horizon", options=[7, 30, 90],
                        index=[7, 30, 90].index(st.session_state["horizon"]),
                        horizontal=True, label_visibility="collapsed")
    st.session_state["horizon"] = horizon

    if st.button("Generate Forecast", type="primary"):
        try:
            forecast_df = recursive_forecast(model_bundle, clean_df, horizon)
        except Exception as exc:  # noqa: BLE001
            import traceback
            tb_text = traceback.format_exc()
            print("=== Forecast generation failed ===", file=sys.stderr)
            print(tb_text, file=sys.stderr)
            try:
                with open("forecast_error.log", "a") as f:
                    f.write("=== Forecast generation failed ===\n" + tb_text + "\n")
            except OSError:
                pass
            st.markdown(
                '<div class="empty-state" style="border-color:%s;"><h4>Forecast generation failed</h4>'
                "The forecasting model could not generate a prediction.<br><br>"
                "Please retrain the model or verify that the uploaded dataset matches the "
                "training format.</div>" % NEGATIVE,
                unsafe_allow_html=True,
            )
            st.caption("Details written to `forecast_error.log` in this folder, and to the terminal.")
            with st.expander("Show technical details"):
                st.code(tb_text)
            return
        st.session_state["forecast_df"] = forecast_df
        st.session_state["retrieved"] = []
        st.session_state["explanation"] = None
        st.session_state["explanation_err"] = None

    forecast_df = st.session_state["forecast_df"]
    if forecast_df is None:
        return

    latest = float(clean_df["value"].iloc[-1])
    forecast_value = float(forecast_df["value"].iloc[-1])
    pct_change = ((forecast_value - latest) / (abs(latest) + 1e-9)) * 100
    delta_class = "kpi-delta-pos" if pct_change >= 0 else "kpi-delta-neg"
    arrow = "\u25B2" if pct_change >= 0 else "\u25BC"

    st.markdown("<br>", unsafe_allow_html=True)
    k1, k2, k3, k4 = st.columns(4)
    k1.markdown(f'<div class="kpi-card"><div class="kpi-label">Latest {value_label}</div>'
                f'<div class="kpi-value">{latest:,.2f}</div></div>', unsafe_allow_html=True)
    k2.markdown(f'<div class="kpi-card"><div class="kpi-label">Forecasted {value_label}</div>'
                f'<div class="kpi-value">{forecast_value:,.2f}</div></div>', unsafe_allow_html=True)
    k3.markdown(f'<div class="kpi-card"><div class="kpi-label">Expected Change</div>'
                f'<div class="kpi-value {delta_class}">{arrow} {pct_change:+.1f}%</div></div>', unsafe_allow_html=True)
    k4.markdown(f'<div class="kpi-card"><div class="kpi-label">Horizon</div>'
                f'<div class="kpi-value">{horizon} steps</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    import plotly.graph_objects as go
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=clean_df["date"], y=clean_df["value"], mode="lines",
                              name="Historical", line=dict(color="rgba(224,231,255,.88)", width=2.0, shape="spline")))
    fig.add_trace(go.Scatter(x=forecast_df["date"], y=forecast_df["value"], mode="lines+markers",
                              name="Forecast", line=dict(color=ACCENT, width=2.8, dash="dash", shape="spline")))
    fig.add_vline(x=forecast_df["date"].iloc[0], line_width=1, line_dash="dot", line_color=MUTED,
                  annotation_text="Forecast Starts", annotation_font_color=MUTED)
    _cinematic_plot(fig, height=460)
    fig.update_layout(
        xaxis=dict(showgrid=False, rangeslider=dict(visible=True), title=date_label),
        yaxis=dict(showgrid=True, gridcolor="rgba(105,126,255,.10)", title=value_label),
    )
    st.plotly_chart(fig, use_container_width=True)
    render_forecast_timeline()

    st.markdown("#### Forecast Details")
    table = forecast_df.copy()
    table["Change %"] = ((table["value"] - latest) / (abs(latest) + 1e-9)) * 100
    table = table.rename(columns={"date": date_label, "value": value_label})
    table[date_label] = table[date_label].dt.strftime("%Y-%m-%d")
    st.dataframe(table.style.format({value_label: "{:.2f}", "Change %": "{:+.1f}%"}), use_container_width=True)

    st.download_button("Download Forecast CSV", table.to_csv(index=False).encode("utf-8"),
                        "forecast.csv", "text/csv")

    if rag is not None and not st.session_state["retrieved"]:
        try:
            index, vectorizer, docs, meta = rag
            query_text = describe_window(clean_df.tail(30))
            st.session_state["retrieved"] = retrieve_top_k(query_text, index, vectorizer, docs, meta, k=3)
        except Exception:  # noqa: BLE001
            st.session_state["retrieved"] = []


def render_train_widget():
    data_files = []
    if os.path.isdir("data/raw"):
        data_files = [f for f in os.listdir("data/raw") if f.endswith(".csv")]
    if not data_files:
        st.info("Put a CSV in data/raw/ first, or run `python train.py --data <file>` from a terminal.")
        return
    normalized = st.session_state.get("normalized_path")
    default_idx = data_files.index(os.path.basename(normalized)) if normalized and os.path.basename(normalized) in data_files else 0
    chosen = st.selectbox("Dataset to train on", data_files, index=default_idx)
    if st.button("Train Model"):
        log = st.empty()
        lines = []
        proc = subprocess.Popen(
            [sys.executable, "train.py", "--data", os.path.join("data/raw", chosen)],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )
        for line in proc.stdout:
            lines.append(line.rstrip())
            log.code("\n".join(lines[-12:]))
        proc.wait()
        if proc.returncode == 0:
            st.success("Training complete.")
            load_model.clear()
            load_metrics_cached.clear()
            load_rag_cached.clear()
        else:
            st.error("Training failed - see log above.")


def render_patterns(rag):
    st.markdown("# Similar Historical Patterns")
    st.markdown(
        f'<p style="color:{MUTED};">TimeRAG searches the series\' own history for periods that '
        "behaved similarly to the current forecasting situation.</p>",
        unsafe_allow_html=True,
    )

    if rag is None:
        st.markdown(
            '<div class="empty-state"><h4>Historical retrieval unavailable</h4>'
            "Train the project to create the FAISS knowledge base.</div>",
            unsafe_allow_html=True,
        )
        return
    if st.session_state["clean_df"] is None:
        st.markdown(
            '<div class="empty-state"><h4>No dataset loaded</h4>Upload a dataset on the Dataset page first.</div>',
            unsafe_allow_html=True,
        )
        return

    retrieved = st.session_state["retrieved"]
    if not retrieved:
        st.markdown(
            '<div class="empty-state"><h4>No retrieval yet</h4>'
            "Generate a forecast on the Forecast page to retrieve similar historical windows.</div>",
            unsafe_allow_html=True,
        )
        return

    st.markdown("### Similarity Ranking")
    for i, r in enumerate(retrieved, start=1):
        closeness = 100 / (1 + r["distance"])
        pct = max(0.0, min(100.0, closeness))
        st.markdown(
            f'<div class="tr-sim-card"><div class="tr-sim-head"><span class="tr-sim-rank">#{i} · {r["start_date"]} → {r["end_date"]}</span><span class="tr-sim-score">{closeness:.1f}% similarity</span></div>'
            f'<div class="tr-progress"><span style="width:{pct:.1f}%"></span></div>'
            f'<div style="color:#9aa5c2;font-size:.75rem">FAISS L2 distance: {r["distance"]:.4f} · lower means more similar</div>'
            f'<div style="color:#cbd3e8;font-size:.82rem;line-height:1.55;margin-top:10px">{r["text"]}</div></div>',
            unsafe_allow_html=True,
        )


def render_explanation(rag):
    st.markdown("# Why does the forecast look like this?")

    if st.session_state["forecast_df"] is None:
        st.markdown(
            '<div class="empty-state"><h4>No forecast yet</h4>'
            "Generate a forecast on the Forecast page first.</div>",
            unsafe_allow_html=True,
        )
        return

    if not llm_configured():
        st.markdown(
            '<div class="empty-state"><h4>AI explanation unavailable</h4>'
            "Configure an OpenAI or Gemini API key to enable explanations.</div>",
            unsafe_allow_html=True,
        )
        return

    if st.session_state["explanation"] is None and st.session_state["explanation_err"] is None:
        if st.button("Generate Explanation", type="primary"):
            clean_df = st.session_state["clean_df"]
            forecast_df = st.session_state["forecast_df"]
            latest = float(clean_df["value"].iloc[-1])
            forecast_value = float(forecast_df["value"].iloc[-1])
            pct_change = ((forecast_value - latest) / (abs(latest) + 1e-9)) * 100
            try:
                explanation, err = generate_explanation(
                    forecast_value=forecast_value,
                    target_date=forecast_df["date"].iloc[-1].strftime("%Y-%m-%d"),
                    pct_change=pct_change,
                    retrieved_docs=st.session_state["retrieved"],
                )
            except Exception as exc:  # noqa: BLE001
                explanation, err = None, str(exc)
            st.session_state["explanation"] = explanation
            st.session_state["explanation_err"] = err
            st.rerun()
        return

    if st.session_state["explanation"]:
        st.markdown("### Forecast Summary")
        st.markdown(f'<div class="tr-ai-card"><div class="tr-ai-label">✦ AI INSIGHT ENGINE</div><div style="font-size:1rem;line-height:1.75;color:#dce3f7">{st.session_state["explanation"]}</div></div>', unsafe_allow_html=True)

        if st.session_state["retrieved"]:
            st.markdown("### Historical Evidence")
            st.caption("The specific retrieved windows this explanation was grounded in:")
            for r in st.session_state["retrieved"]:
                st.markdown(f"- **{r['start_date']} \u2192 {r['end_date']}**: {r['text']}")
        st.markdown("### Grounding Pipeline")
        render_transparency()
        st.info("Forecast generated successfully. This explanation is generated from the "
                "forecast and retrieved historical windows above - the model does not separately "
                "report causal drivers or risk factors beyond what's shown here.")
    else:
        st.info("Forecast generated successfully. The AI explanation is currently unavailable.")
        if st.session_state["explanation_err"]:
            st.caption(st.session_state["explanation_err"])
        if st.button("Retry"):
            st.session_state["explanation"] = None
            st.session_state["explanation_err"] = None
            st.rerun()


def render_actual_vs_predicted(model_bundle, metrics):
    clean_df = st.session_state["clean_df"]
    if clean_df is None:
        st.info("Upload the dataset this model was trained on (Dataset page) to view this chart.")
        return

    n_train, n_test = metrics.get("n_train"), metrics.get("n_test")
    if not isinstance(n_train, int) or not isinstance(n_test, int):
        st.info("metrics.json is missing n_train/n_test - can't reconstruct the holdout split.")
        return

    try:
        featured = add_features(clean_df)
    except Exception as exc:  # noqa: BLE001
        st.info(f"Couldn't recompute features ({exc}).")
        return

    if len(featured) != n_train + n_test:
        st.warning(
            f"The currently loaded dataset has {len(featured)} rows, but this model was "
            f"trained on {n_train + n_test} rows. Upload the exact training dataset on the "
            "Dataset page to see an accurate actual-vs-predicted comparison."
        )
        return

    model = model_bundle["model"]
    slope, intercept = model_bundle["trend_slope"], model_bundle["trend_intercept"]
    test_df = featured.iloc[n_train:]
    X_test = test_df[FEATURE_COLUMNS]
    detrended_preds = model.predict(X_test)
    preds = detrended_preds + trend_at(test_df["date"], slope, intercept).to_numpy()
    actual = test_df["value"].to_numpy()

    from sklearn.metrics import mean_absolute_error
    recomputed_mae = float(mean_absolute_error(actual, preds))
    stored_mae = metrics.get("mae")
    match_ok = stored_mae is not None and abs(recomputed_mae - stored_mae) < 0.01
    st.caption(
        f"Self-check: recomputed MAE {recomputed_mae:.2f} vs. stored {stored_mae:.2f} "
        + ("\u2713 matches" if match_ok else "\u2014 does not match, treat this chart with caution")
    )

    import plotly.graph_objects as go
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=test_df["date"], y=actual, mode="lines", name="Actual",
                              line=dict(color="rgba(224,231,255,.88)", width=2.1, shape="spline")))
    fig.add_trace(go.Scatter(x=test_df["date"], y=preds, mode="lines", name="Predicted",
                              line=dict(color=ACCENT, width=2.8, dash="dash", shape="spline")))
    _cinematic_plot(fig, height=400)
    fig.update_layout(
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor="rgba(105,126,255,.10)"),
    )
    st.plotly_chart(fig, use_container_width=True)


def render_performance(model_bundle, metrics):
    st.markdown("# Model Performance")

    if metrics is None:
        st.markdown(
            '<div class="empty-state"><h4>No evaluation metrics found</h4>'
            "Train the model (python train.py --data &lt;file&gt;) to generate metrics.json.</div>",
            unsafe_allow_html=True,
        )
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f'<div class="kpi-card"><div class="kpi-label">MAE</div><div class="kpi-value">{metrics["mae"]:.2f}</div></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="kpi-card"><div class="kpi-label">RMSE</div><div class="kpi-value">{metrics["rmse"]:.2f}</div></div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="kpi-card"><div class="kpi-label">MAPE</div><div class="kpi-value">{metrics["mape"]:.2f}%</div></div>', unsafe_allow_html=True)
    r2_display = f'{metrics["r2"]:.3f}' if metrics.get("r2") is not None else "n/a"
    c4.markdown(f'<div class="kpi-card"><div class="kpi-label">R\u00b2</div><div class="kpi-value">{r2_display}</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### Actual vs. Predicted (holdout)")
    st.caption(
        "train.py saves only the aggregate metrics above, not the individual holdout "
        "predictions - so this chart is recomputed here, on demand, by re-running the "
        "actual trained model against the actual holdout rows (same split size, same "
        "trend correction, taken directly from the saved model). It only works if the "
        "dataset currently loaded on the Dataset page is the same one this model was "
        "trained on."
    )
    render_actual_vs_predicted(model_bundle, metrics)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### Model Information")
    model_type = type(model_bundle["model"]).__name__ if model_bundle else "n/a"
    n_train = metrics.get("n_train", "n/a")
    n_test = metrics.get("n_test", "n/a")
    test_fraction = (n_test / (n_train + n_test)) if isinstance(n_train, int) and isinstance(n_test, int) and (n_train + n_test) else "n/a"
    info_rows = [
        ("Model type", model_type),
        ("Training rows", n_train),
        ("Test rows", n_test),
        ("Feature count", len(FEATURE_COLUMNS)),
        ("Test fraction", f"{test_fraction:.2f}" if isinstance(test_fraction, float) else test_fraction),
        ("Training status", "Trained" if model_bundle else "Not trained"),
    ]
    st.markdown('<div class="section-card">' + "".join(
        f'<div class="status-row"><span>{k}</span><span>{v}</span></div>' for k, v in info_rows
    ) + '</div>', unsafe_allow_html=True)


def render_reports():
    st.markdown("# Reports & Export")

    forecast_df = st.session_state["forecast_df"]
    if forecast_df is None:
        st.markdown(
            '<div class="empty-state"><h4>Nothing to export yet</h4>'
            "Generate a forecast on the Forecast page first.</div>",
            unsafe_allow_html=True,
        )
        return

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Download Forecast CSV")
        export_df = forecast_df.rename(columns={
            "date": st.session_state["date_label"], "value": st.session_state["value_label"],
        })
        st.download_button("Download CSV", export_df.to_csv(index=False).encode("utf-8"),
                            "forecast.csv", "text/csv", use_container_width=True)

    with c2:
        st.markdown("#### Generate PDF Report")
        if st.button("Generate PDF Report", use_container_width=True):
            pdf_bytes, err = build_pdf_report()
            if pdf_bytes:
                st.download_button("Download PDF", pdf_bytes, "timerag_report.pdf",
                                    "application/pdf", use_container_width=True)
            else:
                st.error(f"Could not generate the PDF: {err}")


def build_pdf_report():
    try:
        from fpdf import FPDF
    except ImportError:
        return None, "fpdf2 is not installed."

    clean_df = st.session_state["clean_df"]
    forecast_df = st.session_state["forecast_df"]
    metrics = load_metrics_cached()
    stats = st.session_state["raw_stats"] or {}
    value_label = st.session_state["value_label"]

    try:
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 18)
        pdf.cell(0, 10, "TimeRAG Forecast Report", ln=True)
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(0, 6, "AI Forecast Intelligence", ln=True)
        pdf.ln(4)

        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, "Dataset", ln=True)
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(0, 6, f"Forecasting: {value_label}", ln=True)
        for label, key in [("Rows", "rows"), ("Date range", "date_range"), ("Frequency", "freq")]:
            pdf.cell(0, 6, f"{label}: {stats.get(key, 'n/a')}", ln=True)
        pdf.ln(3)

        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, f"Forecast (horizon: {st.session_state['horizon']} steps)", ln=True)
        pdf.set_font("Helvetica", "", 10)
        last = clean_df["value"].iloc[-1]
        final = forecast_df["value"].iloc[-1]
        pct = ((final - last) / (abs(last) + 1e-9)) * 100
        pdf.cell(0, 6, f"Latest {value_label}: {last:.2f}", ln=True)
        pdf.cell(0, 6, f"Forecasted {value_label}: {final:.2f}", ln=True)
        pdf.cell(0, 6, f"Expected change: {pct:+.1f}%", ln=True)
        pdf.ln(3)

        if metrics:
            pdf.set_font("Helvetica", "B", 12)
            pdf.cell(0, 8, "Model Metrics", ln=True)
            pdf.set_font("Helvetica", "", 10)
            pdf.cell(0, 6, f"MAE {metrics['mae']:.2f}  RMSE {metrics['rmse']:.2f}  "
                           f"MAPE {metrics['mape']:.2f}%  R2 {metrics.get('r2', 'n/a')}", ln=True)
            pdf.ln(3)

        if st.session_state["retrieved"]:
            pdf.set_font("Helvetica", "B", 12)
            pdf.cell(0, 8, "Retrieved Historical Patterns", ln=True)
            pdf.set_font("Helvetica", "", 10)
            for r in st.session_state["retrieved"]:
                pdf.multi_cell(0, 6, f"{r['start_date']} to {r['end_date']}: {r['text']}")
            pdf.ln(3)

        if st.session_state["explanation"]:
            pdf.set_font("Helvetica", "B", 12)
            pdf.cell(0, 8, "AI Explanation", ln=True)
            pdf.set_font("Helvetica", "", 10)
            pdf.multi_cell(0, 6, st.session_state["explanation"])
            pdf.ln(3)

        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, "Forecast Detail", ln=True)
        pdf.set_font("Helvetica", "", 9)
        for _, row in forecast_df.iterrows():
            pdf.cell(0, 5, f"{row['date'].strftime('%Y-%m-%d')}: {row['value']:.2f}", ln=True)

        return bytes(pdf.output(dest="S")), None
    except Exception as exc:  # noqa: BLE001
        return None, str(exc)


def main():
    init_state()
    model_bundle, model_err = load_model()
    metrics = load_metrics_cached()
    rag = load_rag_cached()

    render_presentation_toggle()
    if st.session_state.get("presentation_mode", False):
        st.markdown('<style>section[data-testid="stSidebar"]{display:none!important}.block-container{max-width:1600px!important}</style>', unsafe_allow_html=True)
    render_sidebar(model_bundle, model_err, metrics, rag)
    render_topbar(model_bundle, metrics, rag)

    page = st.session_state["page"]
    if page == "Overview":
        render_overview(model_bundle, rag)
    elif page == "Dataset":
        render_dataset()
    elif page == "Forecast":
        render_forecast(model_bundle, model_err, rag)
    elif page == "Historical Patterns":
        render_patterns(rag)
    elif page == "AI Explanation":
        render_explanation(rag)
    elif page == "Model Performance":
        render_performance(model_bundle, metrics)
    elif page == "Reports":
        render_reports()


if __name__ == "__main__":
    main()
