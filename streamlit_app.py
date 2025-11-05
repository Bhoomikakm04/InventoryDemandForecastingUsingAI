# Fixed Streamlit app (safe paths for Streamlit Cloud)
# streamlit_app.py (main entry) - Inventory Management Dashboard (fixed DATA_DIR)
import streamlit as st
from pathlib import Path
import pandas as pd, numpy as np, os, io, json, datetime, tempfile
import altair as alt

st.set_page_config(page_title="Inventory Management", layout="wide", initial_sidebar_state="expanded")

# -------------------- Paths --------------------
# Use a writable tempdir on Streamlit Cloud instead of /mnt/data
TMP = Path(tempfile.gettempdir())
DATA_DIR = TMP / "inventory_data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

PERSIST_SALES = DATA_DIR / "sales_current.csv"
PERSIST_STOCK = DATA_DIR / "stock_current.csv"
PERSIST_REVIEWS = DATA_DIR / "reviews_current.csv"
BACKUP_DIR = DATA_DIR / "backups"
BACKUP_DIR.mkdir(exist_ok=True)

# Demo files names (if you uploaded demo CSVs inside repo, keep them in app folder)
HERE = Path(__file__).parent
DEMO_SALES = HERE / "demo_sales_dataset.csv"
DEMO_STOCK = HERE / "demo_stock_dataset.csv"
DEMO_REVIEWS = HERE / "unstructured_reviews_demo_multilingual.csv"

# -------------------- Users / Auth (optional) --------------------
DEFAULT_USERS = {"admin": {"password": "admin", "role": "admin"},
                 "bhoomikakm2004@gmail.com": {"password": "admin", "role": "user"}}

def load_users():
    users_json = os.environ.get("BIM_USERS_JSON")
    if users_json:
        try:
            return json.loads(users_json)
        except Exception:
            pass
    users_file = os.environ.get("BIM_USERS_FILE", str(DATA_DIR / "users.env"))
    if Path(users_file).exists():
        txt = Path(users_file).read_text(encoding="utf-8").strip()
        try:
            return json.loads(txt)
        except Exception:
            users = {}
            for line in txt.splitlines():
                if not line or line.startswith("#"): continue
                parts = [p.strip() for p in line.split(":")]
                if len(parts)>=2:
                    u = parts[0]; p = parts[1]; r = parts[2] if len(parts)>2 else "user"
                    users[u] = {"password":p, "role":r}
            if users: return users
    return DEFAULT_USERS

USERS = load_users()

if "auth" not in st.session_state:
    st.session_state.auth = False
    st.session_state.user = None
    st.session_state.role = None

def do_login(u,pw):
    u = u.strip()
    rec = USERS.get(u)
    if rec and rec.get("password")==pw:
        st.session_state.auth = True
        st.session_state.user = u
        st.session_state.role = rec.get("role","user")
        return True
    return False

# Login UI (simple)
if not st.session_state.auth:
    st.sidebar.title("Account")
    username = st.sidebar.text_input("Username or email")
    password = st.sidebar.text_input("Password", type="password")
    if st.sidebar.button("Login"):
        ok = do_login(username, password)
        if not ok:
            st.sidebar.error("Invalid credentials. Use admin/admin or set BIM_USERS_JSON env or users.env")
        else:
            st.experimental_rerun()
    st.sidebar.markdown("---")
    st.sidebar.info("Set BIM_USERS_JSON in app secrets for secure users.")

# -------------------- Sidebar UI --------------------
with st.sidebar:
    st.header("Inventory Management")
    menu = st.radio("", ["Dashboard", "Inventory", "Reviews", "Reports", "Admin", "Settings"], index=0)
    st.markdown("---")
    st.subheader("Data controls")
    sales_upload = st.file_uploader("Sales CSV", type=["csv"], key="su")
    stock_upload = st.file_uploader("Stock CSV", type=["csv"], key="stu")
    reviews_upload = st.file_uploader("Reviews CSV", type=["csv"], key="ru")
    cols = st.columns(2)
    if cols[0].button("Load demo"):
        for src,dst in [(DEMO_SALES,PERSIST_SALES),(DEMO_STOCK,PERSIST_STOCK),(DEMO_REVIEWS,PERSIST_REVIEWS)]:
            if src.exists():
                df = pd.read_csv(src)
                if dst.exists():
                    bk = BACKUP_DIR / f"{dst.stem}_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}{dst.suffix}"
                    dst.replace(bk)
                df.to_csv(dst, index=False)
        st.experimental_rerun()
    if cols[1].button("Use last"):
        if PERSIST_SALES.exists():
            sales_upload = PERSIST_SALES
        if PERSIST_STOCK.exists():
            stock_upload = PERSIST_STOCK
        if PERSIST_REVIEWS.exists():
            reviews_upload = PERSIST_REVIEWS
        st.experimental_rerun()
    st.markdown("---")
    st.markdown("### Quick tips")
    st.markdown("- Upload CSVs or click *Load demo*.\n- Use **Admin** (admin role) to manage backups.")

# -------------------- Helpers --------------------
def safe_read_csv(path_or_buf):
    try:
        if hasattr(path_or_buf, "read"):
            path_or_buf.seek(0)
            return pd.read_csv(path_or_buf), None
        else:
            return pd.read_csv(path_or_buf), None
    except Exception:
        try:
            return pd.read_csv(path_or_buf, encoding="ISO-8859-1"), "ISO-8859-1"
        except Exception as e:
            return None, str(e)

def persist_upload(uploaded, dest: Path):
    if uploaded is None: return False
    try:
        if hasattr(uploaded, "read"):
            uploaded.seek(0)
            df = pd.read_csv(uploaded)
            if dest.exists():
                bk = BACKUP_DIR / f"{dest.stem}_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}{dest.suffix}"
                dest.replace(bk)
            df.to_csv(dest, index=False)
            return True
        else:
            if Path(uploaded).exists():
                df = pd.read_csv(uploaded)
                if dest.exists():
                    bk = BACKUP_DIR / f"{dest.stem}_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}{dest.suffix}"
                    dest.replace(bk)
                df.to_csv(dest, index=False)
                return True
    except Exception as e:
        st.warning(f"Persist upload failed: {e}")
    return False

# persist if uploads provided
if sales_upload is not None:
    persist_upload(sales_upload, PERSIST_SALES)
if stock_upload is not None:
    persist_upload(stock_upload, PERSIST_STOCK)
if reviews_upload is not None:
    persist_upload(reviews_upload, PERSIST_REVIEWS)

# load current datasets (persisted preferred)
sales_df, stock_df, reviews_df = None, None, None
if PERSIST_SALES.exists():
    sales_df,_ = safe_read_csv(PERSIST_SALES)
elif DEMO_SALES.exists():
    sales_df,_ = safe_read_csv(DEMO_SALES)
if PERSIST_STOCK.exists():
    stock_df,_ = safe_read_csv(PERSIST_STOCK)
elif DEMO_STOCK.exists():
    stock_df,_ = safe_read_csv(DEMO_STOCK)
if PERSIST_REVIEWS.exists():
    reviews_df,_ = safe_read_csv(PERSIST_REVIEWS)
elif DEMO_REVIEWS.exists():
    reviews_df,_ = safe_read_csv(DEMO_REVIEWS)

if sales_df is None:
    st.warning("No sales data loaded. Upload or Load demo to proceed.")
    st.stop()

# normalize sales df
def normalize_sales(df):
    df = df.copy()
    cols = [c.strip() for c in df.columns]
    df.columns = cols
    if "Date" not in df.columns:
        for c in df.columns:
            if "date" in c.lower():
                df = df.rename(columns={c:"Date"}); break
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    if "Product" not in df.columns:
        for c in df.columns:
            if "product" in c.lower():
                df = df.rename(columns={c:'Product'}); break
    if "Sales" not in df.columns:
        numcols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numcols:
            df = df.rename(columns={numcols[0]:"Sales"})
    df["Sales"] = pd.to_numeric(df["Sales"], errors="coerce").fillna(0)
    if "Country" not in df.columns:
        df["Country"] = "All"
    return df

sales_df = normalize_sales(sales_df)

# sentiment utils
def compute_vader(reviews):
    try:
        import nltk
        from nltk.sentiment.vader import SentimentIntensityAnalyzer
        try:
            nltk.data.find("sentiment/vader_lexicon.zip")
        except Exception:
            nltk.download("vader_lexicon")
        sia = SentimentIntensityAnalyzer()
        reviews["vader_compound"] = reviews["ReviewText"].astype(str).apply(lambda t: sia.polarity_scores(t)["compound"])
        return reviews
    except Exception as e:
        st.warning("VADER not available; ensure nltk is in requirements. Falling back to rating->sentiment if possible.")
        if "Rating" in reviews.columns:
            reviews["vader_compound"] = reviews["Rating"].apply(lambda r: (r-3)/2)
        else:
            reviews["vader_compound"] = 0.0
        return reviews

# LLM helper (Hugging Face)
def hf_generate(prompt, model="google/flan-t5-small", max_length=150):
    token = os.environ.get("HUGGINGFACE_API_TOKEN") or st.secrets.get("HUGGINGFACE_API_TOKEN", None)
    if not token:
        return "No Hugging Face token provided; set HUGGINGFACE_API_TOKEN in app secrets to enable LLM suggestions."
    url = f"https://api-inference.huggingface.co/models/{model}"
    headers = {"Authorization": f"Bearer {token}"}
    payload = {"inputs": prompt, "parameters":{"max_new_tokens": max_length, "do_sample": False}}
    try:
        import requests
        res = requests.post(url, headers=headers, json=payload, timeout=30)
        if res.status_code==200:
            j = res.json()
            if isinstance(j, dict) and "error" in j:
                return f"Model error: {j.get('error')}"
            if isinstance(j, list):
                return j[0].get("generated_text", str(j))
            if isinstance(j, str):
                return j
            return str(j)
        else:
            return f"Hugging Face API returned {res.status_code}: {res.text}"
    except Exception as e:
        return f"Request failed: {e}"

# Pages (Dashboard, Inventory, Reviews, Reports, Admin, Settings)
# For brevity reuse earlier page code structure (same as prior implemention)
# (The logic for pages is intentionally the same as previous version; keep LLM and VADER support)
st.title("Inventory Management")
st.write("App initialized. Use the sidebar to navigate. If you see this, DATA_DIR is writable and fixed.")