# Full Streamlit app (Cloud-safe) — Inventory Management (fixed, no experimental_rerun)
# Features:
# - Dashboard, Inventory, Reviews, Reports, Admin, Settings
# - Demo CSV generation if not present
# - Uses tempfile.gettempdir() for writable storage on hosted platforms (Streamlit Cloud)
# - VADER sentiment (auto-download lexicon) with fallback to rating mapping
# - Hugging Face LLM suggestions (optional via HUGGINGFACE_API_TOKEN secret)
# - Role-based Admin access (admin/admin default)
# - Defect breakdown chart and review-driven recommendations
# - No st.experimental_rerun calls (safe on Streamlit Cloud)

import streamlit as st
from pathlib import Path
import pandas as pd, numpy as np, os, io, json, datetime, tempfile, textwrap
import altair as alt

st.set_page_config(page_title="Inventory Management", layout="wide", initial_sidebar_state="expanded")

# ---------------- Paths ----------------
TMP = Path(tempfile.gettempdir())
DATA_DIR = TMP / "inventory_data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

PERSIST_SALES = DATA_DIR / "sales_current.csv"
PERSIST_STOCK = DATA_DIR / "stock_current.csv"
PERSIST_REVIEWS = DATA_DIR / "reviews_current.csv"
BACKUP_DIR = DATA_DIR / "backups"
BACKUP_DIR.mkdir(exist_ok=True)

HERE = Path(__file__).parent

DEMO_SALES = HERE / "demo_sales_dataset.csv"
DEMO_STOCK = HERE / "demo_stock_dataset.csv"
DEMO_REVIEWS = HERE / "unstructured_reviews_demo_multilingual.csv"

# ---------------- Users ----------------
DEFAULT_USERS = {"admin": {"password": "admin", "role": "admin"},
                 "bhoomikakm2004@gmail.com": {"password": "admin", "role": "user"}}

def load_users():
    users_json = os.environ.get("BIM_USERS_JSON")
    if users_json:
        try:
            return json.loads(users_json)
        except Exception:
            pass
    users_file = os.environ.get("BIM_USERS_FILE", str(DATA_DIR/"users.env"))
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
    u = (u or "").strip()
    rec = USERS.get(u)
    if rec and rec.get("password")==pw:
        st.session_state.auth = True
        st.session_state.user = u
        st.session_state.role = rec.get("role","user")
        return True
    return False

# ---------------- Utilities ----------------
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
        else:
            df = pd.read_csv(uploaded)
        if dest.exists():
            bk = BACKUP_DIR / f"{dest.stem}_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}{dest.suffix}"
            dest.replace(bk)
        df.to_csv(dest, index=False)
        return True
    except Exception as e:
        st.warning(f"Persist upload failed: {e}")
        return False

# ---------------- Demo CSV generation (if not present) ----------------
def generate_demo_files():
    # Sales demo
    if not DEMO_SALES.exists():
        dates = pd.date_range(end=pd.Timestamp.today(), periods=180)
        products = ["Widget A","Widget B","Gadget X","Gadget Y","Accessory Z"]
        rows = []
        for d in dates:
            for p in products:
                qty = int(max(0, np.random.poisson(5) + (2 if "A" in p else 0)))
                price = float(20 + 10*np.random.rand())
                rows.append({"Date": d.strftime("%Y-%m-%d"), "Product": p, "Category": "General", "Country": np.random.choice(["USA","India","UK"]), "Sales": qty*price})
        pd.DataFrame(rows).to_csv(DEMO_SALES, index=False)
    # Stock demo
    if not DEMO_STOCK.exists():
        df = pd.DataFrame([{"Product":"Widget A","Stock":120,"ReorderLevel":30},
                           {"Product":"Widget B","Stock":20,"ReorderLevel":50},
                           {"Product":"Gadget X","Stock":5,"ReorderLevel":20},
                           {"Product":"Gadget Y","Stock":60,"ReorderLevel":40},
                           {"Product":"Accessory Z","Stock":200,"ReorderLevel":80}])
        df.to_csv(DEMO_STOCK, index=False)
    # Reviews demo (multilingual + defect labels)
    if not DEMO_REVIEWS.exists():
        reviews = [
            {"Product":"Widget A","ReviewText":"Great product, works as expected.","Rating":5,"DefectLabel":""},
            {"Product":"Widget B","ReviewText":"Stopped working after a week.","Rating":1,"DefectLabel":"electrical"},
            {"Product":"Gadget X","ReviewText":"Buen producto, altamente recomendable.","Rating":5,"DefectLabel":""},
            {"Product":"Gadget Y","ReviewText":"La batería se calienta mucho.","Rating":2,"DefectLabel":"battery"},
            {"Product":"Accessory Z","ReviewText":"Decent value for money.","Rating":4,"DefectLabel":""},
            {"Product":"Widget A","ReviewText":"Entrega rápida, buena calidad.","Rating":5,"DefectLabel":""},
        ]
        pd.DataFrame(reviews).to_csv(DEMO_REVIEWS, index=False)
generate_demo_files()

# ---------------- LLM helper ----------------
def hf_generate(prompt, model="google/flan-t5-small", max_length=150):
    token = os.environ.get("HUGGINGFACE_API_TOKEN") or (st.secrets.get("HUGGINGFACE_API_TOKEN") if hasattr(st, "secrets") else None)
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

# ---------------- Sentiment (VADER) ----------------
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
        # fallback
        if "Rating" in reviews.columns:
            reviews["vader_compound"] = reviews["Rating"].apply(lambda r: (r-3)/2)
        else:
            reviews["vader_compound"] = 0.0
        st.warning("VADER not available; used rating-based fallback.")
        return reviews

# ---------------- Sidebar & Uploads ----------------
if not st.session_state.auth:
    st.sidebar.title("Account")
    username = st.sidebar.text_input("Username or email")
    password = st.sidebar.text_input("Password", type="password")
    if st.sidebar.button("Login"):
        ok = do_login(username, password)
        if not ok:
            st.sidebar.error("Invalid credentials. Use admin/admin")
        else:
            st.sidebar.success("Login successful. Click any menu item to refresh view.")
    st.sidebar.markdown("---")
    st.sidebar.info("Set BIM_USERS_JSON in app secrets or create users.env in data folder for secure users.")

with st.sidebar:
    st.header("Inventory Management")
    menu = st.radio("", ["Dashboard","Inventory","Reviews","Reports","Admin","Settings"], index=0)
    st.markdown("---")
    st.subheader("Data controls")
    sales_upload = st.file_uploader("Sales CSV", type=["csv"], key="su")
    stock_upload = st.file_uploader("Stock CSV", type=["csv"], key="stu")
    reviews_upload = st.file_uploader("Reviews CSV", type=["csv"], key="ru")
    c1, c2 = st.columns([1,1])
    if c1.button("Load demo"):
        # copy demo files to persist location
        copied = False
        for src,dst in [(DEMO_SALES,PERSIST_SALES),(DEMO_STOCK,PERSIST_STOCK),(DEMO_REVIEWS,PERSIST_REVIEWS)]:
            try:
                df = pd.read_csv(src)
                if dst.exists():
                    bk = BACKUP_DIR / f"{dst.stem}_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}{dst.suffix}"
                    dst.replace(bk)
                df.to_csv(dst,index=False)
                copied = True
            except Exception as e:
                st.warning(f"Demo copy failed for {src.name}: {e}")
        if copied:
            st.success("Demo datasets copied to app data. Click another menu to refresh view.")
    if c2.button("Use last"):
        # do nothing special; persisted files will be used automatically
        st.info("Using last persisted files (if they exist).")

# persist uploads
if sales_upload is not None:
    persist_upload(sales_upload, PERSIST_SALES)
if stock_upload is not None:
    persist_upload(stock_upload, PERSIST_STOCK)
if reviews_upload is not None:
    persist_upload(reviews_upload, PERSIST_REVIEWS)

# load data
sales_df = None; stock_df = None; reviews_df = None
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

# normalize sales
def normalize_sales(df):
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    if "Date" not in df.columns:
        for c in df.columns:
            if "date" in c.lower():
                df = df.rename(columns={c:"Date"}); break
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    if "Product" not in df.columns:
        for c in df.columns:
            if "product" in c.lower():
                df = df.rename(columns={c:"Product"}); break
    if "Sales" not in df.columns:
        numcols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numcols:
            df = df.rename(columns={numcols[0]:"Sales"})
    df["Sales"] = pd.to_numeric(df["Sales"], errors="coerce").fillna(0)
    if "Country" not in df.columns:
        df["Country"] = "All"
    return df

sales_df = normalize_sales(sales_df)

# ---------------- Pages ----------------
if menu=="Dashboard":
    st.title("Inventory Management — Dashboard")
    # KPIs
    total_sales = sales_df['Sales'].sum()
    top_item = sales_df.groupby('Product')['Sales'].sum().idxmax() if 'Product' in sales_df.columns else "N/A"
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Sales", f"${total_sales:,.0f}")
    col2.metric("Top Item", top_item)
    col3.metric("Data rows", len(sales_df))
    col4.metric("Products", sales_df['Product'].nunique() if 'Product' in sales_df.columns else 0)
    st.markdown("---")
    # Sales trend
    daily = sales_df.groupby('Date')['Sales'].sum().reset_index()
    st.subheader("Sales trend")
    st.altair_chart(alt.Chart(daily).mark_line(point=True).encode(x='Date:T', y='Sales:Q'), use_container_width=True)
    # Defect breakdown (from reviews)
    if reviews_df is not None:
        rv = reviews_df.copy()
        if 'DefectLabel' in rv.columns and rv['DefectLabel'].notna().sum()>0:
            df_def = rv[rv['DefectLabel'].astype(str)!=""].groupby('DefectLabel').size().reset_index(name='Count')
            st.subheader("Defect breakdown (from reviews)")
            st.altair_chart(alt.Chart(df_def).mark_bar().encode(x='DefectLabel:N', y='Count:Q'), use_container_width=True)
        else:
            st.info("No defect labels in reviews (add 'DefectLabel' column).")
elif menu=="Inventory":
    st.title("Inventory — Stock & Recommendations")
    if stock_df is None:
        st.info("No stock data. Upload a Stock CSV or Load demo.")
    else:
        sd = stock_df.copy()
        sd.columns = [c.strip() for c in sd.columns]
        if 'Product' not in sd.columns:
            for c in sd.columns:
                if 'product' in c.lower(): sd = sd.rename(columns={c:'Product'}); break
        if 'Stock' not in sd.columns:
            for c in sd.columns:
                if 'stock' in c.lower(): sd = sd.rename(columns={c:'Stock'}); break
        sd['Stock'] = pd.to_numeric(sd['Stock'], errors='coerce').fillna(0)
        # compute avg daily sales per product (last 30 days)
        avg = sales_df.groupby('Product')['Sales'].apply(lambda x: x.tail(30).mean() if len(x)>=3 else x.mean()).reset_index().rename(columns={'Sales':'AvgDailySales'})
        merged = pd.merge(sd, avg, on='Product', how='left')
        merged['DaysLeft'] = merged['Stock'] / merged['AvgDailySales'].replace(0, np.nan)
        st.dataframe(merged.sort_values('DaysLeft').fillna("-").head(100))
        st.download_button("Download stock recommendations", data=merged.to_csv(index=False).encode('utf-8'), file_name="stock_recommendations.csv")
elif menu=="Reviews":
    st.title("Reviews & Sentiment")
    if reviews_df is None:
        st.info("No reviews. Upload reviews CSV or Load demo.")
    else:
        rv = reviews_df.copy()
        rv.columns = [c.strip() for c in rv.columns]
        if 'ReviewText' not in rv.columns:
            for c in rv.columns:
                if 'review' in c.lower(): rv = rv.rename(columns={c:'ReviewText'}); break
        if 'Product' not in rv.columns:
            rv['Product'] = rv.get('Product', 'Unknown')
        rv = compute_vader(rv)
        st.subheader("Recent reviews sample")
        st.dataframe(rv[['Product','ReviewText','Rating','vader_compound']].head(100))
        st.subheader("Sentiment over time (per product)")
        if 'Date' in rv.columns:
            rv['Date'] = pd.to_datetime(rv['Date'], errors='coerce')
        rv_ts = rv.groupby(['Product']).agg({'vader_compound':'mean'}).reset_index()
        st.altair_chart(alt.Chart(rv_ts).mark_bar().encode(x='Product:N', y='vader_compound:Q'), use_container_width=True)
        st.subheader("Defect breakdown per product")
        if 'DefectLabel' in rv.columns and rv['DefectLabel'].notna().sum()>0:
            df_defp = rv[rv['DefectLabel'].astype(str)!=""].groupby(['Product','DefectLabel']).size().reset_index(name='Count')
            st.altair_chart(alt.Chart(df_defp).mark_bar().encode(x='Product:N', y='Count:Q', color='DefectLabel:N'), use_container_width=True)
        # LLM suggestions
        st.subheader("Review-driven suggestions (LLM)")
        prod_for = st.text_input("Product name for suggestions")
        prompt = st.text_area("Prompt", value="Suggest 5 actionable improvements for this product based on reviews.")
        if st.button("Get suggestions"):
            if not prod_for:
                st.error("Enter a product name.")
            else:
                subset = rv[rv['Product'].astype(str).str.lower()==prod_for.lower()]
                if subset.shape[0]==0:
                    subset = rv.head(50)
                doc = " ".join(subset['ReviewText'].astype(str).tolist()[:200])
                full_prompt = f"Product: {prod_for}\nReviews: {doc}\nTask: {prompt}\nAnswer concisely."
                out = hf_generate(full_prompt)
                st.text_area("LLM output", value=out, height=240)
elif menu=="Reports":
    st.title("Reports")
    min_date = sales_df['Date'].min(); max_date = sales_df['Date'].max()
    dr = st.date_input("Date range", value=(min_date, max_date), min_value=min_date, max_value=max_date)
    prod = st.selectbox("Product", options=["All"] + sorted(sales_df['Product'].unique().tolist()))
    sdf = sales_df.copy()
    sdf = sdf[(sdf['Date']>=pd.to_datetime(dr[0])) & (sdf['Date']<=pd.to_datetime(dr[1]))]
    if prod!="All": sdf = sdf[sdf['Product']==prod]
    st.download_button("Download filtered sales CSV", data=sdf.to_csv(index=False).encode('utf-8'), file_name="sales_report.csv")
elif menu=="Admin":
    st.title("Admin")
    if st.session_state.role!="admin":
        st.warning("Admin role required.")
    else:
        if PERSIST_SALES.exists(): st.download_button("Download current sales", data=PERSIST_SALES.read_bytes(), file_name=PERSIST_SALES.name)
        if PERSIST_STOCK.exists(): st.download_button("Download current stock", data=PERSIST_STOCK.read_bytes(), file_name=PERSIST_STOCK.name)
        if PERSIST_REVIEWS.exists(): st.download_button("Download current reviews", data=PERSIST_REVIEWS.read_bytes(), file_name=PERSIST_REVIEWS.name)
        if st.button("Create backups now"):
            for p in [PERSIST_SALES,PERSIST_STOCK,PERSIST_REVIEWS]:
                if p.exists():
                    bk = BACKUP_DIR / f"{p.stem}_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}{p.suffix}"
                    p.replace(bk)
            st.success("Backups created.")
        if st.button("Clear current datasets (keep backups)"):
            for p in [PERSIST_SALES,PERSIST_STOCK,PERSIST_REVIEWS]:
                if p.exists(): p.unlink()
            st.success("Cleared current datasets.")
elif menu=="Settings":
    st.title("Settings")
    st.markdown("Set HUGGINGFACE_API_TOKEN in app secrets to enable LLM suggestions. Recommended free models:\n\n- google/flan-t5-small (instruction)\n- distilbert-base-uncased-finetuned-sst-2-english (sentiment)\n- sentence-transformers/all-MiniLM-L6-v2 (embeddings)")
    st.markdown("NOTE: To enable VADER, ensure `nltk` is in `requirements.txt` and the app can download the lexicon at first run.")

st.caption("Built for Streamlit Cloud (writes to a writable tempdir).")