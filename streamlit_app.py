# streamlit_app.py
# Inventory Management Dashboard (full-featured, Cloud-safe)
# - Dashboard with selectable product and multiple charts (line, bar, pie, table)
# - Upload or Load demo CSVs, "Proceed" button to start processing after upload
# - Reviews + VADER sentiment (auto-download lexicon), defect breakdown
# - Stock recommendations, top-seller and place graphs
# - LLM suggestions: tries Hugging Face (with token if available), otherwise uses a free/no-key fallback (heuristic summarizer)
# - Role-based Admin access
# - Writes persistent files to tempfile.gettempdir() (safe for Streamlit Cloud)

import streamlit as st
from pathlib import Path
import pandas as pd, numpy as np, os, json, datetime, tempfile, re, collections
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

# ---------------- Demo CSV generation ----------------
def generate_demo_files():
    if not DEMO_SALES.exists():
        dates = pd.date_range(end=pd.Timestamp.today(), periods=90)
        products = ["Widget A","Widget B","Gadget X","Gadget Y","Accessory Z"]
        rows = []
        for d in dates:
            for p in products:
                qty = int(max(0, np.random.poisson(6) + (2 if "A" in p else 0)))
                price = round(20 + 10*np.random.rand(),2)
                rows.append({"Date": d.strftime("%Y-%m-%d"), "Product": p, "Category":"General", "Place": np.random.choice(["Warehouse1","Warehouse2","StoreA"]), "Country": np.random.choice(["USA","India","UK"]), "Sales": qty*price})
        pd.DataFrame(rows).to_csv(DEMO_SALES, index=False)
    if not DEMO_STOCK.exists():
        df = pd.DataFrame([{"Product":"Widget A","Stock":120,"ReorderLevel":30,"Place":"Warehouse1"},
                           {"Product":"Widget B","Stock":20,"ReorderLevel":50,"Place":"Warehouse2"},
                           {"Product":"Gadget X","Stock":5,"ReorderLevel":20,"Place":"Warehouse1"},
                           {"Product":"Gadget Y","Stock":60,"ReorderLevel":40,"Place":"StoreA"},
                           {"Product":"Accessory Z","Stock":200,"ReorderLevel":80,"Place":"Warehouse2"}])
        df.to_csv(DEMO_STOCK, index=False)
    if not DEMO_REVIEWS.exists():
        reviews = [
            {"Date": pd.Timestamp.today().strftime("%Y-%m-%d"), "Product":"Widget A","ReviewText":"Great product, works as expected.","Rating":5,"DefectLabel":""},
            {"Date": pd.Timestamp.today().strftime("%Y-%m-%d"), "Product":"Widget B","ReviewText":"Stopped working after a week.","Rating":1,"DefectLabel":"electrical"},
            {"Date": pd.Timestamp.today().strftime("%Y-%m-%d"), "Product":"Gadget X","ReviewText":"Buen producto, altamente recomendable.","Rating":5,"DefectLabel":""},
            {"Date": pd.Timestamp.today().strftime("%Y-%m-%d"), "Product":"Gadget Y","ReviewText":"La batería se calienta mucho.","Rating":2,"DefectLabel":"battery"},
            {"Date": pd.Timestamp.today().strftime("%Y-%m-%d"), "Product":"Accessory Z","ReviewText":"Decent value for money.","Rating":4,"DefectLabel":""},
        ]
        pd.DataFrame(reviews).to_csv(DEMO_REVIEWS, index=False)

generate_demo_files()

# ---------------- LLM helper ----------------
def heuristic_summarize(text, topn=5):
    import collections, re
    words = re.findall(r"[a-zA-Z]{3,}", text.lower())
    stop = set(["product","good","great","work","works","please","would","could","also","well"])
    freq = collections.Counter(w for w in words if w not in stop)
    common = [w for w,c in freq.most_common(topn)]
    suggestions = []
    if "battery" in freq:
        suggestions.append("Investigate battery-related issues and consider improving battery cooling or capacity.")
    if "electrical" in freq or "stopped" in freq:
        suggestions.append("Check electrical components and QC processes; consider extended testing on returns.")
    for w in common:
        suggestions.append(f"Check mentions of '{w}' in reviews and improve documentation/quality related to it.")
    if not suggestions:
        suggestions = ["No clear issues from heuristic; consider adding more reviews or using a hosted LLM for deeper analysis."]
    return "\\n- ".join([""] + suggestions)

def hf_generate(prompt, model="google/flan-t5-small", max_length=150):
    token = os.environ.get("HUGGINGFACE_API_TOKEN") or (st.secrets.get("HUGGINGFACE_API_TOKEN") if hasattr(st, "secrets") else None)
    url = f"https://api-inference.huggingface.co/models/{model}"
    import requests
    payload = {"inputs": prompt, "parameters":{"max_new_tokens": max_length, "do_sample": False}}
    try:
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        res = requests.post(url, headers=headers, json=payload, timeout=25)
        if res.status_code==200:
            j = res.json()
            if isinstance(j, list):
                return j[0].get("generated_text", str(j))
            if isinstance(j, dict) and "error" in j:
                return f"Model error: {j.get('error')}"
            return str(j)
        else:
            return f"HF returned {res.status_code}; using local fallback.\\n\\n" + heuristic_summarize(prompt)
    except Exception as e:
        return "HF request failed; using local fallback.\\n\\n" + heuristic_summarize(prompt)

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
        if "Rating" in reviews.columns:
            reviews["vader_compound"] = reviews["Rating"].apply(lambda r: (r-3)/2)
        else:
            reviews["vader_compound"] = 0.0
        st.warning("VADER not available; used rating-based fallback.")
        return reviews

# ---------------- Sidebar & Uploads ----------------
with st.sidebar:
    st.header("Inventory Management")
    st.markdown("Upload your CSVs (Sales, Stock, Reviews) or click **Load demo**. After uploading, click **Proceed** to process.")
    sales_upload = st.file_uploader("Sales CSV", type=["csv"], key="su")
    stock_upload = st.file_uploader("Stock CSV", type=["csv"], key="stu")
    reviews_upload = st.file_uploader("Reviews CSV", type=["csv"], key="ru")
    st.markdown("---")
    if st.button("Load demo"):
        for src,dst in [(DEMO_SALES,PERSIST_SALES),(DEMO_STOCK,PERSIST_STOCK),(DEMO_REVIEWS,PERSIST_REVIEWS)]:
            try:
                df = pd.read_csv(src)
                if dst.exists():
                    bk = BACKUP_DIR / f"{dst.stem}_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}{dst.suffix}"
                    dst.replace(bk)
                df.to_csv(dst,index=False)
            except Exception as e:
                st.warning(f"Demo copy failed: {e}")
        st.success("Demo datasets copied to app data. Click Proceed to start processing.")
    st.markdown("---")
    proceed = st.button("Proceed")  # user must click after upload

# Persist uploads when provided and proceed clicked
if proceed:
    if sales_upload is not None:
        persist_upload(sales_upload, PERSIST_SALES)
    if stock_upload is not None:
        persist_upload(stock_upload, PERSIST_STOCK)
    if reviews_upload is not None:
        persist_upload(reviews_upload, PERSIST_REVIEWS)

# Load datasets
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
    st.warning("No sales data loaded. Upload or Load demo and click Proceed to continue.")
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
                df = df.rename(columns={c:'Product'}); break
    if "Sales" not in df.columns:
        numcols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numcols:
            df = df.rename(columns={numcols[0]:"Sales"})
    df["Sales"] = pd.to_numeric(df["Sales"], errors="coerce").fillna(0)
    if "Country" not in df.columns:
        df["Country"] = "All"
    if "Place" not in df.columns:
        df["Place"] = df.get("Place", "Unknown")
    return df

sales_df = normalize_sales(sales_df)

# Main menu
menu = st.sidebar.radio("Menu", ["Dashboard","Inventory","Reviews","Reports","Admin","Settings"])
products = ["All"] + sorted(sales_df['Product'].dropna().unique().tolist())
selected_product = st.sidebar.selectbox("Product (for charts)", options=products, index=0)

if menu=="Dashboard":
    st.title("Inventory Management — Dashboard")
    total_sales = sales_df['Sales'].sum()
    top_item = sales_df.groupby('Product')['Sales'].sum().idxmax() if 'Product' in sales_df.columns else "N/A"
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Sales", f"${total_sales:,.0f}")
    col2.metric("Top Item", top_item)
    col3.metric("Data rows", len(sales_df))
    col4.metric("Products", sales_df['Product'].nunique() if 'Product' in sales_df.columns else 0)
    st.markdown("---")
    if selected_product!="All":
        sdf = sales_df[sales_df['Product']==selected_product]
    else:
        sdf = sales_df
    daily = sdf.groupby('Date')['Sales'].sum().reset_index()
    st.subheader("Sales trend")
    st.altair_chart(alt.Chart(daily).mark_line(point=True).encode(x='Date:T', y='Sales:Q'), use_container_width=True)
    st.subheader("Top sellers (by Sales)")
    top_s = sdf.groupby('Product')['Sales'].sum().reset_index().sort_values('Sales', ascending=False).head(10)
    st.bar_chart(top_s.set_index('Product')['Sales'])
    st.subheader("Sales by Country (pie)")
    byc = sdf.groupby('Country')['Sales'].sum().reset_index()
    if not byc.empty:
        chart = alt.Chart(byc).mark_arc().encode(theta='Sales:Q', color='Country:N', tooltip=['Country','Sales'])
        st.altair_chart(chart, use_container_width=True)
    st.subheader("Recent sales rows")
    st.dataframe(sdf.sort_values('Date', ascending=False).head(50))
    st.subheader("Sales by Place")
    byp = sdf.groupby('Place')['Sales'].sum().reset_index().sort_values('Sales', ascending=False)
    if not byp.empty:
        st.bar_chart(byp.set_index('Place')['Sales'])

elif menu=="Inventory":
    st.title("Inventory — Stock & Recommendations")
    if stock_df is None:
        st.info("No stock data. Upload or Load demo and click Proceed.")
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
        avg = sales_df.groupby('Product')['Sales'].apply(lambda x: x.tail(30).mean() if len(x)>=3 else x.mean()).reset_index().rename(columns={'Sales':'AvgDailySales'})
        merged = pd.merge(sd, avg, on='Product', how='left')
        merged['DaysLeft'] = merged['Stock'] / merged['AvgDailySales'].replace(0, np.nan)
        merged['DaysLeft'] = merged['DaysLeft'].round(1)
        st.subheader("Stock recommendations")
        st.dataframe(merged.sort_values('DaysLeft').fillna("-").reset_index(drop=True))
        st.download_button("Download stock recommendations", data=merged.to_csv(index=False).encode('utf-8'), file_name="stock_recommendations.csv")

elif menu=="Reviews":
    st.title("Reviews & Sentiment")
    if reviews_df is None:
        st.info("No reviews. Upload reviews CSV or Load demo and click Proceed.")
    else:
        rv = reviews_df.copy()
        rv.columns = [c.strip() for c in rv.columns]
        if 'ReviewText' not in rv.columns:
            for c in rv.columns:
                if 'review' in c.lower(): rv = rv.rename(columns={c:'ReviewText'}); break
        if 'Date' in rv.columns:
            rv['Date'] = pd.to_datetime(rv['Date'], errors='coerce')
        rv = compute_vader(rv)
        st.subheader("Recent reviews sample")
        st.dataframe(rv[['Date','Product','ReviewText','Rating','vader_compound']].head(200))
        st.subheader("Average sentiment by product")
        sbar = rv.groupby('Product')['vader_compound'].mean().reset_index()
        st.altair_chart(alt.Chart(sbar).mark_bar().encode(x='Product:N', y='vader_compound:Q', tooltip=['Product','vader_compound']), use_container_width=True)
        st.subheader("Defect breakdown per product")
        if 'DefectLabel' in rv.columns and rv['DefectLabel'].astype(str).str.strip().any():
            df_defp = rv[rv['DefectLabel'].astype(str)!=""].groupby(['Product','DefectLabel']).size().reset_index(name='Count')
            st.altair_chart(alt.Chart(df_defp).mark_bar().encode(x='Product:N', y='Count:Q', color='DefectLabel:N', tooltip=['Product','DefectLabel','Count']), use_container_width=True)
        else:
            st.info("No defect labels found. Add 'DefectLabel' column to reviews CSV to enable this chart.")
        st.subheader("Automated review suggestions (free/no-key possible)")
        prod_for = st.text_input("Product name for suggestions", value=selected_product if selected_product!="All" else "")
        prompt = st.text_area("Prompt", value="Suggest 5 actionable improvements for this product based on reviews.", height=80)
        if st.button("Get suggestions"):
            if not prod_for:
                st.error("Enter a product name.")
            else:
                subset = rv[rv['Product'].astype(str).str.lower()==prod_for.lower()]
                if subset.shape[0]==0:
                    subset = rv.copy().head(200)
                doc = " ".join(subset['ReviewText'].astype(str).tolist()[:400])
                full_prompt = f"Product: {prod_for}\\nReviews: {doc}\\nTask: {prompt}\\nAnswer concisely."
                out = hf_generate(full_prompt)
                st.text_area("Suggestions", value=out, height=300)

elif menu=="Reports":
    st.title("Reports")
    min_date = sales_df['Date'].min(); max_date = sales_df['Date'].max()
    dr = st.date_input("Date range", value=(min_date, max_date), min_value=min_date, max_value=max_date)
    prod = st.selectbox("Product", options=["All"] + sorted(sales_df['Product'].unique().tolist()))
    sdf = sales_df.copy()
    sdf = sdf[(sdf['Date']>=pd.to_datetime(dr[0])) & (sdf['Date']<=pd.to_datetime(dr[1]))]
    if prod!="All": sdf = sdf[sdf['Product']==prod]
    st.download_button("Download filtered sales CSV", data=sdf.to_csv(index=False).encode('utf-8'), file_name="sales_report.csv")
    st.dataframe(sdf.head(200))

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
    st.markdown("Set HUGGINGFACE_API_TOKEN in app secrets to enable authenticated Hugging Face Inference (recommended).")
    st.markdown("If no token is provided, the app will attempt anonymous HF inference (may be rate-limited) and otherwise use a local heuristic summarizer.")
    st.markdown("Recommended free models:")
    st.write("- google/flan-t5-small (instruction-style summarization)\n- distilbert-base-uncased-finetuned-sst-2-english (sentiment classification)\n- sentence-transformers/all-MiniLM-L6-v2 (embeddings)")

st.caption("Built for Streamlit Cloud: writes to a writable tempdir.")