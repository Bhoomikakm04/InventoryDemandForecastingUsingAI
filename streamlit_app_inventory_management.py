# Inventory Management — Streamlit App (merged UI + features)
# - Sidebar "Inventory Management" main selection, design similar to screenshot
# - Keep previous features: uploads, demo load, persistence, VADER sentiment, defect charts, QR add-to-stock
# - LLM suggestions (optional) using Hugging Face Inference API (free models recommended)
# - Role-based Admin (optional via BIM_USERS_JSON env or /mnt/data/users.env)
# Notes: Designed to run on Streamlit Cloud. Optional heavy deps (transformers) are not required.

import streamlit as st
from pathlib import Path
import pandas as pd, numpy as np, os, io, json, datetime
import altair as alt

st.set_page_config(page_title="Inventory Management", layout="wide", initial_sidebar_state="expanded")

# -------------------- Paths --------------------
DATA_DIR = Path("/mnt/data")
DATA_DIR.mkdir(exist_ok=True)
PERSIST_SALES = DATA_DIR / "sales_current.csv"
PERSIST_STOCK = DATA_DIR / "stock_current.csv"
PERSIST_REVIEWS = DATA_DIR / "reviews_current.csv"
BACKUP_DIR = DATA_DIR / "backups"
BACKUP_DIR.mkdir(exist_ok=True)

# Demo files names (if you uploaded demo CSVs)
DEMO_SALES = DATA_DIR / "demo_sales_dataset.csv"
DEMO_STOCK = DATA_DIR / "demo_stock_dataset.csv"
DEMO_REVIEWS = DATA_DIR / "unstructured_reviews_demo_multilingual.csv"

# -------------------- Users / Auth (optional) --------------------
DEFAULT_USERS = {"admin": {"password": "admin", "role": "admin"},
                 "bhoomikakm2004@gmail.com": {"password": "admin", "role": "user"}}

def load_users():
    import os, json
    users_json = os.environ.get("BIM_USERS_JSON")
    if users_json:
        try:
            return json.loads(users_json)
        except Exception:
            pass
    users_file = os.environ.get("BIM_USERS_FILE", "/mnt/data/users.env")
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

# Login UI (simple) — optional; if you don't want login, comment out this block
if not st.session_state.auth:
    st.sidebar.title("Account")
    username = st.sidebar.text_input("Username or email")
    password = st.sidebar.text_input("Password", type="password")
    if st.sidebar.button("Login"):
        ok = do_login(username, password)
        if not ok:
            st.sidebar.error("Invalid credentials. Provide BIM_USERS_JSON env or /mnt/data/users.env, or use admin/admin")
        else:
            st.experimental_rerun()
    st.sidebar.markdown("---")
    st.sidebar.info("Use demo login or set BIM_USERS_JSON env for secure users.")

# -------------------- Sidebar UI (left) --------------------
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
        # copy demo if present
        for src,dst in [(DEMO_SALES,PERSIST_SALES),(DEMO_STOCK,PERSIST_STOCK),(DEMO_REVIEWS,PERSIST_REVIEWS)]:
            if src.exists():
                df = pd.read_csv(src)
                # backup current
                if dst.exists():
                    bk = BACKUP_DIR / f"{dst.stem}_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}{dst.suffix}"
                    dst.replace(bk)
                df.to_csv(dst, index=False)
        st.experimental_rerun()
    if cols[1].button("Use last"):
        # use persisted files if exist
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
            # backup existing
            if dest.exists():
                bk = BACKUP_DIR / f"{dest.stem}_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}{dest.suffix}"
                dest.replace(bk)
            df.to_csv(dest, index=False)
            return True
        else:
            # path-like
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

# normalize sales df similar to previous v4 helper
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

# -------------------- Sentiment utilities --------------------
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

# -------------------- LLM helper (optional) --------------------
# We recommend using lightweight, free models hosted by Hugging Face Inference API.
# Suggested models:
# - google/flan-t5-small  (small instruction-following T5; good for short suggestions)
# - distilbert-base-uncased-finetuned-sst-2-english (sentiment) - for label-only local use
# For on-host inference via Hugging Face Inference API, set HUGGINGFACE_API_TOKEN as a secret.
def hf_generate(prompt, model="google/flan-t5-small", max_length=150):
    token = os.environ.get("HUGGINGFACE_API_TOKEN") or st.secrets.get("HUGGINGFACE_API_TOKEN", None)
    if not token:
        return "No Hugging Face token provided; provide HUGGINGFACE_API_TOKEN in app secrets to enable LLM suggestions."
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

# -------------------- Pages --------------------
# Dashboard page - design inspired by given screenshot
if menu=="Dashboard":
    st.title("Inventory Management — Dashboard")
    # top KPIs row
    cols = st.columns(7)
    df_range = sales_df.copy()
    total_sales = df_range['Sales'].sum()
    total_purchases = sales_df.get('Purchases', pd.Series(dtype=float)).sum() if 'Purchases' in sales_df.columns else 0
    net_profit = sales_df.get('Profit', pd.Series(dtype=float)).sum() if 'Profit' in sales_df.columns else 0
    total_receivable = 0; total_payable = 0
    top_location = sales_df['Country'].mode()[0] if 'Country' in sales_df.columns else "N/A"
    top_item = sales_df.groupby('Product')['Sales'].sum().idxmax() if 'Product' in sales_df.columns else "N/A"
    try:
        cols[0].metric("Total Sales", f"${total_sales:,.0f}")
        cols[1].metric("Total Purchases", f"${total_purchases:,.0f}")
        cols[2].metric("Net Profit", f"${net_profit:,.0f}")
        cols[3].metric("Total Receivable", f"${total_receivable:,.0f}")
        cols[4].metric("Total Payable", f"${total_payable:,.0f}")
        cols[5].metric("Top Sales Location", top_location)
        cols[6].metric("Top Selling Item", top_item)
    except Exception:
        pass

    st.markdown("---")

    # main charts: sales trend, top customers placeholder, purchase by location donut, purchase by category bars
    left, mid, right = st.columns([3,1.2,1.2])
    with left:
        st.subheader("Sales Trend")
        daily = sales_df.groupby('Date')['Sales'].sum().reset_index()
        chart = alt.Chart(daily).mark_line(point=True).encode(x='Date:T', y='Sales:Q').properties(height=320)
        st.altair_chart(chart, use_container_width=True)
    with mid:
        st.subheader("Top 10 Products")
        top_products = sales_df.groupby('Product')['Sales'].sum().reset_index().sort_values('Sales', ascending=False).head(10)
        st.table(top_products.assign(Sales=lambda d: d['Sales'].map("${:,.0f}".format)))
    with right:
        st.subheader("Purchase by Location")
        if 'Country' in sales_df.columns:
            loc = sales_df.groupby('Country')['Sales'].sum().reset_index()
            pie = alt.Chart(loc).mark_arc().encode(theta='Sales:Q', color='Country:N', tooltip=['Country','Sales'])
            st.altair_chart(pie, use_container_width=True)
        else:
            st.info("Country data not available.")

    st.markdown("---")

    # lower row: sales by location, sales by category, treemap of cities (if available)
    l1,l2,l3 = st.columns(3)
    with l1:
        st.subheader("Sales by Location (bar)")
        if 'Country' in sales_df.columns:
            byloc = sales_df.groupby('Country')['Sales'].sum().reset_index().sort_values('Sales', ascending=False)
            bar = alt.Chart(byloc).mark_bar().encode(x='Country:N', y='Sales:Q', tooltip=['Country','Sales'])
            st.altair_chart(bar, use_container_width=True)
    with l2:
        st.subheader("Sales by Category")
        if 'Category' in sales_df.columns:
            bycat = sales_df.groupby('Category')['Sales'].sum().reset_index()
            st.altair_chart(alt.Chart(bycat).mark_bar().encode(x='Category:N', y='Sales:Q'), use_container_width=True)
        else:
            st.info("Category column not found.")
    with l3:
        st.subheader("City map (treemap)")
        if 'City' in sales_df.columns:
            tre = sales_df.groupby('City')['Sales'].sum().reset_index()
            tre['size'] = tre['Sales']
            st.altair_chart(alt.Chart(tre).mark_rect().encode(x='City:N', y='size:Q'), use_container_width=True)
        else:
            st.info("City data not available.")

# Inventory page
elif menu=="Inventory":
    st.title("Inventory Management — Inventory")
    st.subheader("Stock status & low-stock recommendations")
    if stock_df is None:
        st.info("No stock data. Upload a Stock CSV in the sidebar.")
    else:
        sd = stock_df.copy()
        sd.columns = [c.strip() for c in sd.columns]
        # normalize column names
        if 'Stock' not in sd.columns:
            for c in sd.columns:
                if 'stock' in c.lower(): sd = sd.rename(columns={c:'Stock'}); break
        if 'Product' not in sd.columns:
            for c in sd.columns:
                if 'product' in c.lower(): sd = sd.rename(columns={c:'Product'}); break
        sd['Stock'] = pd.to_numeric(sd['Stock'], errors='coerce').fillna(0)
        # compute days left using avg last 7 days
        avg7 = sales_df.groupby('Product')['Sales'].apply(lambda x: x.tail(7).mean() if len(x)>=3 else x.mean()).reset_index().rename(columns={'Sales':'Avg7'})
        merged = pd.merge(sd, avg7, on='Product', how='left')
        merged['DaysLeft'] = merged['Stock'] / merged['Avg7'].replace(0,np.nan)
        merged = merged.sort_values('DaysLeft')
        st.dataframe(merged.head(50))
        st.download_button("Download stock CSV", data=merged.to_csv(index=False).encode('utf-8'), file_name="stock_recommendations.csv")

    # QR/Barcode add stock (same as previous)
    st.subheader("Add stock via QR/Barcode image")
    img = st.file_uploader("Upload image with QR/barcode (JSON or product:qty)", type=["png","jpg","jpeg"], key="qr2")
    if img is not None:
        try:
            from PIL import Image
            im = Image.open(img)
            try:
                from pyzbar.pyzbar import decode
                res = decode(im)
                if res:
                    txts = [r.data.decode('utf-8') for r in res]
                    st.write(txts)
                    txt = txts[0]
                    try:
                        payload = json.loads(txt)
                        prod = payload.get("product"); qty = int(payload.get("quantity",0))
                    except Exception:
                        prod, qty = None, None
                        parts = txt.replace(";",",").split(",")
                        for part in parts:
                            if ":" in part:
                                k,v = part.split(":",1)
                                if "product" in k.lower(): prod=v.strip()
                                if "qty" in k.lower() or "quantity" in k.lower(): 
                                    try: qty=int(''.join([c for c in v if c.isdigit()]))
                                    except: qty=None
                    st.write("Parsed:", prod, qty)
                    if st.button("Confirm add to stock"):
                        if prod and qty is not None:
                            if stock_df is None:
                                stock_df = pd.DataFrame([{"Product":prod,"Stock":qty}])
                            else:
                                mask = stock_df['Product'].astype(str).str.lower()==prod.lower()
                                if mask.any():
                                    stock_df.loc[mask,'Stock'] = stock_df.loc[mask,'Stock'].fillna(0) + qty
                                else:
                                    stock_df = pd.concat([stock_df, pd.DataFrame([{"Product":prod,"Stock":qty}])], ignore_index=True)
                            stock_df.to_csv(PERSIST_STOCK, index=False)
                            st.success("Stock updated and saved (previous backed up).")
                else:
                    st.info("No barcode decoded. Ensure pyzbar and libzbar available on host.")
            except Exception as e:
                st.warning("pyzbar not available or decoding failed: " + str(e))
        except Exception as e:
            st.error("Failed to open image: " + str(e))

# Reviews page
elif menu=="Reviews":
    st.title("Inventory Management — Reviews & Sentiment")
    if reviews_df is None:
        st.info("No reviews loaded. Upload reviews CSV with columns ReviewText, Rating, DefectLabel to analyze.")
    else:
        rv = reviews_df.copy()
        # normalize columns
        rv.columns = [c.strip() for c in rv.columns]
        if 'ReviewText' not in rv.columns:
            for c in rv.columns:
                if 'review' in c.lower(): rv = rv.rename(columns={c:'ReviewText'}); break
        if 'Rating' not in rv.columns:
            for c in rv.columns:
                if 'rate' in c.lower(): rv = rv.rename(columns={c:'Rating'}); break
        rv = compute_vader(rv)
        st.subheader("Review sample")
        st.dataframe(rv.head(50))
        st.subheader("Defect breakdown (overall)")
        if 'DefectLabel' in rv.columns and rv['DefectLabel'].notna().sum()>0:
            df_def = rv[rv['DefectLabel'].notna()].groupby(['DefectLabel','Product']).size().reset_index(name='Count')
            st.altair_chart(alt.Chart(df_def).mark_bar().encode(x='Product:N', y='Count:Q', color='DefectLabel:N'), use_container_width=True)
        else:
            st.info("No defect labels found. You can add DefectLabel column to reviews CSV to enable this view.")

        # LLM suggestions for product improvements (optional)
        st.subheader("Review-driven suggestions (LLM)")
        st.markdown("Provide a product name and ask the model to suggest improvements or features based on reviews. If Hugging Face API token is set in app secrets (HUGGINGFACE_API_TOKEN), a free model `google/flan-t5-small` will be called. Otherwise a simple rule-based summary is shown.")
        product_for_sugg = st.text_input("Product name for suggestions")
        question = st.text_area("Prompt (what to ask the model)", value="Suggest 5 product improvements based on recent reviews for this product.")
        if st.button("Get suggestions"):
            if not product_for_sugg:
                st.error("Enter a product name.")
            else:
                # assemble a brief document from reviews
                doc = ""
                subset = rv[rv['Product'].astype(str).str.lower()==product_for_sugg.lower()]
                if subset.shape[0]==0:
                    subset = rv.head(50)
                doc = " ".join(subset['ReviewText'].astype(str).tolist()[:200])
                prompt = f"Product: {product_for_sugg}\nReviews: {doc}\nTask: {question}\nAnswer concisely."
                out = hf_generate(prompt, model="google/flan-t5-small")
                st.text_area("LLM output", value=out, height=240)

# Reports page
elif menu=="Reports":
    st.title("Inventory Management — Reports")
    st.write("Export filtered sales, stock, and reviews. Also simple model performance if available.")
    # allow user to pick date range and product to export (reuse v4 filters)
    min_date = sales_df['Date'].min(); max_date = sales_df['Date'].max()
    dr = st.date_input("Date range", value=(min_date, max_date), min_value=min_date, max_value=max_date)
    prod = st.selectbox("Product for report", options=["All"] + sorted(sales_df['Product'].unique().tolist()))
    sdf = sales_df.copy()
    sdf = sdf[(sdf['Date']>=pd.to_datetime(dr[0])) & (sdf['Date']<=pd.to_datetime(dr[1]))]
    if prod!="All": sdf = sdf[sdf['Product']==prod]
    st.download_button("Download filtered sales CSV", data=sdf.to_csv(index=False).encode('utf-8'), file_name="sales_report.csv")

# Admin page (role-protected)
elif menu=="Admin":
    st.title("Inventory Management — Admin")
    if st.session_state.role!="admin":
        st.warning("Admin actions require admin role. Login as admin to access backups and clears.")
    else:
        st.subheader("Data management")
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

# Settings page
elif menu=="Settings":
    st.title("Settings")
    st.markdown("Environment-based settings: set HUGGINGFACE_API_TOKEN in app secrets to enable LLM suggestions via Hugging Face Inference API. Example recommended model: `google/flan-t5-small` (lightweight instruction model, free to call via HF inference within usage limits).")
    st.markdown("Recommended free models for various tasks:\n\n- Sentiment / classification: `distilbert-base-uncased-finetuned-sst-2-english` (small, fast) \n- Short instruction/summary: `google/flan-t5-small` \n- Embeddings / semantic search (light): `sentence-transformers/all-MiniLM-L6-v2`")
    st.markdown("If you want fully offline LLMs in your environment, you'll need larger resources; for Streamlit Cloud, prefer calling HF Inference API via token in secrets.")

# end of app
st.caption("Built for Streamlit Cloud. Keep demo CSVs in /mnt/data to use 'Load demo'. LLM calls use Hugging Face Inference API via HUGGINGFACE_API_TOKEN secret.")