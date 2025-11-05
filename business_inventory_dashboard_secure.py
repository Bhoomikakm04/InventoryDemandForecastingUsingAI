"""
Business Inventory Management Dashboard (Secure + Features)
- Role-based Admin access (only users with role 'admin' see Admin tab)
- Credentials read from environment or a .env/users file (more secure)
- Auto-download NLTK VADER lexicon if nltk is installed
- Defect breakdown widget on Dashboard
- Keep previous database (backups) when new upload happens
- QR/Barcode image upload to decode product info and add stock entries
Notes:
 - This app expects basic packages to be installed in the environment:
   streamlit, pandas, numpy, altair, python-dotenv (optional), pillow, pyzbar (optional), nltk (optional)
 - If packages are missing, the app will degrade gracefully and show instructions.
"""

import streamlit as st
import pandas as pd, numpy as np, os, io, datetime, json
import altair as alt
from pathlib import Path

st.set_page_config(page_title="Business Inventory Management (Secure)", layout="wide")

# --------------------------- Credentials / Users ---------------------------
# Loading users from environment variable or file to avoid hardcoding in code.
# Expected format (env var USERS_JSON or file users.env):
# USERS_JSON='{"admin":{"password":"admin","role":"admin"}, "user1":{"password":"pwd","role":"user"}}'

DEFAULT_USERS = {"admin": {"password": "admin", "role": "admin"},
                 "bhoomikakm2004@gmail.com": {"password":"admin", "role":"user"}}

def load_users():
    # 1) try environment variable USERS_JSON
    import os, json
    users_json = os.environ.get("BIM_USERS_JSON")
    if users_json:
        try:
            return json.loads(users_json)
        except Exception:
            pass
    # 2) try a file path from BIM_USERS_FILE or default /mnt/data/users.env
    users_file = os.environ.get("BIM_USERS_FILE", "/mnt/data/users.env")
    if os.path.exists(users_file):
        try:
            text = Path(users_file).read_text(encoding="utf-8")
            text = text.strip()
            # file may contain JSON or key=value lines
            try:
                return json.loads(text)
            except Exception:
                users = {}
                for line in text.splitlines():
                    line = line.strip()
                    if not line or line.startswith("#"): continue
                    # expect username:password:role  or username=password,role=json
                    if ":" in line:
                        parts = line.split(":")
                        if len(parts) >= 2:
                            username = parts[0].strip()
                            password = parts[1].strip()
                            role = parts[2].strip() if len(parts)>=3 else "user"
                            users[username] = {"password":password, "role":role}
                if users:
                    return users
        except Exception:
            pass
    # fallback
    return DEFAULT_USERS

USERS = load_users()

# --------------------------- Session: login ---------------------------
if 'auth' not in st.session_state:
    st.session_state.auth = False
    st.session_state.user = None
    st.session_state.role = None

def login(username, password):
    u = USERS.get(username)
    if u and u.get("password") == password:
        st.session_state.auth = True
        st.session_state.user = username
        st.session_state.role = u.get("role","user")
        return True
    return False

def logout():
    st.session_state.auth = False
    st.session_state.user = None
    st.session_state.role = None

if not st.session_state.auth:
    st.title("Login — Business Inventory Management (Secure)")
    c1, c2 = st.columns([2,1])
    with c1:
        username = st.text_input("Username or email")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            ok = login(username.strip(), password.strip())
            if not ok:
                st.error("Invalid username or password. Check BIM_USERS_JSON environment variable or /mnt/data/users.env file.")
            else:
                st.experimental_rerun()
    with c2:
        st.info("Provide credentials via environment variable BIM_USERS_JSON or file /mnt/data/users.env for secure storage. Default demo users available.")
    st.stop()

# --------------------------- Paths and demo files ---------------------------
DATA_DIR = Path("/mnt/data")
DATA_DIR.mkdir(parents=True, exist_ok=True)
DEMO_SALES = DATA_DIR / "demo_sales_dataset.csv"
DEMO_STOCK = DATA_DIR / "demo_stock_dataset.csv"
DEMO_REVIEWS = DATA_DIR / "unstructured_reviews_demo_multilingual.csv"

# files used by app (persistent)
SALES_FILE = DATA_DIR / "sales_current.csv"    # single source of truth saved when user uploads or loads demo
STOCK_FILE = DATA_DIR / "stock_current.csv"
REVIEWS_FILE = DATA_DIR / "reviews_current.csv"

BACKUP_DIR = DATA_DIR / "backups"
BACKUP_DIR.mkdir(exist_ok=True)

# --------------------------- Utilities ---------------------------
def safe_read_csv(path_or_buf):
    try:
        if hasattr(path_or_buf, "read"):
            return pd.read_csv(path_or_buf), None
        else:
            return pd.read_csv(path_or_buf), None
    except Exception as e:
        try:
            return pd.read_csv(path_or_buf, encoding="ISO-8859-1"), "ISO-8859-1"
        except Exception as e2:
            return None, str(e2)

def backup_file(src: Path):
    if src.exists():
        ts = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        dest = BACKUP_DIR / f"{src.stem}_{ts}{src.suffix}"
        src.replace(dest)  # move current to backup
        return dest
    return None

def save_df_atomic(df: pd.DataFrame, dest: Path):
    # move existing to backup first
    backup_file(dest)
    tmp = dest.with_suffix(".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(dest)
    return dest

# --------------------------- Top controls ---------------------------
st.markdown(f"### Business Inventory Management — Logged in as `{st.session_state.user}` (role: {st.session_state.role})")
if st.button("Logout"):
    logout()
    st.experimental_rerun()

cols = st.columns([2,2,1,1])
with cols[0]:
    uploaded_sales = st.file_uploader("Upload Sales CSV (will replace current)", type=["csv"])
with cols[1]:
    uploaded_stock = st.file_uploader("Upload Stock CSV (optional)", type=["csv"])
with cols[2]:
    if st.button("Load demo data (preserve current DB)"):
        # only copy demo files into current if they exist; preserve existing by backing up first
        if DEMO_SALES.exists():
            df,_ = safe_read_csv(DEMO_SALES)
            save_df_atomic(df, SALES_FILE)
        if DEMO_STOCK.exists():
            df,_ = safe_read_csv(DEMO_STOCK)
            save_df_atomic(df, STOCK_FILE)
        if DEMO_REVIEWS.exists():
            df,_ = safe_read_csv(DEMO_REVIEWS)
            save_df_atomic(df, REVIEWS_FILE)
        st.success("Demo datasets copied to current DB (previous data backed up).")
with cols[3]:
    if st.button("Backup current DB now"):
        for p in [SALES_FILE, STOCK_FILE, REVIEWS_FILE]:
            if p.exists():
                dest = backup_file(p)
        st.success("Backups created.")

# If user uploads new files, persist them and backup old ones
if uploaded_sales is not None:
    df,enc = safe_read_csv(uploaded_sales)
    if df is None:
        st.error("Could not read uploaded sales file.")
    else:
        saved = save_df_atomic(df, SALES_FILE)
        st.success(f"Uploaded sales saved to {saved}; previous file backed up.")

if uploaded_stock is not None:
    df,enc = safe_read_csv(uploaded_stock)
    if df is None:
        st.error("Could not read uploaded stock file.")
    else:
        saved = save_df_atomic(df, STOCK_FILE)
        st.success(f"Uploaded stock saved to {saved}; previous file backed up.")

# Load current datasets (prefer persisted current files)
sales_df = None
stock_df = None
reviews_df = None
if SALES_FILE.exists():
    sales_df,_ = safe_read_csv(SALES_FILE)
elif DEMO_SALES.exists():
    sales_df,_ = safe_read_csv(DEMO_SALES)
if STOCK_FILE.exists():
    stock_df,_ = safe_read_csv(STOCK_FILE)
elif DEMO_STOCK.exists():
    stock_df,_ = safe_read_csv(DEMO_STOCK)
if REVIEWS_FILE.exists():
    reviews_df,_ = safe_read_csv(REVIEWS_FILE)
elif DEMO_REVIEWS.exists():
    reviews_df,_ = safe_read_csv(DEMO_REVIEWS)

if sales_df is None:
    st.error("No sales data available. Upload or load demo to continue.")
    st.stop()

# normalize and ensure columns
sales_df.columns = [c.strip() for c in sales_df.columns]
if "Date" not in sales_df.columns:
    for c in sales_df.columns:
        if "date" in c.lower():
            sales_df = sales_df.rename(columns={c:"Date"}); break
sales_df["Date"] = pd.to_datetime(sales_df["Date"], errors="coerce")
if "Product" not in sales_df.columns:
    sales_df["Product"] = "ALL_PRODUCTS"
if "Sales" not in sales_df.columns:
    numcols = sales_df.select_dtypes(include=[np.number]).columns.tolist()
    if numcols:
        sales_df = sales_df.rename(columns={numcols[0]:"Sales"})
    else:
        st.error("Sales data must include numeric sales column.")
        st.stop()
sales_df["Sales"] = pd.to_numeric(sales_df["Sales"], errors="coerce").fillna(0)
if "Country" not in sales_df.columns:
    sales_df["Country"] = "All"

# reviews normalization
if reviews_df is not None:
    reviews_df.columns = [c.strip() for c in reviews_df.columns]
    if "ReviewText" not in reviews_df.columns and "Review" in reviews_df.columns:
        reviews_df = reviews_df.rename(columns={"Review":"ReviewText"})
    if "Rating" not in reviews_df.columns:
        # try to infer
        ratings = [c for c in reviews_df.columns if "rate" in c.lower()]
        if ratings:
            reviews_df = reviews_df.rename(columns={ratings[0]:"Rating"})
    # derive VADER later

# --------------------------- Top tabs (upper dashboard) ---------------------------
tab_dashboard, tab_reviews, tab_admin = st.tabs(["Dashboard","Reviews","Admin"])

# --------------------------- Dashboard content ---------------------------
with tab_dashboard:
    st.header("Dashboard — Sales & Defect Overview")
    # country filter and product selector at top of tab
    countries = ["All"] + sorted(sales_df["Country"].dropna().unique().tolist())
    sel_country = st.selectbox("Country filter", countries, index=0, key="dash_country")
    view = sales_df if sel_country=="All" else sales_df[sales_df["Country"]==sel_country].copy()
    top_n = st.number_input("Top N products to show", min_value=1, max_value=50, value=10, key="dash_topn")
    prod_list = view.groupby("Product")["Sales"].sum().reset_index().sort_values("Sales", ascending=False).head(top_n)["Product"].tolist()
    sel_product = st.selectbox("Select product", prod_list, key="dash_product")

    # timeseries and kpis
    prod_ts = view[view["Product"]==sel_product].groupby("Date")["Sales"].sum().reset_index().sort_values("Date")
    col1,col2,col3 = st.columns(3)
    col1.metric("Total sales", f"{int(prod_ts['Sales'].sum()):,}")
    col2.metric("Avg daily", f"{prod_ts['Sales'].mean():.2f}")
    # defect breakdown widget (aggregate from reviews if available)
    if reviews_df is not None and "DefectLabel" in reviews_df.columns and reviews_df["DefectLabel"].notna().sum()>0:
        st.subheader("Defect breakdown (dashboard)")
        df_def = reviews_df[reviews_df["Product"]==sel_product]
        df_def = df_def[df_def["DefectLabel"].notna()]
        if not df_def.empty:
            defect_counts = df_def["DefectLabel"].value_counts().reset_index()
            defect_counts.columns = ["Defect","Count"]
            bar = alt.Chart(defect_counts).mark_bar().encode(x="Defect:N", y="Count:Q", tooltip=["Defect","Count"])
            st.altair_chart(bar, use_container_width=True)
        else:
            st.info("No defect labels for this product.")
    else:
        st.info("No defect labels available to show on dashboard. Upload reviews with DefectLabel to enable this widget.")

    # sales + simple forecast
    st.subheader("Sales time series & simple forecast")
    recent_avg = prod_ts['Sales'].tail(30).mean() if len(prod_ts)>=7 else prod_ts['Sales'].mean()
    last_date = prod_ts['Date'].max()
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=30, freq='D')
    fc = pd.DataFrame([{"Date":d, "Forecast": recent_avg} for d in future_dates])
    plot_df = pd.concat([prod_ts.rename(columns={"Date":"Date"}), fc], ignore_index=True, sort=False).sort_values("Date")
    base = alt.Chart(plot_df).encode(x='Date:T')
    sales_line = base.mark_line().encode(y=alt.Y('Sales:Q', title='Sales'))
    fc_line = base.mark_line(color='orange').encode(y='Forecast:Q')
    st.altair_chart((sales_line + fc_line).properties(width=900, height=350), use_container_width=True)

    # show recommendations (simple rule-based)
    st.subheader("Inventory recommendation (simple)")
    cur_stock = None
    if stock_df is not None:
        # try to find row
        s = stock_df.copy()
        for c in s.columns:
            if "product" in c.lower(): s = s.rename(columns={c:"Product"})
            if "stock" in c.lower(): s = s.rename(columns={c:"Stock"})
        match = s[s["Product"].astype(str).str.lower()==sel_product.lower()]
        if not match.empty and "Stock" in match.columns:
            cur_stock = float(pd.to_numeric(match["Stock"].iloc[0], errors="coerce"))
    est_demand_30 = recent_avg * 30
    if cur_stock is None:
        st.info("No stock data available for recommendation.")
    else:
        if cur_stock < est_demand_30 * 0.9:
            st.warning("Recommendation: Stock Up")
        elif cur_stock > est_demand_30 * 1.5:
            st.info("Recommendation: Reduce/Discount")
        else:
            st.success("Recommendation: Hold")

    # QR/Barcode scanner widget: upload image and decode
    st.subheader("Add stock via QR/Barcode scan")
    st.write("Upload an image of a barcode/QR code that encodes JSON or `product` and `quantity`. Example JSON: {\"product\":\"Widget A\",\"quantity\":5}")
    img_file = st.file_uploader("Upload QR/Barcode image", type=["png","jpg","jpeg"], key="qr_upload")
    if img_file is not None:
        try:
            from PIL import Image
            img = Image.open(img_file)
            decoded = []
            try:
                from pyzbar.pyzbar import decode
                results = decode(img)
                for r in results:
                    txt = r.data.decode("utf-8")
                    decoded.append(txt)
            except Exception as e:
                st.warning("pyzbar not installed or failed to decode. Install pyzbar and pillow in the environment to enable decoding.")
            if decoded:
                st.write("Decoded values:")
                for d in decoded:
                    st.code(d)
                # try to parse first decoded value as JSON or simple "product:qty"
                val = decoded[0]
                try:
                    payload = json.loads(val)
                    product = payload.get("product") or payload.get("Product")
                    qty = int(payload.get("quantity") or payload.get("qty") or 0)
                except Exception:
                    # try key:value pairs split by comma or semicolon
                    product = None; qty = None
                    parts = [p.strip() for p in val.replace(";",",").split(",")]
                    for part in parts:
                        if ":" in part:
                            k,v = part.split(":",1)
                            if "product" in k.lower(): product = v.strip()
                            if "qty" in k.lower() or "quantity" in k.lower(): 
                                try: qty = int(''.join(ch for ch in v if ch.isdigit()))
                                except: qty = None
                st.write("Parsed -> product:", product, " quantity:", qty)
                if st.button("Add to stock database"):
                    if product and qty is not None:
                        # upsert into stock dataframe and save
                        if stock_df is None:
                            stock_df = pd.DataFrame([{"Product":product, "Stock":qty}])
                        else:
                            # find row by product (case-insensitive)
                            mask = stock_df["Product"].astype(str).str.lower()==str(product).lower()
                            if mask.any():
                                stock_df.loc[mask, "Stock"] = stock_df.loc[mask, "Stock"].fillna(0) + qty
                            else:
                                stock_df = pd.concat([stock_df, pd.DataFrame([{"Product":product, "Stock":qty}])], ignore_index=True)
                        save_df_atomic(stock_df, STOCK_FILE)
                        st.success(f"Added {qty} x {product} to stock and saved. Previous stock backed up.")
                    else:
                        st.error("Could not parse product or quantity from the decoded value. Ensure QR encodes JSON {product,quantity} or 'product:... ,qty:...'")
            else:
                st.info("No QR/barcode decoded. If pyzbar is not available, please install it or provide product and quantity manually below.")
        except Exception as e:
            st.error(f"Failed processing image: {e}")

    # manual add stock fallback
    with st.expander("Manual add stock"):
        p_name = st.text_input("Product name (manual)")
        p_qty = st.number_input("Quantity", min_value=1, value=1)
        if st.button("Add manually"):
            if p_name:
                if stock_df is None:
                    stock_df = pd.DataFrame([{"Product":p_name, "Stock":int(p_qty)}])
                else:
                    mask = stock_df["Product"].astype(str).str.lower()==p_name.lower()
                    if mask.any():
                        stock_df.loc[mask, "Stock"] = stock_df.loc[mask, "Stock"].fillna(0) + int(p_qty)
                    else:
                        stock_df = pd.concat([stock_df, pd.DataFrame([{"Product":p_name, "Stock":int(p_qty)}])], ignore_index=True)
                save_df_atomic(stock_df, STOCK_FILE)
                st.success(f"Added {p_qty} x {p_name} to stock and saved. Previous stock backed up.")

# --------------------------- Reviews tab ---------------------------
with tab_reviews:
    st.header("Reviews & Sentiment")
    if reviews_df is None:
        st.info("No reviews available. Upload reviews CSV (with columns ReviewText, Rating, DefectLabel) to enable sentiment and defect analytics.")
    else:
        # attempt to use VADER (download lexicon if nltk installed)
        try:
            import nltk
            from nltk.sentiment.vader import SentimentIntensityAnalyzer
            try:
                nltk.data.find("sentiment/vader_lexicon.zip")
            except Exception:
                with st.spinner("Downloading VADER lexicon..."):
                    nltk.download("vader_lexicon")
            sia = SentimentIntensityAnalyzer()
            reviews_df["VADER_compound"] = reviews_df["ReviewText"].astype(str).apply(lambda t: sia.polarity_scores(t)["compound"])
            st.success("Computed VADER sentiment.")
        except Exception as e:
            st.warning("VADER unavailable: " + str(e) + " — falling back to rating-derived sentiment.")
            if "Rating" in reviews_df.columns:
                reviews_df["VADER_compound"] = reviews_df["Rating"].apply(lambda r: (r-3)/2)
            else:
                reviews_df["VADER_compound"] = 0.0

        st.subheader("Per-product summary")
        summary = reviews_df.groupby("Product").agg(Reviews=("ReviewID","count") if "ReviewID" in reviews_df.columns else ("Rating","count"),
                                                   AvgRating=("Rating","mean") if "Rating" in reviews_df.columns else ("VADER_compound","mean"),
                                                   AvgSentiment=("VADER_compound","mean"),
                                                   DefectReports=("DefectLabel", lambda s: s.notna().sum() if "DefectLabel" in reviews_df.columns else 0)).reset_index()
        st.dataframe(summary.style.format({"AvgRating":"{:.2f}", "AvgSentiment":"{:.2f}"}))

        # defect charts
        st.subheader("Defect breakdown (overall)")
        if "DefectLabel" in reviews_df.columns and reviews_df["DefectLabel"].notna().sum()>0:
            defects = reviews_df[reviews_df["DefectLabel"].notna()].groupby(["DefectLabel","Product"]).size().reset_index(name="Count")
            bar = alt.Chart(defects).mark_bar().encode(x=alt.X("Product:N", sort='-y'), y="Count:Q", color="DefectLabel:N", tooltip=["Product","DefectLabel","Count"])
            st.altair_chart(bar, use_container_width=True)
            pie = alt.Chart(reviews_df[reviews_df["DefectLabel"].notna()]).mark_arc().encode(theta="count()", color="DefectLabel:N", tooltip=["DefectLabel","count()"])
            st.altair_chart(pie, use_container_width=True)
        else:
            st.info("No defect labels present in reviews dataset.")

# --------------------------- Admin tab (role-protected) ---------------------------
with tab_admin:
    st.header("Admin")
    if st.session_state.role != "admin":
        st.warning("Admin tab is visible only to users with role 'admin'. You do not have permission to perform admin actions.")
    else:
        st.subheader("Administrative actions")
        if SALES_FILE.exists():
            st.download_button("Download current sales dataset", data=SALES_FILE.read_bytes(), file_name=SALES_FILE.name)
        if STOCK_FILE.exists():
            st.download_button("Download current stock dataset", data=STOCK_FILE.read_bytes(), file_name=STOCK_FILE.name)
        if REVIEWS_FILE.exists():
            st.download_button("Download current reviews dataset", data=REVIEWS_FILE.read_bytes(), file_name=REVIEWS_FILE.name)
        if st.button("Create immediate backup of all data"):
            for p in [SALES_FILE, STOCK_FILE, REVIEWS_FILE]:
                if p.exists(): backup_file(p)
            st.success("Backups created.")
        if st.button("Clear all persisted current datasets (keep backups)"):
            for p in [SALES_FILE, STOCK_FILE, REVIEWS_FILE]:
                if p.exists(): p.unlink()
            st.success("Current datasets cleared. Backups remain in /mnt/data/backups.")

st.caption("Security notes: credentials are read from BIM_USERS_JSON environment variable or /mnt/data/users.env file if present; otherwise defaults used. For production, provide secure secrets via environment variables or secret manager.")