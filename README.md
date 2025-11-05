# Business Inventory Dashboard

This repository contains a Streamlit application for inventory forecasting, review-driven recommendations, and QR/barcode-assisted stock updates.

## Files included (generated)
- `streamlit_app.py` or `business_inventory_dashboard_secure.py` - your main Streamlit app. (Place your app file in the repo root)
- `.gitignore` - recommended git ignore rules for Streamlit/data projects
- `requirements.txt` - packages needed to run on Streamlit Cloud or Hugging Face Spaces
- `users.env` - sample credentials file (DO NOT commit to public repos; use secrets instead)
- `data/` - optional folder to include demo CSVs (if you want them preloaded)

## Quick deploy (Streamlit Cloud)
1. Rename your app to `streamlit_app.py` (or set the main file path in Streamlit Cloud to your app filename).
2. Push this repo to GitHub.
3. On Streamlit Cloud, click **New app** → connect your GitHub repo → choose branch `main` and set main file to `streamlit_app.py`.
4. In the Streamlit app settings, add the secret `BIM_USERS_JSON` with your user JSON, for example:
```
{"admin":{"password":"admin","role":"admin"},"bhoomikakm2004@gmail.com":{"password":"admin","role":"user"}}
```
5. Deploy. If package installation fails, check app logs and remove problematic packages from `requirements.txt` (e.g., `pyzbar` if it fails to build).

## Security
- Do **not** commit real credentials. Use Streamlit Secrets, GitHub Secrets, or environment variables.
- `users.env` is provided for local testing only.

## Notes & Tips
- NLTK VADER lexicon is downloaded at runtime if NLTK is available. Make sure `nltk` is in `requirements.txt`.
- If you need barcode scanning on hosted platforms, you may need system packages (`libzbar`) — this often requires a Docker-based deployment.
- If you experience `np.float_` errors, ensure `numpy<2.0` is installed (pinned in `requirements.txt`).

## Support
If you want, I can create a ready-to-push zip containing your app file + these files. Reply **"make zip"** and I'll package everything into `inventory_dashboard_repo.zip` for download.
