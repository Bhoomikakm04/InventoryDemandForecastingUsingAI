Fixed Streamlit app that uses a writable temporary directory on hosted platforms (Streamlit Cloud).
Files created:
- streamlit_app.py
- streamlit_app_inventory_management_fixed.py
- requirements.txt
Notes:
- Add HUGGINGFACE_API_TOKEN to app secrets to enable LLM suggestions.
- Do NOT write to /mnt/data on Streamlit Cloud; this app writes to tempfile.gettempdir().
