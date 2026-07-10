# End-to-End Master Analytics Dashboard

A Streamlit app covering: Descriptive & Diagnostic Analysis, Anomaly Detection,
RFM & Clustering (K-Means + Dendrogram + 3D view), Classification, Regression,
Association Rule Mining, and Prescriptive Recommendations.

## Files
- `app.py` — the full dashboard (all 7 tabs)
- `requirements.txt` — Python packages needed
- `test_logic.py` / `test_app.py` — optional smoke tests (not required to deploy)

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```
Then open the local URL it prints (usually http://localhost:8501) and upload
your `.xlsx` or `.csv` file from the sidebar.

## Deploy on Streamlit Community Cloud (share.streamlit.io)
1. Create a new **public GitHub repository** (e.g. `disposable-socks-dashboard`).
2. Upload `app.py` and `requirements.txt` to the **root** of that repo
   (not inside a subfolder — Streamlit Cloud looks for `app.py` at the path you specify).
3. Go to https://share.streamlit.io → **New app**.
4. Pick your repo, branch (`main`), and set **Main file path** to `app.py`.
5. Click **Deploy**. The first build takes 1–3 minutes while it installs
   `requirements.txt`.
6. Once live, open the app's public URL and upload your dataset from the sidebar
   — the data itself is never committed to GitHub, it's uploaded at runtime.

## Common reasons a Streamlit Cloud deploy fails
- **`app.py` not found** — make sure it's at the repo root (or set the correct
  "Main file path" if it's in a subfolder).
- **Missing package errors** — every import in `app.py` must have a matching
  line in `requirements.txt` (already handled here).
- **File-size / import errors on upload** — this app expects `.xlsx`, `.xls`,
  or `.csv`; very large files (tens of MB) can be slow on the free tier.
- **Blank tabs** — most tabs need you to first select columns (features,
  target, item columns) from the dropdowns/multiselects inside that tab —
  they don't auto-run until you do.

## Notes on your data
Some tabs expect specific column types:
- **Classification** needs a column with exactly 2 unique values to use as the target.
- **Association Rule Mining** needs 2+ item columns that are binary/Yes-No flags
  (e.g. `Gym_Bag`, `Towel`, `Pre_Workout`, `Disposable_Socks`).
- **RFM** fields are optional column mappings — pick whichever columns in your
  file represent Recency / Frequency / Monetary, or skip any you don't have.

If a tab shows a warning instead of a chart, it's telling you which column
type it's missing — pick the right columns in that tab and it will render.
